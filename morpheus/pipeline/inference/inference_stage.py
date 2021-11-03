# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import queue
import threading
import typing
from abc import abstractmethod
from concurrent.futures import thread
from functools import reduce

import cupy as cp
import typing_utils
from tornado.ioloop import IOLoop

from morpheus.config import Config
from morpheus.pipeline.messages import MultiInferenceMessage
from morpheus.pipeline.messages import MultiResponseProbsMessage
from morpheus.pipeline.messages import ResponseMemory
from morpheus.pipeline.messages import ResponseMemoryProbs
from morpheus.pipeline.pipeline import MultiMessageStage
from morpheus.pipeline.pipeline import StreamFuture
from morpheus.pipeline.pipeline import StreamPair
from morpheus.utils.producer_consumer_queue import Closed
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue


class InferenceWorker:
    def __init__(self, inf_queue: ProducerConsumerQueue) -> None:
        self._loop = IOLoop.current()
        self._thread: threading.Thread = None
        self._inf_queue = inf_queue

        # Create the complete event here to make sure its on the same thread as the loop
        self._complete_event = asyncio.Event()

    async def start(self):
        ready_event = asyncio.Event()

        self._thread = threading.Thread(
            target=self.main_loop,
            # daemon=True,
            name="Inference Thread",
            args=(ready_event, ))
        self._thread.start()

        await ready_event.wait()

    def init(self):
        # Nothing required in base
        pass

    def stop(self):
        pass

    async def join(self):
        await self._complete_event.wait()

        # Join here just to be safe. Should be closed
        self._thread.join()

    @abstractmethod
    def process(self, batch: MultiInferenceMessage, fut: asyncio.Future):
        pass

    def main_loop(self, ready_event: asyncio.Event = None):
        """
        This function consumes messages (of type `MultiInferenceMessage`) posted to the internal queue and
        initiate calls to Triton inference server calls.

        Parameters
        ----------
        loop : IOLoop
            Loop to send the response generated by future requests
        inf_queue : ProducerConsumerQueue
            Internal queue used as middleware to consume messages by multi threaded TritonInference
        ready_event : asyncio.Event
            ready_event

        """
        self.init()

        if (ready_event is not None):
            self._loop.asyncio_loop.call_soon_threadsafe(ready_event.set)

        # Run while the queue is open
        while not self._inf_queue.is_closed():

            try:
                # Get the next work item
                message: typing.Tuple[MultiInferenceMessage, asyncio.Future] = self._inf_queue.get(block=True)

                batch = message[0]
                fut = message[1]

                # Mark the task as complete when the future is resolved
                fut.add_done_callback(lambda _: self._inf_queue.task_done())

                self.process(batch, fut)
            except Closed:  # Catch closed exceptions
                continue

        self._loop.asyncio_loop.call_soon_threadsafe(self._complete_event.set)


class InferenceStage(MultiMessageStage):
    """
    This class serves as the base for any inference stage. Inference stages operate differently than other
    stages due to the fact that they operate in a separate thread and have their own batch size which is
    separate from the pipeline batch size. Processing the inference work in a separate thread is necessary to
    support inference types that may require exclusive use of a single thread (i.e. TensorRT) without blocking
    the main asyncio thread.

    Changing batch sizes for the inference stage requires breaking messages into smaller slices, running
    inference on the smaller slices, then recombining the inference output into the original batch size. This
    inference base class handles breaking and recombining batches and queing the inference work to be
    processed on another thread.

    Breaking and recombining batches is processed normally on the asyncio Main Thread, while individual
    inference slices are added to a queue which will be processed by the dedicated inference thread. Each
    individual inference slice is added to the queue as a `(asyncio.Future, morpheus.MultiInferenceMessage)`
    tuple. Once the inference thread has completed a `~morpheus.pipeline.messages.MultiInferenceMessage`, it
    signals the work is complete using the `asyncio.Future` object. The inference stage waits for all
    `asyncio.Future` objects in a pipeline batch to be complete before recombining the inference slices into
    larger batchs.

    Inference stages that derive from this class need to only implement the `_get_inference_fn` method. The
    `_get_inference_fn()` should return a callable with the signature `typing.Callable[[tornado.IOLoop,
    asyncio.Queue, asyncio.Event], None]`. The structure of most inference functions will follow the form:

    #.  Perform initialization

        #.  Once complete, signal to the pipeline that the inference stage is ready with:

            .. code-block:: python

                loop.asyncio_loop.call_soon_threadsafe(ready_event.set)

    #.  While the pipeline is running:

        #.  Get an item of work from the `asyncio.Queue`. Block if the queue is empty

            #.  Each item of work will contain a tuple in the form:

                .. code-block:: python

                    (asyncio.Future, morpheus.MultiInferenceMessage)

        #.  Process the `~morpheus.pipeline.messages.MultiInferenceMessage`

            #.  This will be unique to each `InferenceStage`

        #.  Signal the completion of the item of work

            #.  Successful completion or an exception is signaled using the `asyncio.Future` object in the
                unit of work tuple
            #.  Exceptions can be set with :py:meth:`asyncio.Future.set_exception`
            #.  Successful results can be set with :py:meth:`asyncio.Future.set_result`
            #.  Keep in mind that these are `asyncio.Future` objects, not `concurrent.futures.Future`. Setting
                exceptions or results must be done from the Asyncio Main Thread. Therefore, calling
                :py:meth:`~asyncio.Future.set_exception`/:py:meth:`~asyncio.Future.set_result` should be
                queued on the Asyncio Main Thread using:

                .. code-block:: python

                    loop.add_callback(future.set_exception, my_exception)
                    # OR
                    loop.add_callback(future.set_result, my_result)

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance

    """
    def __init__(self, c: Config):
        super().__init__(c)

        self._fea_length = c.feature_length

        self._workers: typing.List[InferenceWorker] = []
        self._inf_queue = ProducerConsumerQueue()

        self._max_batch_size = c.model_max_batch_size

        # Mark these stages to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        return "inference"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types to this stage.

        Returns
        -------
        typing.Tuple(messages.MultiInferenceMessage, StreamFuture[messages.MultiInferenceMessage])
            Accepted input types

        """
        return (MultiInferenceMessage, StreamFuture[MultiInferenceMessage])

    @abstractmethod
    def _get_inference_worker(self, inf_queue: ProducerConsumerQueue) -> InferenceWorker:
        """
        Returns the main inference function to be run in a separate thread.

        :meta public:

        Returns
        -------
            typing.Callable:
                Callable function that takes parameters ``(tornado.IOLoop, asyncio.Queue, asyncio.Event)``
        """
        pass

    def _build_single(self, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]
        out_type = MultiResponseProbsMessage

        if (typing_utils.issubtype(input_stream[1], StreamFuture)):
            # First convert to manageable batches. If the batches are too large, we cant process them
            stream = stream.map(self._split_batches, max_batch_size=self._max_batch_size).gather()

            # Queue the inference work
            stream = stream.async_map(self._queue_inf_work)

            stream = stream.scatter().map(self._convert_response)
            out_type = StreamFuture[MultiResponseProbsMessage]
        else:
            # First convert to manageable batches. If the batches are too large, we cant process them
            stream = stream.async_map(self._split_batches,
                                      executor=self._pipeline.thread_pool,
                                      max_batch_size=self._max_batch_size)

            # Queue the inference work
            stream = stream.async_map(self._queue_inf_work)

            stream = stream.async_map(self._convert_response, executor=self._pipeline.thread_pool)

        return stream, out_type

    async def _start(self):

        wait_events = []

        for _ in range(1):

            self._workers.append(self._get_inference_worker(self._inf_queue))

            # Wait for the inference thread to be ready
            wait_events.append(self._workers[-1].start())

        await asyncio.gather(*wait_events, return_exceptions=True)

        return await super()._start()

    async def stop(self):
        for w in self._workers:
            w.stop()

        # Now stop the _inf_queue to unblock workers
        self._inf_queue.close()

    async def join(self):

        # Wait for queue to be finished. This does block but it should be fine for now
        self._inf_queue.join()

        # Join all workers
        for w in self._workers:
            await w.join()

        return await super().join()

    @staticmethod
    def _split_batches(x: MultiInferenceMessage, max_batch_size: int) -> typing.List[MultiInferenceMessage]:

        out_batches = []

        id_array = cp.concatenate([cp.array([-1]), x.seq_ids[:, 0], cp.array([-1])])

        diff_ids = cp.where(id_array[1:] != id_array[:-1])[0]

        diff_ids = diff_ids.tolist()

        head = 0
        tail = 0

        for i in range(1, len(diff_ids)):

            poss_count = diff_ids[i] - diff_ids[head]

            if (poss_count > max_batch_size):
                out_batches.append((diff_ids[head], diff_ids[tail]))

                head = tail

            tail = i

        out_batches.append((diff_ids[head], diff_ids[tail]))

        out_resp = []

        for start, stop in out_batches:

            out_resp.append(x.get_slice(start, stop))

        assert len(out_resp) > 0

        return out_resp

    async def _queue_inf_work(self, x: typing.List[MultiInferenceMessage]):
        # Get the current event loop.
        loop = asyncio.get_running_loop()

        futures = []

        for y in x:
            fut = loop.create_future()

            self._inf_queue.put((y, fut))

            futures.append(fut)

        res = await asyncio.gather(*futures, return_exceptions=True)

        return x, res

    @staticmethod
    def _convert_response(x: typing.Tuple[typing.List[MultiInferenceMessage], typing.List[ResponseMemoryProbs]]):

        # Convert a MultiResponse into a MultiResponseProbsMessage
        in_message = x[0]
        out_message = x[1]

        assert len(in_message) == len(out_message)

        # Get the total output size
        total_mess_count = reduce(lambda y, z: y + z.mess_count, in_message, 0)

        # Create a message data to store the entire list
        memory = ResponseMemoryProbs(count=total_mess_count,
                                     probs=cp.zeros((total_mess_count, out_message[0].probs.shape[1])))

        saved_meta = in_message[0].meta
        saved_offset = in_message[0].mess_offset
        saved_count = 0

        for inf, res in zip(in_message, out_message):

            # Ensure they all share the same meta object. Otherwise this doesnt work
            # assert inf.meta is saved_meta

            # Make sure we have a continuous list
            assert inf.mess_offset == saved_offset + saved_count

            # Two scenarios:
            if (inf.mess_count == inf.count):
                # In message and out message have same count. Just use probs as is
                memory.probs[inf.mess_offset:inf.mess_offset + inf.mess_count, :] = res.probs
            else:
                assert inf.count == res.count

                mess_ids = inf.seq_ids[:, 0].get().tolist()

                # Out message has more reponses, so we have to do key based blending of probs
                for i, idx in enumerate(mess_ids):
                    memory.probs[idx, :] = cp.maximum(memory.probs[idx, :], res.probs[i, :])

            saved_count += inf.mess_count

        # For now, we assume that this is the whole group of messages so it must start at 0
        assert saved_offset == 0

        return MultiResponseProbsMessage(meta=saved_meta,
                                         mess_offset=saved_offset,
                                         mess_count=saved_count,
                                         memory=memory,
                                         offset=0,
                                         count=memory.count)
