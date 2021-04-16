import asyncio
import queue
import typing

import cupy as cp
from tornado.ioloop import IOLoop

from morpheus.config import Config
from morpheus.pipeline.inference.inference_stage import InferenceStage
from morpheus.pipeline.messages import MultiInferenceMessage
from morpheus.pipeline.messages import ResponseMemory


# This class is exclusively run in the worker thread. Separating the classes helps keeps the threads separate
class IdentityInference:
    def __init__(self, c: Config):

        self._max_batch_size = c.model_max_batch_size
        self._seq_length = c.model_seq_length

    def init(self, loop: IOLoop):

        self._loop = loop

    def process(self, batch: MultiInferenceMessage, fut: asyncio.Future):

        def tmp(b: MultiInferenceMessage, f):
            
            f.set_result(ResponseMemory(
                count=b.count,
                probs=cp.zeros((b.count, 10), dtype=cp.float32),
            ))

        self._loop.add_callback(tmp, batch, fut)

    def main_loop(self, loop: IOLoop, inf_queue: queue.Queue, ready_event: asyncio.Event = None):

        self.init(loop)

        if (ready_event is not None):
            loop.asyncio_loop.call_soon_threadsafe(ready_event.set)

        while True:

            # Get the next work item
            message: typing.Tuple[MultiInferenceMessage, asyncio.Future] = inf_queue.get(block=True)

            batch = message[0]
            fut = message[1]

            self.process(batch, fut)


class IdentityInferenceStage(InferenceStage):
    def __init__(self, c: Config):
        super().__init__(c)

    def _get_inference_fn(self) -> typing.Callable:

        worker = IdentityInference(Config.get())

        return worker.main_loop
