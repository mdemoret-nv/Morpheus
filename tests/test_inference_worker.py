#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import threading
import unittest
from unittest import mock

import cupy as cp

from morpheus.config import Config

Config.get().use_cpp = False

from morpheus.pipeline.inference import inference_stage
from morpheus.utils.producer_consumer_queue import Closed
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue
from tests import BaseMorpheusTest


class TestInferenceWorker(BaseMorpheusTest):
    @mock.patch('asyncio.Event')
    @mock.patch('morpheus.pipeline.inference.inference_stage.IOLoop')
    def test_constructor(self, mock_ioloop, mock_event_cls):
        mock_current_loop = mock.MagicMock()
        mock_ioloop.current.return_value = mock_current_loop

        mock_event = mock.MagicMock()
        mock_event_cls.return_value = mock_event

        pq = ProducerConsumerQueue()
        iw = inference_stage.InferenceWorker(pq)
        self.assertIs(iw._loop, mock_current_loop)
        self.assertIsNone(iw._thread)
        self.assertIs(iw._inf_queue, pq)
        self.assertIs(iw._complete_event, mock_event)

        # Call empty methods
        iw.init()
        iw.stop()

    @mock.patch("asyncio.Event.wait")
    @mock.patch('threading.Thread')
    def test_start(self, mock_thread, mock_event_wait):
        mock_thread.return_value = mock_thread

        pq = ProducerConsumerQueue()
        iw = inference_stage.InferenceWorker(pq)
        asyncio.run(iw.start())
        mock_thread.start.assert_called_once()
        mock_event_wait.assert_awaited()

    @mock.patch("asyncio.Event.wait")
    @mock.patch('threading.Thread')
    def test_join(self, mock_thread, mock_event_wait):
        mock_thread.return_value = mock_thread

        pq = ProducerConsumerQueue()
        iw = inference_stage.InferenceWorker(pq)

        # Initialize the thread to a mock to avoid needing to call start method
        iw._thread = mock_thread
        asyncio.run(iw.join())

        mock_thread.join.assert_called_once()
        mock_event_wait.assert_awaited()


    def test_build_output_message(self):
        # build_output_message calls calc_output_dims which is abstract
        # creating a subclass for the purpose of testing
        config = Config.get()
        config.use_cpp = False
        class TestIW(inference_stage.InferenceWorker):
            def calc_output_dims(self, _):
                return (1, 2)

        pq = ProducerConsumerQueue()
        iw = TestIW(pq)

        mock_message = mock.MagicMock()
        mock_message.count = 2
        mock_message.mess_offset = 11
        mock_message.mess_count = 10
        mock_message.offset = 12

        response = iw.build_output_message(mock_message)
        self.assertEqual(response.count, 2)
        self.assertEqual(response.mess_offset, 11)
        self.assertEqual(response.mess_count, 10)
        self.assertEqual(response.offset, 12)

    @mock.patch('morpheus.pipeline.inference.inference_stage.IOLoop')
    def test_main_loop_short(self, mock_ioloop):
        mock_asyncio_loop =  mock.MagicMock()
        mock_current_loop = mock.MagicMock()
        mock_current_loop.asyncio_loop = mock_asyncio_loop
        mock_ioloop.current.return_value = mock_current_loop

        pq = ProducerConsumerQueue()
        iw = inference_stage.InferenceWorker(pq)

        pq.close()

        # Calling main loop without a ready event and a closed queue
        # should short-circuit most of the function
        iw.main_loop()

        mock_asyncio_loop.call_soon_threadsafe.assert_called_once()

    @mock.patch('morpheus.pipeline.inference.inference_stage.IOLoop')
    def test_main_loop_with_event(self, mock_ioloop):
        mock_asyncio_loop =  mock.MagicMock()
        mock_current_loop = mock.MagicMock()
        mock_current_loop.asyncio_loop = mock_asyncio_loop
        mock_ioloop.current.return_value = mock_current_loop

        pq = ProducerConsumerQueue()
        iw = inference_stage.InferenceWorker(pq)

        pq.close()

        ready_event = asyncio.Event()
        iw.main_loop(ready_event)

        call_args_list = mock_asyncio_loop.call_soon_threadsafe.call_args_list
        self.assertEqual(len(call_args_list), 2)
        self.assertEqual(call_args_list[0], mock.call(ready_event.set))
        self.assertEqual(call_args_list[1], mock.call(iw._complete_event.set))


    @mock.patch('morpheus.pipeline.inference.inference_stage.IOLoop')
    def test_main_loop(self, mock_ioloop):
        mock_asyncio_loop =  mock.MagicMock()
        mock_current_loop = mock.MagicMock()
        mock_current_loop.asyncio_loop = mock_asyncio_loop
        mock_ioloop.current.return_value = mock_current_loop

        class TestIW(inference_stage.InferenceWorker):
            pass

        TestIW.process = mock.MagicMock()

        # Allow for two iterations of the main loop
        mock_queue = mock.MagicMock()
        mock_queue.is_closed.side_effect = [False, False, True]

        batch1 = mock.MagicMock()
        fut1 = mock.MagicMock()
        batch2 = mock.MagicMock()
        fut2 = mock.MagicMock()
        mock_queue.get.side_effect = [(batch1, fut1), (batch2, fut2)]
        iw = TestIW(mock_queue)

        ready_event = asyncio.Event()
        iw.main_loop(ready_event)

        call_args_list = mock_asyncio_loop.call_soon_threadsafe.call_args_list
        self.assertEqual(len(call_args_list), 2)
        self.assertEqual(call_args_list[0], mock.call(ready_event.set))
        self.assertEqual(call_args_list[1], mock.call(iw._complete_event.set))

        fut1.add_done_callback.assert_called_once()
        fut2.add_done_callback.assert_called_once()

        process_calls = TestIW.process.call_args_list
        self.assertEqual(len(process_calls), 2)
        self.assertEqual(process_calls[0], mock.call(batch1, fut1))
        self.assertEqual(process_calls[1], mock.call(batch2, fut2))

    @mock.patch('morpheus.pipeline.inference.inference_stage.IOLoop')
    def test_main_loop_closed_exception(self, mock_ioloop):
        mock_asyncio_loop =  mock.MagicMock()
        mock_current_loop = mock.MagicMock()
        mock_current_loop.asyncio_loop = mock_asyncio_loop
        mock_ioloop.current.return_value = mock_current_loop

        mock_process = mock.MagicMock()
        class TestIW(inference_stage.InferenceWorker):
            def process(self, batch, cb):
                # intentionally calling empty abc method for coverage
                super().process(batch, cb)
                mock_process(batch, cb)

        # Allow for three iterations of the main loop, with the third raising a closed exception on the call to get
        mock_queue = mock.MagicMock()
        mock_queue.is_closed.side_effect = [False, False, False, True]

        batch1 = mock.MagicMock()
        fut1 = mock.MagicMock()
        batch2 = mock.MagicMock()
        fut2 = mock.MagicMock()

        def wrapper():
            counter = [0]
            def inner(*a, **k):
                counter[0]+=1
                if counter[0] == 1:
                    return (batch1, fut1)
                elif counter[0] == 2:
                    return (batch2, fut2)
                else:
                    raise Closed("unittest")
            return inner

        mock_queue.get.side_effect = wrapper()
        iw = TestIW(mock_queue)

        ready_event = asyncio.Event()
        iw.main_loop(ready_event)

        self.assertEqual(mock_queue.get.call_count, 3)
        self.assertEqual(mock_queue.is_closed.call_count, 4)

        call_args_list = mock_asyncio_loop.call_soon_threadsafe.call_args_list
        self.assertEqual(len(call_args_list), 2)
        self.assertEqual(call_args_list[0], mock.call(ready_event.set))
        self.assertEqual(call_args_list[1], mock.call(iw._complete_event.set))

        fut1.add_done_callback.assert_called_once()
        fut2.add_done_callback.assert_called_once()

        process_calls = mock_process.call_args_list
        self.assertEqual(len(process_calls), 2)
        self.assertEqual(process_calls[0], mock.call(batch1, fut1))
        self.assertEqual(process_calls[1], mock.call(batch2, fut2))


if __name__ == '__main__':
    unittest.main()
