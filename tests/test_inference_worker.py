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

    @mock.patch("asyncio.Event.wait")
    @mock.patch('threading.Thread')
    def test_start(self, mock_thread, mock_event_wait):
        mock_thread.return_value = mock_thread

        pq = ProducerConsumerQueue()
        iw = inference_stage.InferenceWorker(pq)
        asyncio.run(iw.start())
        mock_thread.start.assert_called_once()
        mock_event_wait.assert_awaited()

if __name__ == '__main__':
    unittest.main()
