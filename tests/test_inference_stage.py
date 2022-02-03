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


class InferenceStage(inference_stage.InferenceStage):
    # Subclass InferenceStage to implement the abstract methods
    def _get_inference_worker():
        return None


class TestInferenceStage(BaseMorpheusTest):
    def test_constructor(self):
        config = Config.get()
        config.feature_length = 128
        config.num_threads = 17
        config.model_max_batch_size = 256

        inf_stage = InferenceStage(config)
        self.assertEqual(inf_stage._fea_length, 128)
        self.assertEqual(inf_stage._thread_count, 17)
        self.assertEqual(inf_stage._max_batch_size, 256)
        self.assertEqual(inf_stage.name, "inference")

        # Just ensure that we get a valid non-empty tuple
        accepted_types = inf_stage.accepted_types()
        self.assertIsInstance(accepted_types, tuple)
        self.assertGreater(len(accepted_types), 0)

        self.assertRaises(NotImplementedError, inf_stage._get_cpp_inference_node, None)

    def test_build_single(self):
        mock_node = mock.MagicMock()
        mock_segment = mock.MagicMock()
        mock_segment.make_node_full.return_value = mock_node
        mock_input = mock.MagicMock()

        config = Config.get()
        config.num_threads = 17
        inf_stage = InferenceStage(config)
        inf_stage._build_single(mock_segment, mock_input)

        mock_segment.make_node_full.assert_called_once()
        mock_segment.make_edge.assert_called_once()
        self.assertEqual(mock_node.concurrency, 17)

    def test_build_single_cpp(self):
        mock_node = mock.MagicMock()
        mock_segment = mock.MagicMock()
        mock_segment.make_node_full.return_value = mock_node
        mock_input = mock.MagicMock()

        config = Config.get()
        config.use_cpp = True
        config.num_threads = 17
        inf_stage = InferenceStage(config)
        inf_stage.supports_cpp_node = lambda: True
        inf_stage._get_cpp_inference_node = lambda x: mock_node

        inf_stage._build_single(mock_segment, mock_input)

        mock_segment.make_node_full.assert_not_called()
        mock_segment.make_edge.assert_called_once()
        self.assertEqual(mock_node.concurrency, 17)

    def test_build_single_cpp_not_impl(self):
        mock_node = mock.MagicMock()
        mock_segment = mock.MagicMock()
        mock_segment.make_node_full.return_value = mock_node
        mock_input = mock.MagicMock()

        config = Config.get()
        config.use_cpp = True
        inf_stage = InferenceStage(config)
        inf_stage.supports_cpp_node = lambda: True
        self.assertRaises(NotImplementedError, inf_stage._build_single, mock_segment, mock_input)

    def test_start(self):
        mock_start = mock.MagicMock()
        config = Config.get()
        inf_stage = InferenceStage(config)

        inf_stage._start = mock_start

        self.assertRaises(AssertionError, inf_stage.start)

        inf_stage._is_built = True
        inf_stage.start()
        mock_start.assert_called_once()

    def test_stop(self):
        mock_workers = [mock.MagicMock() for _ in range(5)]
        config = Config.get()
        inf_stage = InferenceStage(config)
        inf_stage._workers = mock_workers

        inf_stage.stop()
        for w in mock_workers:
            w.stop.assert_called_once()

        self.assertTrue(inf_stage._inf_queue.is_closed())

    def test_join(self):
        mock_workers = [mock.AsyncMock() for _ in range(5)]
        config = Config.get()
        inf_stage = InferenceStage(config)
        inf_stage._workers = mock_workers

        asyncio.run(inf_stage.join())
        for w in mock_workers:
            w.join.assert_awaited_once()



if __name__ == '__main__':
    unittest.main()
