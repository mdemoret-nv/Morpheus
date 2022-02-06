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

import inspect
import unittest
from unittest import mock

import tqdm

import cudf

from morpheus.config import Config

Config.get().use_cpp = False

from morpheus.pipeline import general_stages as gs
from morpheus.pipeline.messages import MultiMessage
from tests import BaseMorpheusTest


class TestMonitor(BaseMorpheusTest):
    def test_constructor(self):
        config = Config.get()

        # Intentionally not making assumptions about the defaults other than they exist
        # and still create a valid stage.
        m = gs.MonitorStage(config)
        self.assertEqual(m.name, "monitor")

        # Just ensure that we get a valid non-empty tuple
        accepted_types = m.accepted_types()
        self.assertIsInstance(accepted_types, tuple)
        self.assertGreater(len(accepted_types), 0)

        two_x = lambda x: x * 2
        m = gs.MonitorStage(config, description="Test Description", smoothing=0.7, unit='units', determine_count_fn=two_x)
        self.assertEqual(m._description, "Test Description")
        self.assertEqual(m._smoothing, 0.7)
        self.assertEqual(m._unit, "units")
        self.assertIs(m._determine_count_fn, two_x)


    @mock.patch('morpheus.pipeline.general_stages.tqdm')
    def test_on_start(self, mock_tqdm):
        mock_tqdm.return_value = mock_tqdm

        config = Config.get()
        m = gs.MonitorStage(config)
        self.assertIsNone(m._progress)

        m.on_start()
        mock_tqdm.assert_called_once()
        mock_tqdm.reset.assert_called_once()
        self.assertIs(m._progress, mock_tqdm)


    @mock.patch('morpheus.pipeline.general_stages.tqdm')
    def test_stop(self, mock_tqdm):
        mock_tqdm.return_value = mock_tqdm

        config = Config.get()
        m = gs.MonitorStage(config)
        self.assertIsNone(m._progress)

        # Calling on_stop is a noop if we are stopped
        m.stop()
        mock_tqdm.assert_not_called()

        m.on_start()
        m.stop()
        mock_tqdm.refresh.assert_called_once()
        mock_tqdm.close.assert_called_once()


    @mock.patch('morpheus.pipeline.general_stages.tqdm')
    def test_refresh(self, mock_tqdm):
        mock_tqdm.return_value = mock_tqdm

        config = Config.get()
        m = gs.MonitorStage(config)
        self.assertIsNone(m._progress)

        m.on_start()
        m._refresh_progress(None)
        mock_tqdm.refresh.assert_called_once()

    @mock.patch('morpheus.pipeline.general_stages.tqdm')
    def test_build_single(self, mock_tqdm):
        mock_tqdm.return_value = mock_tqdm

        mock_stream = mock.MagicMock()
        mock_segment = mock.MagicMock()
        mock_segment.make_sink.return_value = mock_stream
        mock_input = mock.MagicMock()

        config = Config.get()
        m = gs.MonitorStage(config)
        m._build_single(mock_segment, mock_input)
        m.on_start()

        mock_segment.make_sink.assert_called_once()
        mock_segment.make_edge.assert_called_once()

        sink_on_error = mock_segment.make_sink.call_args.args[2]
        sink_on_completed = mock_segment.make_sink.call_args.args[3]

        # This is currenlty just a log stmt, just verify that its callable
        sink_on_error(RuntimeError("unittest"))

        # Verify we close tqdm propperly on complete
        sink_on_completed()
        mock_tqdm.refresh.assert_called_once()
        mock_tqdm.close.assert_called_once()


    def test_auto_count_fn(self):
        config = Config.get()
        config.use_cpp = False
        m = gs.MonitorStage(config)

        self.assertIsNone(m._auto_count_fn(None))
        self.assertIsNone(m._auto_count_fn([]))

        # Ints not supported, lists are, but lists of unsupported are also unsupported
        self.assertRaises(NotImplementedError, m._auto_count_fn, 1)
        self.assertRaises(NotImplementedError, m._auto_count_fn, [1])

        # Just verify that we get a valid function for each supported type
        self.assertTrue(inspect.isfunction(m._auto_count_fn(['s'])))
        self.assertTrue(inspect.isfunction(m._auto_count_fn('s')))
        self.assertTrue(inspect.isfunction(m._auto_count_fn(cudf.DataFrame())))
        self.assertTrue(inspect.isfunction(m._auto_count_fn(MultiMessage(None, 0, 0))))

        # Other iterables return the len function
        self.assertIs(m._auto_count_fn({}), len)
        self.assertIs(m._auto_count_fn(()), len)
        self.assertIs(m._auto_count_fn(set()), len)


    @mock.patch('morpheus.pipeline.general_stages.tqdm')
    def test_progress_sink(self, mock_tqdm):
        mock_tqdm.return_value = mock_tqdm

        config = Config.get()
        config.use_cpp = False
        m = gs.MonitorStage(config)
        m.on_start()

        m._progress_sink(None)
        self.assertIsNone(m._determine_count_fn)
        mock_tqdm.update.assert_not_called()

        m._progress_sink(MultiMessage(None, 0, 12))
        self.assertTrue(inspect.isfunction(m._determine_count_fn))
        mock_tqdm.update.assert_called_once_with(n=12)

if __name__ == '__main__':
    unittest.main()
