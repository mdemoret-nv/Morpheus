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

import os
import unittest
from unittest import mock

import cupy as cp

from morpheus.config import Config
from morpheus.pipeline.general_stages import AddScoresStage
from tests import BaseMorpheusTest


class TestAddScoresStage(BaseMorpheusTest):
    def test_constructor(self):
        config = Config.get()
        config.class_labels = ['frogs', 'lizards', 'toads']
        config.feature_length = 12

        a = AddScoresStage(config)
        self.assertEqual(a._class_labels, ['frogs', 'lizards', 'toads'])
        self.assertEqual(a._labels, ['frogs', 'lizards', 'toads'])
        self.assertEqual(a._idx2label, {0: 'frogs', 1: 'lizards', 2: 'toads'})
        self.assertEqual(a.name, "add-scores")

        # Just ensure that we get a valid non-empty tuple
        accepted_types = a.accepted_types()
        self.assertIsInstance(accepted_types, tuple)
        self.assertGreater(len(accepted_types), 0)

        a = AddScoresStage(config, labels=['lizards'], prefix='test_')
        self.assertEqual(a._class_labels, ['frogs', 'lizards', 'toads'])
        self.assertEqual(a._labels, ['lizards'])
        self.assertEqual(a._idx2label, {1: 'test_lizards'})

        self.assertRaises(AssertionError, AddScoresStage, config, labels=['missing'])

    def test_add_labels(self):
        mock_message = mock.MagicMock()
        mock_message.probs = cp.array([[0.1, 0.5, 0.8], [0.2, 0.6, 0.9]])

        config = Config.get()
        config.class_labels = ['frogs', 'lizards', 'toads']

        a = AddScoresStage(config)
        a._add_labels(mock_message)

        mock_message.set_meta.assert_has_calls([
            mock.call('frogs', [0.1, 0.2]),
            mock.call('lizards', [0.5, 0.6]),
            mock.call('toads', [0.8, 0.9]),
        ])

        wrong_shape = mock.MagicMock()
        wrong_shape.probs = cp.array([[0.1, 0.5], [0.2, 0.6]])
        self.assertRaises(RuntimeError, a._add_labels, wrong_shape)

    def test_build_single_no_cpp(self):
        mock_stream = mock.MagicMock()
        mock_segment = mock.MagicMock()
        mock_segment.make_node.return_value = mock_stream
        mock_input = mock.MagicMock()

        config = Config.get()
        config.use_cpp = False
        config.class_labels = ['frogs', 'lizards', 'toads']

        a = AddScoresStage(config)
        a._build_single(mock_segment, mock_input)

        mock_segment.make_node.assert_called_once()
        mock_segment.make_edge.assert_called_once()

    def test_build_single_cpp(self):
        self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()
