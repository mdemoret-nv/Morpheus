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
from morpheus.pipeline.general_stages import AddClassificationsStage
from tests import BaseMorpheusTest


class TestAddClassificationsStage(BaseMorpheusTest):
    def test_constructor(self):
        config = Config.get()
        config.class_labels = ['frogs', 'lizards', 'toads']

        ac = AddClassificationsStage(config)
        self.assertEqual(ac._class_labels, ['frogs', 'lizards', 'toads'])
        self.assertEqual(ac._labels, ['frogs', 'lizards', 'toads'])
        self.assertEqual(ac._idx2label, {0: 'frogs', 1: 'lizards', 2: 'toads'})
        self.assertEqual(ac.name, "add-class")

        # Just ensure that we get a valid non-empty tuple
        accepted_types = ac.accepted_types()
        self.assertIsInstance(accepted_types, tuple)
        self.assertGreater(len(accepted_types), 0)

        ac = AddClassificationsStage(config, threshold=1.3, labels=['lizards'], prefix='test_')
        self.assertEqual(ac._class_labels, ['frogs', 'lizards', 'toads'])
        self.assertEqual(ac._labels, ['lizards'])
        self.assertEqual(ac._idx2label, {1: 'test_lizards'})

        self.assertRaises(AssertionError, AddClassificationsStage, config, labels=['missing'])

    def test_add_labels(self):
        mock_message = mock.MagicMock()
        mock_message.probs = cp.array([[0.1, 0.5, 0.8], [0.2, 0.6, 0.9]])

        config = Config.get()
        config.class_labels = ['frogs', 'lizards', 'toads']

        ac = AddClassificationsStage(config, threshold=0.5)
        ac._add_labels(mock_message)

        mock_message.set_meta.assert_has_calls([
            mock.call('frogs', [False, False]),
            mock.call('lizards', [False, True]),
            mock.call('toads', [True, True]),
        ])

        wrong_shape = mock.MagicMock()
        wrong_shape.probs = cp.array([[0.1, 0.5], [0.2, 0.6]])
        self.assertRaises(RuntimeError, ac._add_labels, wrong_shape)

    def test_build_single_no_cpp(self):
        mock_stream = mock.MagicMock()
        mock_segment = mock.MagicMock()
        mock_segment.make_node.return_value = mock_stream
        mock_input = mock.MagicMock()

        config = Config.get()
        config.use_cpp = False
        config.class_labels = ['frogs', 'lizards', 'toads']

        ac = AddClassificationsStage(config)
        ac._build_single(mock_segment, mock_input)

        mock_segment.make_node.assert_called_once()
        mock_segment.make_edge.assert_called_once()

    def test_build_single_cpp(self):
        self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
