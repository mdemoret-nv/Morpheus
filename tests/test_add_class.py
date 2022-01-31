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

from morpheus.config import Config

Config.get().use_cpp = False

from morpheus.config import ConfigAutoEncoder
from morpheus.config import PipelineModes
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.general_stages import AddClassificationsStage
from tests import BaseMorpheusTest


class TestAddClass(BaseMorpheusTest):
    """
    End-to-end test intended to imitate the hammah validation test
    """
    def test_constructor(self):
        config = Config.get()
        config.mode = PipelineModes.FIL
        config.use_cpp = False
        config.class_labels = ['frogs', 'lizards', 'toads']

        ac = AddClassificationsStage(config)
        self.assertEqual(ac._class_labels, ['frogs', 'lizards', 'toads'])
        self.assertEqual(ac._labels, ['frogs', 'lizards', 'toads'])
        self.assertEqual(ac._idx2label, {0: 'frogs', 1: 'lizards', 2: 'toads'})

        ac = AddClassificationsStage(config, threshold=1.3, labels=['lizards'], prefix='test_')
        self.assertEqual(ac._class_labels, ['frogs', 'lizards', 'toads'])
        self.assertEqual(ac._labels, ['lizards'])
        self.assertEqual(ac._idx2label, {1: 'test_lizards'})

        self.assertRaises(AssertionError, AddClassificationsStage, config, labels=['missing'])


if __name__ == '__main__':
    unittest.main()
