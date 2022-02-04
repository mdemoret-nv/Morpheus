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

import json
import os
import unittest
from unittest import mock

import docker
from morpheus import config
from tests import BaseMorpheusTest


class TestConfig(BaseMorpheusTest):
    @mock.patch('docker.from_env')
    def test_auto_determine_bootstrap(self, mock_docker_from_env):
        mock_net = mock.MagicMock()
        mock_net.get.return_value = mock_net
        mock_net.attrs = {'IPAM': {'Config': [{'Gateway': 'test_bridge_ip'}]}}

        # create mock containers
        mc1 = mock.MagicMock()
        mc1.ports = {'9092/tcp': [{'HostIp': 'test_host1', 'HostPort': '47'}]}

        mc2 = mock.MagicMock()
        mc2.ports = {}

        mc3 = mock.MagicMock()
        mc3.ports = {'9092/tcp': [{'HostIp': 'test_host2', 'HostPort': '42'}]}

        mock_net.containers = [mc1, mc2, mc3]

        mock_docker_client = mock.MagicMock()
        mock_docker_client.networks = mock_net
        mock_docker_from_env.return_value = mock_docker_client

        bootstrap_servers = config.auto_determine_bootstrap()
        self.assertEqual(bootstrap_servers, "test_bridge_ip:47,test_bridge_ip:42")

    def test_config_base(self):
        c = config.ConfigBase()

        # Really just test to make sure we are still here
        self.assertIsInstance(c, config.ConfigBase)

    def test_config_onnx_to_trt(self):
        c = config.ConfigOnnxToTRT(input_model='frogs',
                                   output_model='toads',
                                   batches=[(1, 2), (3, 4)],
                                   seq_length=100,
                                   max_workspace_size=512)

        self.assertIsInstance(c, config.ConfigBase)
        self.assertIsInstance(c, config.ConfigOnnxToTRT)
        self.assertEqual(c.input_model, 'frogs')
        self.assertEqual(c.output_model, 'toads')
        self.assertEqual(c.batches, [(1, 2), (3, 4)])
        self.assertEqual(c.seq_length, 100)
        self.assertEqual(c.max_workspace_size, 512)

    def test_auto_encoder(self):
        c = config.ConfigAutoEncoder(feature_columns=['a', 'b', 'c', 'def'],
                                     userid_column_name='def',
                                     userid_filter='testuser')

        self.assertIsInstance(c, config.ConfigBase)
        self.assertIsInstance(c, config.ConfigAutoEncoder)
        self.assertEqual(c.feature_columns, ['a', 'b', 'c', 'def'])
        self.assertEqual(c.userid_column_name, 'def')
        self.assertEqual(c.userid_filter, 'testuser')

    def test_pipeline_modes(self):
        expected = {"OTHER", "NLP", "FIL", "AE"}
        entries = set(pm.name for pm in config.PipelineModes)
        self.assertTrue(entries.issuperset(expected))

    def test_config(self):
        self.assertRaises(Exception, config.Config)

        c = config.Config.default()
        self.assertIsInstance(c, config.ConfigBase)
        self.assertIsInstance(c, config.Config)

        self.assertTrue(hasattr(c, 'debug'))
        self.assertTrue(hasattr(c, 'log_level'))
        self.assertTrue(hasattr(c, 'log_config_file'))
        self.assertTrue(hasattr(c, 'mode'))
        self.assertTrue(hasattr(c, 'feature_length'))
        self.assertTrue(hasattr(c, 'pipeline_batch_size'))
        self.assertTrue(hasattr(c, 'num_threads'))
        self.assertTrue(hasattr(c, 'model_max_batch_size'))
        self.assertTrue(hasattr(c, 'edge_buffer_size'))
        self.assertTrue(hasattr(c, 'class_labels'))
        self.assertTrue(hasattr(c, 'use_cpp'))
        self.assertTrue(hasattr(c, 'ae'))

    def test_config_default(self):
        c1 = config.Config.default()
        c2 = config.Config.default()

        self.assertIsInstance(c1, config.ConfigBase)
        self.assertIsInstance(c1, config.Config)

        self.assertIs(c1, c2)
        self.assertIs(c1, config.Config.default())
        self.assertIsNot(c1, config.Config.get())

    def test_config_get(self):
        c1 = config.Config.get()
        c2 = config.Config.get()

        self.assertIsInstance(c1, config.ConfigBase)
        self.assertIsInstance(c1, config.Config)

        self.assertIs(c1, c2)
        self.assertIs(c1, config.Config.get())
        self.assertIsNot(c1, config.Config.default())

    def test_config_load(self):
        c = config.Config.get()
        self.assertRaises(NotImplementedError, c.load, 'ignored')

    def test_config_save(self):
        temp_dir = self._mk_tmp_dir()
        filename = os.path.join(temp_dir, 'config.json')

        c = config.Config.get()
        c.save(filename)

        self.assertTrue(os.path.exists(filename))
        with open(filename) as fh:
            self.assertIsInstance(json.load(fh), dict)

    def test_to_string(self):
        c = config.Config.get()
        s = c.to_string()
        self.assertIsInstance(s, str)
        self.assertIsInstance(json.loads(s), dict)


if __name__ == '__main__':
    unittest.main()
