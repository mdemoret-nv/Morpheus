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

import csv
import os
import unittest
from unittest import mock

import numpy as np
import tritonclient.grpc

from morpheus.config import Config
from morpheus.config import PipelineModes
from tests import BaseMorpheusTest

FEATURE_LENGTH = 128
MODEL_MAX_BATCH_SIZE = 32


class TestPhishing(BaseMorpheusTest):
    """
    End-to-end test intended to imitate the ABP validation test
    """
    @unittest.skipIf(os.environ.get("MORPHEUS_SKIP_SLOW_TESTS") is not None, "MORPHEUS_SKIP_SLOW_TESTS is defined")
    @mock.patch('tritonclient.grpc.InferenceServerClient')
    def test_email_no_cpp(self, mock_triton_client):
        mock_metadata = {
            "inputs": [{
                "name": "input_ids", "datatype": "INT64", "shape": [-1, FEATURE_LENGTH]
            }, {
                "name": "attention_mask", "datatype": "INT64", "shape": [-1, FEATURE_LENGTH]
            }],
            "outputs": [{
                "name": "output", "datatype": "FP32", "shape": [-1, 2]
            }]
        }
        mock_model_config = {"config": {"max_batch_size": MODEL_MAX_BATCH_SIZE}}

        mock_triton_client.return_value = mock_triton_client
        mock_triton_client.is_server_live.return_value = True
        mock_triton_client.is_server_ready.return_value = True
        mock_triton_client.is_model_ready.return_value = True
        mock_triton_client.get_model_metadata.return_value = mock_metadata
        mock_triton_client.get_model_config.return_value = mock_model_config

        data = np.loadtxt(os.path.join(self._expeced_data_dir, 'triton_phishing_inf_results.csv'), delimiter=',')
        inf_results = self._partition_array(data, MODEL_MAX_BATCH_SIZE)

        mock_infer_result = mock.MagicMock()
        mock_infer_result.as_numpy.side_effect = inf_results

        def async_infer(callback=None, **k):
            callback(mock_infer_result, None)

        mock_triton_client.async_infer.side_effect = async_infer

        config = Config.get()
        config.mode = PipelineModes.NLP
        config.use_cpp = False
        config.class_labels = ["score", "pred"]
        config.model_max_batch_size = MODEL_MAX_BATCH_SIZE
        config.pipeline_batch_size = 1024
        config.feature_length = FEATURE_LENGTH
        config.edge_buffer_size = 128
        config.num_threads = 1

        from morpheus.pipeline import LinearPipeline
        from morpheus.pipeline.general_stages import AddClassificationsStage
        from morpheus.pipeline.general_stages import MonitorStage
        from morpheus.pipeline.inference.inference_triton import TritonInferenceStage
        from morpheus.pipeline.input.from_file import FileSourceStage
        from morpheus.pipeline.output.serialize import SerializeStage
        from morpheus.pipeline.output.to_file import WriteToFileStage
        from morpheus.pipeline.output.validation import ValidationStage
        from morpheus.pipeline.preprocessing import DeserializeStage
        from morpheus.pipeline.preprocessing import PreprocessNLPStage

        val_file_name = os.path.join(self._validation_data_dir, 'phishing-email-validation-data.jsonlines')
        vocab_file_name = os.path.join(self._data_dir, 'bert-base-uncased-hash.txt')

        temp_dir = self._mk_tmp_dir()
        out_file = os.path.join(temp_dir, 'results.csv')
        results_file_name = os.path.join(temp_dir, 'results.json')

        pipe = LinearPipeline(config)
        pipe.set_source(FileSourceStage(config, filename=val_file_name, iterative=False))
        pipe.add_stage(DeserializeStage(config))
        pipe.add_stage(
            PreprocessNLPStage(config,
                               vocab_hash_file=vocab_file_name,
                               truncation=True,
                               do_lower_case=True,
                               add_special_tokens=False))
        pipe.add_stage(
            TritonInferenceStage(config,
                                 model_name='phishing-bert-onnx',
                                 server_url='test:0000',
                                 force_convert_inputs=True))
        pipe.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))
        pipe.add_stage(AddClassificationsStage(config, labels=["pred"], threshold=0.7))
        pipe.add_stage(
            ValidationStage(config, val_file_name=val_file_name, results_file_name=results_file_name, rel_tol=0.05))
        pipe.add_stage(SerializeStage(config, output_type="pandas"))
        pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))

        pipe.run()
        results = self._calc_error_val(results_file_name)
        self.assertLess(results.error_pct, 80)

    @unittest.skipIf(os.environ.get("MORPHEUS_SKIP_SLOW_TESTS") is not None, "MORPHEUS_SKIP_SLOW_TESTS is defined")
    def test_email_cpp(self):
        self._launch_camouflage_triton()

        config = Config.get()
        config.mode = PipelineModes.NLP
        config.use_cpp = True
        config.class_labels = ["score", "pred"]
        config.model_max_batch_size = MODEL_MAX_BATCH_SIZE
        config.pipeline_batch_size = 1024
        config.feature_length = FEATURE_LENGTH
        config.edge_buffer_size = 128
        config.num_threads = 1

        from morpheus.pipeline import LinearPipeline
        from morpheus.pipeline.general_stages import AddClassificationsStage
        from morpheus.pipeline.general_stages import MonitorStage
        from morpheus.pipeline.inference.inference_triton import TritonInferenceStage
        from morpheus.pipeline.input.from_file import FileSourceStage
        from morpheus.pipeline.output.serialize import SerializeStage
        from morpheus.pipeline.output.to_file import WriteToFileStage
        from morpheus.pipeline.output.validation import ValidationStage
        from morpheus.pipeline.preprocessing import DeserializeStage
        from morpheus.pipeline.preprocessing import PreprocessNLPStage

        val_file_name = os.path.join(self._validation_data_dir, 'phishing-email-validation-data.jsonlines')
        vocab_file_name = os.path.join(self._data_dir, 'bert-base-uncased-hash.txt')

        temp_dir = self._mk_tmp_dir()
        out_file = os.path.join(temp_dir, 'results.csv')
        results_file_name = os.path.join(temp_dir, 'results.json')

        pipe = LinearPipeline(config)
        pipe.set_source(FileSourceStage(config, filename=val_file_name, iterative=False))
        pipe.add_stage(DeserializeStage(config))
        pipe.add_stage(
            PreprocessNLPStage(config,
                               vocab_hash_file=vocab_file_name,
                               truncation=True,
                               do_lower_case=True,
                               add_special_tokens=False))
        pipe.add_stage(
            TritonInferenceStage(config,
                                 model_name='phishing-bert-onnx',
                                 server_url='localhost:8001',
                                 force_convert_inputs=True))
        pipe.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))
        pipe.add_stage(AddClassificationsStage(config, labels=["pred"], threshold=0.7))
        pipe.add_stage(
            ValidationStage(config, val_file_name=val_file_name, results_file_name=results_file_name, rel_tol=0.05))
        pipe.add_stage(SerializeStage(config, output_type="pandas"))
        pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))

        pipe.run()
        results = self._calc_error_val(results_file_name)
        self.assertLess(results.error_pct, 80)


if __name__ == '__main__':
    unittest.main()
