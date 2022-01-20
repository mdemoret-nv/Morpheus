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
from morpheus.config import PipelineModes
from tests.base_test import BaseMorpheusTest


class TestAPB(BaseMorpheusTest):
    """
    End-to-end test intended to imitate the APB validation test
    """
    def test_apb(self):
        config = Config.get()
        config.mode = PipelineModes.FIL
        config.use_cpp = True
        config.class_labels = ["mining"]
        config.model_max_batch_size = 1024
        config.pipeline_batch_size = 1024
        config.feature_length = 29
        config.edge_buffer_size = 128
        config.num_threads = 1

        from morpheus.pipeline import LinearPipeline
        from morpheus.pipeline.general_stages import AddClassificationsStage
        from morpheus.pipeline.general_stages import AddScoresStage
        from morpheus.pipeline.general_stages import MonitorStage
        from morpheus.pipeline.inference.inference_ae import AutoEncoderInferenceStage
        from morpheus.pipeline.inference.inference_triton import TritonInferenceStage
        from morpheus.pipeline.input.from_file import FileSourceStage
        from morpheus.pipeline.output.serialize import SerializeStage
        from morpheus.pipeline.output.to_file import WriteToFileStage
        from morpheus.pipeline.output.validation import ValidationStage
        from morpheus.pipeline.postprocess.timeseries import TimeSeriesStage
        from morpheus.pipeline.preprocess.autoencoder import TrainAEStage
        from morpheus.pipeline.preprocessing import DeserializeStage
        from morpheus.pipeline.preprocessing import PreprocessFILStage

        temp_dir = self._mk_tmp_dir()
        val_file_name = os.path.join(self._validation_data_dir, 'abp-validation-data.jsonlines')

        out_file = os.path.join(temp_dir, 'results.csv')
        results_file_name = os.path.join(temp_dir, 'results.json')

        pipe = LinearPipeline(config)
        pipe.set_source(FileSourceStage(config, filename=val_file_name, iterative=False))
        pipe.add_stage(DeserializeStage(config))
        pipe.add_stage(PreprocessFILStage(config))
        pipe.add_stage(
            TritonInferenceStage(config, model_name='abp-nvsmi-xgb', server_url='test:0000', force_convert_inputs=True))
        pipe.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))
        pipe.add_stage(AddClassificationsStage(config))
        pipe.add_stage(
            ValidationStage(config, val_file_name=val_file_name, results_file_name=results_file_name, rel_tol=0.05))

        pipe.add_stage(SerializeStage(config, output_type="pandas"))
        pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))

        pipe.run()
        results = self._calc_error_val(results_file_name)
        self.assertEqual(results.error_pct, 0)


if __name__ == '__main__':
    unittest.main()
