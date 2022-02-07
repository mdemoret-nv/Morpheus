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
from morpheus.config import ConfigAutoEncoder
from morpheus.config import PipelineModes
from tests import BaseMorpheusTest


class TestHammah(BaseMorpheusTest):
    """
    End-to-end test intended to imitate the hammah validation test
    """
    def test_hammah_roleg(self):
        config = Config.get()
        config.mode = PipelineModes.AE
        config.use_cpp = False
        config.class_labels = ["ae_anomaly_score"]
        config.model_max_batch_size = 1024
        config.pipeline_batch_size = 1024
        config.feature_length = 256
        config.edge_buffer_size = 128
        config.num_threads = 1

        config.ae = ConfigAutoEncoder()
        config.ae.userid_column_name = "userIdentitysessionContextsessionIssueruserName"
        config.ae.userid_filter = "role-g"

        with open(os.path.join(self._data_dir, 'columns_ae.txt')) as fh:
            config.ae.feature_columns = [x.strip() for x in fh.readlines()]

        from morpheus.pipeline import LinearPipeline
        from morpheus.pipeline.general_stages import AddScoresStage
        from morpheus.pipeline.general_stages import MonitorStage
        from morpheus.pipeline.inference.inference_ae import AutoEncoderInferenceStage
        from morpheus.pipeline.input.from_cloudtrail import CloudTrailSourceStage
        from morpheus.pipeline.output.serialize import SerializeStage
        from morpheus.pipeline.output.to_file import WriteToFileStage
        from morpheus.pipeline.output.validation import ValidationStage
        from morpheus.pipeline.postprocess.timeseries import TimeSeriesStage
        from morpheus.pipeline.preprocess.autoencoder import PreprocessAEStage
        from morpheus.pipeline.preprocess.autoencoder import TrainAEStage

        temp_dir = self._mk_tmp_dir()
        input_glob = os.path.join(self._validation_data_dir, "hammah-*.csv")
        train_data_glob = os.path.join(self._training_data_dir, "hammah-*.csv")
        out_file = os.path.join(temp_dir, 'results.csv')
        val_file_name = os.path.join(self._validation_data_dir, 'hammah-role-g-validation-data.csv')
        results_file_name = os.path.join(temp_dir, 'results.json')

        pipe = LinearPipeline(config)
        pipe.set_source(CloudTrailSourceStage(config, input_glob=input_glob))
        pipe.add_stage(TrainAEStage(config, train_data_glob=train_data_glob, seed=42))
        pipe.add_stage(PreprocessAEStage(config))
        pipe.add_stage(AutoEncoderInferenceStage(config))
        pipe.add_stage(AddScoresStage(config))
        pipe.add_stage(
            TimeSeriesStage(config,
                            resolution="10m",
                            min_window="12 h",
                            hot_start=False,
                            cold_end=False,
                            filter_percent=90.0,
                            zscore_threshold=8.0))
        pipe.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))
        pipe.add_stage(
            ValidationStage(config,
                            val_file_name=val_file_name,
                            results_file_name=results_file_name,
                            index_col="_index_",
                            exclude=("event_dt", ),
                            rel_tol=0.15))
        pipe.add_stage(SerializeStage(config, include=[], output_type="pandas"))
        pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))

        pipe.run()
        results = self._calc_error_val(results_file_name)
        self.assertEqual(results.diff_rows, 3)

    def test_hammah_user123(self):
        config = Config.get()
        config.mode = PipelineModes.AE
        config.use_cpp = False
        config.class_labels = ["ae_anomaly_score"]
        config.model_max_batch_size = 1024
        config.pipeline_batch_size = 1024
        config.edge_buffer_size = 128
        config.num_threads = 1

        config.ae = ConfigAutoEncoder()
        config.ae.userid_column_name = "userIdentitysessionContextsessionIssueruserName"
        config.ae.userid_filter = "user123"

        with open(os.path.join(self._data_dir, 'columns_ae.txt')) as fh:
            config.ae.feature_columns = [x.strip() for x in fh.readlines()]

        from morpheus.pipeline import LinearPipeline
        from morpheus.pipeline.general_stages import AddScoresStage
        from morpheus.pipeline.general_stages import MonitorStage
        from morpheus.pipeline.inference.inference_ae import AutoEncoderInferenceStage
        from morpheus.pipeline.input.from_cloudtrail import CloudTrailSourceStage
        from morpheus.pipeline.output.serialize import SerializeStage
        from morpheus.pipeline.output.to_file import WriteToFileStage
        from morpheus.pipeline.output.validation import ValidationStage
        from morpheus.pipeline.postprocess.timeseries import TimeSeriesStage
        from morpheus.pipeline.preprocess.autoencoder import PreprocessAEStage
        from morpheus.pipeline.preprocess.autoencoder import TrainAEStage

        temp_dir = self._mk_tmp_dir()
        input_glob = os.path.join(self._validation_data_dir, "hammah-*.csv")
        train_data_glob = os.path.join(self._training_data_dir, "hammah-*.csv")
        out_file = os.path.join(temp_dir, 'results.csv')
        val_file_name = os.path.join(self._validation_data_dir, 'hammah-user123-validation-data.csv')
        results_file_name = os.path.join(temp_dir, 'results.json')

        pipe = LinearPipeline(config)
        pipe.set_source(CloudTrailSourceStage(config, input_glob=input_glob))
        pipe.add_stage(TrainAEStage(config, train_data_glob=train_data_glob, seed=42))
        pipe.add_stage(PreprocessAEStage(config))
        pipe.add_stage(AutoEncoderInferenceStage(config))
        pipe.add_stage(AddScoresStage(config))
        pipe.add_stage(
            TimeSeriesStage(config,
                            resolution="1m",
                            min_window="12 h",
                            hot_start=True,
                            cold_end=False,
                            filter_percent=90.0,
                            zscore_threshold=8.0))
        pipe.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))
        pipe.add_stage(
            ValidationStage(config,
                            val_file_name=val_file_name,
                            results_file_name=results_file_name,
                            index_col="_index_",
                            exclude=("event_dt", ),
                            rel_tol=0.1))
        pipe.add_stage(SerializeStage(config, include=[], output_type="pandas"))
        pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))

        pipe.run()
        results = self._calc_error_val(results_file_name)
        self.assertEqual(results.diff_rows, 48)


if __name__ == '__main__':
    unittest.main()
