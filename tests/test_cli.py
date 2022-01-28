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

import click
from click.testing import CliRunner

from morpheus import cli
from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.config import PipelineModes
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
from tests import WORKSPACE_DIR
from tests import BaseMorpheusTest

GENERAL_ARGS = ['run', '--num_threads=12', '--pipeline_batch_size=1024', '--model_max_batch_size=1024', '--use_cpp=0']
PIPE_AE_ARGS = ['pipeline-ae', '--userid_filter=user321', '--userid_column_name=user_col']
CLOUD_TRAIL_ARGS = ['from-cloudtrail', '--input_glob=input_glob*.csv']
TRAIN_AE_ARGS = ['train-ae', '--train_data_glob=train_glob*.csv', '--seed', '47']
TIME_SERIES_ARGS = ['timeseries', '--resolution=1m', '--zscore_threshold=8.0', '--hot_start']
MONITOR_ARGS = ['monitor', '--description', 'Unittest', '--smoothing=0.001', '--unit', 'inf']
VALIDATE_ARGS = ['validate', '--val_file_name',
                 os.path.join(WORKSPACE_DIR, 'models/datasets/validation-data/hammah-role-g-validation-data.csv'),
                 '--results_file_name=results.json', '--index_col=_index_', '--exclude', 'event_dt', '--rel_tol=0.1']
TO_FILE_ARGS = ['to-file', '--filename=out.csv']

class TestCli(BaseMorpheusTest):
    def _replace_results_callback(self, group, exit_val=47):
        """
        Replaces the results_callback in cli which executes the pipeline.
        Allowing us to examine/verify that cli built us a propper pipeline
        without actually running it. When run the callback will update the
        `callback_values` dictionary with the context and stages constructed.
        """
        callback_values = {}
        @group.result_callback(replace=True)
        @click.pass_context
        def mock_post_callback(ctx, stages, *a, **k):
            callback_values.update({'ctx': ctx, 'stages': stages})
            ctx.exit(exit_val)

        return callback_values

    def test_pipeline_ae(self):
        args = GENERAL_ARGS + PIPE_AE_ARGS + CLOUD_TRAIL_ARGS + TRAIN_AE_ARGS + \
               ['preprocess', 'inf-pytorch', 'add-scores'] + \
               TIME_SERIES_ARGS + MONITOR_ARGS + VALIDATE_ARGS + ['serialize'] + TO_FILE_ARGS

        callback_values = self._replace_results_callback(cli.pipeline_ae)

        runner = CliRunner()
        result = runner.invoke(cli.cli, args)
        self.assertEqual(result.exit_code, 47, result.output)

        # Ensure our config is populated correctly
        config = Config.get()
        self.assertEqual(config.mode, PipelineModes.AE)
        self.assertFalse(config.use_cpp)
        self.assertEqual(config.class_labels, ["ae_anomaly_score"])
        self.assertEqual(config.model_max_batch_size, 1024)
        self.assertEqual(config.pipeline_batch_size, 1024)
        self.assertEqual(config.num_threads, 12)

        self.assertIsInstance(config.ae, ConfigAutoEncoder)
        config.ae.userid_column_name = "user_col"
        config.ae.userid_filter = "user321"

        ctx = callback_values['ctx']
        pipe = ctx.find_object(LinearPipeline)
        self.assertIsNotNone(pipe)

        stages = callback_values['stages']
        # Verify the stages are as we expect them, if there is a size-mismatch python will raise a Value error
        [cloud_trail, train_ae, process_ae, auto_enc, add_scores, time_series, monitor,
            validation, serialize, to_file] = stages

        self.assertIsInstance(cloud_trail, CloudTrailSourceStage)
        self.assertEqual(cloud_trail._input_glob, "input_glob*.csv")

        self.assertIsInstance(train_ae, TrainAEStage)
        self.assertEqual(train_ae._train_data_glob, "train_glob*.csv")
        self.assertEqual(train_ae._seed, 47)

        self.assertIsInstance(process_ae, PreprocessAEStage)
        self.assertIsInstance(auto_enc, AutoEncoderInferenceStage)
        self.assertIsInstance(add_scores, AddScoresStage)

        self.assertIsInstance(time_series, TimeSeriesStage)
        self.assertEqual(time_series._resolution, '1m')
        self.assertEqual(time_series._zscore_threshold, 8.0)
        self.assertTrue(time_series._hot_start)

        self.assertIsInstance(monitor, MonitorStage)
        self.assertEqual(monitor._description,  'Unittest')
        self.assertEqual(monitor._smoothing,  0.001)
        self.assertEqual(monitor._unit, 'inf')

        self.assertIsInstance(validation, ValidationStage)
        self.assertEqual(validation._val_file_name, os.path.join(self._validation_data_dir, 'hammah-role-g-validation-data.csv'))
        self.assertEqual(validation._results_file_name, 'results.json')
        self.assertEqual(validation._index_col, '_index_')

        # Click appears to be converting this into a tuple
        self.assertEqual(list(validation._exclude_columns), ['event_dt'])
        self.assertEqual(validation._rel_tol, 0.1)

        self.assertIsInstance(serialize, SerializeStage)
        self.assertEqual(serialize._output_type, 'pandas')

        self.assertIsInstance(to_file, WriteToFileStage)
        self.assertEqual(to_file._output_file, 'out.csv')


if __name__ == '__main__':
    unittest.main()
