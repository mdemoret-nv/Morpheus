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
import warnings

import click
from click.testing import CliRunner

from morpheus.config import Config

Config.get().use_cpp = False

from morpheus import cli
from morpheus.config import ConfigAutoEncoder
from morpheus.config import PipelineModes
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.general_stages import AddClassificationsStage
from morpheus.pipeline.general_stages import AddScoresStage
from morpheus.pipeline.general_stages import FilterDetectionsStage
from morpheus.pipeline.general_stages import MonitorStage
from morpheus.pipeline.inference.inference_ae import AutoEncoderInferenceStage
from morpheus.pipeline.inference.inference_identity import IdentityInferenceStage
from morpheus.pipeline.inference.inference_pytorch import PyTorchInferenceStage
from morpheus.pipeline.inference.inference_triton import TritonInferenceStage
from morpheus.pipeline.input.from_cloudtrail import CloudTrailSourceStage
from morpheus.pipeline.input.from_file import FileSourceStage
from morpheus.pipeline.input.from_kafka import KafkaSourceStage
from morpheus.pipeline.output.serialize import SerializeStage
from morpheus.pipeline.output.to_file import WriteToFileStage
from morpheus.pipeline.output.to_kafka import WriteToKafkaStage
from morpheus.pipeline.output.validation import ValidationStage
from morpheus.pipeline.postprocess.timeseries import TimeSeriesStage
from morpheus.pipeline.preprocess.autoencoder import PreprocessAEStage
from morpheus.pipeline.preprocess.autoencoder import TrainAEStage
from morpheus.pipeline.preprocessing import DeserializeStage
from morpheus.pipeline.preprocessing import DropNullStage
from morpheus.pipeline.preprocessing import PreprocessFILStage
from morpheus.pipeline.preprocessing import PreprocessNLPStage
from tests import VALIDATION_DATA_DIR
from tests import BaseMorpheusTest

GENERAL_ARGS = ['run', '--num_threads=12', '--pipeline_batch_size=1024', '--model_max_batch_size=1024', '--use_cpp=0']
MONITOR_ARGS = ['monitor', '--description', 'Unittest', '--smoothing=0.001', '--unit', 'inf']
VALIDATE_ARGS = ['validate', '--val_file_name',
                 os.path.join(VALIDATION_DATA_DIR, 'hammah-role-g-validation-data.csv'),
                 '--results_file_name=results.json', '--index_col=_index_', '--exclude', 'event_dt', '--rel_tol=0.1']
TO_FILE_ARGS = ['to-file', '--filename=out.csv']
FILE_SRC_ARGS = ['from-file', '--filename', os.path.join(VALIDATION_DATA_DIR, 'abp-validation-data.jsonlines')]
INF_TRITON_ARGS = ['inf-triton', '--model_name=test-model', '--server_url=test:123', '--force_convert_inputs=True']

KAFKA_BOOTS = ['--bootstrap_servers', 'kserv1:123,kserv2:321']
FROM_KAFKA_ARGS = ['from-kafka', '--input_topic', 'test_topic'] + KAFKA_BOOTS
TO_KAFKA_ARGS = ['to-kafka',  '--output_topic', 'test_topic'] + KAFKA_BOOTS

class TestCli(BaseMorpheusTest):
    def _replace_results_callback(self, group, exit_val=47):
        """
        Replaces the results_callback in cli which executes the pipeline.
        Allowing us to examine/verify that cli built us a propper pipeline
        without actually running it. When run the callback will update the
        `callback_values` dictionary with the context, pipeline & stages constructed.
        """
        callback_values = {}
        @group.result_callback(replace=True)
        @click.pass_context
        def mock_post_callback(ctx, stages, *a, **k):
            callback_values.update({'ctx': ctx, 'stages': stages, 'pipe': ctx.find_object(LinearPipeline)})
            ctx.exit(exit_val)

        return callback_values

    def test_pipeline_ae(self):
        """
        Build a pipeline roughly ressembles the hammah validation script
        """
        args = GENERAL_ARGS + \
               ['pipeline-ae', '--userid_filter=user321', '--userid_column_name=user_col',
                'from-cloudtrail', '--input_glob=input_glob*.csv',
                'train-ae', '--train_data_glob=train_glob*.csv', '--seed', '47',
                'preprocess', 'inf-pytorch', 'add-scores',
                'timeseries', '--resolution=1m', '--zscore_threshold=8.0', '--hot_start'] + \
               MONITOR_ARGS + VALIDATE_ARGS + ['serialize'] + TO_FILE_ARGS

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

        pipe = callback_values['pipe']
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


    def test_pipeline_ae_all(self):
        """
        Attempt to add all possible stages to the pipeline_ae, even if the pipeline doesn't
        actually make sense, just test that cli could assemble it
        """
        args = GENERAL_ARGS + \
               ['pipeline-ae', '--userid_filter=user321', '--userid_column_name=user_col',
                'from-cloudtrail', '--input_glob=input_glob*.csv',
                'add-class', 'filter',
                'train-ae', '--train_data_glob=train_glob*.csv', '--seed', '47',
                'preprocess', 'inf-pytorch', 'add-scores'] + \
               INF_TRITON_ARGS + \
               ['timeseries', '--resolution=1m', '--zscore_threshold=8.0', '--hot_start'] + \
               MONITOR_ARGS + VALIDATE_ARGS + ['serialize'] + TO_FILE_ARGS + TO_KAFKA_ARGS

        callback_values = self._replace_results_callback(cli.pipeline_ae)

        runner = CliRunner()
        result = runner.invoke(cli.cli, args)

        self.assertEqual(result.exit_code, 47, result.output)

        stages = callback_values['stages']
        # Verify the stages are as we expect them, if there is a size-mismatch python will raise a Value error
        [cloud_trail, add_class, filter_stage, train_ae, process_ae, auto_enc, add_scores, triton_inf, time_series,
         monitor, validation, serialize, to_file, to_kafka] = stages

        self.assertIsInstance(cloud_trail, CloudTrailSourceStage)
        self.assertEqual(cloud_trail._input_glob, "input_glob*.csv")

        self.assertIsInstance(add_class, AddClassificationsStage)
        self.assertIsInstance(filter_stage, FilterDetectionsStage)

        self.assertIsInstance(train_ae, TrainAEStage)
        self.assertEqual(train_ae._train_data_glob, "train_glob*.csv")
        self.assertEqual(train_ae._seed, 47)

        self.assertIsInstance(process_ae, PreprocessAEStage)
        self.assertIsInstance(auto_enc, AutoEncoderInferenceStage)
        self.assertIsInstance(add_scores, AddScoresStage)

        self.assertIsInstance(triton_inf, TritonInferenceStage)
        self.assertEqual(triton_inf._kwargs['model_name'], 'test-model')
        self.assertEqual(triton_inf._kwargs['server_url'], 'test:123')
        self.assertTrue(triton_inf._kwargs['force_convert_inputs'])

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

        self.assertIsInstance(to_kafka, WriteToKafkaStage)
        self.assertEqual(to_kafka._kafka_conf['bootstrap.servers'], 'kserv1:123,kserv2:321')
        self.assertEqual(to_kafka._output_topic, 'test_topic')


    def test_pipeline_fil(self):
        """
        Creates a pipeline roughly matching that of the abp validation test
        """
        args = GENERAL_ARGS + ['pipeline-fil'] + FILE_SRC_ARGS + ['deserialize', 'preprocess'] + INF_TRITON_ARGS + \
               MONITOR_ARGS + ['add-class'] + VALIDATE_ARGS + ['serialize'] + TO_FILE_ARGS

        callback_values = self._replace_results_callback(cli.pipeline_fil)

        runner = CliRunner()
        result = runner.invoke(cli.cli, args)
        self.assertEqual(result.exit_code, 47, result.output)

        # Ensure our config is populated correctly
        config = Config.get()
        self.assertEqual(config.mode, PipelineModes.FIL)
        self.assertEqual(config.class_labels, ["mining"])

        self.assertIsNone(config.ae)

        pipe = callback_values['pipe']
        self.assertIsNotNone(pipe)

        stages = callback_values['stages']
        # Verify the stages are as we expect them, if there is a size-mismatch python will raise a Value error
        [file_source, deserialize, process_fil, triton_inf, monitor, add_class, validation, serialize, to_file] = stages

        self.assertIsInstance(file_source, FileSourceStage)
        self.assertEqual(file_source._filename, os.path.join(self._validation_data_dir, 'abp-validation-data.jsonlines'))
        self.assertFalse(file_source._iterative)

        self.assertIsInstance(deserialize, DeserializeStage)
        self.assertIsInstance(process_fil, PreprocessFILStage)

        self.assertIsInstance(triton_inf, TritonInferenceStage)
        self.assertEqual(triton_inf._kwargs['model_name'], 'test-model')
        self.assertEqual(triton_inf._kwargs['server_url'], 'test:123')
        self.assertTrue(triton_inf._kwargs['force_convert_inputs'])

        self.assertIsInstance(monitor, MonitorStage)
        self.assertEqual(monitor._description,  'Unittest')
        self.assertEqual(monitor._smoothing,  0.001)
        self.assertEqual(monitor._unit, 'inf')

        self.assertIsInstance(add_class, AddClassificationsStage)

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


    def test_pipeline_fil_all(self):
        """
        Attempt to add all possible stages to the pipeline_fil, even if the pipeline doesn't
        actually make sense, just test that cli could assemble it
        """
        tmp_dir = self._mk_tmp_dir()
        tmp_model = os.path.join(tmp_dir, 'fake-model.file')
        with open(tmp_model, 'w') as fh:
            pass

        args = GENERAL_ARGS + ['pipeline-fil'] + FILE_SRC_ARGS + FROM_KAFKA_ARGS +\
               ['deserialize', 'filter',
                'dropna', '--column', 'xyz',
                'preprocess', 'add-scores', 'inf-identity',
                'inf-pytorch', '--model_filename', tmp_model,
                # 'mflow-drift', <-- Missing dep
                ] + \
               INF_TRITON_ARGS + MONITOR_ARGS + ['add-class'] + VALIDATE_ARGS + ['serialize'] + TO_FILE_ARGS + TO_KAFKA_ARGS

        callback_values = self._replace_results_callback(cli.pipeline_fil)

        runner = CliRunner()
        result = runner.invoke(cli.cli, args)
        self.assertEqual(result.exit_code, 47, result.output)

        # Ensure our config is populated correctly
        config = Config.get()
        self.assertEqual(config.mode, PipelineModes.FIL)
        self.assertEqual(config.class_labels, ["mining"])

        self.assertIsNone(config.ae)

        pipe = callback_values['pipe']
        self.assertIsNotNone(pipe)

        stages = callback_values['stages']
        # Verify the stages are as we expect them, if there is a size-mismatch python will raise a Value error
        [file_source, from_kafka, deserialize, filter_stage, dropna, process_fil, add_scores, inf_ident, inf_pytorch,
         triton_inf, monitor, add_class, validation, serialize, to_file, to_kafka] = stages

        self.assertIsInstance(file_source, FileSourceStage)
        self.assertEqual(file_source._filename, os.path.join(self._validation_data_dir, 'abp-validation-data.jsonlines'))
        self.assertFalse(file_source._iterative)

        self.assertIsInstance(from_kafka, KafkaSourceStage)
        self.assertEqual(from_kafka._consumer_conf['bootstrap.servers'], 'kserv1:123,kserv2:321')
        self.assertEqual(from_kafka._input_topic, 'test_topic')

        self.assertIsInstance(deserialize, DeserializeStage)
        self.assertIsInstance(filter_stage, FilterDetectionsStage)

        self.assertIsInstance(dropna, DropNullStage)
        self.assertEqual(dropna._column, 'xyz')

        self.assertIsInstance(process_fil, PreprocessFILStage)

        self.assertIsInstance(add_scores, AddScoresStage)
        self.assertIsInstance(inf_ident, IdentityInferenceStage)

        self.assertIsInstance(inf_pytorch, PyTorchInferenceStage)
        self.assertEqual(inf_pytorch._model_filename, tmp_model)

        self.assertIsInstance(triton_inf, TritonInferenceStage)
        self.assertEqual(triton_inf._kwargs['model_name'], 'test-model')
        self.assertEqual(triton_inf._kwargs['server_url'], 'test:123')
        self.assertTrue(triton_inf._kwargs['force_convert_inputs'])

        self.assertIsInstance(monitor, MonitorStage)
        self.assertEqual(monitor._description,  'Unittest')
        self.assertEqual(monitor._smoothing,  0.001)
        self.assertEqual(monitor._unit, 'inf')

        self.assertIsInstance(add_class, AddClassificationsStage)

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

        self.assertIsInstance(to_kafka, WriteToKafkaStage)
        self.assertEqual(to_kafka._kafka_conf['bootstrap.servers'], 'kserv1:123,kserv2:321')
        self.assertEqual(to_kafka._output_topic, 'test_topic')


    def test_pipeline_nlp(self):
        """
        Build a pipeline roughly ressembles the phishing validation script
        """
        labels_file = os.path.join(self._data_dir, 'labels_phishing.txt')
        vocab_file_name = os.path.join(self._data_dir, 'bert-base-uncased-hash.txt')
        args = GENERAL_ARGS + \
               ['pipeline-nlp', '--model_seq_length=128', '--labels_file', labels_file] + \
               FILE_SRC_ARGS +  \
               ['deserialize',
                'preprocess', '--vocab_hash_file', vocab_file_name, '--truncation=True',
                '--do_lower_case=True', '--add_special_tokens=False'] + \
               INF_TRITON_ARGS + MONITOR_ARGS + \
               ['add-class', '--label=pred', '--threshold=0.7'] + \
               VALIDATE_ARGS + ['serialize'] + TO_FILE_ARGS

        callback_values = self._replace_results_callback(cli.pipeline_nlp)

        runner = CliRunner()
        result = runner.invoke(cli.cli, args)
        self.assertEqual(result.exit_code, 47, result.output)

        # Ensure our config is populated correctly
        config = Config.get()
        self.assertEqual(config.mode, PipelineModes.NLP)
        self.assertEqual(config.class_labels, ["score", "pred"])
        self.assertEqual(config.feature_length, 128)

        self.assertIsNone(config.ae)

        pipe = callback_values['pipe']
        self.assertIsNotNone(pipe)

        stages = callback_values['stages']
        # Verify the stages are as we expect them, if there is a size-mismatch python will raise a Value error
        [file_source, deserialize, process_nlp, triton_inf, monitor, add_class, validation, serialize, to_file] = stages

        self.assertIsInstance(file_source, FileSourceStage)
        self.assertEqual(file_source._filename, os.path.join(self._validation_data_dir, 'abp-validation-data.jsonlines'))
        self.assertFalse(file_source._iterative)

        self.assertIsInstance(deserialize, DeserializeStage)

        self.assertIsInstance(process_nlp, PreprocessNLPStage)
        self.assertEqual(process_nlp._vocab_hash_file, vocab_file_name)
        self.assertTrue(process_nlp._truncation)
        self.assertTrue(process_nlp._do_lower_case)
        self.assertFalse(process_nlp._add_special_tokens)

        self.assertIsInstance(triton_inf, TritonInferenceStage)
        self.assertEqual(triton_inf._kwargs['model_name'], 'test-model')
        self.assertEqual(triton_inf._kwargs['server_url'], 'test:123')
        self.assertTrue(triton_inf._kwargs['force_convert_inputs'])

        self.assertIsInstance(monitor, MonitorStage)
        self.assertEqual(monitor._description,  'Unittest')
        self.assertEqual(monitor._smoothing,  0.001)
        self.assertEqual(monitor._unit, 'inf')

        self.assertIsInstance(add_class, AddClassificationsStage)
        self.assertEqual(add_class._labels, ['pred'])
        self.assertEqual(add_class._threshold, 0.7)

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


    def test_pipeline_nlp_all(self):
        """
        Attempt to add all possible stages to the pipeline_nlp, even if the pipeline doesn't
        actually make sense, just test that cli could assemble it
        """
        tmp_dir = self._mk_tmp_dir()
        tmp_model = os.path.join(tmp_dir, 'fake-model.file')
        with open(tmp_model, 'w') as fh:
            pass

        labels_file = os.path.join(self._data_dir, 'labels_phishing.txt')
        vocab_file_name = os.path.join(self._data_dir, 'bert-base-uncased-hash.txt')
        args = GENERAL_ARGS + \
               ['pipeline-nlp', '--model_seq_length=128', '--labels_file', labels_file] + \
               FILE_SRC_ARGS + FROM_KAFKA_ARGS +  \
               ['deserialize', 'filter',
                'dropna', '--column', 'xyz',
                'preprocess', '--vocab_hash_file', vocab_file_name, '--truncation=True',
                '--do_lower_case=True', '--add_special_tokens=False',
                'add-scores', 'inf-identity',
                'inf-pytorch', '--model_filename', tmp_model,
                # 'mflow-drift', <-- Missing dep
                ] + \
               INF_TRITON_ARGS + MONITOR_ARGS + \
               ['add-class', '--label=pred', '--threshold=0.7'] + \
               VALIDATE_ARGS + ['serialize'] + TO_FILE_ARGS + TO_KAFKA_ARGS

        callback_values = self._replace_results_callback(cli.pipeline_nlp)

        runner = CliRunner()
        result = runner.invoke(cli.cli, args)
        self.assertEqual(result.exit_code, 47, result.output)

        # Ensure our config is populated correctly
        config = Config.get()
        self.assertEqual(config.mode, PipelineModes.NLP)
        self.assertEqual(config.class_labels, ["score", "pred"])
        self.assertEqual(config.feature_length, 128)

        self.assertIsNone(config.ae)

        pipe = callback_values['pipe']
        self.assertIsNotNone(pipe)

        stages = callback_values['stages']
        # Verify the stages are as we expect them, if there is a size-mismatch python will raise a Value error
        [file_source, from_kafka, deserialize, filter_stage, dropna, process_nlp, add_scores, inf_ident, inf_pytorch,
         triton_inf, monitor, add_class, validation, serialize, to_file, to_kafka] = stages

        self.assertIsInstance(file_source, FileSourceStage)
        self.assertEqual(file_source._filename, os.path.join(self._validation_data_dir, 'abp-validation-data.jsonlines'))
        self.assertFalse(file_source._iterative)

        self.assertIsInstance(from_kafka, KafkaSourceStage)
        self.assertEqual(from_kafka._consumer_conf['bootstrap.servers'], 'kserv1:123,kserv2:321')
        self.assertEqual(from_kafka._input_topic, 'test_topic')

        self.assertIsInstance(deserialize, DeserializeStage)
        self.assertIsInstance(filter_stage, FilterDetectionsStage)

        self.assertIsInstance(dropna, DropNullStage)
        self.assertEqual(dropna._column, 'xyz')

        self.assertIsInstance(process_nlp, PreprocessNLPStage)
        self.assertEqual(process_nlp._vocab_hash_file, vocab_file_name)
        self.assertTrue(process_nlp._truncation)
        self.assertTrue(process_nlp._do_lower_case)
        self.assertFalse(process_nlp._add_special_tokens)

        self.assertIsInstance(add_scores, AddScoresStage)
        self.assertIsInstance(inf_ident, IdentityInferenceStage)

        self.assertIsInstance(inf_pytorch, PyTorchInferenceStage)
        self.assertEqual(inf_pytorch._model_filename, tmp_model)

        self.assertIsInstance(triton_inf, TritonInferenceStage)
        self.assertEqual(triton_inf._kwargs['model_name'], 'test-model')
        self.assertEqual(triton_inf._kwargs['server_url'], 'test:123')
        self.assertTrue(triton_inf._kwargs['force_convert_inputs'])

        self.assertIsInstance(monitor, MonitorStage)
        self.assertEqual(monitor._description,  'Unittest')
        self.assertEqual(monitor._smoothing,  0.001)
        self.assertEqual(monitor._unit, 'inf')

        self.assertIsInstance(add_class, AddClassificationsStage)
        self.assertEqual(add_class._labels, ['pred'])
        self.assertEqual(add_class._threshold, 0.7)

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

        self.assertIsInstance(to_kafka, WriteToKafkaStage)
        self.assertEqual(to_kafka._kafka_conf['bootstrap.servers'], 'kserv1:123,kserv2:321')
        self.assertEqual(to_kafka._output_topic, 'test_topic')


if __name__ == '__main__':
    unittest.main()
