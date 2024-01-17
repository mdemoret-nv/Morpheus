#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from _utils import TEST_DIRS
from _utils import assert_results
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import source
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.dataframe_loader_stage import DataFrameLoaderStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage


@pytest.mark.use_cpp
def test_dataframe_loader(config: Config):
    """
    End to end test for DeserializeStage
    """

    filenames = [
        os.path.join(TEST_DIRS.tests_data_dir, 'examples/abp_pcap_detection/abp_pcap.jsonlines'),
        os.path.join(TEST_DIRS.tests_data_dir, 'examples/developer_guide/email_with_addresses_first_10.jsonlines'),
        os.path.join(TEST_DIRS.validation_data_dir, 'abp-validation-data.jsonlines')
    ]

    @source
    def emit_df_filenames() -> str:

        for filename in filenames:
            yield filename

    pipe = LinearPipeline(config)
    pipe.set_source(emit_df_filenames(config=config))
    pipe.add_stage(DataFrameLoaderStage(config))
    pipe.add_stage(MonitorStage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    assert len(sink.get_messages()) == len(filenames)
