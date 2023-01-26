#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import typing
from unittest import mock

import mrc
import pytest

from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.postprocess.validation_stage import ValidationStage
from utils import TEST_DIRS


def test_forking_pipeline(config):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    # out_file = os.path.join(tmp_path, 'results.{}'.format(output_type))

    pipe = Pipeline(config)
    source = pipe.add_node(FileSourceStage(config, filename=input_file))

    val1 = pipe.add_node(ValidationStage(config, val_file_name=input_file))
    val2 = pipe.add_node(ValidationStage(config, val_file_name=input_file))

    # Create the edges
    pipe.add_edge(source, val1)
    pipe.add_edge(source, val2)

    pipe.run()

    # Get the results
    results1 = val1.get_results()
    results2 = val1.get_results()

    assert results1.diff_rows == 0
    assert results2.diff_rows == 0
