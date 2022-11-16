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

import numpy as np
import pandas as pd
import pytest
import srf

import cudf

from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.messages import MultiResponseProbsMessage
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from utils import TEST_DIRS
from utils import ConvMsg
from utils import assert_file_exists_with_timeout
from utils import get_column_names_from_file


class CheckPreAlloc(SinglePortStage):
    """
    Acts like add-class/add-scores in that it requests a preallocation, the node will assert that the preallocation
    occurred with the correct type.
    """

    def __init__(self, c, probs_type):
        super().__init__(c)
        self._expected_type = cudf.dtype(probs_type)
        self._class_labels = c.class_labels
        self.needed_columns.update({label: probs_type for label in c.class_labels})

    @property
    def name(self):
        return "check-prealloc"

    def accepted_types(self):
        return (MultiMessage, )

    def supports_cpp_node(self):
        return False

    def _check_prealloc(self, m):
        df = m.get_meta()
        for label in self._class_labels:
            assert label in df.columns
            assert df[label].dtype == self._expected_type

        return m

    def _build_single(self, builder: srf.Builder, input_stream):
        stream = builder.make_node(self.unique_name, self._check_prealloc)
        builder.make_edge(input_stream[0], stream)

        return stream, input_stream[1]


@pytest.mark.slow
@pytest.mark.parametrize('probs_type', ['f4', 'f8'])
def test_preallocation(config, tmp_path, probs_type):
    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']

    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, 'results.csv')

    input_cols = get_column_names_from_file(input_file)

    file_src = FileSourceStage(config, filename=input_file, iterative=False)
    assert len(file_src.needed_columns) == 0

    pipe = LinearPipeline(config)
    pipe.set_source(file_src)
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(ConvMsg(config, columns=input_cols, probs_type=probs_type))
    pipe.add_stage(CheckPreAlloc(config, probs_type=probs_type))
    pipe.add_stage(SerializeStage(config, include=["^{}$".format(c) for c in config.class_labels]))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    assert file_src.needed_columns == {
        'frogs': probs_type, 'lizards': probs_type, 'toads': probs_type, 'turtles': probs_type
    }

    # There seems to be some sort of race between the sync to the output file when cpp=True and repeat=100
    assert_file_exists_with_timeout(out_file, 1.0)


@pytest.mark.slow
@pytest.mark.parametrize('probs_type', ['f4', 'f8'])
def test_preallocation_multi_segment_pipe(config, tmp_path, probs_type):
    """
    Test ensures that when columns are needed for preallocation in a multi-segment pipeline, the preallocagtion will
    always be performed on the closest source to the stage that requested preallocation. Which in cases where the
    requesting stage is not in the first segment, then the preallocation will be performed on the segment ingress
    """
    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']

    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, 'results.csv')

    file_src = FileSourceStage(config, filename=input_file, iterative=False)
    assert len(file_src.needed_columns) == 0

    pipe = LinearPipeline(config)
    pipe.set_source(file_src)
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(DeserializeStage(config))
    pipe.add_segment_boundary(MultiMessage)
    pipe.add_stage(ConvMsg(config, columns=get_column_names_from_file(input_file), probs_type=probs_type))
    (_, boundary_ingress) = pipe.add_segment_boundary(MultiResponseProbsMessage)
    pipe.add_stage(CheckPreAlloc(config, probs_type=probs_type))
    pipe.add_segment_boundary(MultiResponseProbsMessage)
    pipe.add_stage(SerializeStage(config, include=["^{}$".format(c) for c in config.class_labels]))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    assert len(file_src.needed_columns) == 0
    boundary_ingress.needed_columns == {
        'frogs': probs_type, 'lizards': probs_type, 'toads': probs_type, 'turtles': probs_type
    }

    assert_file_exists_with_timeout(out_file, 1.0)