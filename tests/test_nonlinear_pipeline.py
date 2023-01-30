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
from mrc.core.node import Broadcast
import mrc.core.operators as ops
import pytest
from morpheus.messages import MessageMeta

from morpheus.pipeline.pipeline import Pipeline
from morpheus.pipeline.stage import Stage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.postprocess.validation_stage import ValidationStage
from utils import TEST_DIRS

class SplitStage(Stage):
    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned. Derived classes should override this method. An
        error will be generated if the input types to the stage do not match one of the available types
        returned from this method.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (MessageMeta, )

    def _build(self, builder: mrc.Builder, in_ports_streams: typing.List[StreamPair]) -> typing.List[StreamPair]:

        assert len(in_ports_streams) == 1, "Only 1 input supported"

        # Create a broadcast node
        broadcast = Broadcast(builder, "broadcast")
        builder.make_edge(in_ports_streams[0][0], broadcast)

        # Create a node that only passes on rows >= 0.5
        filter_higher = builder.make_node("filter_higher", ops.map(lambda data: data[data["v2"] >= 0.5]))
        builder.make_edge(broadcast, filter_higher)

        # Create a node that only passes on rows < 0.5
        filter_lower = builder.make_node("filter_lower", ops.map(lambda data: data[data["v2"] < 0.5]))
        builder.make_edge(broadcast, filter_lower)

        return [(filter_higher, in_ports_streams[0][1]), (filter_lower, in_ports_streams[0][1])]


def test_forking_pipeline(config):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    # out_file = os.path.join(tmp_path, 'results.{}'.format(output_type))

    pipe = Pipeline(config)
    source = pipe.add_stage(FileSourceStage(config, filename=input_file))

    split_stage = pipe.add_stage(SplitStage(config))

    val_higher = pipe.add_stage(ValidationStage(config, val_file_name=input_file))
    val_lower = pipe.add_stage(ValidationStage(config, val_file_name=input_file))

    # Create the edges
    pipe.add_edge(source, split_stage)
    pipe.add_edge(split_stage, val_higher)
    pipe.add_edge(split_stage, val_lower)

    pipe.run()

    # Get the results
    results1 = val_higher.get_results()
    results2 = val_lower.get_results()

    assert results1.diff_rows == 0
    assert results2.diff_rows == 0
