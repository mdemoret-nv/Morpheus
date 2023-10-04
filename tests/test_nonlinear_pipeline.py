#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import dataclasses
import typing

import mrc
import mrc.core.operators as ops
import pytest
from mrc.core.node import Broadcast

from _utils import assert_results
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.pipeline import Pipeline
from morpheus.pipeline.stage import Stage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage


@dataclasses.dataclass(init=False)
class LoopedMessageMeta(MessageMeta):
    loop_count: int = dataclasses.field(init=False, default=0)

    def __init__(self, other: MessageMeta) -> None:
        super().__init__(other.df)

        if (isinstance(other, LoopedMessageMeta)):
            self.loop_count = other.loop_count

        self.loop_count += 1


class SplitStage(Stage):

    def __init__(self, c: Config):
        super().__init__(c)

        self._create_ports(1, 2)

    @property
    def name(self) -> str:
        return "split"

    def supports_cpp_node(self):
        return False

    def _build(self, builder: mrc.Builder, in_ports_streams: typing.List[StreamPair]) -> typing.List[StreamPair]:

        assert len(in_ports_streams) == 1, "Only 1 input supported"

        # Create a broadcast node
        broadcast = Broadcast(builder, "broadcast")
        builder.make_edge(in_ports_streams[0][0], broadcast)

        def filter_higher_fn(data: MessageMeta):
            return MessageMeta(data.df[data.df["v2"] >= 0.5])

        def filter_lower_fn(data: MessageMeta):
            return MessageMeta(data.df[data.df["v2"] < 0.5])

        # Create a node that only passes on rows >= 0.5
        filter_higher = builder.make_node("filter_higher", ops.map(filter_higher_fn))
        builder.make_edge(broadcast, filter_higher)

        # Create a node that only passes on rows < 0.5
        filter_lower = builder.make_node("filter_lower", ops.map(filter_lower_fn))
        builder.make_edge(broadcast, filter_lower)

        return [(filter_higher, in_ports_streams[0][1]), (filter_lower, in_ports_streams[0][1])]


class CyclicStage(Stage):

    def __init__(self, c: Config):
        super().__init__(c)

        self._create_ports(1, 2)

    @property
    def name(self) -> str:
        return "cyclic"

    def supports_cpp_node(self):
        return False

    def _build(self, builder: mrc.Builder, in_ports_streams: typing.List[StreamPair]) -> typing.List[StreamPair]:

        assert len(in_ports_streams) == 1, "Only 1 input supported"

        def create_loop_fn(data: MessageMeta):
            return LoopedMessageMeta(data)

        # Convert the incoming message to the LoopedMessageMeta
        create_loop = builder.make_node("create_loop", ops.map(create_loop_fn))
        builder.make_edge(in_ports_streams[0][0], create_loop)

        # Create a broadcast node
        broadcast = Broadcast(builder, "broadcast")
        builder.make_edge(create_loop, broadcast)

        def filter_higher_fn(data: LoopedMessageMeta):
            return data.loop_count > 3

        def filter_lower_fn(data: LoopedMessageMeta):
            return data.loop_count <= 3

        # Create a node that only passes on rows >= 0.5
        filter_higher = builder.make_node("filter_higher", ops.filter(filter_higher_fn))
        builder.make_edge(broadcast, filter_higher)

        # Create a node that only passes on rows < 0.5
        filter_lower = builder.make_node("filter_lower", ops.filter(filter_lower_fn))
        builder.make_edge(broadcast, filter_lower)

        return [(filter_higher, in_ports_streams[0][1]), (filter_lower, in_ports_streams[0][1])]


def test_forking_pipeline(config, dataset_cudf: DatasetManager):
    filter_probs_df = dataset_cudf["filter_probs.csv"]
    compare_higher_df = filter_probs_df[filter_probs_df["v2"] >= 0.5]
    compare_lower_df = filter_probs_df[filter_probs_df["v2"] < 0.5]

    pipe = Pipeline(config)

    # Create the stages
    source = pipe.add_stage(InMemorySourceStage(config, [filter_probs_df]))

    split_stage = pipe.add_stage(SplitStage(config))

    comp_higher = pipe.add_stage(CompareDataFrameStage(config, compare_df=compare_higher_df))
    comp_lower = pipe.add_stage(CompareDataFrameStage(config, compare_df=compare_lower_df))

    # Create the edges
    pipe.add_edge(source, split_stage)
    pipe.add_edge(split_stage.output_ports[0], comp_higher)
    pipe.add_edge(split_stage.output_ports[1], comp_lower)

    pipe.run()

    # Get the results
    assert_results(comp_higher.get_results())
    assert_results(comp_lower.get_results())


@pytest.mark.parametrize("source_count, expected_count", [(1, 1), (2, 2), (3, 3)])
def test_port_multi_sender(config, dataset_cudf: DatasetManager, source_count, expected_count):

    filter_probs_df = dataset_cudf["filter_probs.csv"]

    pipe = Pipeline(config)

    input_ports = []
    for x in range(source_count):
        input_port = f"input_{x}"
        input_ports.append(input_port)

    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    for x in range(source_count):
        source_stage = pipe.add_stage(InMemorySourceStage(config, [filter_probs_df]))
        pipe.add_edge(source_stage, sink_stage)

    pipe.run()

    assert len(sink_stage.get_messages()) == expected_count


def test_cyclic_pipeline(config, dataset_cudf: DatasetManager):
    filter_probs_df = dataset_cudf["filter_probs.csv"]

    pipe = Pipeline(config)

    # Create the stages
    source = pipe.add_stage(InMemorySourceStage(config, [filter_probs_df, filter_probs_df]))

    cyclic_stage = pipe.add_stage(CyclicStage(config))

    sink = pipe.add_stage(InMemorySinkStage(config))

    # Create the edges
    pipe.add_edge(source, cyclic_stage)
    pipe.add_edge(cyclic_stage.output_ports[0], sink)
    pipe.add_edge(cyclic_stage.output_ports[1], cyclic_stage)

    pipe.run()

    # Get the results
    messages: list[LoopedMessageMeta] = sink.get_messages()

    assert len(messages) == 2

    for message in messages:
        assert message.loop_count == 4
