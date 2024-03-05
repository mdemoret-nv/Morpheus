#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import typing

import mrc

from morpheus._lib.stages.operators import ControlMessageRouter
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.stages.input.http_client_source_stage import HttpClientSourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage


class TestStage(SinglePortStage):

    def __init__(self,
                 config: Config,
                 build_test_fn: typing.Callable[[mrc.Builder, mrc.SegmentObject], mrc.SegmentObject]):
        super().__init__(config)

        self._build_test_fn = build_test_fn

    @property
    def name(self):
        return "test-stage"

    def accepted_types(self):
        return (ControlMessage, MessageMeta)

    def supports_cpp_node(self):
        return False

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:

        return self._build_test_fn(builder, input_node)


def test_get_source():

    def create_router(builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:

        router = ControlMessageRouter(builder, "my-router")

        router_source1 = router.get_source("source1")
        router_source2 = router.get_source("source2")

        sink1 = builder.make_sink("sink1", lambda x: print("Got message from source1"))
        sink2 = builder.make_sink("sink2", lambda x: print("Got message from source2"))

        builder.make_edge(router_source1, sink1)
        builder.make_edge(router_source2, sink2)

        return sink1

    config = Config()

    pipe = LinearPipeline(config)
    pipe.set_source(
        HttpClientSourceStage(config=config, url="http://localhost", max_retries=1, lines=True, stop_after=1))
    comp_stage = pipe.add_stage(TestStage(config, create_router))
    pipe.run()
