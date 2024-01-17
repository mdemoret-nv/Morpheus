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

import logging
import typing

import mrc

import morpheus._lib.stages as _stages
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema

logger = logging.getLogger(__name__)


@register_stage("dataframe_loader")
class DataFrameLoaderStage(SinglePortStage):

    def __init__(self, c: Config):
        super().__init__(c)

    @property
    def name(self) -> str:
        return "dataframe_loader"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        """
        return (str, )

    def supports_cpp_node(self):
        # Enable support by default
        return True

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:

        if self._build_cpp_node():
            node = _stages.DataFrameLoaderStage(builder, self.unique_name)

            node.launch_options.pe_count = self._config.num_threads
        else:
            raise NotImplementedError("Python node not implemented")

        builder.make_edge(input_node, node)

        return node
