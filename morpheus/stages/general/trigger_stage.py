# Copyright (c) 2021-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import typing

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


@register_stage("trigger")
class TriggerStage(SinglePortStage):
    """
    Buffer data until the previous stage has completed.

    This stage will buffer all inputs until the source stage is complete. At that point all messages
    will be dumped into downstream stages. Useful for testing performance of one stage at a time.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        super().__init__(c)

    @property
    def name(self) -> str:
        return "trigger"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (typing.Any, )

    def supports_cpp_node(self):
        return False

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        # Store all messages until on_complete is called and then push them
        def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):

            obs.pipe(ops.to_list(), ops.flatten()).subscribe(sub)

        node = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]
