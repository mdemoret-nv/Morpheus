# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils.module_utils import load_module

logger = logging.getLogger(__name__)


class LinearModulesStage(SinglePortStage):
    """
    Loads an existing, registered, MRC SegmentModule and wraps it as a Morpheus SinglePortStage.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    module_config : typing.Dict
        Module configuration.
    input_port_name : str
        Name of the input port for the registered module.
    output_port_name : str
        Name of the output port for the registered module
    input_type : default `typing.Any`
        The stage acceptable input type.
    output_type : default `typing.Any`
        The output type that the stage produces.

    """

    def __init__(self,
                 c: Config,
                 module_config: typing.Dict,
                 input_port_name: str,
                 output_port_name: str,
                 input_type=typing.Any,
                 output_type=typing.Any):

        super().__init__(c)

        self._input_type = input_type
        self._ouput_type = output_type
        self._module_config = module_config
        self._input_port_name = input_port_name
        self._output_port_name = output_port_name

    @property
    def name(self) -> str:
        return self._module_config.get("module_name", "linear_module")

    def supports_cpp_node(self):
        return False

    def input_types(self) -> typing.Tuple:
        """
        Returns input type for the current stage.
        """

        return (typing.Any, )

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (self._input_type, )

    def _get_cpp_module_node(self, builder: mrc.Builder) -> mrc.SegmentObject:
        raise NotImplementedError("No C++ node is available for this module type")

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        # Laod module from registry.
        module = load_module(self._module_config, builder=builder)

        mod_in_stream = module.input_port(self._input_port_name)
        mod_out_stream = module.output_port(self._output_port_name)

        builder.make_edge(input_stream[0], mod_in_stream)

        return mod_out_stream, self._ouput_type
