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

import os
import typing
from functools import partial

import cupy as cp
import mrc
import numpy as np
from mrc.core import operators as ops
from nemollm.api import NemoLLM

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import InferenceMemoryFIL
from morpheus.messages import MultiInferenceFILMessage
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiMessage
from morpheus.messages import TensorMemory
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.inference.inference_stage import InferenceStage
from morpheus.stages.inference.inference_stage import InferenceWorker
from morpheus.stages.preprocess.preprocess_base_stage import PreprocessBaseStage
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue

# class _NeMoInferenceWorker(InferenceWorker):
#     """
#     Inference worker used by PyTorchInferenceStage.

#     Parameters
#     ----------
#     inf_queue : `morpheus.utils.producer_consumer_queue.ProducerConsumerQueue`
#         Inference queue.
#     c : `morpheus.config.Config`
#         Pipeline configuration instance.
#     model_filename : str
#         Model file path.
#     """

#     def __init__(self, inf_queue: ProducerConsumerQueue, c: Config, model_filename: str):
#         super().__init__(inf_queue)

#         self._max_batch_size = c.model_max_batch_size
#         self._seq_length = c.feature_length
#         self._model_filename: str = model_filename

#         self._conn: NemoLLM = None

#         # Use this to cache the output size
#         self._output_size = None

#     def init(self):

#         self._conn = NemoLLM(
#             # The client must configure the authentication and authorization parameters
#             # in accordance with the API server security policy.
#             # Configure Bearer authorization
#             api_key=os.environ["NGC_API_KEY"],

#             # If you are in more than one LLM-enabled organization, you must
#             # specify your org ID in the form of a header. This is optional
#             # if you are only in one LLM-enabled org.
#             org_id="bwbg3fjn7she")

#     def calc_output_dims(self, x: MultiInferenceMessage) -> typing.Tuple:

#         # If we haven't cached the output dimension, do that here
#         if (not self._output_size):
#             test_intput_ids_shape = (self._max_batch_size, ) + x.get_input("input_ids").shape[1:]
#             test_input_mask_shape = (self._max_batch_size, ) + x.get_input("input_mask").shape[1:]

#             test_outputs = self._model(torch.randint(65000, (test_intput_ids_shape), dtype=torch.long).cuda(),
#                                        token_type_ids=None,
#                                        attention_mask=torch.ones(test_input_mask_shape).cuda())

#             # Send random input through the model
#             self._output_size = test_outputs[0].data.shape

#         return (x.count, self._outputs[list(self._outputs.keys())[0]].shape[1])

#     def process(self, batch: MultiInferenceMessage, cb: typing.Callable[[TensorMemory], None]):

#         response = self._conn.generate(
#             prompt=
#             "Alex is a cheerful AI assistant and companion, created by NVIDIA engineers. Alex is clever and helpful, and will do everything it can to cheer you up:\n\nYou: How are you feeling?\nAlex: I'm feeling great, how may I help you today?\nYou: Can you please suggest a movie?\nAlex: How about \"The Martian\". It's a sci-fi movie about an astronaut getting stranded on Mars!\nYou: That's cool! But I'm in the mood for watching comedy today\nAlex:",
#             model="gpt530b",
#             stop=["You:"],
#             tokens_to_generate=40,
#             temperature=0.5,
#             top_k=2,
#             top_p=1.0,
#             random_seed=194640,
#             beam_search_diversity_rate=0.0,
#             beam_width=1,
#             repetition_penalty=1.0,
#             length_penalty=1.0,
#         )

#         response_mem = TensorMemory(count=batch.count, tensors={'probs': probs_cp})

#         # Return the response
#         cb(response_mem)

# @register_stage("inf-nemo", modes=[PipelineModes.OTHER])
# class NeMoInferenceStage(InferenceStage):

#     def __init__(self, c: Config):

#         super().__init__(c)

#         self._fea_length = c.feature_length
#         self.features = [
#             "ack",
#             "psh",
#             "rst",
#             "syn",
#             "fin",
#             "ppm",
#             "data_len",
#             "bpp",
#             "all",
#             "ackpush/all",
#             "rst/all",
#             "syn/all",
#             "fin/all",
#         ]
#         assert self._fea_length == len(
#             self.features
#         ), f"Number of features in preprocessing {len(self.features)}, does not match configuration {self._fea_length}"

#         # columns required to be added to input message meta
#         self.req_cols = ["flow_id", "rollup_time"]

#         for req_col in self.req_cols:
#             self._needed_columns[req_col] = TypeId.STRING

#     @property
#     def name(self) -> str:
#         return "inf-nemo"

#     def supports_cpp_node(self):
#         return False

#     def _get_inference_worker(self, inf_queue: ProducerConsumerQueue) -> InferenceWorker:

#         return _NeMoInferenceWorker(inf_queue, self._config, model_filename=self._model_filename)


@register_stage("inf-nemo", modes=[PipelineModes.OTHER])
class NeMoInferenceStage(SinglePortStage):

    def __init__(self, c: Config, model_name: str):

        super().__init__(c)

        self._model_name = model_name

    @property
    def name(self) -> str:
        return "inf-nemo"

    def accepted_types(self) -> typing.Tuple:
        return (MultiMessage, )

    def supports_cpp_node(self):
        return False

    def _process_message(self, message: MultiMessage):
        response = self._conn.generate(
            prompt=
            "Alex is a cheerful AI assistant and companion, created by NVIDIA engineers. Alex is clever and helpful, and will do everything it can to cheer you up:\n\nYou: How are you feeling?\nAlex: I'm feeling great, how may I help you today?\nYou: Can you please suggest a movie?\nAlex: How about \"The Martian\". It's a sci-fi movie about an astronaut getting stranded on Mars!\nYou: That's cool! But I'm in the mood for watching comedy today\nAlex:",
            model=self._model_name,
            stop=["You:"],
            tokens_to_generate=40,
            temperature=0.5,
            top_k=2,
            top_p=1.0,
            random_seed=194640,
            beam_search_diversity_rate=0.0,
            beam_width=1,
            repetition_penalty=1.0,
            length_penalty=1.0,
        )

        return message

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        self._conn = NemoLLM(
            # The client must configure the authentication and authorization parameters
            # in accordance with the API server security policy.
            # Configure Bearer authorization
            api_key=os.environ["NGC_API_KEY"],

            # If you are in more than one LLM-enabled organization, you must
            # specify your org ID in the form of a header. This is optional
            # if you are only in one LLM-enabled org.
            org_id="bwbg3fjn7she")

        node = builder.make_node(self.unique_name, ops.map(self._process_message))
        builder.make_edge(input_stream[0], node)
        return node, MultiMessage
