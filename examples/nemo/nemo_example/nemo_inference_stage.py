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

import dataclasses
import os
import typing

import mrc
import pandas as pd
from mrc.core import operators as ops
from nemollm.api import NemoLLM
from pydantic import BaseModel

import cudf

from morpheus._lib.messages import MessageMeta
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.messages import MultiMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from .llm_engine import LlmEngine
from .nemo_service import NeMoService

subclass_registry = {}

class LLMTask(BaseModel):
    task_type: str

    def __init_subclass__(cls, **kwargs: dict) -> None:
        super().__init_subclass__(**kwargs)
        subclass_registry[cls.__name__] = cls


class SpearPhishingGenerateEmailTask(LLMTask):
    task_type: typing.Literal["spear_phishing_generate_email"] = "spear_phishing_generate_email"
    template: str = "This is my email template. I am asking you to do something for me. Please do it. {stuff}"


@register_stage("inf-nemo-preproc", modes=[PipelineModes.OTHER])
class NeMoPreprocessingStage(SinglePortStage):

    def __init__(self, c: Config):

        super().__init__(c)

    @property
    def name(self) -> str:
        return "inf-nemo-preproc"

    def accepted_types(self) -> typing.Tuple:
        return (MultiMessage, )

    def supports_cpp_node(self):
        return False

    def _process_message(self, message: MultiMessage):

        def apply_fn(x: pd.Series):
            labels = "\n".join([f"{context}: {label}" for context, label in zip(x.LABELS, x.CONTEXTS)])

            return f"Provided context:\n{labels}\nQuestion: {x.QUESTION}\nAnswer (yes / no / maybe):"

        # Make the prompt and completion columns
        prompt = message.get_meta(["CONTEXTS", "LABELS", "QUESTION"]).to_pandas().apply(apply_fn, axis=1)

        message.set_meta("prompt", prompt)

        message.set_meta("completion", message.get_meta("final_decision"))

        return message

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        node = builder.make_node(self.unique_name, ops.map(self._process_message))
        builder.make_edge(input_stream[0], node)
        return node, MultiMessage


@register_stage("inf-nemo", modes=[PipelineModes.OTHER])
class NeMoInferenceStage(SinglePortStage):

    def __init__(self, c: Config, model_name: str, customization_id: str = None):

        super().__init__(c)

        self._model_name = model_name
        self._customization_id = customization_id

        self._nemo_service: NeMoService = None

        self._llm_engine = LlmEngine()

    @property
    def name(self) -> str:
        return "inf-nemo"

    def accepted_types(self) -> typing.Tuple:
        return (MultiMessage, )

    def supports_cpp_node(self):
        return False

    def _process_message(self, message: MultiMessage):

        # Create a temp ControlMessage with a prompt task
        control_message = ControlMessage()

        control_message.add_task(
            "llm_query",
            dataclasses.asdict(
                LLMTask(template="Generate me an {adjective} email targeting {subject}", model_name=self._model_name)))

        payload = cudf.DataFrame({
            "adjective": ["marketing", "HR", "friendly"],
            "subject": ["a bank", "a bank", "a bank"],
        })

        control_message.payload(MessageMeta(payload))

        engine_response = self._llm_engine.run(control_message)

        prompts = message.get_meta("prompt").to_pandas().tolist()

        client = self._nemo_service.get_client(model_name=self._model_name,
                                               customization_id=self._customization_id,
                                               infer_kwargs={
                                                   "tokens_to_generate": 5,
                                                   "stop": ["yes", "no", "maybe"],
                                                   "temperature": 0.1,
                                               })

        response = client.generate(prompts)

        # # Call the NeMo API and generate responses for the current batch
        # response = self._conn.generate_multiple(
        #     model=self._model_name,
        #     prompts=prompts,
        #     customization_id=self._customization_id,
        #     return_type="text",
        #     tokens_to_generate=5,
        #     stop=["yes", "no", "maybe"],
        #     temperature=0.1,
        # )

        message.set_meta("response", [x.lower().strip() for x in response])

        return message

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        self._nemo_service = NeMoService.instance(org_id=os.environ["NGC_ORG_ID"])

        self._nemo_service.start()

        # self._conn = NemoLLM(
        #     # The client must configure the authentication and authorization parameters
        #     # in accordance with the API server security policy.
        #     # Configure Bearer authorization
        #     api_key=os.environ["NGC_API_KEY"],

        #     # If you are in more than one LLM-enabled organization, you must
        #     # specify your org ID in the form of a header. This is optional
        #     # if you are only in one LLM-enabled org.
        # )

        node = builder.make_node(self.unique_name, ops.map(self._process_message))
        builder.make_edge(input_stream[0], node)
        return node, MultiMessage
