import logging
import os
from abc import ABC
from abc import abstractmethod

import pandas as pd
from langchain import LLMChain
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic import root_validator

import cudf

from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta

from .nemo_service import NeMoService

logger = logging.getLogger(f"morpheus.{__name__}")


def get_from_dict_or_env(values: dict, dict_key: str, env_key: str) -> str:
    if dict_key in values:
        return values[dict_key]
    elif env_key in os.environ:
        return os.environ[env_key]
    else:
        raise ValueError(f"Could not find {dict_key} in values or {env_key} in os.environ")


class NeMoLangChain(LLM):
    _nemo_service: NeMoService

    api_key: str | None = None
    org_id: str | None = None

    model_name: str
    customization_id: str | None = None

    class Config:
        underscore_attrs_are_private = True

    def __init__(self, api_key: str | None = None, org_id: str | None = None, **data):

        # Before initializing the base class, see if we can get the api key and org id from the environment.
        api_key = api_key if api_key is not None else os.environ.get("NGC_API_KEY", None)
        org_id = org_id if org_id is not None else os.environ.get("NGC_ORG_ID", None)

        super().__init__(api_key=api_key, org_id=org_id, **data)

        self._nemo_service: NeMoService = NeMoService.instance(api_key=api_key, org_id=org_id)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "nemo"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
    ) -> str:
        # if self.stop:
        #     # append manual stop words
        #     stop = self.stop if stop is None else stop + self.stop

        client = self._nemo_service.get_client(model_name=self.model_name,
                                               customization_id=self.customization_id,
                                               infer_kwargs={
                                                   "tokens_to_generate": 100,
                                                   "temperature": 0.0,
                                                   "stop": stop,
                                               })

        response: str = client.generate(prompt=prompt)

        # If it ends with one of the stop words, remove it
        for s in stop:
            if (response.endswith(s)):
                response = response.removesuffix(s)
                break

        logger.debug("Prompt: '%s', Response: '%s'", prompt, response)

        return response


class LlmPropmtGenerator(ABC):

    @abstractmethod
    def can_handle(self, input_tasks: list[dict], responses: list[cudf.DataFrame]) -> bool:
        pass

    @abstractmethod
    def try_handle(self, input_message: ControlMessage, input_tasks: list[dict]) -> list[ControlMessage] | None:
        pass


class LangChainPromptGenerator(LlmPropmtGenerator):

    def can_handle(self, input_tasks: list[dict], responses: list[cudf.DataFrame]) -> bool:
        return True

    def try_handle(self, input_message: ControlMessage, input_tasks: list[dict]) -> list[ControlMessage] | None:
        pass


class LlmTaskHandler(ABC):

    @abstractmethod
    def can_handle(self, input_tasks: list[dict], responses: list[cudf.DataFrame]) -> bool:
        pass

    @abstractmethod
    def try_handle(self, input_message: ControlMessage, input_tasks: list[dict],
                   responses: list[cudf.DataFrame]) -> list[ControlMessage] | None:
        pass


class SpearPhishingTaskHandler(LlmTaskHandler):

    def can_handle(self, input_tasks: list[dict], responses: list[cudf.DataFrame]) -> bool:
        return True

    def try_handle(self, input_message: ControlMessage, input_tasks: list[dict],
                   responses: list[cudf.DataFrame]) -> list[ControlMessage] | None:
        pass


class LlmEngine:

    def __init__(self):

        self._nemo_service = NeMoService.instance()

        self._llm = NeMoLangChain(model_name="gpt5b")

        self._prompt_generator: list[LlmTaskHandler] = self._prompt_generator
        self._task_handlers: list[LlmTaskHandler] = []

    def run(self, message: ControlMessage):
        if (not message.has_task("llm_query")):
            raise RuntimeError("llm_query task not found on message")

        input_tasks = []

        while (message.has_task("llm_query")):
            input_tasks.append(message.remove_task("llm_query"))

        # Prompt generator to build the input queries into the model
        prompts = self._prompt_generator(input_tasks, message.payload())

        # Execute the LLM model
        # client = self._nemo_service.get_client(model_name="")

        # responses = client.generate(prompts)

        # Generate the output tasks
        output_tasks = self._task_generator(input_tasks, prompts)

        return output_tasks

    def _prompt_generator(self, input_tasks: list[dict], input_payload: MessageMeta) -> list[cudf.DataFrame]:

        responses: list[cudf.DataFrame] = []

        for task in input_tasks:

            llm_chain: LLMChain = LLMChain.from_string(llm=self._llm, template=task["template"])

            # Create the input dict from the payload
            with input_payload.mutable_dataframe() as df:
                input_dict = df[llm_chain.input_keys].to_dict(orient="records")

            result = llm_chain.generate(input_dict)

            # Convert it to a list of dicts
            # Only support one output for now
            result_dict = [x[0].dict() for x in result.generations]

            responses.append(cudf.DataFrame(data=result_dict))

        return responses

    def _task_generator(self, input_message: ControlMessage, input_tasks: list[dict],
                        responses: list[cudf.DataFrame]) -> list[ControlMessage]:

        tasks: list[ControlMessage] = []

        for handler in self._task_handlers:

            new_tasks = handler.try_handle(input_message=input_message, input_tasks=input_tasks, response=responses)

            if new_tasks is not None:
                tasks.extend(new_tasks)
                break

        return tasks
