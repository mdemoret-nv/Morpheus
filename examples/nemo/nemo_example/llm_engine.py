import dataclasses
import logging
import os
import typing
from abc import ABC
from abc import abstractmethod

from langchain.agents import Agent
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic import BaseModel

import cudf

from morpheus.messages import ControlMessage

from .nemo_service import NeMoService

logger = logging.getLogger(f"morpheus.{__name__}")


def get_from_dict_or_env(values: dict, dict_key: str, env_key: str) -> str:
    if dict_key in values:
        return values[dict_key]
    elif env_key in os.environ:
        return os.environ[env_key]
    else:
        raise ValueError(f"Could not find {dict_key} in values or {env_key} in os.environ")


subclass_registry = {}


class LLMTask(BaseModel):
    task_type: str

    model_name: str
    model_kwargs: dict = {}

    def __init_subclass__(cls, **kwargs: dict) -> None:
        super().__init_subclass__(**kwargs)
        subclass_registry[cls.__name__] = cls


class LLMTemplateTask(LLMTask):
    task_type: typing.Literal["template"] = "template"
    template: str
    input_keys: list[str]


class SpearPhishingGenerateEmailTask(LLMTask):
    task_type: typing.Literal["spear_phishing_generate_email"] = "spear_phishing_generate_email"
    template: str = "This is my email template. I am asking you to do something for me. Please do it. {stuff}"


@dataclasses.dataclass
class LLMGeneratePrompt:
    model_name: str
    model_kwargs: dict
    prompts: list[str]


@dataclasses.dataclass
class LLMGenerateResult(LLMGeneratePrompt):
    responses: list[str]


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
    def try_handle(self, input_task: dict,
                   input_message: ControlMessage) -> LLMGeneratePrompt | LLMGenerateResult | None:
        pass


class TemplatePromptGenerator(LlmPropmtGenerator):

    def __init__(self):
        super().__init__()

    def try_handle(self, input_task: dict,
                   input_message: ControlMessage) -> LLMGeneratePrompt | LLMGenerateResult | None:

        if (input_task["task_type"] != "template"):
            return None

        input_keys = input_task["input_keys"]

        with input_message.payload().mutable_dataframe() as df:
            input_dict: list[dict] = df[input_keys].to_dict(orient="records")

        template: str = input_task["template"]

        prompts = [template.format(**x) for x in input_dict]

        return LLMGeneratePrompt(model_name=input_task["model_name"],
                                 model_kwargs=input_task["model_kwargs"],
                                 prompts=prompts)


class LangChainAgentPromptGenerator(LlmPropmtGenerator):

    def __init__(self, agent: Agent):
        self._agent = agent

    def try_handle(self, input_task: dict,
                   input_message: ControlMessage) -> LLMGeneratePrompt | LLMGenerateResult | None:
        pass


class LlmTaskHandler(ABC):

    @abstractmethod
    def try_handle(self, input_task: dict, input_message: ControlMessage,
                   responses: LLMGenerateResult) -> list[ControlMessage] | None:
        pass


class DefaultTaskHandler(LlmTaskHandler):

    def try_handle(self, input_task: dict, input_message: ControlMessage,
                   responses: LLMGenerateResult) -> list[ControlMessage] | None:

        with input_message.payload().mutable_dataframe() as df:
            df["response"] = responses.responses

        return [input_message]


class SpearPhishingTaskHandler(LlmTaskHandler):

    def try_handle(self, input_task: dict, input_message: ControlMessage,
                   responses: LLMGenerateResult) -> list[ControlMessage] | None:
        pass


class LlmEngine:

    def __init__(self):

        self._nemo_service = NeMoService.instance()

        self._llm = NeMoLangChain(model_name="gpt5b")

        self._prompt_generators: list[LlmPropmtGenerator] = []
        self._task_handlers: list[LlmTaskHandler] = []

    def add_prompt_generator(self, prompt_generator: LlmPropmtGenerator):
        self._prompt_generators.append(prompt_generator)

    def add_task_handler(self, task_handler: LlmTaskHandler):
        self._task_handlers.append(task_handler)

    def run(self, message: ControlMessage):
        if (not message.has_task("llm_query")):
            raise RuntimeError("llm_query task not found on message")

        output_tasks: list[ControlMessage] = []

        while (message.has_task("llm_query")):
            current_task = message.remove_task("llm_query")

            # Prompt generator to build the input queries into the model
            prompts = self._generate_prompts(current_task, message)

            if (not isinstance(prompts, LLMGenerateResult)):
                # Execute the LLM model
                prompts = self._execute_model(prompts)

            # Generate the output tasks
            output_tasks.extend(self._handle_tasks(current_task, message, prompts))

        return output_tasks

    def _generate_prompts(self, input_task: dict,
                          input_message: ControlMessage) -> LLMGeneratePrompt | LLMGenerateResult:

        for generator in self._prompt_generators:

            prompt_result = generator.try_handle(input_task, input_message)

            if (prompt_result is not None):
                return prompt_result

        raise RuntimeError(f"No prompt generator found for task: {input_task}")

    def _execute_model(self, prompts: LLMGeneratePrompt) -> LLMGenerateResult:
        # Execute the LLM model
        client = self._nemo_service.get_client(model_name=prompts.model_name, infer_kwargs=prompts.model_kwargs)

        responses = client.generate(prompts.prompts)

        return LLMGenerateResult(model_name=prompts.model_name,
                                 model_kwargs=prompts.model_kwargs,
                                 prompts=prompts.prompts,
                                 responses=responses)

    def _handle_tasks(self, input_task: dict, input_message: ControlMessage,
                      response: LLMGenerateResult) -> list[ControlMessage]:

        for handler in self._task_handlers:

            new_tasks = handler.try_handle(input_task=input_task, input_message=input_message, responses=response)

            if new_tasks is not None:
                return new_tasks

        raise RuntimeError(f"No prompt generator found for task: {input_task}")
