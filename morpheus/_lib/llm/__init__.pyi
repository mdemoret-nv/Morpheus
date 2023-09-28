"""
        -----------------------
        .. currentmodule:: morpheus.llm
        .. autosummary::
           :toctree: _generate

        """
from __future__ import annotations
import morpheus._lib.llm
import typing
import morpheus._lib.messages

__all__ = [
    "CoroAwaitable",
    "LLMContext",
    "LLMEngine",
    "LLMGeneratePrompt",
    "LLMGenerateResult",
    "LLMNode",
    "LLMNodeBase",
    "LLMNodeRunner",
    "LLMPromptGenerator",
    "LLMService",
    "LLMTask",
    "LLMTaskHandler"
]


class CoroAwaitable():
    def __await__(self) -> CoroAwaitable: ...
    def __init__(self) -> None: ...
    def __iter__(self) -> CoroAwaitable: ...
    def __next__(self) -> None: ...
    pass
class LLMContext():
    def get_input(self) -> object: ...
    def message(self) -> morpheus._lib.messages.ControlMessage: ...
    def set_output(self, arg0: object) -> None: ...
    def task(self) -> LLMTask: ...
    @property
    def full_name(self) -> str:
        """
        :type: str
        """
    @property
    def name(self) -> str:
        """
        :type: str
        """
    pass
class LLMNodeBase():
    def __init__(self) -> None: ...
    def execute(self, arg0: LLMContext) -> CoroAwaitable: ...
    pass
class LLMGeneratePrompt():
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, model_name: str, model_kwargs: dict, prompts: typing.List[str]) -> None: ...
    @property
    def model_kwargs(self) -> object:
        """
        :type: object
        """
    @model_kwargs.setter
    def model_kwargs(self, arg1: dict) -> None:
        pass
    @property
    def model_name(self) -> str:
        """
        :type: str
        """
    @model_name.setter
    def model_name(self, arg0: str) -> None:
        pass
    @property
    def prompts(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @prompts.setter
    def prompts(self, arg0: typing.List[str]) -> None:
        pass
    pass
class LLMGenerateResult(LLMGeneratePrompt):
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: LLMGeneratePrompt, arg1: typing.List[str]) -> None: ...
    @property
    def responses(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @responses.setter
    def responses(self, arg0: typing.List[str]) -> None:
        pass
    pass
class LLMNode(LLMNodeBase):
    def __init__(self) -> None: ...
    def add_node(self, name: str, input_names: typing.List[str], node: LLMNodeBase) -> LLMNodeRunner: ...
    pass
class LLMEngine(LLMNode, LLMNodeBase):
    def __init__(self) -> None: ...
    def add_prompt_generator(self, prompt_generator: LLMPromptGenerator) -> None: ...
    def add_task_handler(self, task_handler: LLMTaskHandler) -> None: ...
    def arun2(self, arg0: function) -> CoroAwaitable: ...
    def arun3(self, arg0: function) -> CoroAwaitable: ...
    def run(self, input_message: morpheus._lib.messages.ControlMessage) -> CoroAwaitable: ...
    def run_async(self, arg0: object) -> CoroAwaitable: ...
    pass
class LLMNodeRunner():
    @property
    def input_names(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @property
    def name(self) -> str:
        """
        :type: str
        """
    pass
class LLMPromptGenerator():
    def __init__(self) -> None: ...
    def try_handle(self, arg0: LLMEngine, arg1: LLMTask, arg2: morpheus._lib.messages.ControlMessage) -> typing.Optional[typing.Union[LLMGeneratePrompt, LLMGenerateResult]]: ...
    pass
class LLMService():
    def __init__(self) -> None: ...
    def generate(self, arg0: LLMGeneratePrompt) -> LLMGenerateResult: ...
    pass
class LLMTask():
    def __getitem__(self, arg0: str) -> object: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: str, arg1: dict) -> None: ...
    def __len__(self) -> int: ...
    def __setitem__(self, arg0: str, arg1: object) -> None: ...
    def get(self, arg0: str, arg1: object) -> object: ...
    @property
    def task_type(self) -> str:
        """
        :type: str
        """
    pass
class LLMTaskHandler():
    def __init__(self) -> None: ...
    def try_handle(self, arg0: LLMEngine, arg1: LLMTask, arg2: morpheus._lib.messages.ControlMessage, arg3: LLMGenerateResult) -> typing.Optional[typing.List[morpheus._lib.messages.ControlMessage]]: ...
    pass
__version__ = '23.11.0'