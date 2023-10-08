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
    "InputMap",
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
    "LLMTaskHandler",
    "LangChainTemplateNodeCpp"
]


class CoroAwaitable():
    def __await__(self) -> CoroAwaitable: ...
    def __init__(self) -> None: ...
    def __iter__(self) -> CoroAwaitable: ...
    def __next__(self) -> None: ...
    pass
class InputMap():
    def __init__(self) -> None: ...
    @property
    def input_name(self) -> str:
        """
        :type: str
        """
    @input_name.setter
    def input_name(self, arg0: str) -> None:
        pass
    @property
    def node_name(self) -> str:
        """
        :type: str
        """
    @node_name.setter
    def node_name(self, arg0: str) -> None:
        pass
    pass
class LLMContext():
    @typing.overload
    def get_input(self) -> object: ...
    @typing.overload
    def get_input(self, arg0: str) -> object: ...
    def get_inputs(self) -> dict: ...
    def message(self) -> morpheus._lib.messages.ControlMessage: ...
    def set_output(self, arg0: object) -> None: ...
    def task(self) -> LLMTask: ...
    @property
    def all_outputs(self) -> object:
        """
        :type: object
        """
    @property
    def full_name(self) -> str:
        """
        :type: str
        """
    @property
    def input_map(self) -> typing.List[InputMap]:
        """
        :type: typing.List[InputMap]
        """
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def parent(self) -> LLMContext:
        """
        :type: LLMContext
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
    @typing.overload
    def add_node(self, name: str, *, inputs: typing.List[typing.Union[str, typing.Tuple[str, str]]], node: LLMNodeBase, is_output: bool = False) -> LLMNodeRunner: ...
    @typing.overload
    def add_node(self, name: str, *, node: LLMNodeBase, is_output: bool = False) -> LLMNodeRunner: ...
    pass
class LLMEngine(LLMNode, LLMNodeBase):
    def __init__(self) -> None: ...
    def add_prompt_generator(self, prompt_generator: LLMPromptGenerator) -> None: ...
    def add_task_handler(self, inputs: typing.List[typing.Union[str, typing.Tuple[str, str]]], handler: LLMTaskHandler) -> None: ...
    def arun2(self, arg0: function) -> CoroAwaitable: ...
    def arun3(self, arg0: function) -> CoroAwaitable: ...
    def run(self, input_message: morpheus._lib.messages.ControlMessage) -> CoroAwaitable: ...
    def run_async(self, arg0: object) -> CoroAwaitable: ...
    pass
class LLMNodeRunner():
    @property
    def inputs(self) -> typing.List[InputMap]:
        """
        :type: typing.List[InputMap]
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
    def try_handle(self, context: LLMContext) -> CoroAwaitable: ...
    pass
class LangChainTemplateNodeCpp(LLMNodeBase):
    def __init__(self, template: str) -> None: ...
    def execute(self, arg0: LLMContext) -> CoroAwaitable: ...
    def get_input_names(self) -> typing.List[str]: ...
    @property
    def template(self) -> str:
        """
        :type: str
        """
    pass
__version__ = '23.11.0'
