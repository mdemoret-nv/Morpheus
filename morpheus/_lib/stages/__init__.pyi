"""
        -----------------------
        .. currentmodule:: morpheus.stages
        .. autosummary::
           :toctree: _generate

        """
from __future__ import annotations
import morpheus._lib.stages
import typing
from morpheus._lib.common import FilterSource
import morpheus._lib.common
import morpheus._lib.messages
import mrc.core.segment

__all__ = [
    "AddClassificationsStage",
    "AddScoresStage",
    "CoroAwaitable",
    "DeserializeStage",
    "FileSourceStage",
    "FilterDetectionsStage",
    "FilterSource",
    "InferenceClientStage",
    "KafkaSourceStage",
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
    "PreallocateMessageMetaStage",
    "PreallocateMultiMessageStage",
    "PreprocessFILStage",
    "PreprocessNLPStage",
    "SerializeStage",
    "WriteToFileStage"
]


class AddClassificationsStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, idx2label: typing.Dict[int, str], threshold: float) -> None: ...
    pass
class AddScoresStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, idx2label: typing.Dict[int, str]) -> None: ...
    pass
class CoroAwaitable():
    def __await__(self) -> CoroAwaitable: ...
    def __init__(self) -> None: ...
    def __iter__(self) -> CoroAwaitable: ...
    def __next__(self) -> None: ...
    pass
class DeserializeStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, batch_size: int, ensure_sliceable_index: bool = True) -> None: ...
    pass
class FileSourceStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, filename: str, repeat: int, parser_kwargs: dict) -> None: ...
    pass
class FilterDetectionsStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, threshold: float, copy: bool, filter_source: morpheus._lib.common.FilterSource, field_name: str = 'probs') -> None: ...
    pass
class InferenceClientStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, model_name: str, server_url: str, force_convert_inputs: bool, use_shared_memory: bool, needs_logits: bool, inout_mapping: typing.Dict[str, str] = {}) -> None: ...
    pass
class KafkaSourceStage(mrc.core.segment.SegmentObject):
    @typing.overload
    def __init__(self, builder: mrc.core.segment.Builder, name: str, max_batch_size: int, topic: str, batch_timeout_ms: int, config: typing.Dict[str, str], disable_commits: bool = False, disable_pre_filtering: bool = False, stop_after: int = 0, async_commits: bool = True) -> None: ...
    @typing.overload
    def __init__(self, builder: mrc.core.segment.Builder, name: str, max_batch_size: int, topics: typing.List[str], batch_timeout_ms: int, config: typing.Dict[str, str], disable_commits: bool = False, disable_pre_filtering: bool = False, stop_after: int = 0, async_commits: bool = True) -> None: ...
    pass
class LLMContext():
    def get_input(self) -> object: ...
    def message(self) -> morpheus._lib.messages.ControlMessage: ...
    def set_output(self, arg0: dict) -> None: ...
    def task(self) -> LLMTask: ...
    pass
class LLMEngine():
    def __init__(self) -> None: ...
    def add_node(self, name: str, input_names: typing.List[str], node: LLMNodeBase) -> LLMNodeRunner: ...
    def add_prompt_generator(self, prompt_generator: LLMPromptGenerator) -> None: ...
    def add_task_handler(self, task_handler: LLMTaskHandler) -> None: ...
    def arun(self, arg0: morpheus._lib.messages.ControlMessage) -> CoroAwaitable: 
        """
        async def arun(self, arg0: morpheus._lib.messages.ControlMessage) -> CoroAwaitable
        """
    def run(self, input_message: morpheus._lib.messages.ControlMessage) -> typing.List[morpheus._lib.messages.ControlMessage]: ...
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
class LLMNodeBase():
    def __init__(self) -> None: ...
    def execute(self, arg0: LLMContext) -> None: ...
    pass
class LLMNode(LLMNodeBase):
    def __init__(self) -> None: ...
    def add_node(self, arg0: str, arg1: typing.List[str], arg2: LLMNodeBase) -> LLMNodeRunner: ...
    def execute(self, arg0: LLMContext) -> None: ...
    pass
class LLMNodeRunner():
    def execute(self, arg0: LLMContext) -> None: ...
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
class PreallocateMessageMetaStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, needed_columns: typing.List[typing.Tuple[str, morpheus._lib.common.TypeId]]) -> None: ...
    pass
class PreallocateMultiMessageStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, needed_columns: typing.List[typing.Tuple[str, morpheus._lib.common.TypeId]]) -> None: ...
    pass
class PreprocessFILStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, features: typing.List[str]) -> None: ...
    pass
class PreprocessNLPStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, vocab_hash_file: str, sequence_length: int, truncation: bool, do_lower_case: bool, add_special_token: bool, stride: int, column: str) -> None: ...
    pass
class SerializeStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, include: typing.List[str], exclude: typing.List[str], fixed_columns: bool = True) -> None: ...
    pass
class WriteToFileStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, filename: str, mode: str = 'w', file_type: morpheus._lib.common.FileTypes = FileTypes.Auto, include_index_col: bool = True, flush: bool = False) -> None: ...
    pass
__version__ = '23.11.0'
