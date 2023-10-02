import typing

from nemo_example.nemo_service import NeMoService
from pydantic import BaseModel

from morpheus._lib.llm import LLMContext
from morpheus._lib.llm import LLMGeneratePrompt
from morpheus._lib.llm import LLMGenerateResult
from morpheus._lib.llm import LLMNodeBase
from morpheus._lib.llm import LLMService

subclass_registry = {}


class LLMTask(BaseModel):
    task_type: str

    model_name: str
    model_kwargs: dict = {}

    def __init_subclass__(cls, **kwargs: dict) -> None:
        super().__init_subclass__(**kwargs)
        subclass_registry[cls.__name__] = cls


class LLMDictTask(LLMTask):
    task_type: typing.Literal["dictionary"] = "dictionary"
    input_keys: list[str]


class LLMTemplateTask(LLMTask):
    task_type: typing.Literal["template"] = "template"
    template: str
    input_keys: list[str]


class SpearPhishingGenerateEmailTask(LLMTask):
    task_type: typing.Literal["spear_phishing_generate_email"] = "spear_phishing_generate_email"
    template: str = "This is my email template. I am asking you to do something for me. Please do it. {stuff}"


class ExtracterNode(LLMNodeBase):

    def __init__(self) -> None:
        super().__init__()

    async def execute(self, context: LLMContext):

        # Get the keys from the task
        input_keys = context.task()["input_keys"]

        with context.message().payload().mutable_dataframe() as df:
            input_dict: list[dict] = df[input_keys].to_dict(orient="list")

        if (len(input_keys) == 1):
            # Extract just the first key if there is only 1
            context.set_output(input_dict[input_keys[0]])
        else:
            context.set_output(input_dict)

        return context


class NeMoLLMService(LLMService):

    def __init__(self, *, api_key: str | None = None, org_id: str | None = None):

        super().__init__()

        self._api_key = api_key
        self._org_id = org_id

        self._nemo_service: NeMoService = NeMoService.instance(api_key=api_key, org_id=org_id)

    async def generate(self, prompt: LLMGeneratePrompt) -> LLMGenerateResult:

        client = self._nemo_service.get_client(model_name=prompt.model_name, infer_kwargs=prompt.model_kwargs)

        responses = await client.generate(prompt.prompts)

        return LLMGenerateResult(prompt, responses)
