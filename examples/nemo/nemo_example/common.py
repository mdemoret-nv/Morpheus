import typing

from pydantic import BaseModel

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
