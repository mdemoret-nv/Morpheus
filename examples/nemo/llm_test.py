import asyncio
import typing

import pybind11_stubgen

from morpheus._lib.stages import LLMEngine
from morpheus._lib.stages import LLMGeneratePrompt
from morpheus._lib.stages import LLMGenerateResult
from morpheus._lib.stages import LLMPromptGenerator
from morpheus._lib.stages import LLMTask
from morpheus._lib.stages import LLMTaskHandler
from morpheus.messages import ControlMessage


class MyPromptGenerator(LLMPromptGenerator):

    def try_handle(self, engine: LLMEngine, task: LLMTask,
                   message: ControlMessage) -> typing.Optional[typing.Union[LLMGeneratePrompt, LLMGenerateResult]]:

        print("MyPromptGenerator.try_handle")

        print("MyPromptGenerator.try_handle... done")

        return None


class MyPromptGeneratorAsync(LLMPromptGenerator):

    async def try_handle(
            self, engine: LLMEngine, task: LLMTask,
            message: ControlMessage) -> typing.Optional[typing.Union[LLMGeneratePrompt, LLMGenerateResult]]:

        print("MyPromptGenerator.try_handle")

        await asyncio.sleep(5)

        print("MyPromptGenerator.try_handle... done")

        return LLMGeneratePrompt()


class MyTaskHandlerAsync(LLMTaskHandler):

    async def try_handle(self, engine: LLMEngine, task: LLMTask, message: ControlMessage,
                         responses: LLMGenerateResult) -> typing.Optional[list[ControlMessage]]:

        print("MyTaskHandler.try_handle")

        await asyncio.sleep(5)

        print("MyTaskHandler.try_handle... done")

        return [ControlMessage()]


engine = LLMEngine()

engine.add_prompt_generator(MyPromptGenerator())
engine.add_prompt_generator(MyPromptGeneratorAsync())

engine.add_task_handler(MyTaskHandlerAsync())

message = ControlMessage()

message.add_task("llm_engine", {})

result = engine.run(message)

print("Done")
