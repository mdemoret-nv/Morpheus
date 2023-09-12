import asyncio
import typing

import pybind11_stubgen

from morpheus._lib.stages import LLMEngine
from morpheus._lib.stages import LLMGeneratePrompt
from morpheus._lib.stages import LLMGenerateResult
from morpheus._lib.stages import LLMPromptGenerator
from morpheus._lib.stages import LLMTask
from morpheus.messages import ControlMessage


class MyPromptGenerator(LLMPromptGenerator):

    def try_handle(self, engine: LLMEngine, task: LLMTask,
                   message: ControlMessage) -> typing.Optional[typing.Union[LLMGeneratePrompt, LLMGenerateResult]]:

        print("MyPromptGenerator.try_handle")

        print("MyPromptGenerator.try_handle... done")

        return LLMGeneratePrompt()


class MyPromptGeneratorAsync(LLMPromptGenerator):

    async def try_handle(
            self, engine: LLMEngine, task: LLMTask,
            message: ControlMessage) -> typing.Optional[typing.Union[LLMGeneratePrompt, LLMGenerateResult]]:

        print("MyPromptGenerator.try_handle")

        await asyncio.sleep(5)

        print("MyPromptGenerator.try_handle... done")

        return LLMGeneratePrompt()


engine = LLMEngine()

my_prompt = MyPromptGenerator()

engine.add_prompt_generator(my_prompt)

del my_prompt

engine.run(ControlMessage())

print("Done")
