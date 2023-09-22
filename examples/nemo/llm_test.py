import asyncio
import typing

from langchain.agents import AgentExecutor
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from nemo_example.llm_engine import LLMDictTask
from nemo_example.nemo_service import NeMoService

import cudf

from morpheus._lib.stages import LLMEngine
from morpheus._lib.stages import LLMGeneratePrompt
from morpheus._lib.stages import LLMGenerateResult
from morpheus._lib.stages import LLMPromptGenerator
from morpheus._lib.stages import LLMService
from morpheus._lib.stages import LLMTask
from morpheus._lib.stages import LLMTaskHandler
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta


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


def run_langchain_example():

    class MyPromptGeneratorAsync(LLMPromptGenerator):

        async def try_handle(
                self, engine: LLMEngine, task: LLMTask,
                message: ControlMessage) -> typing.Optional[typing.Union[LLMGeneratePrompt, LLMGenerateResult]]:

            print("MyPromptGenerator.try_handle")

            await asyncio.sleep(5)

            print("MyPromptGenerator.try_handle... done")

            return LLMGeneratePrompt()

    # Simple class which uses the sync implementation of try_handle but always fails
    class AlwaysFailPromptGenerator(LLMPromptGenerator):

        def try_handle(self, engine: LLMEngine, task: LLMTask,
                       message: ControlMessage) -> typing.Optional[typing.Union[LLMGeneratePrompt, LLMGenerateResult]]:
            # Always return None to skip this generator
            return None

    # Prompt generator wrapper around a LangChain agent executor
    class LangChainAgentExectorPromptGenerator(LLMPromptGenerator):

        def __init__(self, agent_executor: AgentExecutor):
            self._agent_executor = agent_executor

        async def try_handle(self, input_task: LLMTask,
                             input_message: ControlMessage) -> LLMGeneratePrompt | LLMGenerateResult | None:

            if (input_task["task_type"] != "dictionary"):
                return None

            input_keys = input_task["input_keys"]

            with input_message.payload().mutable_dataframe() as df:
                input_dict: list[dict] = df[input_keys].to_dict(orient="records")

            results = []

            for x in input_dict:
                # Await the result of the agent executor
                single_result = await self._agent_executor.arun(**x)

                results.append(single_result)

            return LLMGenerateResult(model_name=input_task["model_name"],
                                     model_kwargs=input_task["model_kwargs"],
                                     prompts=[],
                                     responses=results)

    class SimpleTaskHandler(LLMTaskHandler):

        def try_handle(self, engine: LLMEngine, task: LLMTask, message: ControlMessage,
                       result: LLMGenerateResult) -> typing.Optional[list[ControlMessage]]:

            with message.payload().mutable_dataframe() as df:
                df["response"] = result.responses

            return [message]

    # Create the NeMo LLM Service using our API key and org ID
    llm_service = NeMoLLMService(api_key="my_api_key", org_id="my_org_id")

    engine = LLMEngine(llm_service=llm_service)

    # Create a LangChain agent executor using the NeMo LLM Service and specified tools
    agent = initialize_agent(tools,
                             NeMoLangChainWrapper(engine.llm_service.get_client(model_name="gpt-43b-002")),
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True)

    # Add 2 prompt generators to test fallback
    engine.add_prompt_generator(AlwaysFailPromptGenerator())
    engine.add_prompt_generator(LangChainAgentExectorPromptGenerator(agent))

    # Add our task handler
    engine.add_task_handler(SimpleTaskHandler())

    # Create a control message with a single task which uses the LangChain agent executor
    message = ControlMessage()

    message.add_task("llm_query", LLMDictTask(input_keys=["input"], model_name="gpt-43b-002").dict())

    payload = cudf.DataFrame({
        "input": [
            "What is the product of the 998th, 999th and 1000th prime numbers?",
            "What is the product of the 998th and 1000th prime numbers?"
        ],
    })
    message.payload(MessageMeta(payload))

    # Finally, run the engine
    result = engine.run(message)

    print("Done")


def run_spearphishing_example():

    class TemplatePromptGenerator(LLMPromptGenerator):

        def __init__(self, template: str) -> None:
            super().__init__()

            self._template = template

        def try_handle(self, engine: LLMEngine, input_task: LLMTask,
                       input_message: ControlMessage) -> LLMGeneratePrompt | LLMGenerateResult | None:

            if (input_task.task_type != "template"):
                return None

            input_keys = input_task["input_keys"]

            with input_message.payload().mutable_dataframe() as df:
                input_dict: list[dict] = df[input_keys].to_dict(orient="records")

            template: str = input_task.get("template", self._template)

            prompts = [template.format(**x) for x in input_dict]

            return LLMGeneratePrompt(model_name=input_task["model_name"],
                                     model_kwargs=input_task["model_kwargs"],
                                     prompts=prompts)

    class QualityCheckTaskHandler(LLMTaskHandler):

        def _check_quality(self, response: str) -> bool:
            # Some sort of check here
            return True

        def try_handle(self, engine: LLMEngine, task: LLMTask, message: ControlMessage,
                       result: LLMGenerateResult) -> typing.Optional[list[ControlMessage]]:

            # Loop over all responses and check if they pass the quality check
            passed_check = [self._check_quality(r) for r in result.responses]

            with message.payload().mutable_dataframe() as df:
                df["emails"] = result.responses

                if (not all(passed_check)):
                    # Need to separate into 2 messages
                    good_message = ControlMessage()
                    good_message.payload(MessageMeta(df[passed_check]))

                    bad_message = ControlMessage()
                    bad_message.payload(MessageMeta(df[~passed_check]))

                    # Set a new llm_engine task on the bad message
                    bad_message.add_task("llm_query",
                                         LLMDictTask(input_keys=["input"], model_name="gpt-43b-002").dict())

                    return [good_message, bad_message]

                else:
                    # Return a single message
                    return [message]

    llm_service = NeMoLLMService()

    engine = LLMEngine(llm_service=llm_service)

    # Add a templating prompt generator to convert our payloads into prompts
    engine.add_prompt_generator(
        TemplatePromptGenerator(
            template=("Write a brief summary of the email below to use as a subject line for the email. "
                      "Be as brief as possible.\n\n{body}")))

    # Add our task handler
    engine.add_task_handler(QualityCheckTaskHandler())

    # Create a control message with a single task which uses the LangChain agent executor
    message = ControlMessage()

    message.add_task("llm_engine",
                     {
                         "task_type": "template",
                         "task_dict": LLMDictTask(input_keys=["body"], model_name="gpt-43b-002").dict(),
                     })

    payload = cudf.DataFrame({
        "body": [
            "Email body #1...",
            "Email body #2...",
        ],
    })
    message.payload(MessageMeta(payload))

    # Finally, run the engine
    result = engine.run(message)

    print(result)


if __name__ == "__main__":
    run_langchain_example()
    # run_spearphishing_example()
