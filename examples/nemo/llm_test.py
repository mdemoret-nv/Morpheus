# Speed up the import of cudf. From https://github.com/nv-morpheus/Morpheus/issues/1101
import logging
import textwrap
from ctypes import byref
from ctypes import c_int

from numba import cuda

from morpheus.utils.logger import configure_logging

dv = c_int(0)
cuda.cudadrv.driver.driver.cuDriverGetVersion(byref(dv))
drv_major = dv.value // 1000
drv_minor = (dv.value - (drv_major * 1000)) // 10
run_major, run_minor = cuda.runtime.get_version()
print(f'{drv_major} {drv_minor} {run_major} {run_minor}')

import os

os.environ["PTXCOMPILER_CHECK_NUMBA_CODEGEN_PATCH_NEEDED"] = "0"
os.environ["PTXCOMPILER_KNOWN_DRIVER_VERSION"] = f"{drv_major}.{drv_minor}"
os.environ["PTXCOMPILER_KNOWN_RUNTIME_VERSION"] = f"{run_major}.{run_minor}"

import time

start_time = time.time()

print(f"Starting at {start_time}")

import asyncio
import typing

print(f"Import 1 took: t {time.time() - start_time}")

# from langchain.agents import AgentExecutor
# from langchain.agents import AgentType
# from langchain.agents import initialize_agent
# from nemo_example.llm_engine import LLMDictTask
from nemo_example.nemo_service import NeMoService

print(f"Import 2 took: t {time.time() - start_time}")

import morpheus._lib

print(f"Import 3a took: t {time.time() - start_time}")

# from morpheus._lib.llm import LLMNodeBase
# from morpheus._lib.llm import LLMPromptGenerator
# from morpheus._lib.llm import LLMContext
# from morpheus._lib.llm import LLMEngine
from morpheus._lib.llm import LLMContext
from morpheus._lib.llm import LLMGeneratePrompt
from morpheus._lib.llm import LLMGenerateResult
from morpheus._lib.llm import LLMNodeBase
from morpheus._lib.llm import LLMService

print(f"Import 3 took: t {time.time() - start_time}")

# import cudf

# from morpheus._lib.llm import LLMTask
# from morpheus._lib.llm import LLMTaskHandler
# from morpheus.messages import ControlMessage
# from morpheus.messages import MessageMeta

# Enable the default logger.
configure_logging(log_level=logging.DEBUG)


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


async def run_haystack_example():

    import pydantic
    from haystack import Pipeline
    from haystack.agents import Agent
    from haystack.agents import Tool
    from haystack.agents.base import ToolsManager
    from haystack.nodes import PromptModel
    from haystack.nodes import PromptNode
    from haystack.nodes import PromptTemplate
    from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer
    from haystack.nodes.retriever.web import WebRetriever
    from haystack.pipelines import WebQAPipeline
    from nemo_example.common import LLMDictTask
    from nemo_example.llm_engine import NeMoLangChain

    import cudf

    from morpheus._lib.llm import LLMContext
    from morpheus._lib.llm import LLMEngine
    from morpheus._lib.llm import LLMNodeBase
    from morpheus._lib.llm import LLMPromptGenerator
    from morpheus._lib.llm import LLMTask
    from morpheus._lib.llm import LLMTaskHandler
    from morpheus.messages import ControlMessage
    from morpheus.messages import MessageMeta

    class NeMoHaystackInvocationLayer(PromptModelInvocationLayer):

        def __init__(self, model_name_or_path: str, **kwargs):
            super().__init__(model_name_or_path, **kwargs)

            self._nemo_service: NeMoService = NeMoService.instance()

            self._max_length = kwargs.get("max_length", None)

        def invoke(self, prompt: str, *args, **kwargs):
            """
            It takes a prompt and returns a list of generated text using the underlying model.
            :return: A list of generated text.
            """

            client = self._nemo_service.get_client(model_name=self.model_name_or_path,
                                                   infer_kwargs={
                                                       "tokens_to_generate": self._max_length,
                                                       "stop": kwargs.get("stop_words", None),
                                                       "top_k": kwargs.get("top_k", None),
                                                   })

            return [client.query(prompt=prompt.rstrip())]

        @classmethod
        def supports(cls, model_name_or_path: str, **kwargs) -> bool:
            """
            Checks if the given model is supported by this invocation layer.

            :param model_name_or_path: The name or path of the model.
            :param kwargs: Additional keyword arguments passed to the underlying model which might be used to determine
            if the model is supported.
            :return: True if this invocation layer supports the model, False otherwise.
            """
            return True

        def _ensure_token_limit(
            self, prompt: typing.Union[str, typing.List[typing.Dict[str, str]]]
        ) -> typing.Union[str, typing.List[typing.Dict[str, str]]]:
            """Ensure that length of the prompt and answer is within the maximum token length of the PromptModel.

            :param prompt: Prompt text to be sent to the generative model.
            """
            return prompt

    # Prompt generator wrapper around a LangChain agent executor
    class HaystackAgentNode(LLMNodeBase):

        def __init__(self, agent: Agent):
            super().__init__()

            self._agent = agent

        async def execute(self, context: LLMContext):

            input_prompts = context.get_input()

            results = [self._agent.run(p) for p in input_prompts]

            context.set_output(results)

    class SimpleTaskHandler(LLMTaskHandler):

        async def try_handle(self, context: LLMContext):

            with message.payload().mutable_dataframe() as df:
                df["response"] = context.get_input()

            return [message]

    # Create the NeMo LLM Service using our API key and org ID
    # llm = NeMoLangChain(model_name="llama-2-70b-hf")
    # llm = OpenAI(temperature=0)

    search_key = os.environ.get("SERPER_API_KEY")
    if not search_key:
        raise ValueError("Please set the SERPER_API_KEY environment variable")

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    haystack_model = PromptModel(model_name_or_path="gpt-43b-002", invocation_layer_class=NeMoHaystackInvocationLayer)

    pn = PromptNode(
        model_name_or_path=haystack_model,
        api_key=openai_key,
        max_length=256,
        default_prompt_template="question-answering-with-document-scores",
    )
    web_retriever = WebRetriever(api_key=search_key)
    pipeline = WebQAPipeline(retriever=web_retriever, prompt_node=pn)

    few_shot_prompt = """
    You are a helpful and knowledgeable agent. To achieve your goal of answering complex questions correctly, you have access to the following tools:

    Search: useful for when you need to Google questions. You should ask targeted questions, for example, Who is Anthony Dirrell's brother?

    To answer questions, you'll need to go through multiple steps involving step-by-step thinking and selecting appropriate tools and their inputs; tools will respond with observations. When you are ready for a final answer, respond with the `Final Answer:`
    Examples:
    ##
    Question: Anthony Dirrell is the brother of which super middleweight title holder?
    Thought: Let's think step by step. To answer this question, we first need to know who Anthony Dirrell is.
    Tool: Search
    Tool Input: Who is Anthony Dirrell?
    Observation: Boxer
    Thought: We've learned Anthony Dirrell is a Boxer. Now, we need to find out who his brother is.
    Tool: Search
    Tool Input: Who is Anthony Dirrell brother?
    Observation: Andre Dirrell
    Thought: We've learned Andre Dirrell is Anthony Dirrell's brother. Now, we need to find out what title Andre Dirrell holds.
    Tool: Search
    Tool Input: What is the Andre Dirrell title?
    Observation: super middleweight
    Thought: We've learned Andre Dirrell title is super middleweight. Now, we can answer the question.
    Final Answer: Andre Dirrell
    ##
    Question: What year was the party of the winner of the 1971 San Francisco mayoral election founded?
    Thought: Let's think step by step. To answer this question, we first need to know who won the 1971 San Francisco mayoral election.
    Tool: Search
    Tool Input: Who won the 1971 San Francisco mayoral election?
    Observation: Joseph Alioto
    Thought: We've learned Joseph Alioto won the 1971 San Francisco mayoral election. Now, we need to find out what party he belongs to.
    Tool: Search
    Tool Input: What party does Joseph Alioto belong to?
    Observation: Democratic Party
    Thought: We've learned Democratic Party is the party of Joseph Alioto. Now, we need to find out when the Democratic Party was founded.
    Tool: Search
    Tool Input: When was the Democratic Party founded?
    Observation: 1828
    Thought: We've learned the Democratic Party was founded in 1828. Now, we can answer the question.
    Final Answer: 1828
    ##
    Question: {query}
    Thought:
    {transcript}
    """

    few_shot_agent_template = PromptTemplate(textwrap.dedent(few_shot_prompt))
    prompt_node = PromptNode(model_name_or_path=haystack_model,
                             api_key=openai_key,
                             max_length=512,
                             stop_words=["Observation:"])

    web_qa_tool = Tool(
        name="Search",
        pipeline_or_node=pipeline,
        description="useful for when you need to Google questions.",
        output_variable="results",
    )

    agent = Agent(prompt_node=prompt_node,
                  prompt_template=few_shot_agent_template,
                  tools_manager=ToolsManager([web_qa_tool]))

    hotpot_questions = [
        "What year was the father of the Princes in the Tower born?",
        "Name the movie in which the daughter of Noel Harrison plays Violet Trefusis.",
        "Where was the actress who played the niece in the Priest film born?",
        "Which author is English: John Braine or Studs Terkel?",
    ]

    # for question in hotpot_questions:
    #     result = agent.run(query=question)
    #     print(f"\n{result}")

    engine = LLMEngine()

    engine.add_node("extract_prompt", [], ExtracterNode())
    engine.add_node("haystack", [("query", "/extract_prompt")], HaystackAgentNode(agent=agent))

    # Add our task handler
    engine.add_task_handler(inputs=["/haystack"], handler=SimpleTaskHandler())

    # Create a control message with a single task which uses the LangChain agent executor
    message = ControlMessage()

    message.add_task("llm_engine",
                     {
                         "task_type": "template",
                         "task_dict": LLMDictTask(input_keys=["input"], model_name="gpt-43b-002").dict(),
                     })

    payload = cudf.DataFrame({"input": hotpot_questions})
    message.payload(MessageMeta(payload))

    # Finally, run the engine
    result = await engine.run(message)

    print(result)


async def run_langchain_example():

    import pydantic
    from langchain import LLMMathChain
    from langchain import OpenAI
    from langchain.agents import AgentExecutor
    from langchain.agents import AgentType
    from langchain.agents import initialize_agent
    from langchain.agents.tools import Tool
    from nemo_example.common import LLMDictTask
    from nemo_example.llm_engine import NeMoLangChain

    import cudf

    from morpheus._lib.llm import LLMContext
    from morpheus._lib.llm import LLMEngine
    from morpheus._lib.llm import LLMNodeBase
    from morpheus._lib.llm import LLMPromptGenerator
    from morpheus._lib.llm import LLMTask
    from morpheus._lib.llm import LLMTaskHandler
    from morpheus.messages import ControlMessage
    from morpheus.messages import MessageMeta

    # Prompt generator wrapper around a LangChain agent executor
    class LangChainAgentExectorPromptGenerator(LLMNodeBase):

        def __init__(self, agent_executor: AgentExecutor):
            super().__init__()

            self._agent_executor = agent_executor

        async def execute(self, context: LLMContext):

            input_dict = context.get_input()

            results_async = [self._agent_executor.arun(**x) for x in input_dict]

            results = await asyncio.gather(*results_async)

            context.set_output(results)

    class SimpleTaskHandler(LLMTaskHandler):

        async def try_handle(self, context: LLMContext):

            with message.payload().mutable_dataframe() as df:
                df["response"] = result.responses

            return [message]

    # Create the NeMo LLM Service using our API key and org ID
    llm = NeMoLangChain(model_name="llama-2-70b-hf")
    # llm = OpenAI(temperature=0)

    engine = LLMEngine()

    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    primes = {998: 7901, 999: 7907, 1000: 7919}

    class CalculatorInput(pydantic.BaseModel):
        question: str = pydantic.Field()

    class PrimeInput(pydantic.BaseModel):
        n: int = pydantic.Field()

    def is_prime(n: int) -> bool:
        if n <= 1 or (n % 2 == 0 and n > 2):
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    def get_prime(n: int, primes: dict = primes) -> str:
        return str(primes.get(int(n)))

    async def aget_prime(n: int, primes: dict = primes) -> str:
        return str(primes.get(int(n)))

    tools = [
        Tool(
            name="GetPrime",
            func=get_prime,
            description="A tool that returns the `n`th prime number",
            args_schema=PrimeInput,
            coroutine=aget_prime,
        ),
        Tool.from_function(
            func=llm_math_chain.run,
            name="Calculator",
            description="Useful for when you need to compute mathematical expressions",
            args_schema=CalculatorInput,
            coroutine=llm_math_chain.arun,
        ),
    ]

    # Create a LangChain agent executor using the NeMo LLM Service and specified tools
    agent = initialize_agent(tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    engine.add_node("extract_prompt", [], ExtracterNode())
    engine.add_node("langchain", [("prompt", "/extract_prompt")],
                    LangChainAgentExectorPromptGenerator(agent_executor=agent))

    # Add our task handler
    engine.add_task_handler(inputs=["langchain"], handler=SimpleTaskHandler())

    # Create a control message with a single task which uses the LangChain agent executor
    message = ControlMessage()

    message.add_task("llm_engine",
                     {
                         "task_type": "template",
                         "task_dict": LLMDictTask(input_keys=["input"], model_name="gpt-43b-002").dict(),
                     })

    payload = cudf.DataFrame({
        "input": [
            "What is the product of the 998th, 999th and 1000th prime numbers?",
            "What is the product of the 998th and 1000th prime numbers?"
        ],
    })
    message.payload(MessageMeta(payload))

    # Finally, run the engine
    result = await engine.run(message)


async def run_langchain_example2():

    import pydantic
    from aiohttp import ClientSession
    from langchain import LLMMathChain
    from langchain import OpenAI
    from langchain.agents import AgentExecutor
    from langchain.agents import AgentType
    from langchain.agents import initialize_agent
    from langchain.agents import load_tools
    from langchain.agents.tools import Tool
    from langchain.callbacks.stdout import StdOutCallbackHandler
    from langchain.callbacks.tracers import LangChainTracer
    from langchain.llms import OpenAI
    from nemo_example.common import LLMDictTask
    from nemo_example.llm_engine import NeMoLangChain

    import cudf

    from morpheus._lib.llm import LLMContext
    from morpheus._lib.llm import LLMEngine
    from morpheus._lib.llm import LLMNodeBase
    from morpheus._lib.llm import LLMPromptGenerator
    from morpheus._lib.llm import LLMTask
    from morpheus._lib.llm import LLMTaskHandler
    from morpheus.messages import ControlMessage
    from morpheus.messages import MessageMeta

    # Prompt generator wrapper around a LangChain agent executor
    class LangChainAgentExectorPromptGenerator(LLMNodeBase):

        def __init__(self, agent_executor: AgentExecutor):
            super().__init__()

            self._agent_executor = agent_executor

        async def execute(self, context: LLMContext):

            input_dict: dict = context.get_input()

            if (isinstance(input_dict, list)):
                input_dict = {"input": input_dict}

            # Transform from dict[str, list[Any]] to list[dict[str, Any]]
            input_list = [dict(zip(input_dict, t)) for t in zip(*input_dict.values())]

            results_async = [self._agent_executor.arun(**x) for x in input_list]

            results = await asyncio.gather(*results_async)

            context.set_output(results)

    class SimpleTaskHandler(LLMTaskHandler):

        async def try_handle(self, context: LLMContext):

            with message.payload().mutable_dataframe() as df:
                df["response"] = context.get_input()

            return [message]

    # Create the NeMo LLM Service using our API key and org ID
    # llm = NeMoLangChain(model_name="llama-2-70b-hf")
    llm = NeMoLangChain(model_name="gpt-43b-002")
    # llm = OpenAI(temperature=0)
    tools = load_tools(["google-serper", "llm-math"], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    engine = LLMEngine()

    engine.add_node("extract_prompt", [], ExtracterNode())
    engine.add_node("langchain", [("prompt", "/extract_prompt")],
                    LangChainAgentExectorPromptGenerator(agent_executor=agent))

    # Add our task handler
    engine.add_task_handler(inputs=["/langchain"], handler=SimpleTaskHandler())

    # Create a control message with a single task which uses the LangChain agent executor
    message = ControlMessage()

    message.add_task("llm_engine",
                     {
                         "task_type": "template",
                         "task_dict": LLMDictTask(input_keys=["input"], model_name="gpt-43b-002").dict(),
                     })

    questions = [
        "Who won the US Open men's final in 2019?",
        # "Who won the US Open men's final in 2019? What is his age raised to the 0.334 power?",
        "Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?",
        # "Who won the most recent formula 1 grand prix? What is their age raised to the 0.23 power?",
        # "Who won the US Open women's final in 2019? What is her age raised to the 0.34 power?",
        # "Who is Beyonce's husband? What is his age raised to the 0.19 power?",
    ]

    payload = cudf.DataFrame({
        "input": questions,
    })
    message.payload(MessageMeta(payload))

    # Finally, run the engine
    result = await engine.run(message)


def run_spearphishing_example():

    from nemo_example.common import LLMDictTask

    import cudf

    from morpheus._lib.llm import LLMEngine
    from morpheus._lib.llm import LLMPromptGenerator
    from morpheus._lib.llm import LLMTask
    from morpheus._lib.llm import LLMTaskHandler
    from morpheus.messages import ControlMessage
    from morpheus.messages import MessageMeta

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


async def run_spearphishing_example2():

    from nemo_example.common import LLMDictTask

    import cudf

    from morpheus._lib.llm import LLMContext
    from morpheus._lib.llm import LLMEngine
    from morpheus._lib.llm import LLMNodeBase
    from morpheus._lib.llm import LLMTaskHandler
    from morpheus.messages import ControlMessage
    from morpheus.messages import MessageMeta

    class FunctionWrapperNode(LLMNodeBase):

        def __init__(self, node_fn: typing.Callable) -> None:
            super().__init__()

            self._node_fn = node_fn

        async def execute(self, context: LLMContext):

            inputs = context.get_input()

            result = await self._node_fn(inputs)

            context.set_output(result)

            return context

    class TemplatePromptGenerator(LLMNodeBase):

        def __init__(self, template: str) -> None:
            super().__init__()

            self._template = template

        async def execute(self, context: LLMContext):

            if (context.task().task_type != "template"):
                return None

            input_keys = context.task()["input_keys"]

            with context.message().payload().mutable_dataframe() as df:
                input_dict: list[dict] = df[input_keys].to_dict(orient="records")

            template: str = context.task().get("template", self._template)

            prompts = [template.format(**x) for x in input_dict]

            context.set_output(prompts)

            return context

    class LLMGenerateNode(LLMNodeBase):

        def __init__(self, llm_service: LLMService) -> None:
            super().__init__()

            self._llm_service = llm_service

        async def execute(self, context: LLMContext):

            prompts = context.get_input()

            gen_prompt = LLMGeneratePrompt(context.task().get("model_name", "gpt-43b-002"), {}, prompts)

            print("Running LLM generate...")
            result = await self._llm_service.generate(gen_prompt)
            print("Running LLM generate... Done")

            context.set_output(result.responses)

            return context

    async def extract_subject(llm_output: list[str]):

        # Remove leading "Subject: "
        subjects = [x[9:] for x in llm_output]

        return subjects

    async def quality_check(subjects: list[str]):

        def _check_quality(response: str) -> bool:
            # Some sort of check here
            return True

        # Some sort of check here
        passed_check = [_check_quality(r) for r in subjects]

        return passed_check

    class QualityCheckTaskHandler(LLMTaskHandler):

        async def try_handle(self, context: LLMContext):

            all_input = context.get_input()

            with context.message().payload().mutable_dataframe() as df:
                df["subjects"] = all_input["subject"]

                passed_check = all_input["quality_check"]

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
                    return [context.message()]

    llm_service = NeMoLLMService()

    engine = LLMEngine()

    engine.add_node(name="template",
                    inputs=[],
                    node=TemplatePromptGenerator(
                        ("Write a brief summary of the email below to use as a subject line for the email. "
                         "Be as brief as possible.\n\n{body}")))
    engine.add_node(name="nemo", inputs=["/template"], node=LLMGenerateNode(llm_service=llm_service))
    engine.add_node(name="extract_subject", inputs=["/nemo"], node=FunctionWrapperNode(extract_subject))
    engine.add_node(name="quality_check", inputs=["/extract_subject"], node=FunctionWrapperNode(quality_check))

    engine.add_task_handler(inputs=[("subject", "/extract_subject"), ("quality_check", "/quality_check")],
                            handler=QualityCheckTaskHandler())

    def build_message():

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

        return message

    # Finally, run the engine
    # result = await engine.run(message)
    results = [engine.run(build_message()), engine.run(build_message())]

    for result in asyncio.as_completed(results):

        awaited_result = await result
        print(f"Got result: {awaited_result}")


async def test_async():

    from morpheus._lib.llm import LLMEngine

    engine = LLMEngine()

    async def inner_async_fn():

        print("Sleeping from python...")

        await asyncio.sleep(1)

        print("Sleeping from python... Done")

        return 5

    async def inner_async_fn_with_ex():

        print("Sleeping from python...")

        await asyncio.sleep(1)

        print("Sleeping from python... Done")

        raise RuntimeError("Inside inner_async_fn_with_ex")

    async def inner_async_fn_with_arg(fn_to_call):

        print("Sleeping from python...")

        await asyncio.sleep(1)

        print("Sleeping from python... Done")

        print("Calling function argument")

        return await fn_to_call(5)

    result = await engine.arun2(inner_async_fn)

    print(result)

    # Run with an exception
    try:
        result = await engine.arun2(inner_async_fn_with_ex)

        print(result)

    except RuntimeError as ex:
        print(f"Exception returned: {ex}")

    result = await engine.arun3(inner_async_fn_with_arg)

    print(result)


if __name__ == "__main__":
    # asyncio.run(run_haystack_example())
    # asyncio.run(run_langchain_example())
    asyncio.run(run_langchain_example2())
    # asyncio.run(run_spearphishing_example2())
    # asyncio.run(test_async())

    print("Done")
