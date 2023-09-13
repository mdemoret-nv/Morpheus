# Copyright (c) 2021-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import json
import logging
import os

import click
import pydantic
from langchain import LLMChain
from langchain import LLMMathChain
from langchain import OpenAI
from langchain import PromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.agents import tool
from langchain.agents.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from nemo_example.llm_engine import DefaultTaskHandler
from nemo_example.llm_engine import LLMEngine
from nemo_example.llm_engine import LLMTask
from nemo_example.llm_engine import LLMTemplateTask
from nemo_example.llm_engine import TemplatePromptGenerator
from nemo_example.nemo_inference_stage import NeMoInferenceStage
from nemo_example.nemo_inference_stage import NeMoPreprocessingStage
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import cudf

from morpheus.cli.commands import FILE_TYPE_NAMES
from morpheus.cli.utils import str_to_file_type
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.logger import configure_logging


@click.command()
@click.option(
    "--num_threads",
    default=os.cpu_count(),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use.",
)
@click.option(
    "--pipeline_batch_size",
    default=128,
    type=click.IntRange(min=1),
    help=("Internal batch size for the pipeline. Can be much larger than the model batch size. "
          "Also used for Kafka consumers."),
)
@click.option(
    "--model_max_batch_size",
    default=32,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model.",
)
@click.option(
    "--input_file",
    type=click.Path(exists=True, readable=True),
    default="pcap.jsonlines",
    required=True,
    help="Input filepath.",
)
@click.option(
    "--output_file",
    default="./pcap_out.jsonlines",
    help="The path to the file where the inference output will be saved.",
)
@click.option(
    "--model_fea_length",
    default=13,
    type=click.IntRange(min=1),
    help="Features length to use for the model.",
)
@click.option(
    "--model_name",
    default="abp-pcap-xgb",
    help="The name of the model that is deployed on Tritonserver.",
)
@click.option(
    "--iterative",
    is_flag=True,
    default=False,
    help=("Iterative mode will emit dataframes one at a time. Otherwise a list of dataframes is emitted. "
          "Iterative mode is good for interleaving source stages."),
)
@click.option("--server_url", help="Tritonserver url.")
@click.option(
    "--file_type",
    type=click.Choice(FILE_TYPE_NAMES, case_sensitive=False),
    default="auto",
    help=("Indicates what type of file to read. "
          "Specifying 'auto' will determine the file type from the extension."),
)
def run_pipeline(
    num_threads,
    pipeline_batch_size,
    model_max_batch_size,
    model_fea_length,
    input_file,
    output_file,
    model_name,
    iterative,
    server_url,
    file_type,
):

    # Enable the default logger.
    configure_logging(log_level=logging.DEBUG)

    # ============= Testing code =============
    # prompt_template = "What is a good name for a company that makes {product}?"
    # llm = NeMoLangChain(model_name="gpt5b")
    # llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
    # llm_chain("colorful socks")
    # ====================================
    # llm = NeMoLangChain(model_name="gpt530b")
    # # llm = OpenAI(temperature=0)
    # llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    # primes = {998: 7901, 999: 7907, 1000: 7919}
    # class CalculatorInput(pydantic.BaseModel):
    #     question: str = pydantic.Field()
    # class PrimeInput(pydantic.BaseModel):
    #     n: int = pydantic.Field()
    # def is_prime(n: int) -> bool:
    #     if n <= 1 or (n % 2 == 0 and n > 2):
    #         return False
    #     for i in range(3, int(n**0.5) + 1, 2):
    #         if n % i == 0:
    #             return False
    #     return True
    # def get_prime(n: int, primes: dict = primes) -> str:
    #     return str(primes.get(int(n)))
    # async def aget_prime(n: int, primes: dict = primes) -> str:
    #     return str(primes.get(int(n)))
    # tools = [
    #     Tool(
    #         name="GetPrime",
    #         func=get_prime,
    #         description="A tool that returns the `n`th prime number",
    #         args_schema=PrimeInput,
    #         coroutine=aget_prime,
    #     ),
    #     Tool.from_function(
    #         func=llm_math_chain.run,
    #         name="Calculator",
    #         description="Useful for when you need to compute mathematical expressions",
    #         args_schema=CalculatorInput,
    #         coroutine=llm_math_chain.arun,
    #     ),
    # ]
    # agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    # question = "What is the product of the 998th, 999th and 1000th prime numbers?"
    # agent.run(question)
    # llm = NeMoLangChain(model_name="gpt-43b-002")
    # # llm = OpenAI(temperature=0)
    # tools = load_tools(["serpapi", "llm-math"], llm=llm)
    # agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    # agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
    # ====================================
    # engine = LlmEngine()
    # engine.add_prompt_generator(TemplatePromptGenerator())
    # engine.add_task_handler(DefaultTaskHandler())
    # control_message = ControlMessage()
    # control_message.add_task(
    #     "llm_query",
    #     LLMTemplateTask(template="Generate me an {adjective} email targeting {subject}",
    #                     input_keys=["adjective", "subject"],
    #                     model_name="gpt-43b-002").dict())
    # payload = cudf.DataFrame({
    #     "adjective": ["marketing", "HR", "friendly"],
    #     "subject": ["a bank", "a bank", "a bank"],
    # })
    # control_message.payload(MessageMeta(payload))
    # engine.run(control_message)
    # ====================================
    from nemo_example.llm_engine import LangChainAgentExectorPromptGenerator
    from nemo_example.llm_engine import LLMDictTask
    from nemo_example.llm_engine import LlmEngine
    from nemo_example.llm_engine import NeMoLangChain

    llm = NeMoLangChain(model_name="gpt-43b-002")
    # llm = OpenAI(temperature=0)
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
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    engine = LLMEngine()

    engine.add_prompt_generator(LangChainAgentExectorPromptGenerator(agent))

    engine.add_task_handler(DefaultTaskHandler())

    control_message = ControlMessage()

    control_message.add_task("llm_query", LLMDictTask(input_keys=["input"], model_name="gpt-43b-002").dict())

    payload = cudf.DataFrame({
        "input": [
            "What is the product of the 998th, 999th and 1000th prime numbers?",
            "What is the product of the 998th and 1000th prime numbers?"
        ],
    })

    control_message.payload(MessageMeta(payload))

    engine.run(control_message)

    # ========================================

    CppConfig.set_should_use_cpp(False)

    # Its necessary to get the global config object and configure it for FIL mode.
    config = Config()
    config.mode = PipelineModes.OTHER

    # Below properties are specified by the command line.
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = 1
    config.feature_length = model_fea_length
    config.class_labels = ["probs"]

    # Create a linear pipeline object.
    pipeline = LinearPipeline(config)

    # Set source stage.
    # In this stage, messages were loaded from a file.
    pipeline.set_source(
        FileSourceStage(config,
                        filename=input_file,
                        iterative=iterative,
                        file_type=str_to_file_type(file_type.lower()),
                        filter_null=False,
                        parser_kwargs={
                            "orient": "index",
                            "lines": False,
                        }))

    # Add a deserialize stage.
    # At this stage, messages were logically partitioned based on the 'pipeline_batch_size'.
    pipeline.add_stage(DeserializeStage(config))

    pipeline.add_stage(NeMoPreprocessingStage(config))

    pipeline.add_stage(
        NeMoInferenceStage(config, model_name="gpt5b", customization_id="7436ca66-ec34-42f2-8261-83fe9155fb13"))

    # Add a monitor stage.
    pipeline.add_stage(MonitorStage(config, description="Inference rate"))

    output_stage = pipeline.add_stage(InMemorySinkStage(config))

    # Run the pipeline.
    pipeline.run()

    # Convert the output to a dictionary.
    output_df = cudf.concat([x.get_meta(["response", "_index_"]) for x in output_stage.get_messages()]).to_pandas()
    output_df["_index_"] = output_df["_index_"].astype(str)
    output_df = output_df.set_index("_index_")
    output_dict = output_df["response"].to_dict()

    # ====================== EVALUATION ======================
    ground_truth = json.load(open('./pubmedqa/data/test_ground_truth.json'))

    assert set(list(ground_truth)) == set(list(output_dict)), 'IDs in the dataset must match the validation set.'

    # Load the truth and prediction into the right format
    pmids = list(ground_truth)
    truth = [ground_truth[pmid] for pmid in pmids]
    preds = [output_dict[pmid] for pmid in pmids]

    # Calc the score
    acc = accuracy_score(truth, preds)
    maf = f1_score(truth, preds, average='macro')

    print('Accuracy %f' % acc)
    print('Macro-F1 %f' % maf)


if __name__ == "__main__":
    run_pipeline()
