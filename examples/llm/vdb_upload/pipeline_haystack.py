# Copyright (c) 2023, NVIDIA CORPORATION.
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
import asyncio
import functools
import inspect
import logging
import os
import pickle
import time
import typing

import networkx as nx
import pandas as pd
from haystack import BaseComponent
from haystack import Document
from haystack import Pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from milvus_haystack import MilvusDocumentStore  # pylint: disable=unused-import # noqa: F401

import cudf

from morpheus._lib.messages import MessageMeta
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.llm import LLMContext
from morpheus.llm import LLMEngine
from morpheus.llm import LLMNodeBase
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus.messages import ControlMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.general.trigger_stage import TriggerStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.input.rss_source_stage import RSSSourceStage
from morpheus.stages.llm.llm_engine_stage import LLMEngineStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.output.write_to_vector_db_stage import WriteToVectorDBStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage

from ..common.utils import build_milvus_config
from ..common.utils import build_rss_urls
from ..common.utils import convert_docs_langchain_to_haystack
from ..common.web_scraper_stage import WebScraperStage

logger = logging.getLogger(__name__)


class TestHaystackEmbeddingNode(LLMNodeBase):
    """
    Extracts fields from the DataFrame contained by the message attached to the `LLMContext` and copies them directly
    to the context.

    The list of fields to be extracted is provided by the task's `input_keys` attached to the `LLMContext`.
    """

    def __init__(self, cache_file: str, component: BaseComponent, **component_kwargs) -> None:
        super().__init__()

        with open(cache_file, "rb") as f:
            documents = pickle.load(f)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=128 // 10)

        documents = text_splitter.split_documents(documents)

        self._documents = convert_docs_langchain_to_haystack(documents)

        self._component = component
        self._component_kwargs = component_kwargs

    def get_input_names(self) -> list[str]:
        # This node does not receive its inputs from upstream nodes, but rather from the task itself
        return []

    async def execute(self, context: LLMContext) -> LLMContext:

        self._component._dispatch_run(documents=self._documents, **self._component_kwargs)

        return context


class HaystackFileNode(LLMNodeBase):
    """
    Extracts fields from the DataFrame contained by the message attached to the `LLMContext` and copies them directly
    to the context.

    The list of fields to be extracted is provided by the task's `input_keys` attached to the `LLMContext`.
    """

    def get_input_names(self) -> list[str]:
        # This node does not receive its inputs from upstream nodes, but rather from the task itself
        return []

    async def execute(self, context: LLMContext) -> LLMContext:

        # Get the keys from the task
        input_keys: list[str] = typing.cast(list[str], context.task()["input_keys"])

        with context.message().payload().mutable_dataframe() as df:
            input_dict: list[dict] = df[input_keys].to_dict(orient="records")

        # if (len(input_keys) == 1):
        #     # Extract just the first key if there is only 1
        #     context.set_output(input_dict[input_keys[0]])
        # else:
        context.set_output({"documents": input_dict, "root_node": "File"})

        return context


class HaystackNode(LLMNodeBase):
    """
    Executes a Haystack agent in an LLMEngine
    Parameters
    ----------
    agent : Agent
        The agent to use to execute.
    return_only_answer : bool
        Return only the answer if this flag is set to True; otherwise, return the transcript, query, and answer.
        Default value is True.
    """

    def __init__(self, component: BaseComponent, return_only_answer: bool = True):
        super().__init__()

        self._component = component
        self._return_only_answer = return_only_answer

        run_signature_args = inspect.signature(self._component.run)

        self._all_inputs = list(run_signature_args.parameters.keys())

        self._optional_inputs = [
            p_name for p_name,
            p_value in run_signature_args.parameters.items() if p_value.default != inspect.Parameter.empty
        ]

    def get_input_names(self) -> list[str]:
        return self._all_inputs

    def get_optional_input_names(self) -> list[str]:
        return self._optional_inputs

    async def _run_single(self, **kwargs: typing.Any) -> dict[str, typing.Any]:

        if (hasattr(self._component, "arun")):
            return self._component.arun(**kwargs)

        result = self._component._dispatch_run(**kwargs)

        # loop = asyncio.get_event_loop()

        # result = await loop.run_in_executor(None, functools.partial(self._component._dispatch_run, **kwargs))

        return result[0]

        # all_lists = all(isinstance(v, list) for v in kwargs.values())

        # # Check if all values are a list
        # if all_lists:
        #     # Transform from dict[str, list[Any]] to list[dict[str, Any]]
        #     input_list = [dict(zip(kwargs, t)) for t in zip(*kwargs.values())]

        #     results = []

        #     # Run multiple queries asynchronously
        #     loop = asyncio.get_event_loop()

        #     # The asyncio loop utilizes the default executor when 'None' is passed.
        #     tasks = [loop.run_in_executor(None, self._agent.run, x) for x in input_list]
        #     results = await asyncio.gather(*tasks)

        #     return results

        # # We are not dealing with a list, so run single
        # return [self._agent.run(**kwargs)]

    def _prepare_arguments(self, arguments: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Prepares the arguments for the agent's run method.
        Parameters
        ----------
        arguments : dict[str, typing.Any]
            The arguments to prepare.
        Returns
        -------
        dict[str, typing.Any]
            The prepared arguments.
        """

        if ("documents" in arguments):
            # Convert the documents to Haystack documents
            documents = [Document.from_dict(doc) for doc in arguments["documents"]]

            arguments["documents"] = documents

        return arguments

    def _parse_results(self, results: dict[str, list[typing.Any]]) -> dict[str, list[typing.Any]]:

        if ("documents" in results):
            doc_dict = [doc.to_dict() for doc in results["documents"]]

            # Fix up the numpy arrays
            for doc in doc_dict:
                if ("embedding" in doc and doc["embedding"] is not None):
                    doc["embedding"] = doc["embedding"].tolist()

            results["documents"] = doc_dict

        return results
        # parsed_results = []

        # for item in results:
        #     parsed_result = {}

        #     if self._return_only_answer:
        #         parsed_result["answers"] = [answer.to_dict()["answer"] for answer in item['answers']]
        #     else:
        #         parsed_result["query"] = item['query']['query']
        #         parsed_result["transcript"] = item['transcript']
        #         parsed_result["answers"] = [answer.to_dict() for answer in item['answers']]

        #     parsed_results.append(parsed_result)

        # return parsed_results

    async def execute(self, context: LLMContext) -> LLMContext:
        import mrc.core.options
        print(f"Current CPU set: {mrc.core.options.Config.get_cpuset()}")

        input_dict = context.get_inputs()

        try:
            # Call _run_single asynchronously
            results = await self._run_single(**self._prepare_arguments(input_dict))
            parsed_results = self._parse_results(results)

            context.set_output(parsed_results)

        except KeyError as exe:
            logger.error("Parsing of results encountered an error: %s", exe)
        except Exception as exe:
            logger.error("Processing input encountered an error: %s", exe)

        return context


def test_run():
    from haystack.nodes.retriever import EmbeddingRetriever
    context = LLMContext()

    node = TestHaystackEmbeddingNode(cache_file=".cache/langchain.pkl",
                                     component=EmbeddingRetriever(
                                         document_store=None,
                                         embedding_model="intfloat/e5-small-v2",
                                     ),
                                     root_node="File")

    engine = LLMEngine()

    engine.add_node("Test", inputs=[], node=node)

    async def main_fn():

        message = ControlMessage()
        message.payload(MessageMeta(df=cudf.DataFrame()))
        message.add_task("llm_engine", {
            "task_type": "completion", "task_dict": {
                "input_keys": ["title", "link", "content"],
            }
        })

        await engine.run(message)

    asyncio.run(main_fn())


def pipeline(num_threads: int,
             pipeline_batch_size: int,
             model_max_batch_size: int,
             model_fea_length: int,
             embedding_size: int,
             model_name: str,
             isolate_embeddings: bool,
             stop_after: int,
             enable_cache: bool,
             interval_secs: int,
             run_indefinitely: bool,
             vector_db_uri: str,
             vector_db_service: str,
             vector_db_resource_name: str,
             triton_server_url: str):

    # test_run()

    config = Config()
    config.mode = PipelineModes.NLP

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_fea_length
    config.edge_buffer_size = 128

    CppConfig.set_should_use_cpp(False)

    config.class_labels = [str(i) for i in range(embedding_size)]

    pipe = LinearPipeline(config)

    # Load the documents from disk

    cache_file = ".cache/langchain.pkl"

    if (os.path.exists(cache_file)):
        with open(cache_file, "rb") as f:
            documents = pickle.load(f)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=128 // 10)

        documents = text_splitter.split_documents(documents)

        docs_dicts = [doc.dict() for doc in documents]

        df = pd.json_normalize(docs_dicts)

        df.rename(columns={'metadata.title': 'title', 'metadata.link': 'link', "page_content": "content"}, inplace=True)

        df = df.iloc[0:2 * config.pipeline_batch_size]

        pipe.set_source(InMemorySourceStage(config, dataframes=[df]))

        pipe.add_stage(MonitorStage(config, description="Source rate", unit='documents', delayed_start=False))

    else:

        # add rss source stage
        pipe.set_source(
            RSSSourceStage(config,
                           feed_input=build_rss_urls(),
                           stop_after=stop_after,
                           run_indefinitely=run_indefinitely,
                           enable_cache=enable_cache,
                           interval_secs=interval_secs))

        pipe.add_stage(MonitorStage(config, description="Source rate", unit='pages'))

        pipe.add_stage(WebScraperStage(config, chunk_size=128, enable_cache=enable_cache))

        pipe.add_stage(MonitorStage(config, description="Download rate", unit='pages'))

    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["title", "link", "content"], }}

    # add deserialize stage
    pipe.add_stage(
        DeserializeStage(config, message_type=ControlMessage, task_type="llm_engine", task_payload=completion_task))

    if (isolate_embeddings):
        pipe.add_stage(TriggerStage(config))

    # Load YAML to Haystack pipeline
    hay_pipe = Pipeline.load_from_yaml("dense_milvus_index.yaml")

    engine = LLMEngine()
    engine.run
    root_node = None

    # Copy components into nodes
    for name in nx.topological_sort(hay_pipe.graph):
        node_info = hay_pipe.graph.nodes[name]

        if (name == "File"):
            # Add an extractor node
            root_node = engine.add_node("File", node=HaystackFileNode())
        else:
            inputs = [(f"/{n}/*", "*") for n in node_info["inputs"]]
            node = HaystackNode(component=node_info["component"])

            if ("root_node" in node.get_input_names()):
                # Prepend pulling root_node from the root_node
                inputs = [f"/{root_node.name}/root_node"] + inputs

            engine.add_node(name, inputs=inputs, node=node)

    engine.add_task_handler(inputs=[f"/{name}/documents"], handler=SimpleTaskHandler())

    # pipe.add_stage(LLMEngineStage(config, engine=engine))
    sink = pipe.add_stage(InMemorySinkStage(config))

    # # add preprocessing stage
    # pipe.add_stage(
    #     PreprocessNLPStage(config,
    #                        vocab_hash_file="data/bert-base-uncased-hash.txt",
    #                        do_lower_case=True,
    #                        truncation=True,
    #                        add_special_tokens=False,
    #                        column='page_content'))

    # pipe.add_stage(MonitorStage(config, description="Tokenize rate", unit='events', delayed_start=True))

    # pipe.add_stage(
    #     TritonInferenceStage(config,
    #                          model_name=model_name,
    #                          server_url=triton_server_url,
    #                          force_convert_inputs=True,
    #                          use_shared_memory=True))
    # pipe.add_stage(MonitorStage(config, description="Inference rate", unit="events", delayed_start=True))

    # # pipe.add_stage(TriggerStage(config))

    # pipe.add_stage(
    #     WriteToVectorDBStage(config,
    #                          resource_name=vector_db_resource_name,
    #                          resource_kwargs=build_milvus_config(embedding_size=embedding_size),
    #                          recreate=True,
    #                          service=vector_db_service,
    #                          uri=vector_db_uri))

    pipe.add_stage(MonitorStage(config, description="Upload rate", unit="events", delayed_start=False))

    message = ControlMessage()
    message.payload(MessageMeta(df=df))
    message.add_task("llm_engine", {
        "task_type": "completion", "task_dict": {
            "input_keys": ["title", "link", "content"],
        }
    })

    start_time = time.time()

    print("Starting pipeline...")

    # pipe.run()

    # message = sink.get_messages()[0]
    async def test(m):
        result = await engine.run(m)

        print(result)

    asyncio.run(test(message))

    return start_time
