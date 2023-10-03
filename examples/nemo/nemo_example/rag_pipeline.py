import asyncio
import functools
import logging
import os
import textwrap
import time
import typing
from typing import Any
from typing import Coroutine
from typing import List
from typing import Optional

import mrc
import mrc.core.operators as ops
import pydantic
from aiohttp import ClientSession
from langchain import LLMMathChain
from langchain import OpenAI
from langchain import PromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.agents.tools import Tool
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain.document_loaders import ConfluenceLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Milvus
from pypdf.errors import PdfStreamError

import cudf

from morpheus._lib.llm import LLMContext
from morpheus._lib.llm import LLMEngine
from morpheus._lib.llm import LLMNodeBase
from morpheus._lib.llm import LLMPromptGenerator
from morpheus._lib.llm import LLMTask
from morpheus._lib.llm import LLMTaskHandler
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.general.monitor_stage import MonitorStage

from .common import ExtracterNode
from .common import LLMDictTask
from .llm_engine import NeMoLangChain
from .logging_timer import log_time

logger = logging.getLogger(f"morpheus.{__name__}")

MILVUS_COLLECTION_NAME = "Test"


class ConfluenceSource(SingleOutputSource):

    def __init__(self, c: Config):

        super().__init__(c)

        self._max_pages = 10000

        self._loader = ConfluenceLoader(
            url="https://confluence.nvidia.com",
            token=os.environ.get("CONFLUENCE_API_KEY", None),
        )

        self._embeddings = get_hf_embeddings("intfloat/e5-large-v2")
        self._vdb = getVDB("Milvus", embeddings=self._embeddings, collection_name=MILVUS_COLLECTION_NAME)

        self._text_splitter1 = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self._text_splitter2 = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20, length_function=len)

    @property
    def name(self) -> str:
        """Return the name of the stage"""
        return "from-confluence"

    def supports_cpp_node(self) -> bool:
        """Indicates whether or not this stage supports a C++ node"""
        return False

    def _build_source(self, builder: mrc.Builder) -> StreamPair:

        download_pages = builder.make_source(self.unique_name + "-download", self._generate_frames())

        process_pages = builder.make_node(self.unique_name + "-process", ops.map(self._process_pages))
        process_pages.launch_options.pe_count = 4

        builder.make_edge(download_pages, process_pages)

        splitting_pages = builder.make_node(self.unique_name + "-split", ops.map(self._splitting_pages))
        splitting_pages.launch_options.pe_count = 4

        builder.make_edge(process_pages, splitting_pages)

        out_type = typing.List[Document]

        return splitting_pages, out_type

    def _generate_frames(self) -> typing.Iterable[MessageMeta]:

        limit = 50
        count = 0

        for i in range(0, self._max_pages, limit):
            pages = self._loader.confluence.get_all_pages_from_space(space="PRODSEC",
                                                                     start=i,
                                                                     limit=limit,
                                                                     status="current",
                                                                     expand="body.storage.value")

            logger.debug(f"Got {len(pages)} pages")

            yield pages

            count += len(pages)

            if (len(pages) < limit):
                logger.debug(f"Only returned {len(pages)}/{limit} pages, assuming no more pages to download")
                break

        logger.debug(f"Downloading complete {count} pages")

    def _process_pages(self, pages):

        logger.debug(f"Processing {len(pages)} pages")

        documents = self._loader.process_pages(pages,
                                               include_restricted_content=False,
                                               include_attachments=False,
                                               include_comments=True)

        return documents

    def _splitting_pages(self, documents: list[Document]):

        texts1 = self._text_splitter1.split_documents(documents)
        texts = self._text_splitter2.split_documents(texts1)

        return texts


class ArxivSource(SingleOutputSource):

    def __init__(self, c: Config):

        super().__init__(c)

        self._max_pages = 10000

        self._loader = ConfluenceLoader(
            url="https://confluence.nvidia.com",
            token=os.environ.get("CONFLUENCE_API_KEY", None),
        )

        # self._text_splitter1 = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self._text_splitter2 = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)

        self._total_pdfs = 0
        self._total_pages = 0
        self._total_chunks = 0

    @property
    def name(self) -> str:
        """Return the name of the stage"""
        return "from-confluence"

    def supports_cpp_node(self) -> bool:
        """Indicates whether or not this stage supports a C++ node"""
        return False

    def _build_source(self, builder: mrc.Builder) -> StreamPair:

        download_pages = builder.make_source(self.unique_name + "-download", self._generate_frames())

        process_pages = builder.make_node(self.unique_name + "-process", ops.map(self._process_pages))
        process_pages.launch_options.pe_count = 6

        builder.make_edge(download_pages, process_pages)

        splitting_pages = builder.make_node(self.unique_name + "-split", ops.map(self._splitting_pages))
        # splitting_pages.launch_options.pe_count = 4

        builder.make_edge(process_pages, splitting_pages)

        out_type = typing.List[Document]

        return splitting_pages, out_type

    def _generate_frames(self):

        import arxiv

        search_results = arxiv.Search(
            query="large language models",
            max_results=50,
        )

        dir_path = "./shared-dir/dataset/pdfs/"

        for x in search_results.results():

            full_path = os.path.join(dir_path, x._get_default_filename())

            if (not os.path.exists(full_path)):
                x.download_pdf(dir_path)
                logger.debug(f"Downloaded: {full_path}")
                # time.sleep(0.1)

            yield full_path

            self._total_pdfs += 1

        logger.debug(f"Downloading complete {self._total_pdfs} pages")

    def _process_pages(self, pdf_path: str):

        from langchain.document_loaders import PyPDFLoader

        for _ in range(5):
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                self._total_pages += len(documents)

                logger.debug(f"Processing {len(documents)}/{self._total_pages}: {pdf_path}")

                return documents
            except PdfStreamError:
                logger.error(f"Failed to load PDF (retrying): {pdf_path}")
                documents = []

        raise RuntimeError(f"Failed to load PDF: {pdf_path}")

    def _splitting_pages(self, documents: list[Document]):

        # texts1 = self._text_splitter1.split_documents(documents)
        texts = self._text_splitter2.split_documents(documents)

        self._total_chunks += len(texts)

        return texts


class WriteToMilvus(SinglePortStage):

    def __init__(self, c: Config):

        super().__init__(c)

        # self._embeddings = get_hf_embeddings("intfloat/e5-large-v2")
        self._embeddings = get_hf_embeddings("sentence-transformers/all-mpnet-base-v2")
        self._vdb = getVDB("Milvus", embeddings=self._embeddings, collection_name=MILVUS_COLLECTION_NAME)

    @property
    def name(self) -> str:
        """Returns the name of this stage."""
        return "to-milvus"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple(`morpheus.pipeline.messages.MessageMeta`, )
            Accepted input types.

        """
        return (list[Document], )

    def supports_cpp_node(self):
        """Indicates whether this stage supports a C++ node."""
        return False

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
            print("Creating embeddings")
            # Create embedding per thread
            embeddings = get_hf_embeddings("sentence-transformers/all-mpnet-base-v2")
            vdb = getVDB("Milvus", embeddings=embeddings, collection_name=MILVUS_COLLECTION_NAME)

            obs.pipe(ops.map(functools.partial(self._add_documents, vdb=vdb))).subscribe(sub)

        node = builder.make_node(self.unique_name, ops.build(node_fn))
        node.launch_options.pe_count = 1

        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]

    def _add_documents(self, documents: list[Document], vdb: Milvus):

        total_chars = sum([len(x.page_content) for x in documents])

        with log_time(msg=f"Writing {len(documents)} documents to VDB took {{duration}}. {{rate_per_sec}} chars/sec",
                      count=total_chars,
                      log_fn=logger.debug):

            vdb.add_documents(documents)

        return documents


# A version of LlamaCpp that supports async calls
class LlamaCppAsync(LlamaCpp):

    _mutex: asyncio.Lock = asyncio.Lock()

    class Config:
        underscore_attrs_are_private = True

    async def _acall(self,
                     prompt: str,
                     stop: List[str] | None = None,
                     run_manager: AsyncCallbackManagerForLLMRun | None = None) -> str:

        # Can only have one running at a time
        async with self._mutex:
            return await asyncio.to_thread(self._call, prompt, stop, None)


# Prompt generator wrapper around a LangChain agent executor
class LangChainChainNode(LLMNodeBase):

    def __init__(self, chain: Chain):
        super().__init__()

        self._chain = chain

    async def execute(self, context: LLMContext):

        input_dict: dict = context.get_input()

        if (isinstance(input_dict, list)):
            input_dict = {"query": input_dict}

        # Transform from dict[str, list[Any]] to list[dict[str, Any]]
        input_list = [dict(zip(input_dict, t)) for t in zip(*input_dict.values())]

        # outputs = []

        # for x in input_list:
        #     result = await self._chain.acall(inputs=x)

        #     outputs.append(result)

        output_coros = [self._chain.acall(inputs=x) for x in input_list]

        outputs = await asyncio.gather(*output_coros)

        # Uncomment to run synchronously
        # results = [self._chain(inputs=x) for x in input_list]

        # Extract the results from the output
        results = [x["result"] for x in outputs]

        context.set_output(results)


class SimpleTaskHandler(LLMTaskHandler):

    async def try_handle(self, context: LLMContext):

        with context.message().payload().mutable_dataframe() as df:
            df["response"] = context.get_input()

        return [context.message()]


def get_hf_embeddings(model):

    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {
        # 'normalize_embeddings': True, # set True to compute cosine similarity
        "batch_size": 100,
    }

    embeddings = HuggingFaceEmbeddings(model_name=model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    return embeddings


def getVDB(db_name, *, embeddings, collection_name, host="localhost", port="19530"):
    if db_name == 'Chroma':
        #vectordb = Chroma("langchain_store",embeddings, persist_directory="./data-chroma")
        vectordb = Chroma(persist_directory="./data-chroma", embedding_function=embeddings)
        print("Chroma")
    elif db_name == 'FAISS':
        print("FAISS")
        vectordb = FAISS.load_local("vdb_chunks", embeddings, index_name="nv-index")
    elif db_name == 'Milvus':

        # Get existing collection from Milvus
        vectordb: Milvus = Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args={
                "host": host, "port": port
            },
            drop_old=True,
        )
        return vectordb

    raise RuntimeError("Unknown db_type: {db_name}}")


def get_modelpath(model):
    MODEL_DIR = "./shared-dir/llama2/"
    if model == 'llama-2-13b-chat.Q4_K_M':
        print(MODEL_DIR + model + ".gguf")
        return (MODEL_DIR + model + ".gguf")


def get_llm(model, n_ctx, n_gpu_layers, n_batch):
    model_path = get_modelpath(model)
    # llm = LlamaCppAsync(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, temperature=0, n_batch=n_batch)
    # llm = NeMoLangChain(model_name="llama-2-70b-hf")
    llm = NeMoLangChain(model_name="gpt-43b-002")
    return llm


def build_langchain():

    embeddings = get_hf_embeddings("all-mpnet-base-v2")

    vdb = getVDB("Milvus", embeddings=embeddings, collection_name=MILVUS_COLLECTION_NAME)

    retriever = vdb.as_retriever(include_metadata=True, metadata_key='source')

    llm = get_llm("llama-2-13b-chat.Q4_K_M", 2048, 80, 512)

    # Define a prompt template. This is a format for the text input we'll give to our model.
    # It tells the model how to structure its response and what to do in different situations.
    template = textwrap.dedent("""
    I will provide you pieces of [Context] to answer the [Question]. \
    If you don't know the answer based on [Context] just say that you don't know, don't try to make up an answer. \
    [Context]: {context} \
    [Question]: {question} \
    Helpful Answer:""").lstrip()

    # Create a PromptTemplate object from our string template.
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    print(qa_chain.dict())

    return qa_chain


def build_confluence_langchain():

    embeddings = get_hf_embeddings("intfloat/e5-large-v2")

    vectordb = getVDB("Milvus", embeddings=embeddings, collection_name=MILVUS_COLLECTION_NAME)

    llm = get_llm("llama-2-13b-chat.Q4_K_M", 3000, 60, 512)

    retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=retriever,
                                     verbose=True,
                                     return_source_documents=True)

    return qa


async def seed_vdb():

    loader = ConfluenceLoader(
        url="https://confluence.nvidia.com",
        token=os.environ.get("CONFLUENCE_API_KEY", None),
    )

    async def download_pages():

        count = 0
        limit = 50

        coros = []

        for i in range(0, 200, limit):

            coros.append(
                asyncio.to_thread(
                    functools.partial(loader.confluence.get_all_pages_from_space,
                                      space="PRODSEC",
                                      start=i,
                                      limit=limit,
                                      status="current",
                                      expand="body.storage.value")))

        for pages in asyncio.as_completed(coros):

            yield await pages

    async def process_pages(pages):

        print("Processing pages...")

        # Process the docs
        return await asyncio.to_thread(
            functools.partial(loader.process_pages,
                              pages,
                              include_restricted_content=False,
                              include_attachments=False,
                              include_comments=True))

    embeddings = get_hf_embeddings("intfloat/e5-large-v2")
    vdb = getVDB("Milvus", embeddings=embeddings, collection_name=MILVUS_COLLECTION_NAME)

    async for pages in download_pages():
        documents = await process_pages(pages)

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts1 = text_splitter.split_documents(documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20, length_function=len)
        #text_splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=10, encoding_name="cl100k_base")  # This the encoding for text-embedding-ada-002
        texts = text_splitter.split_documents(texts1)

        await asyncio.to_thread(functools.partial(vdb.add_documents, texts))


async def seed_vdb_pipeline():

    c = Config()

    c.num_threads = 12

    pipeline = LinearPipeline(c)

    # pipeline.set_source(ConfluenceSource(c))
    pipeline.set_source(ArxivSource(c))

    pipeline.add_stage(MonitorStage(c, description="Downloading and processing pages", determine_count_fn=len))

    pipeline.add_stage(WriteToMilvus(c))

    pipeline.add_stage(MonitorStage(c, description="Uploading to VDB", determine_count_fn=len))

    await pipeline.run_async()


async def run_rag_pipeline():

    seed_db = True

    if (seed_db):
        await seed_vdb_pipeline()

        return

    # chain = build_langchain()
    chain = build_confluence_langchain()

    # Test run the chain if needed
    # result = await chain.acall(inputs={"query": "What was the GAAP gross margin compared to last year?"})

    # Create the NeMo LLM Service using our API key and org ID
    # llm = NeMoLangChain(model_name="llama-2-70b-hf")
    llm = NeMoLangChain(model_name="gpt-43b-002")
    # llm = OpenAI(temperature=0)
    # engine = LLMEngine()

    # engine.add_node("extract_prompt", [], ExtracterNode())
    # engine.add_node("langchain", [("query", "/extract_prompt")], LangChainChainNode(chain=chain))

    # # Add our task handler
    # engine.add_task_handler(inputs=["/langchain"], handler=SimpleTaskHandler())

    # # Create a control message with a single task which uses the LangChain agent executor
    # message = ControlMessage()

    # message.add_task("llm_engine",
    #                  {
    #                      "task_type": "template",
    #                      "task_dict": LLMDictTask(input_keys=["input"], model_name="gpt-43b-002").dict(),
    #                  })

    questions = [
        "Why is a Security Incident Escalation Test performed?",
        "What is the NSPECT-ID field for? And what section is it found in?",
        "What is ProdSec's team mission?",
        "Who is the VP of ProdSec?",
        "What is the nSpect health grade?"
    ]

    # payload = cudf.DataFrame({
    #     "input": questions,
    # })
    # message.payload(MessageMeta(payload))

    # Finally, run the engine
    # result = await engine.run(message)

    coros = [chain.acall(inputs=x) for x in questions]

    result = await asyncio.gather(*coros)

    # result = [chain(inputs=x) for x in questions]

    print(f"Got results: {result}")
