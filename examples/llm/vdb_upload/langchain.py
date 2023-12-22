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
import logging
import os
import pickle
import typing
from datetime import datetime

from langchain.document_loaders.rss import RSSFeedLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.milvus import Milvus

from llm.common.utils import build_rss_urls
from morpheus.llm import LLMContext
from morpheus.llm import LLMNodeBase
from morpheus.utils.logging_timer import log_time

logger = logging.getLogger(__name__)


def haystack_pipeline(model_name, save_cache):

    from haystack import Document as HaystackDocument
    from haystack import Pipeline
    from haystack.nodes.preprocessor import PreProcessor
    from haystack.nodes.retriever import EmbeddingRetriever
    from milvus_haystack import MilvusDocumentStore

    with log_time(msg="Seeding with Haystack pipeline took {duration} ms. {rate_per_sec} docs/sec",
                  log_fn=logger.debug) as log:

        url_list = build_rss_urls()

        if (save_cache is not None and os.path.exists(save_cache)):
            with open(save_cache, "rb") as f:
                documents = pickle.load(f)
        else:

            loader = RSSFeedLoader(urls=url_list)

            documents = loader.load()

            if (save_cache is not None):
                with open(save_cache, "wb") as f:
                    pickle.dump(documents, f)

        logger.info("Loaded %s documents from %s URLs", len(documents), len(url_list))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=128 // 10)

        pre_chunk_count = len(documents)

        documents = text_splitter.split_documents(documents)

        logger.info("Split %s documents into %s chunks", pre_chunk_count, len(documents))

        log.count = len(documents)

        # Convert documents to Haystack docs
        haystack_documents = []
        for document in documents:
            metadata = document.metadata or {}

            for key, value in document.metadata.items():
                if isinstance(value, datetime):
                    metadata[key] = value.isoformat()

            haystack_document = HaystackDocument(
                content=document.page_content,
                meta=metadata,
            )
            haystack_document.meta["_doc_id"] = haystack_document.id  # used for deletion
            haystack_documents.append(haystack_document)

        # pipeline = Pipeline.load_from_yaml("dense_milvus_index.yaml")

        pipeline = Pipeline()

        pipeline.add_node(component=PreProcessor(split_length=128,
                                                 split_respect_sentence_boundary=False,
                                                 split_overlap=128 // 10),
                          name="Preprocessor",
                          inputs=["File"])

        pipeline.add_node(component=EmbeddingRetriever(
            document_store=None,
            progress_bar=False,
            embedding_model="intfloat/e5-small-v2",
        ),
                          name="Dense_Retriever",
                          inputs=["Preprocessor"])

        pipeline.add_node(component=MilvusDocumentStore(
            embedding_dim=384,
            recreate_index=True,
            uri="http://localhost:19530/default",
        ),
                          name="Dense_Document_Store",
                          inputs=["Dense_Retriever"])

        pipeline.save_to_yaml("dense_milvus_index.yaml")

        result = pipeline.run(documents=haystack_documents)

        print("Done. Result:")
        print(len(result))

        # embeddings = HuggingFaceEmbeddings(
        #     model_name=model_name,
        #     model_kwargs={'device': 'cuda'},
        #     encode_kwargs={
        #         # 'normalize_embeddings': True, # set True to compute cosine similarity
        #         "batch_size": 100,
        #     })

        # with log_time(msg="Adding to Milvus took {duration} ms. Doc count: {count}. {rate_per_sec} docs/sec",
        #               count=log.count,
        #               log_fn=logger.debug):

        #     Milvus.from_documents(documents, embeddings, collection_name="LangChain", drop_old=True)


def chain(model_name, save_cache):

    with log_time(msg="Seeding with chain took {duration} ms. {rate_per_sec} docs/sec", log_fn=logger.debug) as log:

        loader = RSSFeedLoader(urls=build_rss_urls())

        documents = loader.load()

        if (save_cache is not None):
            with open(save_cache, "wb") as f:
                pickle.dump(documents, f)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20, length_function=len)

        documents = text_splitter.split_documents(documents)

        log.count = len(documents)

        logger.info("Loaded %s documents", len(documents))

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={
                # 'normalize_embeddings': True, # set True to compute cosine similarity
                "batch_size": 100,
            })

        with log_time(msg="Adding to Milvus took {duration} ms. Doc count: {count}. {rate_per_sec} docs/sec",
                      count=log.count,
                      log_fn=logger.debug):

            Milvus.from_documents(documents, embeddings, collection_name="LangChain", drop_old=True)
