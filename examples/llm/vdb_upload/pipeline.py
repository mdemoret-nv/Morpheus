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
import logging
import os
import pickle
import time

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

import cudf

from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.general.trigger_stage import TriggerStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.input.rss_source_stage import RSSSourceStage
from morpheus.stages.output.write_to_vector_db_stage import WriteToVectorDBStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage

from ..common.utils import build_milvus_config
from ..common.utils import build_rss_urls
from ..common.web_scraper_stage import WebScraperStage

logger = logging.getLogger(__name__)


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

    config = Config()
    config.mode = PipelineModes.NLP

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_fea_length
    config.edge_buffer_size = 128

    # CppConfig.set_should_use_cpp(False)

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

        df.rename(columns={'metadata.title': 'title', 'metadata.link': 'link'}, inplace=True)

        # df = df.iloc[0:config.pipeline_batch_size]

        df = cudf.DataFrame.from_pandas(df)

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

    # add deserialize stage
    pipe.add_stage(DeserializeStage(config))

    if (isolate_embeddings):
        pipe.add_stage(TriggerStage(config))

    # add preprocessing stage
    pipe.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file="data/bert-base-uncased-hash.txt",
                           do_lower_case=True,
                           truncation=True,
                           add_special_tokens=False,
                           column='page_content'))

    pipe.add_stage(MonitorStage(config, description="Tokenize rate", unit='events', delayed_start=True))

    pipe.add_stage(
        TritonInferenceStage(config,
                             model_name=model_name,
                             server_url=triton_server_url,
                             force_convert_inputs=True,
                             use_shared_memory=True))
    pipe.add_stage(MonitorStage(config, description="Inference rate", unit="events", delayed_start=True))

    # pipe.add_stage(TriggerStage(config))

    pipe.add_stage(
        WriteToVectorDBStage(config,
                             resource_name=vector_db_resource_name,
                             resource_kwargs=build_milvus_config(embedding_size=embedding_size),
                             recreate=True,
                             service=vector_db_service,
                             uri=vector_db_uri))

    pipe.add_stage(MonitorStage(config, description="Upload rate", unit="events", delayed_start=False))

    start_time = time.time()

    print("Starting pipeline...")

    pipe.run()

    return start_time
