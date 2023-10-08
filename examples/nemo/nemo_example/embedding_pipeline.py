import logging
import os
import pickle
import threading
import time
import typing

import click
import mrc
import mrc.core.operators as ops
import pandas as pd
from langchain.document_loaders import ConfluenceLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus

import cudf

from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.multi_response_message import MultiResponseMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.general.trigger_stage import TriggerStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage
from morpheus.utils.logger import configure_logging

from .logging_timer import log_time
from .rag_pipeline import get_hf_embeddings
from .rag_pipeline import getVDB

logger = logging.getLogger(f"morpheus.{__name__}")


class ConfluenceSource(SingleOutputSource):

    def __init__(self, c: Config, use_cache: str = None, chunk_size=600):

        super().__init__(c)

        self._batch_size = 100
        self._max_pages = 10000
        self._use_cache = use_cache
        self._chunk_size = chunk_size

        self._loader = ConfluenceLoader(
            url="https://confluence.nvidia.com",
            token=os.environ.get("CONFLUENCE_API_KEY", None),
        )

        self._space = "PRODSEC"

        self._cache_dir = f"./.cache/llm/confluence/{self._space}"

        # Ensure the directory exists
        os.makedirs(self._cache_dir, exist_ok=True)

        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=self._chunk_size,
                                                             chunk_overlap=self._chunk_size // 10,
                                                             length_function=len)

        self._mutex = threading.Lock()
        self._end_page_number = -1

    @property
    def name(self) -> str:
        """Return the name of the stage"""
        return "from-confluence"

    def supports_cpp_node(self) -> bool:
        """Indicates whether or not this stage supports a C++ node"""
        return False

    def _build_source(self, builder: mrc.Builder) -> StreamPair:

        # Calc the number of threads to use as the total threads divided by 2 clamped to [1, 4]
        engines_per_node = max(min(self._config.num_threads // 2, 4), 1)

        if (self._use_cache is None):
            create_slices = builder.make_source(self.unique_name + "-slices", self._generate_slices())

            download_pages = builder.make_node(self.unique_name + "-download",
                                               ops.map(self._download_pages),
                                               ops.filter(lambda x: len(x) > 0))
            download_pages.launch_options.pe_count = self._config.num_threads

            builder.make_edge(create_slices, download_pages)

            # download_pages = builder.make_source(self.unique_name + "-download", self._download_pages2())

            if (self._disable_cache):
                add_comments = builder.make_node(self.unique_name + "-comments", ops.map(self._add_comments))
                add_comments.launch_options.pe_count = engines_per_node

                builder.make_edge(download_pages, add_comments)
            else:
                add_comments = download_pages

            process_pages = builder.make_node(self.unique_name + "-process", ops.map(self._process_pages))
            process_pages.launch_options.pe_count = engines_per_node

            builder.make_edge(add_comments, process_pages)

            document_node = process_pages
        else:
            from_cache = builder.make_source(self.unique_name + "-cache", self._load_documents_from_cache())

            document_node = from_cache

        splitting_pages = builder.make_node(self.unique_name + "-split", ops.map(self._splitting_pages))
        splitting_pages.launch_options.pe_count = engines_per_node

        builder.make_edge(document_node, splitting_pages)

        out_type = MessageMeta

        return splitting_pages, out_type

    def _generate_slices(self):
        limit = self._batch_size

        # First figure out how many pages are in the space
        cql_response = self._loader.confluence.cql(cql="space=PRODSEC and type=page", start=0, limit=1)

        total_pages = cql_response["totalSize"]

        max_pages = min(total_pages, self._max_pages)

        logger.info(f"There are {total_pages} pages in the space. Downloading {max_pages}")

        for i in range(0, max_pages, limit):
            yield {"start": i, "limit": min(limit, self._max_pages - i)}

    def _download_pages(self, slice_info: dict):

        start = slice_info["start"]
        limit = slice_info["limit"]

        # Check if we have already found the end

        # We add children.comment to see if there are any comments on the page. If there are, we will need to make a
        # separate query to get them. Wish we could use children.comment.body.view.value but this only populates the top
        # level comment
        pages = self._loader.confluence.get_all_pages_from_space(space=self._space,
                                                                 start=start,
                                                                 limit=limit,
                                                                 status="current",
                                                                 expand="body.storage.value,children.comment")

        if (len(pages) == limit):
            logger.debug(f"Got {len(pages)} pages")
        elif (len(pages) > 0 and len(pages) < limit):
            logger.debug(f"Only returned {len(pages)}/{limit} pages, assuming no more pages to download")
        else:
            pass

        return pages

    def _load_documents_from_cache(self):

        with open(self._use_cache, "rb") as f:
            documents = pickle.load(f)

        for i in range(0, len(documents), self._batch_size):
            yield documents[i:min(i + self._batch_size, len(documents))]

        logger.debug(f"Loaded {len(documents)} from cache")

    def _add_comments(self, pages: list[dict]):
        with log_time(msg="Adding comments to {count} pages took {duration:.2f}ms. {rate_per_sec:.2f} pages/sec",
                      log_fn=logger.debug,
                      count=len(pages)):

            for page in pages:
                comments = []

                if ("children" in page and "comment" in page["children"] and "size" in page["children"]["comment"]):

                    # We have already fetched if there are any comments
                    comment_count = page["children"]["comment"]["size"]

                    if (comment_count > 0):
                        # There are comments on the page. Need to fetch them with depth=all to get all of the replies
                        comments = self._loader.confluence.get_page_comments(page["id"],
                                                                             expand="body.view.value",
                                                                             depth="all")["results"]
                else:
                    # Have not fetched comments yet
                    comments = self._loader.confluence.get_page_comments(page["id"],
                                                                         expand="body.view.value",
                                                                         depth="all")["results"]

                # Add to the page for processing later
                page.get("children", {}).get("comment", {})["results"] = comments

        return pages

    def _process_one_page(
        self,
        page: dict,
        include_attachments: bool,
        include_comments: bool,
    ) -> Document:
        # Adopted from ConfluenceLoader.process_page. Improved to use existing comments instead of making a new query
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except ImportError:
            raise ImportError("`beautifulsoup4` package not found, please run "
                              "`pip install beautifulsoup4`")

        if include_attachments:
            attachment_texts = self._loader.process_attachment(page["id"])
        else:
            attachment_texts = []
        text = BeautifulSoup(page["body"]["storage"]["value"], "lxml").get_text(" ",
                                                                                strip=True) + "".join(attachment_texts)

        if ("children" in page and "comment" in page["children"] and "results" in page["children"]["comment"]):

            comments = page["children"]["comment"]["results"]

            comment_texts = [
                BeautifulSoup(comment["body"]["view"]["value"], "lxml").get_text(" ", strip=True)
                for comment in comments
            ]
            text = text + "".join(comment_texts)

        return Document(
            page_content=text,
            metadata={
                "title": page["title"],
                "id": page["id"],
                "source": self._loader.base_url.strip("/") + page["_links"]["webui"],
            },
        )

    def _process_pages(self, pages: list[dict]):

        with log_time(msg="Processing {count} pages took {duration:.2f}ms. {rate_per_sec:.2f} pages/sec",
                      log_fn=logger.debug,
                      count=len(pages)):
            documents = []

            if (self._disable_cache):
                for p in pages:
                    # documents.append(self._loader.process_page(p, include_attachments=False, include_comments=True))
                    documents.append(self._process_one_page(p, include_attachments=False, include_comments=True))
            else:
                for p in pages:
                    cache_path = os.path.join(self._cache_dir, f"{p['id']}.pkl")

                    if (os.path.exists(cache_path)):
                        # Load it from the cache
                        with open(cache_path, "rb") as f:
                            doc = pickle.load(f)
                    else:
                        doc = self._loader.process_page(p, include_attachments=False, include_comments=True)

                        # Now save to the cache
                        with open(cache_path, "wb") as f:
                            pickle.dump(doc, f)

                    documents.append(doc)

            return documents

    def _splitting_pages(self, documents: list[Document]):

        texts = self._text_splitter.split_documents(documents)

        df = pd.json_normalize([x.dict() for x in texts])

        # Rename the columns to remove the metadata prefix
        map_cols = {name: name.removeprefix("metadata.") for name in df.columns if name.startswith("metadata.")}

        df.rename(columns=map_cols, inplace=True)

        return MessageMeta(cudf.from_pandas(df))


class WriteToMilvus(SinglePortStage):

    def __init__(self, c: Config, model_name: str, content_column: str = "page_content"):

        super().__init__(c)

        self._content_column = content_column

        model_name = f"sentence-transformers/{model_name}"

        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {
            # 'normalize_embeddings': True, # set True to compute cosine similarity
            "batch_size": 100,
        }

        embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                           model_kwargs=model_kwargs,
                                           encode_kwargs=encode_kwargs)

        self._milvus = Milvus(
            embedding_function=embeddings,
            collection_name="Confluence",
            connection_args={
                "host": "localhost",
                "port": "19530",
            },
            drop_old=True,
        )

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
        return (MessageMeta, MultiResponseMessage)

    def supports_cpp_node(self):
        """Indicates whether this stage supports a C++ node."""
        return False

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        if (issubclass(input_stream[1], MultiResponseMessage)):
            node = builder.make_node(self.unique_name, ops.map(self._add_documents_pre_embeded))
        elif (issubclass(input_stream[1], MessageMeta)):
            node = builder.make_node(self.unique_name, ops.map(self._add_documents))
        else:
            raise ValueError(f"Unexpected input type {input_stream[1]}")

        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]

    def _add_documents_pre_embeded(self, message: MultiResponseMessage):

        # Probs tensor contains all of the embeddings
        embeddings = message.get_probs_tensor()

        # Build up the metadata from the message
        metadata = message.get_meta().to_pandas()

        # metadata: pd.DataFrame = metadata[metadata.columns.difference(["page_content"])].to_pandas()

        # map_cols = {name: name.removeprefix("metadata.") for name in metadata.columns if name.startswith("metadata.")}

        # metadata = metadata.rename(columns=map_cols)

        embeddings_list = embeddings.tolist()

        # Init the collection on first pass
        if (self._milvus.col is None):
            meta_cols = metadata.columns.difference([self._content_column])

            self._milvus._init(embeddings=embeddings_list, metadatas=metadata[meta_cols].to_dict(orient="records"))

        # Build the dataframe for all data to be inserted
        df = metadata.rename(columns={self._content_column: "text"})

        # Add embeddings
        df["vector"] = embeddings_list

        # Now use the col object to add to the collection
        self._milvus.col.insert(data=df)

        return message

    def _add_documents(self, message: MessageMeta):

        # Recreate the documents from the dataframe
        df = message.df.to_pandas()

        # metadata: pd.DataFrame = full_metadata[full_metadata.columns.difference([self._self._content_column
        #                                                                          ])].to_pandas()

        df_list = df.to_dict(orient="records")
        documents = []

        for row in df_list:

            content = row[self._content_column]

            # Remove it from the dict
            del row[self._content_column]

            doc = Document.parse_obj({
                "page_content": content,
                "metadata": row,
            })

            documents.append(doc)

        self._milvus.add_documents(documents)

        return message


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--num_threads",
    default=os.cpu_count(),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use",
)
@click.option(
    "--pipeline_batch_size",
    default=1024,
    type=click.IntRange(min=1),
    help=("Internal batch size for the pipeline. Can be much larger than the model batch size. "
          "Also used for Kafka consumers"),
)
@click.option(
    "--model_max_batch_size",
    default=64,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model",
)
@click.option(
    "--model_fea_length",
    default=256,
    type=click.IntRange(min=1),
    help="Features length to use for the model",
)
@click.option(
    "--input_file",
    default="output.csv",
    help="The path to input event stream",
)
@click.option(
    "--output_file",
    default="output.csv",
    help="The path to the file where the inference output will be saved.",
)
@click.option("--server_url", required=True, default='192.168.0.69:8000', help="Tritonserver url")
@click.option(
    "--model_name",
    required=True,
    default='all-mpnet-base-v2',
    help="The name of the model that is deployed on Triton server",
)
@click.option("--pre_calc_embeddings",
              is_flag=True,
              default=False,
              help="Whether to pre-calculate the embeddings using Triton")
@click.option("--isolate_embeddings",
              is_flag=True,
              default=False,
              help="Whether to pre-calculate the embeddings using Triton")
@click.option("--use_cache",
              type=click.Path(file_okay=True, dir_okay=False),
              default=None,
              help="What cache to use for the confluence documents")
def pipeline(num_threads,
             pipeline_batch_size,
             model_max_batch_size,
             model_fea_length,
             input_file,
             output_file,
             server_url,
             model_name,
             pre_calc_embeddings,
             isolate_embeddings,
             use_cache):

    CppConfig.set_should_use_cpp(True)

    config = Config()
    config.mode = PipelineModes.NLP

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_fea_length
    config.mode = PipelineModes.NLP
    config.edge_buffer_size = 128

    config.class_labels = [str(i) for i in range(384)]

    pipeline = LinearPipeline(config)

    # add doca source stage
    # pipeline.set_source(FileSourceStage(config, filename=input_file, repeat=1))
    pipeline.set_source(ConfluenceSource(config, use_cache=use_cache))
    pipeline.add_stage(MonitorStage(config, description="File source rate", unit='events'))

    if (isolate_embeddings):
        pipeline.add_stage(TriggerStage(config))

    if (pre_calc_embeddings):

        # add deserialize stage
        pipeline.add_stage(DeserializeStage(config))

        # add preprocessing stage
        pipeline.add_stage(
            PreprocessNLPStage(config,
                               vocab_hash_file="data/bert-base-uncased-hash.txt",
                               do_lower_case=True,
                               truncation=True,
                               add_special_tokens=False,
                               column='page_content'))

        pipeline.add_stage(MonitorStage(config, description="Tokenize rate", unit='events', delayed_start=True))

        pipeline.add_stage(
            TritonInferenceStage(config,
                                 model_name=model_name,
                                 server_url="localhost:8001",
                                 force_convert_inputs=True,
                                 use_shared_memory=True))
        pipeline.add_stage(MonitorStage(config, description="Inference rate", unit="events", delayed_start=True))

    pipeline.add_stage(WriteToMilvus(config, model_name=model_name))

    pipeline.add_stage(MonitorStage(config, description="Upload rate", unit="events", delayed_start=True))

    # # Secondary monitor to track the entire pipeline throughput
    # pipeline.add_stage(MonitorStage(config, description="Total rate", unit="events", delayed_start=False))

    start_time = time.time()

    pipeline.run()

    duration = time.time() - start_time

    print(f"Total duration: {duration:.2f} seconds")


@cli.command()
@click.option(
    "--model_name",
    required=True,
    default='all-mpnet-base-v2',
    help="The name of the model that is deployed on Triton server",
)
@click.option(
    "--save_cache",
    default=None,
    type=click.Path(file_okay=True, dir_okay=False),
    help="Location to save the cache to",
)
def chain(model_name, save_cache):
    with log_time(msg="Seeding with chain took {duration} ms. {rate_per_sec} docs/sec", log_fn=logger.debug) as l:

        loader = ConfluenceLoader(
            url="https://confluence.nvidia.com",
            token=os.environ.get("CONFLUENCE_API_KEY", None),
        )

        documents = loader.load(space_key="PRODSEC", max_pages=2000)

        if (save_cache is not None):
            with open(save_cache, "wb") as f:
                pickle.dump(documents, f)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20, length_function=len)

        documents = text_splitter.split_documents(documents)

        l.count = len(documents)

        print(f"Loaded {len(documents)} documents")

        with log_time(msg="Adding to Milvus took {duration} ms. Doc count: {count}. {rate_per_sec} docs/sec",
                      count=l.count,
                      log_fn=logger.debug):

            Milvus.from_documents(documents,
                                  get_hf_embeddings(f"sentence-transformers/{model_name}"),
                                  collection_name="LangChain",
                                  drop_old=True)


if __name__ == "__main__":
    cli()
