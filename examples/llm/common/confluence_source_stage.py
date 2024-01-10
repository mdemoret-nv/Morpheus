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
import threading

import mrc
import mrc.core.operators as ops
import pandas as pd
from langchain.document_loaders import ConfluenceLoader
from langchain.document_loaders.confluence import ContentFormat
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import cudf

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.utils.logging_timer import log_time

logger = logging.getLogger(f"morpheus.{__name__}")


class ConfluenceSource(SingleOutputSource):

    def __init__(self, c: Config, confluence_base_url: str, space: str, use_cache: bool, chunk_size=600):

        super().__init__(c)

        self._batch_size = 100
        self._max_pages = 10000
        self._use_cache = use_cache
        self._chunk_size = chunk_size
        self._confluence_base_url = confluence_base_url

        self._loader = ConfluenceLoader(
            url=self._confluence_base_url,
            token=os.environ.get("CONFLUENCE_API_KEY", None),
        )

        self._space = space

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

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def _build_source(self, builder: mrc.Builder):

        # Calc the number of threads to use as the total threads divided by 2 clamped to [1, 4]
        engines_per_node = max(min(self._config.num_threads // 2, 4), 1)

        # Disable loading all documents from the cache. This should happen outside this stage
        if (True):
            create_slices = builder.make_source(self.unique_name + "-slices", self._generate_slices())

            download_pages = builder.make_node(self.unique_name + "-download",
                                               ops.map(self._download_pages),
                                               ops.filter(lambda x: len(x) > 0))
            download_pages.launch_options.pe_count = self._config.num_threads

            builder.make_edge(create_slices, download_pages)

            # download_pages = builder.make_source(self.unique_name + "-download", self._download_pages2())

            if (not self._use_cache):
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

        return splitting_pages

    def _generate_slices(self):
        limit = self._batch_size

        # First figure out how many pages are in the space
        cql_response = self._loader.confluence.cql(cql=f"space={self._space} and type=page", start=0, limit=1)

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

        with open(self._cache_dir, "rb") as f:
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

            if (not self._use_cache):
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
                        doc = self._loader.process_page(p,
                                                        content_format=ContentFormat.STORAGE,
                                                        include_attachments=False,
                                                        include_comments=True)

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
