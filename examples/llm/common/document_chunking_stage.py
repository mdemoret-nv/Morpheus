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
import typing

import mrc
import mrc.core.operators as ops
import pandas as pd
from langchain.document_loaders import ConfluenceLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import cudf

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.utils.logging_timer import log_time

logger = logging.getLogger(f"morpheus.{__name__}")


class DocumentChunkingStage(SinglePortStage):

    def __init__(self, c: Config, content_column: str, chunker=None):

        super().__init__(c)

        if (chunker is None):
            chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)

        self._chunker = chunker
        self._content_column = content_column

    @property
    def name(self) -> str:
        """Return the name of the stage"""
        return "document-chunker"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple(`morpheus.pipeline.messages.MessageMeta`, )
            Accepted input types.

        """
        return (
            cudf.DataFrame,
            pd.DataFrame,
            MessageMeta,
        )

    def supports_cpp_node(self) -> bool:
        """Indicates whether or not this stage supports a C++ node"""
        return False

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(schema.input_schema.get_type())

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject):

        splitting_pages = builder.make_node(self.unique_name, ops.map(self._splitting_documents))
        splitting_pages.launch_options.pe_count = self._config.num_threads

        builder.make_edge(input_node, splitting_pages)

        return splitting_pages

    def _splitting_documents(self, doc_df: pd.DataFrame | MessageMeta):

        convert_to_meta = False

        if isinstance(doc_df, MessageMeta):
            doc_df = doc_df.df
            convert_to_meta = True

        if (isinstance(doc_df, cudf.DataFrame)):
            doc_df = doc_df.to_pandas()

        df_list = doc_df.to_dict(orient="records")
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

        # Convert from a dataframe to Documents
        split_docs = self._chunker.split_documents(documents)

        df = pd.json_normalize([x.dict() for x in split_docs])

        # Rename the columns to remove the metadata prefix
        map_cols = {name: name.removeprefix("metadata.") for name in df.columns if name.startswith("metadata.")}

        df.rename(columns=map_cols, inplace=True)

        if convert_to_meta:
            return MessageMeta(cudf.from_pandas(df))
        else:
            return cudf.from_pandas(df)
