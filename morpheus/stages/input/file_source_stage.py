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
"""File source stage."""

import logging
import pathlib
import typing

import mrc

from morpheus.cli import register_stage
from morpheus.common import FileTypes
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MessageMeta
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


@register_stage("from-file", modes=[PipelineModes.FIL, PipelineModes.NLP, PipelineModes.OTHER])
class FileSourceStage(PreallocatorMixin, SingleOutputSource):
    """
    Load messages from a file.

    Source stage is used to load messages from a file and dumping the contents into the pipeline immediately. Useful for
    testing performance and accuracy of a pipeline.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    filename : pathlib.Path, exists = True, dir_okay = False
        Name of the file from which the messages will be read.
    iterative : boolean, default = False, is_flag = True
        Iterative mode will emit dataframes one at a time. Otherwise a list of dataframes is emitted. Iterative mode is
        good for interleaving source stages.
    file_type : `morpheus.common.FileTypes`, optional, case_sensitive = False
        Indicates what type of file to read. Specifying 'auto' will determine the file type from the extension.
        Supported extensions: 'csv', 'json', 'jsonlines' and 'parquet'.
    repeat : int, default = 1, min = 1
        Repeats the input dataset multiple times. Useful to extend small datasets for debugging.
    filter_null : bool, default = True
        Whether or not to filter rows with null 'data' column. Null values in the 'data' column can cause issues down
        the line with processing. Setting this to True is recommended.
    """

    def __init__(self,
                 c: Config,
                 filename: pathlib.Path,
                 iterative: bool = False,
                 file_type: FileTypes = FileTypes.Auto,
                 repeat: int = 1,
                 filter_null: bool = True,
                 parser_kwargs: dict = None):

        super().__init__(c)

        self._batch_size = c.pipeline_batch_size

        self._filename = filename
        self._file_type = file_type
        self._filter_null = filter_null

        self._input_count = None
        self._max_concurrent = c.num_threads

        # Iterative mode will emit dataframes one at a time. Otherwise a list of dataframes is emitted. Iterative mode
        # is good for interleaving source stages.
        self._iterative = iterative
        self._repeat_count = repeat

        self._parser_kwargs = parser_kwargs

    @property
    def name(self) -> str:
        """Return the name of the stage"""
        return "from-file"

    @property
    def input_count(self) -> int:
        """Return None for no max intput count"""
        return self._input_count

    def supports_cpp_node(self) -> bool:
        """Indicates whether or not this stage supports a C++ node"""
        return True

    def _build_source(self, builder: mrc.Builder) -> StreamPair:

        if self._build_cpp_node():
            import morpheus._lib.stages as _stages
            out_stream = _stages.FileSourceStage(builder, self.unique_name, self._filename, self._repeat_count)
        else:
            out_stream = builder.make_source(self.unique_name, self._generate_frames())

        out_type = MessageMeta

        return out_stream, out_type

    def _generate_frames(self) -> typing.Iterable[MessageMeta]:

        df = read_file_to_df(
            self._filename,
            self._file_type,
            filter_nulls=self._filter_null,
            df_type="cudf",
            parser_kwargs=self._parser_kwargs,
        )

        for i in range(self._repeat_count):

            x = MessageMeta(df)

            # If we are looping, copy the object. Do this before we push the object in case it changes
            if (i + 1 < self._repeat_count):
                df = df.copy()

                # Shift the index to allow for unique indices without reading more data
                df.index += len(df)

            yield x
