#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest
from unittest import mock

import cupy as cp
import numpy as np

import cudf as pd

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.file_types import FileTypes
from morpheus.pipeline.general_stages import FilterDetectionsStage
from morpheus.pipeline.input.from_file import FileSourceStage
from morpheus.pipeline.input.utils import read_file_to_df
from morpheus.pipeline.messages import MessageMeta
from morpheus.pipeline.messages import MultiResponseProbsMessage
from morpheus.pipeline.messages import ResponseMemoryProbs
from morpheus.pipeline.output.serialize import SerializeStage
from morpheus.pipeline.output.to_file import WriteToFileStage
from morpheus.pipeline.pipeline import SingleOutputSource
from morpheus.pipeline.preprocessing import DeserializeStage
from tests import BaseMorpheusTest


def _build_mpm_from_file(filename):
    probs = cp.array(np.loadtxt(filename, delimiter=",", skiprows=1))

    mm = MessageMeta(read_file_to_df(filename, FileTypes.Csv))
    rmp = ResponseMemoryProbs(1, probs)
    return MultiResponseProbsMessage(mm, 0, 1, rmp, 0, len(probs))

class CupySrc(SingleOutputSource):
    def __init__(self, c: Config, filename: str):
        super().__init__(c)
        self._filename = filename

    @property
    def name(self) -> str:
        return "test"

    def _read_file(self):
        yield _build_mpm_from_file(self._filename)

    def _build_source(self, seg):
        out_stream = seg.make_source(self.unique_name, self._read_file())
        out_type = MultiResponseProbsMessage

        return out_stream, out_type


class TestFilterDetectionsStage(BaseMorpheusTest):
    def test_constructor(self):
        config = Config.get()

        fds = FilterDetectionsStage(config)
        self.assertEqual(fds.name, "filter")

        # Just ensure that we get a valid non-empty tuple
        accepted_types = fds.accepted_types()
        self.assertIsInstance(accepted_types, tuple)
        self.assertGreater(len(accepted_types), 0)

        fds = FilterDetectionsStage(config, threshold=0.2)
        self.assertEqual(fds._threshold, 0.2)


    def test_filter_no_cpp(self):
        config = Config.get()
        config.use_cpp = False
        fds = FilterDetectionsStage(config, threshold=0.5)

        mock_message = mock.MagicMock()
        mock_message.mess_offset = 8
        mock_message.probs = cp.array([[0.1, 0.5, 0.3], [0.2, 0.3, 0.4]])

        # All values are below the threshold
        self.assertEqual(fds.filter(mock_message), [])

        # Only one row has a value above the threshold
        mock_message.probs = cp.array([
            [0.2, 0.4, 0.3],
            [0.1, 0.5, 0.8],
            [0.2, 0.4, 0.3],
        ])

        output_list = fds.filter(mock_message)
        self.assertEqual(len(output_list), 1)
        self.assertEqual(output_list[0].offset, 1)
        self.assertEqual(output_list[0].mess_offset, 9)
        self.assertEqual(output_list[0].mess_count, 1)

        # Two adjacent rows have a value above the threashold
        mock_message.probs = cp.array([
            [0.2, 0.4, 0.3],
            [0.1, 0.2, 0.3],
            [0.1, 0.5, 0.8],
            [0.1, 0.9, 0.2],
            [0.2, 0.4, 0.3],
        ])

        output_list = fds.filter(mock_message)
        self.assertEqual(len(output_list), 1)
        self.assertEqual(output_list[0].offset, 2)
        self.assertEqual(output_list[0].mess_offset, 10)
        self.assertEqual(output_list[0].mess_count, 2)

        # Two non-adjacent rows have a value above the threashold
        mock_message.probs = cp.array([
            [0.2, 0.4, 0.3],
            [0.1, 0.2, 0.3],
            [0.1, 0.5, 0.8],
            [0.4, 0.3, 0.2],
            [0.1, 0.9, 0.2],
            [0.2, 0.4, 0.3],
        ])
        output_list = fds.filter(mock_message)
        self.assertEqual(len(output_list), 2)
        self.assertEqual(output_list[0].offset, 2)
        self.assertEqual(output_list[0].mess_offset, 10)
        self.assertEqual(output_list[0].mess_count, 1)

        self.assertEqual(output_list[1].offset, 4)
        self.assertEqual(output_list[1].mess_offset, 12)
        self.assertEqual(output_list[1].mess_count, 1)

    def test_filter_no_mock_no_cpp(self):
        config = Config.get()
        config.use_cpp = False

        # FilterDetectionStage will return any row if at least one value is above the threshold
        # The input data contains 20 rows, all rows have one value above the threshold except rows 9 & 14
        # Expecting 3 slices: 0:9, 10:14, 15:21
        mpm = _build_mpm_from_file(os.path.join(self._expeced_data_dir, "filter_probs.csv"))

        fds = FilterDetectionsStage(config, threshold=0.5)
        output_list = fds.filter(mpm)

        self.assertEqual(len(output_list), 3)
        (out1, out2, out3) = output_list

        self.assertEqual(out1.mess_offset, 0)
        self.assertEqual(out1.mess_count, 9)

        self.assertEqual(out2.mess_offset, 10)
        self.assertEqual(out2.mess_count, 4)

        self.assertEqual(out3.mess_offset, 15)
        self.assertEqual(out3.mess_count, 5)

    def test_build_single_no_cpp(self):
        mock_stream = mock.MagicMock()
        mock_segment = mock.MagicMock()
        mock_segment.make_node.return_value = mock_stream
        mock_input = mock.MagicMock()

        config = Config.get()
        config.use_cpp = False
        fds = FilterDetectionsStage(config)
        fds._build_single(mock_segment, mock_input)

        mock_segment.make_node_full.assert_called_once()
        mock_segment.make_edge.assert_called_once()

    def _test_filter_pipe(self, use_cpp):
        config = Config.get()
        config.use_cpp = use_cpp

        input_file = os.path.join(self._expeced_data_dir, "filter_probs.csv")

        temp_dir = self._mk_tmp_dir()
        out_file = os.path.join(temp_dir, 'results.csv')


        pipe = LinearPipeline(config)
        pipe.set_source(CupySrc(config, filename=input_file))
        pipe.add_stage(FilterDetectionsStage(config, threshold=0.5))
        pipe.add_stage(SerializeStage(config, output_type="csv", include=['^v\d$']))
        pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
        pipe.run()

        self.assertTrue(os.path.exists(out_file))

        input_data = np.loadtxt(input_file, delimiter=",", skiprows=1)
        output_data = np.loadtxt(out_file, delimiter=",")

        expected = input_data[0:9, :].tolist() + input_data[10:14, :].tolist() + input_data[15:, :].tolist()

        # The output data will contain an additional id column that we will need to slice off
        self.assertEqual(expected, output_data[:, 1:].tolist())

    def test_filter_pipe_no_cpp(self):
        self._test_filter_pipe(False)

    def test_filter_pipe_cpp(self):
        self._test_filter_pipe(True)

if __name__ == '__main__':
    unittest.main()