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

import cupy as cp
import numpy as np

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.general_stages import FilterDetectionsStage
from morpheus.pipeline.input.from_file import FileSourceStage
from morpheus.pipeline.messages import MultiMessage
from morpheus.pipeline.messages import MultiResponseProbsMessage
from morpheus.pipeline.messages import ResponseMemoryProbs
from morpheus.pipeline.output.serialize import SerializeStage
from morpheus.pipeline.output.to_file import WriteToFileStage
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.preprocessing import DeserializeStage
from tests import BaseMorpheusTest


class ConvMsg(SinglePortStage):
    def __init__(self, c: Config):
        super().__init__(c)

    @property
    def name(self):
        return "test"

    def accepted_types(self):
        return (MultiMessage, )

    def _conv_message(self, m):
        df = m.meta.df
        probs = cp.asanyarray(df.astype(cp.float32).values)
        memory = ResponseMemoryProbs(count=len(probs), probs=probs)
        return MultiResponseProbsMessage(m.meta, 0, len(probs), memory, 0, len(probs))

    def _build_single(self, seg, input_stream):
        stream = seg.make_node(self.unique_name, self._conv_message)
        seg.make_edge(input_stream[0], stream)

        return stream, MultiResponseProbsMessage


class TestFilterDetectionsStagePipe(BaseMorpheusTest):
    def _test_filter_pipe(self, use_cpp):
        config = Config.get()
        config.use_cpp = use_cpp

        input_file = os.path.join(self._expeced_data_dir, "filter_probs.csv")

        temp_dir = self._mk_tmp_dir()
        out_file = os.path.join(temp_dir, 'results.csv')


        pipe = LinearPipeline(config)
        pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
        pipe.add_stage(DeserializeStage(config))
        pipe.add_stage(ConvMsg(config))
        pipe.add_stage(FilterDetectionsStage(config, threshold=0.5))
        pipe.add_stage(SerializeStage(config, output_type="csv"))
        pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
        pipe.run()

        self.assertTrue(os.path.exists(out_file))

        input_data = np.loadtxt(input_file, delimiter=",", skiprows=1)
        output_data = np.loadtxt(out_file, delimiter=",")

        # The output data will contain an additional id column that we will need to slice off
        # also somehow 0.7 ends up being 0.7000000000000001
        output_data = np.around(output_data[:, 1:], 2)

        expected = np.concatenate((input_data[0:9, :], input_data[10:14, :], input_data[15:, :]))
        self.assertEqual(output_data.tolist(), expected.tolist())

    @unittest.skipIf(os.environ.get("MORPHEUS_RUN_SLOW_TESTS") is None, "MORPHEUS_RUN_SLOW_TESTS is not defined")
    def test_filter_pipe_no_cpp(self):
        self._test_filter_pipe(False)

    @unittest.skipIf(os.environ.get("MORPHEUS_RUN_SLOW_TESTS") is None, "MORPHEUS_RUN_SLOW_TESTS is not defined")
    def test_filter_pipe_cpp(self):
        self._test_filter_pipe(True)

if __name__ == '__main__':
    unittest.main()
