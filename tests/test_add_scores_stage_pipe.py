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
import pandas as pd

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.general_stages import AddScoresStage
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


class WriteMetaToFileStage(SinglePortStage):
    def __init__(self, c, filename):
        super().__init__(c)

        self._output_file = filename
        self._columns = c.class_labels

    @property
    def name(self):
        return "meta-to-file"

    def accepted_types(self):
        return (MultiMessage, )

    def _write_meta_to_file(self, x):
        df = x.get_meta()
        with open(self._output_file, "a") as f:
            idx = df.columns.intersection(self._columns)
            df[idx].to_csv(f)

        return x

    def _build_single(self, seg, input_stream):
        stream = input_stream[0]
        to_file = seg.make_node(self.unique_name, self._write_meta_to_file)
        seg.make_edge(stream, to_file)
        stream = to_file

        return input_stream

class TestAddScoresStagePipe(BaseMorpheusTest):
    def _test_pipe(self, use_cpp):
        config = Config.get()
        config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']
        config.use_cpp = use_cpp

        input_file = os.path.join(self._expeced_data_dir, "filter_probs.csv")

        temp_dir = self._mk_tmp_dir()
        out_file = os.path.join(temp_dir, 'results.csv')


        pipe = LinearPipeline(config)
        pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
        pipe.add_stage(DeserializeStage(config))
        pipe.add_stage(ConvMsg(config))
        pipe.add_stage(AddScoresStage(config))
        pipe.add_stage(WriteMetaToFileStage(config, filename=out_file))
        pipe.run()

        self.assertTrue(os.path.exists(out_file))

        expected = np.loadtxt(input_file, delimiter=",", skiprows=1)

        # The output data will contain an additional id column that we will need to slice off
        # also somehow 0.7 ends up being 0.7000000000000001
        output_data = pd.read_csv(out_file)
        idx = output_data.columns.intersection(config.class_labels)
        self.assertEqual(idx.to_list(), config.class_labels)

        output_np = np.around(output_data[idx].to_numpy(), 2)

        self.assertEqual(output_np.tolist(), expected.tolist())

    @unittest.skipIf(os.environ.get("MORPHEUS_RUN_SLOW_TESTS") is None, "MORPHEUS_RUN_SLOW_TESTS is not defined")
    def test_pipe_no_cpp(self):
        self._test_pipe(False)

    @unittest.skipIf(os.environ.get("MORPHEUS_RUN_SLOW_TESTS") is None, "MORPHEUS_RUN_SLOW_TESTS is not defined")
    def test_pipe_cpp(self):
        self._test_pipe(True)

if __name__ == '__main__':
    unittest.main()
