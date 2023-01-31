# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import glob
import os

import GPUtil
from test_bench_e2e_pipelines import E2E_TEST_CONFIGS


def pytest_benchmark_update_json(config, benchmarks, output_json):

    gpus = GPUtil.getGPUs()

    for i, gpu in enumerate(gpus):
        # output_json["machine_info"]["gpu_" + str(i)] = gpu.name
        output_json["machine_info"]["gpu_" + str(i)] = {}
        output_json["machine_info"]["gpu_" + str(i)]["id"] = gpu.id
        output_json["machine_info"]["gpu_" + str(i)]["name"] = gpu.name
        output_json["machine_info"]["gpu_" + str(i)]["load"] = f"{gpu.load*100}%"
        output_json["machine_info"]["gpu_" + str(i)]["free_memory"] = f"{gpu.memoryFree}MB"
        output_json["machine_info"]["gpu_" + str(i)]["used_memory"] = f"{gpu.memoryUsed}MB"
        output_json["machine_info"]["gpu_" + str(i)]["temperature"] = f"{gpu.temperature} C"
        output_json["machine_info"]["gpu_" + str(i)]["uuid"] = gpu.uuid

    line_count = 0
    byte_count = 0
    for bench in output_json['benchmarks']:
        if "file_path" in E2E_TEST_CONFIGS[bench["name"]]:
            source_file = E2E_TEST_CONFIGS[bench["name"]]["file_path"]
            line_count = len(open(source_file).readlines())
            byte_count = os.path.getsize(source_file)

        elif "glob_path" in E2E_TEST_CONFIGS[bench["name"]]:
            for fn in glob.glob(E2E_TEST_CONFIGS[bench["name"]]["glob_path"]):
                line_count += len(open(fn).readlines())
                byte_count += os.path.getsize(fn)

        repeat = E2E_TEST_CONFIGS[bench["name"]]["repeat"]

        bench["morpheus_config"] = {}
        bench["morpheus_config"]["num_threads"] = E2E_TEST_CONFIGS[bench["name"]]["num_threads"]
        bench["morpheus_config"]["pipeline_batch_size"] = E2E_TEST_CONFIGS[bench["name"]]["pipeline_batch_size"]
        bench["morpheus_config"]["model_max_batch_size"] = E2E_TEST_CONFIGS[bench["name"]]["model_max_batch_size"]
        bench["morpheus_config"]["feature_length"] = E2E_TEST_CONFIGS[bench["name"]]["feature_length"]
        bench["morpheus_config"]["edge_buffer_size"] = E2E_TEST_CONFIGS[bench["name"]]["edge_buffer_size"]

        bench['stats']["input_lines"] = line_count * repeat
        bench['stats']['min_throughput_lines'] = (line_count * repeat) / bench['stats']['max']
        bench['stats']['max_throughput_lines'] = (line_count * repeat) / bench['stats']['min']
        bench['stats']['mean_throughput_lines'] = (line_count * repeat) / bench['stats']['mean']
        bench['stats']['median_throughput_lines'] = (line_count * repeat) / bench['stats']['median']
        bench['stats']["input_bytes"] = byte_count * repeat
        bench['stats']['min_throughput_bytes'] = (byte_count * repeat) / bench['stats']['max']
        bench['stats']['max_throughput_bytes'] = (byte_count * repeat) / bench['stats']['min']
        bench['stats']['mean_throughput_bytes'] = (byte_count * repeat) / bench['stats']['mean']
        bench['stats']['median_throughput_bytes'] = (byte_count * repeat) / bench['stats']['median']
