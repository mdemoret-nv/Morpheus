# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

import time

start_time = time.time()

print(f"morpheus._lib Starting at {start_time}")

from . import common

print(f"morpheus._lib.common took: t {time.time() - start_time}")

from . import messages

print(f"morpheus._lib.messages took: t {time.time() - start_time}")

from . import modules

print(f"morpheus._lib.modules took: t {time.time() - start_time}")

from . import stages

print(f"morpheus._lib.stages took: t {time.time() - start_time}")
