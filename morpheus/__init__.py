# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
"""Root module for the Morpheus library."""

import logging
import os

# ########################### CVE-2023-47248 Mitigation ############################
# Import pyarrow_hotfix as early as possible to ensure that the pyarrow hotfix is applied before any code can use it
# Can be removed after upgrading to pyarrow 14.0.1 or later (which is dictated by cudf)
import pyarrow_hotfix

# ##################################################################################

from ctypes import byref
from ctypes import c_int

from numba import cuda

dv = c_int(0)
cuda.cudadrv.driver.driver.cuDriverGetVersion(byref(dv))
drv_major = dv.value // 1000
drv_minor = (dv.value - (drv_major * 1000)) // 10
run_major, run_minor = cuda.runtime.get_version()
print(f'{drv_major} {drv_minor} {run_major} {run_minor}')

import os

os.environ["PTXCOMPILER_CHECK_NUMBA_CODEGEN_PATCH_NEEDED"] = "0"
os.environ["PTXCOMPILER_KNOWN_DRIVER_VERSION"] = f"{drv_major}.{drv_minor}"
os.environ["PTXCOMPILER_KNOWN_RUNTIME_VERSION"] = f"{run_major}.{run_minor}"

import cudf

# Create a default null logger to prevent log messages from being propagated to users of this library unless otherwise
# configured. Use the `utils.logging` module to configure Morpheus logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

from . import _version  # pylint: disable=wrong-import-position

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
__version__ = _version.get_versions()['version']
