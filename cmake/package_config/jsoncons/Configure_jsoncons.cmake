#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
#=============================================================================

include_guard(GLOBAL)

function(morpheus_utils_configure_jsoncons)
  list(APPEND CMAKE_MESSAGE_CONTEXT "jsoncons")

  morpheus_utils_assert_cpm_initialized()
  set(JSONCONS_VERSION "0.171.0" CACHE STRING "Version of jsoncons to use")

  # Try to find hwloc and download from source if not found
  rapids_cpm_find(jsoncons ${JSONCONS_VERSION}
    GLOBAL_TARGETS
      jsoncons jsoncons::jsoncons
    BUILD_EXPORT_SET
      ${PROJECT_NAME}-core-exports
    INSTALL_EXPORT_SET
      ${PROJECT_NAME}-core-exports
    CPM_ARGS
      GIT_REPOSITORY          https://github.com/danielaparker/jsoncons.git
      GIT_TAG                 "v${JSONCONS_VERSION}"
      FIND_PACKAGE_ARGUMENTS  "EXACT"
      OPTIONS                 JSONCONS_BUILD_TESTS OFF
                              CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}
  )

  list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endfunction()
