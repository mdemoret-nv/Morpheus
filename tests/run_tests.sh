#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Work-around for issue #71 we can't run more than one pipeline in a single
# process. Once that is resolved we can run the tests through something like
# pytest
TEST_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

pytest --cov=morpheus \
    ${TEST_DIR}/test_add_classifications_stage.py \
    ${TEST_DIR}/test_cli.py \
    ${TEST_DIR}/test_config.py \
    ${TEST_DIR}/test_filter_detections_stage.py \
    ${TEST_DIR}/test_monitor_stage.py \
    ${TEST_DIR}/test_package.py

if [ -n "${MORPHEUS_RUN_SLOW_TESTS}" ]; then
    pytest ${TEST_DIR}/test_abp.py::TestABP::test_abp_no_cpp
    pytest ${TEST_DIR}/test_abp.py::TestABP::test_abp_cpp
    pytest ${TEST_DIR}/test_hammah.py::TestHammah::test_hammah_roleg
    pytest ${TEST_DIR}/test_hammah.py::TestHammah::test_hammah_user123
    pytest ${TEST_DIR}/test_phishing.py::TestPhishing::test_email_no_cpp
    pytest ${TEST_DIR}/test_phishing.py::TestPhishing::test_email_cpp
    pytest ${TEST_DIR}/test_sid.py::TestSid::test_minibert_no_cpp
    pytest ${TEST_DIR}/test_sid.py::TestSid::test_minibert_cpp
fi
