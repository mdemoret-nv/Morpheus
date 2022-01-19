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
import shutil
import tempfile
import unittest


class BaseMorpheusTest(unittest.TestCase):
    def _mk_tmp_dir(self):
        """
        Creates a temporary directory for use by tests, directory is deleted after the test is run unless the
        MORPHEUS_NO_TEST_CLEANUP environment variable is defined.
        """
        tmp_dir = tempfile.mkdtemp(prefix='morpheus_test_')
        if os.environ.get('MORPHEUS_NO_TEST_CLEANUP') is None:
            self.addCleanup(shutil.rmtree, tmp_dir)

        return tmp_dir

    def _save_env_vars(self):
        """
        Save the current environment variables and restore them at the end of the test, removing any newly added values
        """
        orig_vars = os.environ.copy()
        self.addCleanup(self._restore_env_vars, orig_vars)

    def _restore_env_vars(self, orig_vars):
        # Iterating over a copy of the keys as we will potentially be deleting keys in the loop
        for key in list(os.environ.keys()):
            orig_val = orig_vars.get(key)
            if orig_val is not None:
                os.environ[key] = orig_val
            else:
                del (os.environ[key])
