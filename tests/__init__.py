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

import asyncio
import collections
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import unittest

import requests

from morpheus.config import Config

TESTS_DIR = os.path.dirname(__file__)
WORKSPACE_DIR = os.path.dirname(TESTS_DIR)


class BaseMorpheusTest(unittest.TestCase):
    Results = collections.namedtuple('Results', ['total_rows', 'diff_rows', 'error_pct'])

    def setUp(self) -> None:
        super().setUp()

        # reset the config singleton
        Config._Config__default = None
        Config._Config__instance = None

        # Reset the asyncio event loop
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())

        self._morpheus_root = os.environ.get('MORPHEUS_ROOT', WORKSPACE_DIR)
        self._data_dir = os.path.join(self._morpheus_root, 'data')
        self._models_dir = os.path.join(self._morpheus_root, 'models')
        self._datasets_dir = os.path.join(self._models_dir, 'datasets')
        self._training_data_dir = os.path.join(self._datasets_dir, 'training-data')
        self._validation_data_dir = os.path.join(self._datasets_dir, 'validation-data')

        self._expeced_data_dir = os.path.join(TESTS_DIR, 'expected_data')
        self._mock_triton_servers_dir = os.path.join(TESTS_DIR, 'mock_triton_servers')


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

    def _calc_error_val(self, results_file):
        """
        Based on the calc_error_val function in val-utils.sh
        """
        with open(results_file) as fh:
            results = json.load(fh)

        total_rows = results['total_rows']
        diff_rows = results['diff_rows']
        return self.Results(total_rows=total_rows, diff_rows=diff_rows, error_pct=(diff_rows / total_rows) * 100)

    def _wait_for_camouflage(self, host="localhost", port=8000, timeout=5):
        ready = False
        elapsed_time = 0.0
        sleep_time = 0.1
        url = "http://{}:{}/v2/health/ready".format(host, port)
        while not ready and elapsed_time < timeout:
            try:
                r = requests.get(url)
                ready = (r.status_code == 200)
            except Exception as e:
                pass

            if not ready:
                time.sleep(sleep_time)
                elapsed_time += sleep_time

        return ready

    def _kill_proc(self, proc, timeout=1):
        logging.debug("killing pid {}".format(proc.pid))

        elapsed_time = 0.0
        sleep_time = 0.1
        stopped = False

        # It takes a little while to shutdown
        while not stopped and elapsed_time < timeout:
            proc.kill()
            stopped = (proc.poll() is not None)
            if not stopped:
                time.sleep(sleep_time)
                elapsed_time += sleep_time

    def _launch_camouflage_triton(self, root_dir, config="config.yml", timeout=5):
        """
        Launches a mock triton server using camouflage (https://testinggospels.github.io/camouflage/) with a package
        rooted at `root_dir` and configured with `config`.

        This function will wait for up to `timeout` seconds for camoflauge to startup

        This function is a no-op if the `MORPHEUS_NO_LAUNCH_CAMOUFLAGE` environment variable is defined, which can
        be useful during test development to run camouflage by hand.
        """
        if os.environ.get('MORPHEUS_NO_LAUNCH_CAMOUFLAGE') is None:
            popen = subprocess.Popen(["camouflage", "--config", config],
                                     cwd=root_dir,
                                     stderr=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL)

            logging.debug("Launching camouflage in {} with pid: {}".format(root_dir, popen.pid))
            self.addCleanup(self._kill_proc, popen)

            if timeout > 0:
                if not self._wait_for_camouflage(timeout=timeout):
                    raise RuntimeError("Failed to launch camouflage server")
