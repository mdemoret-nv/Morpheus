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

# See the docstring in versioneer.py for instructions. Note that you must
# re-run 'versioneer.py setup' after changing this section, and commit the
# resulting files.

from setuptools import find_namespace_packages  # noqa: E402
from setuptools import setup  # noqa: E402

setup(
    name="morpheus_ex_simple_cpp_stage",
    version="0.1",
    description="Morpheus Example 3 - Simple C++ Stage",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: System :: Networking :: Monitoring",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    author="NVIDIA Corporation",
    include_package_data=True,
    packages=find_namespace_packages(include=["morpheus_ex.*"], exclude=['tests']),
    install_requires=[
        # Only list the packages which cannot be installed via conda here. Should mach the requirements in
        # docker/conda/environments/requirements.txt
    ],
    license="Apache",
    python_requires='>=3.10, <4',
    entry_points='''
        [console_scripts]
        morpheus_ex3=morpheus_ex.simple_cpp_stage.pipeline:cli
        ''',
)
