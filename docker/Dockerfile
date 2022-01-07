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

# syntax=docker/dockerfile:1

# Args used in FROM commands must come first
ARG FROM_IMAGE="gpuci/miniconda-cuda"
ARG CUDA_VER=11.4
ARG LINUX_DISTRO=ubuntu
ARG LINUX_VER=20.04

# Stage 0 - Vcpkg build
# Builds vcpkg so dependencies can be easily installed on the command line
FROM ${LINUX_DISTRO}:${LINUX_VER} as vcpkg_build

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Denver

RUN apt update && apt -y install build-essential cmake git curl unzip tar zip

RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg

WORKDIR /opt/vcpkg

# Build vcpkg and delete any data in packages/buildtrees/downloads
RUN ./bootstrap-vcpkg.sh -disableMetrics &&\
    rm -rf "/opt/vcpkg/packages" "/opt/vcpkg/buildtrees" "/opt/vcpkg/downloads"

# Stage 1 - Dependencies
FROM ${FROM_IMAGE}:${CUDA_VER}-devel-${LINUX_DISTRO}${LINUX_VER} AS base

# Required arguments
ARG IMAGE_TYPE=base
ARG RAPIDS_CHANNEL=rapidsai-nightly
ARG RAPIDS_VER=21.10
ARG PYTHON_VER=3.8

# The TensorRT Version must be the full version (i.e. 7.2.2.3) and match Triton EXACTLY
ARG TENSORRT_VERSION=8.2.1.3

# Capture argument used for FROM
ARG CUDA_VER

# Install dependencies to build vcpkg dependencies
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
      build-essential pkg-config curl unzip tar zip openssh-client bc jq \
    && rm -rf /var/lib/apt/lists/*

# Vcpkg Install
COPY --from=vcpkg_build /opt/vcpkg/ /opt/vcpkg/
ENV VCPKG_ROOT=/opt/vcpkg

# Run the install and bash integrate scripts. Also add VCPKG_BUILD_TYPE
RUN mkdir -p $HOME/.vcpkg && \
    /opt/vcpkg/vcpkg integrate install && \
    /opt/vcpkg/vcpkg integrate bash && \
    echo 'export PATH=$PATH:/opt/vcpkg' >>~/.bashrc &&\
    echo 'export VCPKG_ROOT="/opt/vcpkg/"' >>~/.bashrc

# Set the default cache to be in the work folder. Ensure it exists
ENV VCPKG_DEFAULT_BINARY_CACHE=/workspace/.cache/vcpkg

# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

# Accept all gitlab-master.nvidia.com hosts
RUN mkdir -p $HOME/.ssh && ssh-keyscan -t rsa -p 12051 gitlab-master.nvidia.com > ~/.ssh/known_hosts

# All code will be under /workspace
WORKDIR /workspace

# Install mamba to speed the solve up
RUN conda config --set ssl_verify false \
    && /opt/conda/bin/conda install -y -n base -c conda-forge mamba && conda clean -afy

# Conda dependencies install
COPY docker/conda/environments/dev_cuda${CUDA_VER}.yml ./docker/conda/environments/

RUN /opt/conda/bin/mamba env create -n morpheus --file docker/conda/environments/dev_cuda${CUDA_VER}.yml &&\
    # Install only the dependencies for cudf. Need to build cudf manually
    /opt/conda/bin/mamba install -y -n morpheus -c rapidsai -c nvidia -c conda-forge --only-deps cudf=${RAPIDS_VER} libcudf=${RAPIDS_VER} custreamz=${RAPIDS_VER} cudf_kafka=${RAPIDS_VER} libcudf_kafka=${RAPIDS_VER} &&\
    # Clean and activate
    conda clean -afy && \
    sed -i 's/conda activate base/conda activate morpheus/g' ~/.bashrc

# Enables all RUN commands to use the conda env
SHELL ["/opt/conda/bin/conda", "run", "-n", "morpheus", "/bin/bash", "-c"]

# Set the permenant conda channes to use for morpheus
RUN conda config --env --add channels conda-forge &&\
    conda config --env --add channels nvidia &&\
    conda config --env --add channels rapidsai

# Install nvidia-pyindex which is needed for tritonclient[all] but must be
# installed before setup is run. Also force reinstall of just streamz since it
# gets installed as a dependency of custreamz. The arguments `--upgrade
# --no-deps --force-reinstall` are required to force reinstalling just streamz
# without all its dependencies
RUN pip install nvidia-pyindex && \
  # Removing the tensorrt install for 0.2-EA
  # pip install nvidia-tensorrt==${TENSORRT_VERSION} && \
  pip install --upgrade --no-deps --force-reinstall git+https://github.com/mdemoret-nv/streamz.git@on_completed#egg=streamz &&\
  pip install git+https://github.com/efajardo-nv/dfencoder.git@nv-updates#egg=dfencoder &&\
  pip install dill

# Set the CUDAToolkit_ROOT variable to help CMake find CUDA
ENV CUDAToolkit_ROOT=/usr/local/cuda-${CUDA_VER}

# Set the entrypoint to use the entrypoint.sh script which sets the conda env
COPY docker/entrypoint.sh ./docker/
ENTRYPOINT [ "/opt/conda/bin/tini", "--", "docker/entrypoint.sh" ]

# Reset the shell back to normal
SHELL ["/bin/bash", "-c"]

# Manually build cuDF to get around libcudacxx issues when building as CPM package. See
# https://github.com/NVIDIA/libcudacxx/issues/209
FROM base as cudf_build

# Recapture args again
ARG RAPIDS_VER
ARG CUDF_DIR=/opt/cudf

COPY cmake/deps/patches/cudf.patch /workspace/cmake/deps/patches/

# Clone cudf, patch and build
RUN git clone -b branch-${RAPIDS_VER} --depth 1 https://github.com/rapidsai/cudf ${CUDF_DIR} &&\
    cd ${CUDF_DIR} &&\
    git apply --whitespace=fix /workspace/cmake/deps/patches/cudf.patch &&\
    source activate morpheus &&\
    ./build.sh --ptds libcudf cudf libcudf_kafka cudf_kafka custreamz &&\
    cd -

FROM cudf_build as build

# Copy the source
COPY . ./

# Build/Install neo/morpheus C++ libraries. Need buildkit and ssh auth to get private repos
RUN --mount=type=ssh source activate morpheus && ./scripts/compile.sh

# Install libraries to alternate folder and package into a single tar to preserve symlinks (Docker will remove these)
RUN source activate morpheus &&\
    DESTDIR="/workspace/.cache/install" cmake --build build -j --target install &&\
    mkdir -p /workspace/.cache/dist &&\
    cd /workspace/.cache/install/opt/conda/envs/morpheus &&\
    tar -zcvf /workspace/.cache/dist/install.tar.gz ./lib ./bin ./share &&\
    cd /workspace

# Build/Install morpheus CLI and put on PATH. Export wheels for installation
RUN source activate morpheus && \
    pip install . && \
    cd /workspace/.cache/install/workspace/build &&\
    python setup.py bdist_wheel --dist-dir /workspace/.cache/dist && \
    cd /workspace/.cache/install/workspace/build/_deps/trtlab-build/trtlab/neo/python/ && \
    python setup.py bdist_wheel --dist-dir /workspace/.cache/dist && \
    cd /workspace

# Setup container for runtime environment
FROM cudf_build as runtime

# Copy the install directory over and the pip packages
COPY --from=build ["/workspace/.cache/dist/*", "/tmp/"]

# Install the binaries by untarring
RUN source activate morpheus &&\
    cd /opt/conda/envs/morpheus &&\
    tar -zxvf /tmp/install.tar.gz &&\
    rm /tmp/install.tar.gz &&\
    cd -

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
      patchelf \
    && rm -rf /var/lib/apt/lists/*

# TMP: Hard coded the installed RPATH for UCX
RUN source activate morpheus &&\
    patchelf --set-rpath '/opt/conda/envs/morpheus/lib' /opt/conda/envs/morpheus/lib/libucm.so.0.0.0 &&\
    patchelf --set-rpath '/opt/conda/envs/morpheus/lib' /opt/conda/envs/morpheus/lib/libucs.so.0.0.0 &&\
    patchelf --set-rpath '/opt/conda/envs/morpheus/lib' /opt/conda/envs/morpheus/lib/libuct.so.0.0.0 &&\
    patchelf --set-rpath '/opt/conda/envs/morpheus/lib' /opt/conda/envs/morpheus/lib/libucp.so.0.0.0

# Install the Python packages
RUN source activate morpheus && \
    pip install /tmp/neo-* && \
    pip install /tmp/morpheus-* && \
    rm /tmp/neo-* && \
    rm /tmp/morpheus-*

# Only copy specific files/folders over that are necessary for runtime
COPY ["*.md", "LICENSE", "./"]
COPY "./data" "./data"
COPY "./docker" "./docker"
COPY "./docs" "./docs"
COPY "./examples" "./examples"
COPY "./scripts" "./scripts"

# Use morpheus by default
CMD [ "morpheus" ]
