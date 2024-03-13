# syntax=docker/dockerfile:1.3

# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG MORPHEUS_CONTAINER=nvcr.io/nvidia/morpheus/morpheus
ARG MORPHEUS_CONTAINER_VERSION=v24.03.00-dev

FROM ${MORPHEUS_CONTAINER}:${MORPHEUS_CONTAINER_VERSION} as base

WORKDIR /workspace

# # Use a separate build directory to avoid clashes on the host
ENV BUILD_DIR=build-docker

# Make it easier to run commands in the container
ENV SHELL=/bin/bash

# # Copy over just the environment file to install the dependencies without busting the cache from source file changes
# COPY ./examples/cyber_dev_day/requirements.yaml ./examples/cyber_dev_day/

# # Install the example's dependencies
# RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
#     source activate morpheus &&\
#     mamba env update -n morpheus -f ./examples/cyber_dev_day/requirements.yaml

# Copy everything over to the container to build
COPY . ./

# Build morpheus
RUN --mount=type=cache,id=workspace_cache,target=/workspace/.cache,sharing=locked \
    source activate morpheus &&\
    # Force a full reconfigure (build artifacts will still be found)
    rm -rf ${BUILD_DIR}/CMakeCache.txt ${BUILD_DIR}/CMakeFiles &&\
    INSTALL_PREFIX=${CONDA_PREFIX} \
    CMAKE_CONFIGURE_EXTRA_ARGS="-DMORPHEUS_PYTHON_BUILD_STUBS=OFF -DMORPHEUS_PYTHON_PERFORM_INSTALL=ON \
                                -DMORPHEUS_CUDA_ARCHITECTURES=75;80" ./scripts/compile.sh

# Set the workdir to the dev day example
WORKDIR /workspace/examples/cyber_dev_day/

# # If any changes have been made from the base image, recopy the sources
# COPY ./examples/cyber_dev_day /workspace/examples/cyber_dev_day/

# ===== Setup for running unattended =====
FROM condaforge/mambaforge:latest as runtime

ARG PYTHON_VER=3.10

# Create a base environment
RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    # Update the base environment
    /opt/conda/bin/mamba install -y -n base -c conda-forge "git >=2.35.3" "python=${PYTHON_VER}" "tini=0.19" &&\
    # Create the environment and install as little dependencies as possible
    CONDA_ALWAYS_YES=true /opt/conda/bin/mamba create -n morpheus -c conda-forge  \
      python=${PYTHON_VER} && \
    # Activate the environment automatically from .bashrc
    conda init bash && \
    conda clean -afy &&\
    echo "conda activate morpheus" >> ~/.bashrc

# Set the permenant conda channes to use for morpheus
RUN . /opt/conda/etc/profile.d/conda.sh &&\
    conda activate morpheus &&\
    conda config --env --add channels conda-forge &&\
    conda config --env --add channels nvidia &&\
    conda config --env --add channels rapidsai

# Set the entrypoint to use the entrypoint.sh script which sets the conda env
COPY ${MORPHEUS_ROOT_HOST}/docker/entrypoint.sh ./docker/
ENTRYPOINT [ "/opt/conda/bin/tini", "--", "/workspace/docker/entrypoint.sh" ]

# Reset the shell back to normal
SHELL ["/bin/bash", "-c"]

# Copy over just the environment file to install the dependencies without busting the cache from source file changes
COPY ./examples/cyber_dev_day/requirements.yaml ./examples/cyber_dev_day/

# Install the required dependencies (this does not include Morpheus' deps)
RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    source activate morpheus &&\
    mamba env update -n morpheus -f ./examples/cyber_dev_day/requirements.yaml

# Mount the build folder and then run just the install command follow by pip
# RUN --mount=type=cache,id=workspace_cache,target=/workspace/.cache,sharing=locked \
#     --mount=type=bind,from=base,source=/workspace/build-docker,target=/workspace/build-docker \
#     source activate morpheus &&\
#     cd /workspace/${BUILD_DIR:-build-docker} &&\
#     cmake --install . --prefix ${CONDA_PREFIX} &&\
#     pip install .

# Finally, copy all source files over
COPY . ./

COPY --from=base /workspace/build-docker /workspace/build-docker

# ===== Setup for running Jupyter =====
FROM runtime as jupyter

# Install the jupyter specific requirements
RUN source activate morpheus &&\
    mamba install -y -c conda-forge \
        ipywidgets \
        jupyter_contrib_nbextensions \
        # notebook v7 is incompatible with jupyter_contrib_nbextensions
        notebook=6 &&\
    jupyter contrib nbextension install --user &&\
    pip install jupyterlab_nvdashboard==0.9

# Launch jupyter
CMD ["jupyter-lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
