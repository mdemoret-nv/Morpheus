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

FROM nvidia/cuda:11.5.1-devel-ubuntu20.04 AS base

RUN apt-get update && \
    apt-get install --no-install-recommends --yes \
        wget git-lfs gdb iputils-ping

ENV CONDA_DIR=/opt/conda

RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh" && \
    bash Mambaforge-$(uname)-$(uname -m).sh -b -p ${CONDA_DIR} && \
    /opt/conda/bin/conda init --system

ENV PATH="${CONDA_DIR}/bin:${PATH}"

ENV MAMBA_NO_BANNER=1

# Update the max user watches according to
# https://code.visualstudio.com/docs/setup/linux#_visual-studio-code-is-unable-to-watch-for-file-changes-in-this-large-workspace-error-enospc
RUN echo "fs.inotify.max_user_watches=524288" >> /etc/sysctl.conf
