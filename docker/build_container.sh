#!/bin/bash

DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:?"Must set \$DOCKER_IMAGE_NAME to build. Use the dev/release scripts to set these automatically"}
DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:?"Must set \DOCKER_IMAGE_TAG to build. Use the dev/release scripts to set these automatically"}
DOCKER_TARGET=${DOCKER_TARGET:-"runtime"}
DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1}
DOCKER_EXTRA_ARGS=${DOCKER_EXTRA_ARGS:-""}

# Build args
FROM_IMAGE=${FROM_IMAGE:-"gpuci/miniconda-cuda"}
CUDA_VER=${CUDA_VER:-11.4}
LINUX_DISTRO=${LINUX_DISTRO:-ubuntu}
LINUX_VER=${LINUX_VER:-20.04}
RAPIDS_VER=${RAPIDS_VER:-21.10}
PYTHON_VER=${PYTHON_VER:-3.8}
TENSORRT_VERSION=${TENSORRT_VERSION:-8.2.1.3}

DOCKER_ARGS="-t ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
DOCKER_ARGS="${DOCKER_ARGS} --target ${DOCKER_TARGET}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg FROM_IMAGE=${FROM_IMAGE}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg CUDA_VER=${CUDA_VER}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg LINUX_DISTRO=${LINUX_DISTRO}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg LINUX_VER=${LINUX_VER}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg RAPIDS_VER=${RAPIDS_VER}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg PYTHON_VER=${PYTHON_VER}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg TENSORRT_VERSION=${TENSORRT_VERSION}"
DOCKER_ARGS="${DOCKER_ARGS} --network=host"

if [[ "${DOCKER_BUILDKIT}" = "1" ]]; then
   # If using BUILDKIT, add the necessary args to pull private containers
   DOCKER_ARGS="${DOCKER_ARGS} --ssh default --add-host gitlab-master.nvidia.com:$(dig +short gitlab-master.nvidia.com | tail -1)"
fi

# Last add any extra args (duplicates override earlier ones)
DOCKER_ARGS="${DOCKER_ARGS} ${DOCKER_EXTRA_ARGS}"

# Export buildkit variable
export DOCKER_BUILDKIT

echo "Building morpheus:${DOCKER_TAG}..."
echo "   FROM_IMAGE      : ${FROM_IMAGE}"
echo "   CUDA_VER        : ${CUDA_VER}"
echo "   LINUX_DISTRO    : ${LINUX_DISTRO}"
echo "   LINUX_VER       : ${LINUX_VER}"
echo "   RAPIDS_VER      : ${RAPIDS_VER}"
echo "   PYTHON_VER      : ${PYTHON_VER}"
echo "   TENSORRT_VERSION: ${TENSORRT_VERSION}"
echo ""
echo "   COMMAND: docker build ${DOCKER_ARGS} -f docker/Dockerfile ."

docker build ${DOCKER_ARGS} -f docker/Dockerfile .
