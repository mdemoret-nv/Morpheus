# To enable
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

export MORPHEUS_HOME=$(git rev-parse --show-toplevel)

# Change to morpheus home
cd ${MORPHEUS_HOME}

export DOCKER_IMAGE_TAG="sid-viz"

# Build the dev container
./docker/build_container_dev.sh

# Change to the example directory
cd ${MORPHEUS_HOME}/examples/sid_visualization

# Launch the containers
docker-compose up -d

# Exec into the morpheus container
docker-compose exec morpheus bash

# Inside the container, compile morpheus
BUILD_DIR=build-docker ./scripts/compile.sh

# Install morpheus with an extra dependency
pip install -e . && pip install websockets

# Now run the demo
python examples/sid_visualization/run.py \
  --use_cpp=False \
  --num_threads=1 \
  --triton_server_url=triton:8001 \
  --input_file=./examples/data/sid_visualization/group1-benign-2nodes-v2.jsonlines \
  --input_file=./examples/data/sid_visualization/group2-benign-50nodes.jsonlines \
  --input_file=./examples/data/sid_visualization/group3-si-50nodes.jsonlines \
  --input_file=./examples/data/sid_visualization/group4-benign-49nodes.jsonlines
