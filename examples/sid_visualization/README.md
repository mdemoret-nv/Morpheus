# SID Visualization Example

## Prerequisites

To run the demo you will need the following:
- Docker
- NodeJS
  - See the installation guide [here](https://nodejs.org/en/download/)
- `yarn`
  - Once NodeJS is installed with NPM, run `npm install --global yarn` to install yarn
- `docker-compose`

## Setup

To run this demo, ensure all submodules are checked out

```bash
git submodule update --init --recursive
```

### Build Morpheus Dev Container

Before launching the demo, we need the dev container for Morpheus to be created:

```bash
export DOCKER_IMAGE_TAG="sid-viz"

# Build the dev container
./docker/build_container_dev.sh
```

### Launch User Interface

We will use docker-compose to build and run the entire demo. To launch everything, run the following from the repo root:

```bash
# Save the Morpheus repo directory
export MORPHEUS_HOME=$(git rev-parse --show-toplevel)

# Change to the example directory
cd ${MORPHEUS_HOME}/examples/sid_visualization

# Launch the containers
DOCKER_BUILDKIT=1 docker-compose up --build -d
```

### Build Morpheus

Once docker-compose has been launched, exec into the container to build and run Morpheus:

```bash
# Exec into the morpheus container
docker-compose exec morpheus bash

# Inside the container, compile morpheus
BUILD_DIR=build-docker ./scripts/compile.sh

# Install morpheus with an extra dependency
pip install -e . && pip install websockets

# Verify Morpheus is installed
morpheus --version

# Ensure the data has been downloaded
./scripts/fetch_data.py fetch examples
```

**Note: ** Keep the shell running the Morpheus Dev container running. It will be used later to start Morpheus.

## Running the Demo

### Running Morpheus

After the GUI has been launched, Morpheus now needs to be started. In the same shell used to build Morpheus (the one running the Morpheus Dev container), run the following:

```bash
python examples/sid_visualization/run.py \
  --debug --use_cpp=False --num_threads=1 \
  --triton_server_url=triton:8001 \
  --input_file=./examples/data/sid_visualization/group1-benign-2nodes-v2.jsonlines \
  --input_file=./examples/data/sid_visualization/group2-benign-50nodes.jsonlines \
  --input_file=./examples/data/sid_visualization/group3-si-50nodes.jsonlines \
  --input_file=./examples/data/sid_visualization/group4-benign-49nodes.jsonlines
```

This launch will use all of the available datasets. Each dataset will show up as one batch in the visualization. Here is a description of each dataset:

- `examples/data/sid_visualization/group1-benign-2nodes.jsonlines`
  - Small scale with 2 nodes, no SID
- `examples/data/sid_visualization/group2-benign-50nodes.jsonlines`
  - Scale up to 50 nodes, no SID
- `examples/data/sid_visualization/group3-si-50nodes.jsonlines`
  - 50 nodes, with SID from a single node
- `examples/data/sid_visualization/group4-benign-49nodes.jsonlines`
  - Isolate bad node leaving 49 nodes, no SID

Changing the dataset does not require relaunching the GUI. Simply re-run Morpheus with the new dataset and the GUI will be updated.

It's also possible to launch the demo using the Morpheus CLI using the following:

```bash
DEMO_DATASET="examples/data/sid_visualization/group1-benign-2nodes.jsonlines"

morpheus --log_level=DEBUG \
   run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=32 --edge_buffer_size=4 --use_cpp=False \
      pipeline-nlp --model_seq_length=256 \
         from-file --filename=${DEMO_DATASET} \
         deserialize \
         preprocess --vocab_hash_file=morpheus/data/bert-base-uncased-hash.txt --truncation=True --do_lower_case=True --add_special_tokens=False \
         inf-triton --model_name=sid-minibert-onnx --server_url=localhost:8001 --force_convert_inputs=True \
         monitor --description Inference\ Rate --unit=inf \
         add-class \
         gen-viz
```

Note, this launch method is more useful for showing performance than showing capability,
