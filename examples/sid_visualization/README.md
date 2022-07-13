# SID Visualization Example

## Prerequisites

To run the demo you will need the following:
- Docker
- NodeJS
  - See the installation guide [here](https://nodejs.org/en/download/)
- `yarn`
  - Once NodeJS is installed with NPM, run `npm install --global yarn` to install yarn
-

## Setup

To run this demo, you will need two repos:

```bash
# Clone a Morpheus Repo fork and checkout the Demo branch
git clone -b mdd_demo-socket https://github.com/mdemoret-nv/Morpheus.git

# Clone the Demo GUI repo
git clone -b mdd_ubuntu18.04 https://github.com/trxcllnt/morpheus-ebc.git
```

### Build User Interface

Run the following to build the GUI

```bash
# Change to the GUI repo directory
cd morpheus-ebc

# Run the following to perform a clean build (Can be skipped for incremental builds)
rm -rf node_modules rapidsai

# Download and build all dependencies
yarn bootstrap

# Build the demo
yarn make
```

### Build Morpheus

Run the following to build and install Morpheus

```bash
# Change to the Morpheus repo directory
cd Morpheus

# Build the dev container
./docker/build_container_dev.sh

# Run the dev container
./docker/run_container_dev.sh

# Build the Morpheus library
./scripts/compile.sh

# Install the Morpheus library
pip install -e .

# Verify Morpheus is installed
morpheus --version

# Ensure the data has been downloaded
./scripts/fetch_data.py fetch examples

# Install an extra dependencies
pip install websockets
```

**Note: ** Keep the shell running the Morpheus Dev container running. It will be used later to start Morpheus.

## Running the Demo

### Running the GUI

The GUI must be launched before running Morpheus. In a different shell than the one used to build Morpheus, run the following:

```bash
# Change directory to the GUI repo
cd morpheus-ebc

# Run the GUI
yarn start
```

### Running Morpheus

After the GUI has been launched, Morpheus now needs to be started. In the same shell used to build Morpheus (the one running the Morpheus Dev container), run the following:

```bash
morpheus --log_level=DEBUG \
   run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=32 --edge_buffer_size=4 --use_cpp=False \
      pipeline-nlp --model_seq_length=256 \
         from-file --filename=examples/data/sid_visualization/group1-benign-2nodes.jsonlines \
         deserialize \
         preprocess --vocab_hash_file=morpheus/data/bert-base-uncased-hash.txt --truncation=True --do_lower_case=True --add_special_tokens=False \
         inf-triton --model_name=sid-minibert-onnx --server_url=localhost:8001 --force_convert_inputs=True \
         monitor --description Inference\ Rate --unit=inf \
         add-class \
         gen-viz
```
