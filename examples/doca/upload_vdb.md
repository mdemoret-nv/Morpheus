
## Run Milvus

Download the milvus docker-compose file from the [Milvus GitHub repository]()

```bash
mkdir milvus
cd milvus
wget https://github.com/milvus-io/milvus/releases/download/v2.3.3/milvus-standalone-docker-compose-gpu.yml -O docker-compose.yml
```

Start Milvus

```bash
docker-compose up -d
```

## Launch Triton Inference Server

To serve the embedding model, we will use Triton:

```bash
cd ${MORPHEUS_ROOT}
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD/models:/models nvcr.io/nvidia/tritonserver:24.01-py3 tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit --load-model all-MiniLM-L6-v2
```

## Populate the Milvus database

```bash
cd ${MORPHEUS_ROOT}

python examples/doca/run_udp_convert.py --nic_addr=ca:00.0 --gpu_addr=17:00.0
```

## Send data to the NIC to be indexed

On another machine, run the following command:

```bash
sudo python3 send2.py
```

On the original machine, wait for the "Upload rate" to match the "DOCA GPUNetIO Source rate" and then press `Ctrl+C` to stop the script. The output should look like the following

```
====Building Segment Complete!====
Accumulated 1 rows for collection: vdb_doca
Accumulated 2 rows for collection: vdb_doca
Accumulated 3 rows for collection: vdb_doca
Accumulated 1 rows for collection: vdb_doca
Accumulated 2 rows for collection: vdb_doca
Accumulated 3 rows for collection: vdb_doca
Stopping pipeline. Please wait... Press Ctrl+C again to kill.
====Stopping Pipeline====
====Pipeline Stopped====
DOCA GPUNetIO Source rate[Complete]: 229 pkts [04:29,  1.18s/ pkts]
Embedding rate[Complete]: 229 pkts [05:51,  1.53s/ pkts]
====Pipeline Complete====
```

## Query the Milvus database

First, set the NeMo LLM API Key:

```bash
export NGC_API_KEY="<YOUR_NGC_API>"
```

Run the RAG example to query the Milvus database:

```bash
cd ${MORPHEUS_ROOT}
python examples/llm/main.py --use_cpp=True --log_level=DEBUG rag pipeline --vdb_resource_name=vdb_doca
```

You should see the answer to the query in the output:

```
Pipeline complete. Received 3 responses
Question:
What is DOCA?
Response:
 DOCA is a library that provides a set of APIs for creating and managing network devices on GPUs.
Question:
What is the DOCA SDK?
Response:
 The DOCA Software Development Kit (SDK) is a software development kit that provides a set of libraries, tools, and documentation to help developers create and deploy network applications on Mellanox network adapters.
Question:
What does DOCA GPUNetIO to remove the CPU from the critical path?
Response:
 DOCA GPUNetIO enables GPU-centric solutions that remove the CPU from the critical path by providing the following features:
   GPUDirect Async Kernel-Initiated Network (GDAKIN) communications – a CUDA kernel can invoke GPUNetIO device functions to receive or send, directly interacting with the NIC
       CPU intervention is not needed in the application critical path
   GPUDirect RDMA – receive packets directly into a contiguous GPU memory​ area
   Semaphores – provide a standardized I/O communication protocol between the receiving entity and the CUDA kernel real-time packet processing​
   Smart memory allocation – allocate aligned GPU memory buffers exposing them to direct CPU access
       Combination of CUDA and DPDK gpudev library (with the DOCA GPUNetIO shared library is doca-gpu.pc. However, there is no pkgconfig file for the DOCA GPUNetIO CUDA device's static library /opt/mellanox/d
Total time: 10.61 sec
Pipeline runtime: 4.12 sec
```
