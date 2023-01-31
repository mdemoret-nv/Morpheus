<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Building a Pipeline
> **Prerequisites**
> The following examples assume that the example dataset has been fetched, from the root of the Morpheus repo run:
>```bash
>./scripts/fetch_data.py fetch examples
>```

To build a pipeline via the CLI, users must first specify the type of pipeline, a source object, followed by a sequential list of stages. For each stage, options can be specified to configure the particular stage. Since stages are listed sequentially the output of one stage becomes the input to the next. Unless heavily customized, pipelines will start with either:

```bash
# For NLP Pipelines
morpheus run pipeline-nlp ...
# For FIL Pipelines
morpheus run pipeline-fil ...
```

While each stage will have configuration options, there are options that apply to the pipeline as a whole as well. Check
``morpheus run --help``, ``morpheus run pipeline-nlp --help`` and ``morpheus run pipeline-fil --help`` for these global
Pipeline options.

## Source Stages

All pipelines configured with the CLI need to start with a source object. Two commonly used source stages included with Morpheus are:

* `from-file`
  - Reads from a local file into the Pipeline
  - Supports JSON lines format
  - All lines are read at the start and queued into the pipeline at one time. Useful for performance testing.
  - Refer to `morpheus.stages.input.file_source_stage.FileSourceStage` for more information
* `from-kafka`
  - Pulls messages from a Kafka cluster into the Pipeline
  - Kafka cluster can be running on the localhost or remotely
  - Refer to `morpheus.stages.input.kafka_source_stage.KafkaSourceStage` for more information

## Stages

From this point on, any number of stages can be sequentially added to the command line from start to finish. For example, we could build a trivial pipeline that reads from a file, deserializes messages, serializes them, and then writes to a file, use the following:
```bash
morpheus --log_level=DEBUG run pipeline-nlp --viz_file=.tmp/simple_identity.png \
  from-file --filename=examples/data/pcap_dump.jsonlines \
  deserialize \
  serialize \
  to-file --overwrite --filename .tmp/temp_out.json
```
![../img/simple_identity.png](../img/simple_identity.png)

The output should be similar to:
```console
Configuring Pipeline via CLI
Parameter, 'labels_file', with relative path, 'data/labels_nlp.txt', does not exist. Using package relative location: '/home/dagardner/work/morpheus/morpheus/data/labels_nlp.txt'
Loaded labels file. Current labels: [['address', 'bank_acct', 'credit_card', 'email', 'govt_id', 'name', 'password', 'phone_num', 'secret_keys', 'user']]
Starting pipeline via CLI... Ctrl+C to Quit
Config:
{
  "ae": null,
  "class_labels": [
    "address",
    "bank_acct",
    "credit_card",
    "email",
    "govt_id",
    "name",
    "password",
    "phone_num",
    "secret_keys",
    "user"
  ],
  "debug": false,
  "edge_buffer_size": 128,
  "feature_length": 256,
  "fil": null,
  "log_config_file": null,
  "log_level": 10,
  "mode": "NLP",
  "model_max_batch_size": 8,
  "num_threads": 64,
  "pipeline_batch_size": 256,
  "plugins": []
}
CPP Enabled: True
====Registering Pipeline====
====Building Pipeline====
====Building Segment: linear_segment_0====
====Building Segment Complete!====
====Building Pipeline Complete!====
Starting! Time: 1672959248.7163541
====Registering Pipeline Complete!====
====Starting Pipeline====
====Pipeline Started====
Added source: <from-file-0; FileSourceStage(filename=examples/data/pcap_dump.jsonlines, iterative=False, file_type=FileTypes.Auto, repeat=1, filter_null=True, cudf_kwargs=None)>
  └─> morpheus.MessageMeta
Added stage: <deserialize-1; DeserializeStage()>
  └─ morpheus.MessageMeta -> morpheus.MultiMessage
Added stage: <serialize-2; SerializeStage(include=(), exclude=('^ID$', '^_ts_'), fixed_columns=True)>
  └─ morpheus.MultiMessage -> morpheus.MessageMeta
Added stage: <to-file-3; WriteToFileStage(filename=.tmp/temp_out.json, overwrite=True, file_type=FileTypes.Auto, include_index_col=True)>
  └─ morpheus.MessageMeta -> morpheus.MessageMeta
====Pipeline Complete====
Pipeline visualization saved to .tmp/simple_identity.png
```

### Pipeline Build Checks
After the `====Building Pipeline====` message, if logging is `INFO` or greater, the CLI will print a list of all stages and the type transformations of each stage. To be a valid Pipeline, the output type of one stage must match the input type of the next. Many stages are flexible and will determine their type at runtime but some stages require a specific input type. If your Pipeline is configured incorrectly, Morpheus will report the error. For example, if we run the same command as above but forget the `serialize` stage:
```bash
morpheus --log_level=DEBUG run pipeline-nlp \
  from-file --filename=examples/data/pcap_dump.jsonlines \
  deserialize \
  to-file --overwrite --filename .tmp/temp_out.json
```

Then the following error will be displayed:
```
RuntimeError: The to-file stage cannot handle input of <class 'morpheus.messages.multi_message.MultiMessage'>. Accepted input types: (<class 'morpheus.messages.message_meta.MessageMeta'>,)
```

This indicates that the ``to-file`` stage cannot accept the input type of `morpheus.pipeline.messages.MultiMessage`.
This is because the ``to-file`` stage has no idea how to write that class to a file, it only knows how to write instances of `morpheus.messages.message_meta.MessageMeta`. To ensure you have a valid pipeline, examine the `Accepted input types: (<class 'morpheus.messages.message_meta.MessageMeta'>,)` portion of the message. This indicates you need a stage that converts from the output type of the `deserialize` stage, `MultiMessage`, to `MessageMeta`, which is exactly what the `serialize` stage does.

### Kafka Source Example
The above example essentially just copies a file. However it is an important to note that most Morpheus pipelines are similar in structure, in that they begin with a source stage (`from-file`) followed by a `deserialize` stage, end with a `serialize` stage followed by a sink stage (`to-file`), with the actual training or inference logic occurring in between.

We could also easily swap out the source or sink stages in the above example without any impact to the pipeline as a whole. For example, to read from a Kafka topic, simply replace the `from-file` stage with `from-kafka`:

> **Note**: This assumes a Kafka broker running on the localhost listening to port 9092. For testing Morpheus with Kafka follow steps 1-8 in [Quick Launch Kafka Cluster](../developer_guide/contributing.md#quick-launch-kafka-cluster) section of [contributing.md](../developer_guide/contributing.md), creating a topic named `test_pcap` then replace port `9092` with the port your Kafka instance is listening on.

```bash
morpheus --log_level=DEBUG run pipeline-nlp \
  from-kafka --input_topic test_pcap --bootstrap_servers localhost:9092 \
  deserialize \
  serialize \
  to-file --filename .tmp/temp_out.json
```

## Available Stages
For a complete list of available stages, use the CLI help commands. The available stages can also be queried from the CLI using ``morpheus run pipeline-nlp --help`` or ``morpheus run pipeline-fil --help``.

## Basic Usage Examples

### Remove Fields from JSON Objects
This example will only copy the fields 'timestamp', 'src_ip' and 'dest_ip' from `examples/data/pcap_dump.jsonlines` to
`out.jsonlines`.

![../img/remove_fields_from_json_objects.png](../img/remove_fields_from_json_objects.png)

```bash
morpheus run pipeline-nlp --viz_file=.tmp/remove_fields_from_json_objects.png \
   from-file --filename examples/data/pcap_dump.jsonlines \
   deserialize \
   serialize --include 'timestamp' --include 'src_ip' --include 'dest_ip' \
   to-file --overwrite --filename out.jsonlines
```

### Monitor Throughput

This example will report the throughput on the command line.

![../img/monitor_throughput.png](../img/monitor_throughput.png)

```bash
morpheus run pipeline-nlp --viz_file=.tmp/monitor_throughput.png  \
   from-file --filename examples/data/pcap_dump.jsonlines \
   deserialize \
   monitor --description "Lines Throughput" --smoothing 0.1 --unit "lines" \
   serialize \
   to-file --overwrite --filename out.jsonlines
```

Output:
```console
Configuring Pipeline via CLI
Starting pipeline via CLI... Ctrl+C to Quit
Lines Throughput[Complete]: 93085 lines [00:03, 29446.18 lines/s]
Pipeline visualization saved to .tmp/monitor_throughput.png
```

### Multi-Monitor Throughput

This example will report the throughput for each stage independently.

![../img/multi_monitor_throughput.png](../img/multi_monitor_throughput.png)

```bash
morpheus run pipeline-nlp --viz_file=.tmp/multi_monitor_throughput.png  \
   from-file --filename examples/data/pcap_dump.jsonlines \
   monitor --description "From File Throughput" \
   deserialize \
   monitor --description "Deserialize Throughput" \
   serialize \
   monitor --description "Serialize Throughput" \
   to-file --filename out.jsonlines --overwrite
```

Output:
```console
Configuring Pipeline via CLI
Starting pipeline via CLI... Ctrl+C to Quit
From File Throughput[Complete]: 93085 messages [00:00, 168118.35 messages/s]
Deserialize Throughput[Complete]: 93085 messages [00:04, 22584.37 messages/s]
Serialize Throughput[Complete]: 93085 messages [00:06, 14095.36 messages/s]
Pipeline visualization saved to .tmp/multi_monitor_throughput.png
```

### NLP Kitchen Sink
This example shows an NLP Pipeline which uses several stages available in Morpheus. This example utilizes the Triton Inference Server to perform inference, and writes the output to a Kafka topic named `inference_output`. Both of which need to be started prior to launching Morpheus.

#### Launching Triton
From the Morpheus repo root directory, run the following to launch Triton and load the `sid-minibert` model:
```bash
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD/models:/models nvcr.io/nvidia/tritonserver:22.08-py3 tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit --load-model sid-minibert-onnx
```

#### Launching Kafka
Follow steps 1-8 in [Quick Launch Kafka Cluster](../developer_guide/contributing.md#quick-launch-kafka-cluster) section of [contributing.md](../developer_guide/contributing.md), creating a topic named `inference_output` then replace port `9092` with the port your Kafka instance is listening on.

![../img/nlp_kitchen_sink.png](../img/nlp_kitchen_sink.png)

```bash
morpheus run --num_threads=8 --pipeline_batch_size=1024 --model_max_batch_size=32 \
   pipeline-nlp --viz_file=.tmp/nlp_kitchen_sink.png  \
   from-file --filename examples/data/pcap_dump.jsonlines \
   deserialize \
   preprocess \
   inf-triton --model_name=sid-minibert-onnx --server_url=localhost:8001 \
   monitor --description "Inference Rate" --smoothing=0.001 --unit "inf" \
   add-class \
   filter --threshold=0.8 \
   serialize --include 'timestamp' --exclude '^_ts_' \
   to-kafka --bootstrap_servers localhost:9092 --output_topic "inference_output" \
   monitor --description "ToKafka Rate" --smoothing=0.001 --unit "msg"
```

Output:
```console
Configuring Pipeline via CLI
Starting pipeline via CLI... Ctrl+C to Quit
Inference Rate[Complete]: 93085 inf [00:07, 12334.49 inf/s]
ToKafka Rate[Complete]: 93085 msg [00:07, 13297.85 msg/s]
Pipeline visualization saved to .tmp/nlp_kitchen_sink.png
```
