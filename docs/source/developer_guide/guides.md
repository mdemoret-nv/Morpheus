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

# Developer Guides

Morpheus includes several stages to choose from when building a custom pipeline, which can be included and configured to suit your needs. However, there are likely going to be situations that require writing a custom stage. Morpheus stages are written in Python and optionally may include a C++ implementation. The following guides outline how to create your own stages in both Python and C++.

* [Simple Python Stage](./guides/1_simple_python_stage.md)
* [Real-World Application: Phishing Detection](./guides/2_real_world_phishing.md)
* [Simple C++ Stage](./guides/3_simple_cpp_stage.md)
* [Creating a C++ Source Stage](./guides/4_source_cpp_stage.md)
* [Digital Fingerprinting (DFP)](./guides/5_digital_fingerprinting.md)
* [Digital Fingerprinting (DFP) Reference](./guides/6_digital_fingerprinting_reference.md)
