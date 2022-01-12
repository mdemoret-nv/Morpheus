/**
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "morpheus/stages.hpp"
#include <pybind11/cast.h>

#include <memory>
#include <vector>

#include "morpheus/cudf_helpers.hpp"
#include "pyneo/utils.hpp"

namespace morpheus {

cudf::io::table_with_metadata load_table(const std::string& filename)
{
    auto file_path = fs::path(filename);

    if (file_path.extension() == ".json" || file_path.extension() == ".jsonlines")
    {
        // First, load the file into json
        auto options = cudf::io::json_reader_options::builder(cudf::io::source_info{filename}).lines(true);

        return cudf::io::read_json(options.build());
    }
    else if (file_path.extension() == ".csv")
    {
        auto options = cudf::io::csv_reader_options::builder(cudf::io::source_info{filename});

        return cudf::io::read_csv(options.build());
    }
    else
    {
        LOG(FATAL) << "Unknown extension for file: " << filename;
        throw std::runtime_error("Unknown extension");
    }
}

// Define the pybind11 module m, as 'pipeline'.
PYBIND11_MODULE(stages, m)
{
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: scikit_build_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    // Load the cudf helpers
    load_cudf_helpers();

    pyneo::import(m, "cupy");
    pyneo::import(m, "morpheus._lib.messages");

    py::class_<FileSourceStage, neo::SegmentObject, std::shared_ptr<FileSourceStage>>(
        m, "FileSourceStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent, const std::string& name, std::string filename, int repeat = 1) {
                 auto stage = std::make_shared<FileSourceStage>(parent, name, filename, repeat);

                 parent.register_node<FileSourceStage::source_type_t, FileSourceStage::source_type_t>(stage);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("filename"),
             py::arg("repeat"));

    py::class_<KafkaSourceStage, neo::SegmentObject, std::shared_ptr<KafkaSourceStage>>(
        m, "KafkaSourceStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent,
                           const std::string& name,
                           size_t max_batch_size,
                           std::string topic,
                           int32_t batch_timeout_ms,
                           std::map<std::string, std::string> config) {
                 auto stage =
                     std::make_shared<KafkaSourceStage>(parent, name, max_batch_size, topic, batch_timeout_ms, config);

                 parent.register_node<FileSourceStage::source_type_t, FileSourceStage::source_type_t>(stage);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("max_batch_size"),
             py::arg("topic"),
             py::arg("batch_timeout_ms"),
             py::arg("config"));

    py::class_<DeserializeStage, neo::SegmentObject, std::shared_ptr<DeserializeStage>>(
        m, "DeserializeStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent, const std::string& name, size_t batch_size) {
                 auto stage = std::make_shared<DeserializeStage>(parent, name, batch_size);

                 parent.register_node<DeserializeStage::sink_type_t, DeserializeStage::source_type_t>(stage);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("batch_size"));

    py::class_<PreprocessNLPStage, neo::SegmentObject, std::shared_ptr<PreprocessNLPStage>>(
        m, "PreprocessNLPStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent,
                           const std::string& name,
                           std::string vocab_hash_file,
                           uint32_t sequence_length,
                           bool truncation,
                           bool do_lower_case,
                           bool add_special_token,
                           int stride = -1) {
                 auto stage = std::make_shared<PreprocessNLPStage>(parent,
                                                                   name,
                                                                   vocab_hash_file,
                                                                   sequence_length,
                                                                   truncation,
                                                                   do_lower_case,
                                                                   add_special_token,
                                                                   stride);

                 parent.register_node<PreprocessNLPStage::sink_type_t, PreprocessNLPStage::source_type_t>(stage);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("vocab_hash_file"),
             py::arg("sequence_length"),
             py::arg("truncation"),
             py::arg("do_lower_case"),
             py::arg("add_special_token"),
             py::arg("stride"));

    py::class_<PreprocessFILStage, neo::SegmentObject, std::shared_ptr<PreprocessFILStage>>(
        m, "PreprocessFILStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent, const std::string& name) {
                 auto stage = std::make_shared<PreprocessFILStage>(parent, name);

                 parent.register_node<PreprocessFILStage::sink_type_t, PreprocessFILStage::source_type_t>(stage);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"));

    py::class_<InferenceClientStage, neo::SegmentObject, std::shared_ptr<InferenceClientStage>>(
        m, "InferenceClientStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent,
                           const std::string& name,
                           std::string model_name,
                           std::string server_url,
                           bool force_convert_inputs,
                           bool use_shared_memory,
                           bool needs_logits,
                           std::map<std::string, std::string> inout_mapping) {
                 auto stage = std::make_shared<InferenceClientStage>(parent,
                                                                     name,
                                                                     model_name,
                                                                     server_url,
                                                                     force_convert_inputs,
                                                                     use_shared_memory,
                                                                     needs_logits,
                                                                     inout_mapping);

                 parent.register_node<InferenceClientStage::sink_type_t, InferenceClientStage::source_type_t>(stage);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("model_name"),
             py::arg("server_url"),
             py::arg("force_convert_inputs"),
             py::arg("use_shared_memory"),
             py::arg("needs_logits"),
             py::arg("inout_mapping") = py::dict());

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace morpheus
