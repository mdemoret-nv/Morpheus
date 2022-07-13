/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <morpheus/stages/add_classification.hpp>
#include <morpheus/stages/add_scores.hpp>
#include <morpheus/stages/deserialization.hpp>
#include <morpheus/stages/file_source.hpp>
#include <morpheus/stages/filter_detection.hpp>
#include <morpheus/stages/kafka_source.hpp>
#include <morpheus/stages/monitor.hpp>
#include <morpheus/stages/preprocess_fil.hpp>
#include <morpheus/stages/preprocess_nlp.hpp>
#include <morpheus/stages/serialize.hpp>
#include <morpheus/stages/triton_inference.hpp>
#include <morpheus/stages/write_to_file.hpp>
#include <morpheus/utilities/cudf_util.hpp>

#include <srf/segment/object.hpp>

namespace morpheus {
namespace py = pybind11;

// Define the pybind11 module m, as 'pipeline'.
PYBIND11_MODULE(stages, m)
{
    m.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: morpheus.stages
        .. autosummary::
           :toctree: _generate
            TODO(Documentation)
        )pbdoc";

    // Load the cudf helpers
    load_cudf_helpers();

    srf::pysrf::import(m, "cupy");
    srf::pysrf::import(m, "morpheus._lib.messages");
    srf::pysrf::import(m, "morpheus._lib.file_types");

    py::class_<srf::segment::Object<AddClassificationsStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<AddClassificationsStage>>>(
        m, "AddClassificationsStage", py::multiple_inheritance())
        .def(py::init<>(&AddClassificationStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("threshold"),
             py::arg("num_class_labels"),
             py::arg("idx2label"));

    py::class_<srf::segment::Object<AddScoresStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<AddScoresStage>>>(m, "AddScoresStage", py::multiple_inheritance())
        .def(py::init<>(&AddScoresStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("num_class_labels"),
             py::arg("idx2label"));

    py::class_<srf::segment::Object<DeserializeStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<DeserializeStage>>>(
        m, "DeserializeStage", py::multiple_inheritance())
        .def(py::init<>(&DeserializeStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("batch_size"));

    py::class_<srf::segment::Object<FileSourceStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<FileSourceStage>>>(m, "FileSourceStage", py::multiple_inheritance())
        .def(py::init<>(&FileSourceStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("filename"),
             py::arg("repeat"))
        .def("get_total_lines", [](srf::segment::Object<FileSourceStage>& self) {
            // Return the objects property
            return self.object().get_total_lines();
        });

    py::class_<srf::segment::Object<FilterDetectionsStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<FilterDetectionsStage>>>(
        m, "FilterDetectionsStage", py::multiple_inheritance())
        .def(py::init<>(&FilterDetectionStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("threshold"));

    py::class_<srf::segment::Object<InferenceClientStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<InferenceClientStage>>>(
        m, "InferenceClientStage", py::multiple_inheritance())
        .def(py::init<>(&InferenceClientStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("model_name"),
             py::arg("server_url"),
             py::arg("force_convert_inputs"),
             py::arg("use_shared_memory"),
             py::arg("needs_logits"),
             py::arg("inout_mapping") = py::dict());

    py::class_<srf::segment::Object<KafkaSourceStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<KafkaSourceStage>>>(
        m, "KafkaSourceStage", py::multiple_inheritance())
        .def(py::init<>(&KafkaSourceStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("max_batch_size"),
             py::arg("topic"),
             py::arg("batch_timeout_ms"),
             py::arg("config"),
             py::arg("disable_commits")       = false,
             py::arg("disable_pre_filtering") = false);

    py::class_<srf::segment::Object<PreprocessFILStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<PreprocessFILStage>>>(
        m, "PreprocessFILStage", py::multiple_inheritance())
        .def(py::init<>(&PreprocessFILStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("features"));

    py::class_<srf::segment::Object<PreprocessNLPStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<PreprocessNLPStage>>>(
        m, "PreprocessNLPStage", py::multiple_inheritance())
        .def(py::init<>(&PreprocessNLPStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("vocab_hash_file"),
             py::arg("sequence_length"),
             py::arg("truncation"),
             py::arg("do_lower_case"),
             py::arg("add_special_token"),
             py::arg("stride"));

    py::class_<srf::segment::Object<SerializeStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<SerializeStage>>>(m, "SerializeStage", py::multiple_inheritance())
        .def(py::init<>(&SerializeStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("include"),
             py::arg("exclude"),
             py::arg("fixed_columns") = true);

    py::class_<srf::segment::Object<WriteToFileStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<WriteToFileStage>>>(
        m, "WriteToFileStage", py::multiple_inheritance())
        .def(py::init<>(&WriteToFileStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("filename"),
             py::arg("mode")      = "w",
             py::arg("file_type") = 0);  // Setting this to FileTypes::AUTO throws a conversion error at runtime

    py::class_<srf::segment::Object<MonitorStageBase>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<MonitorStageBase>>>(m, "MonitorStage", py::multiple_inheritance())
        .def(py::init<>(&MonitorStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("input_type"),
             py::arg("description"),
             py::arg("smoothing"),
             py::arg("unit"),
             py::arg("delayed_start"))
        .def("get_throughput",
             [](srf::segment::Object<MonitorStageBase>& self) { return self.object().get_throughput(); });

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace morpheus
