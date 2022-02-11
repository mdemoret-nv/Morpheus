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

#include "morpheus/messages.hpp"
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <cstddef>
#include <memory>
#include <sstream>
#include <vector>

#include <cudf/io/csv.hpp>
#include <cudf/io/json.hpp>
#include "cudf_helpers_api.h"
#include "pybind11/pybind11.h"
#include "pyneo/utils.hpp"
#include "trtlab/neo/channels/channel.h"

namespace morpheus {

namespace neo = trtlab::neo;
namespace py  = pybind11;

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
PYBIND11_MODULE(messages, m)
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
    pyneo::import(m, "morpheus._lib.common");

    // Required for SegmentObject
    pyneo::import(m, "neo.core.node");

    // Allows python objects to keep DataTable objects alive
    py::class_<IDataTable, std::shared_ptr<IDataTable>>(m, "DataTable");

    neo::EdgeConnector<std::shared_ptr<MessageMeta>, py::object>::register_converter();
    neo::EdgeConnector<py::object, std::shared_ptr<MessageMeta>>::register_converter();

    py::class_<MessageMeta, std::shared_ptr<MessageMeta>>(m, "MessageMeta")
        .def(py::init<>([](py::object df) { return MessageMeta::create_from_python(std::move(df)); }), py::arg("df"))
        .def_property_readonly("count", [](MessageMeta& self) { return self.count(); })
        .def_property_readonly(
            "df",
            [](MessageMeta& self) {
                // // Get the column and convert to cudf
                // auto py_table_struct = make_table_from_view_and_meta(self.m_pydf;.tbl->view(),
                // self.m_pydf;.metadata); py::object py_table  =
                // py::reinterpret_steal<py::object>((PyObject*)py_table_struct);

                // // py_col.inc_ref();

                // return py_table;
                return self.get_py_table();
            },
            py::return_value_policy::move)
        .def_static("make_from_file", [](std::string filename) {
            // Load the file
            auto df_with_meta = load_table(filename);

            return MessageMeta::create_from_cpp(std::move(df_with_meta));
        });

    neo::EdgeConnector<std::shared_ptr<MultiMessage>, py::object>::register_converter();
    neo::EdgeConnector<py::object, std::shared_ptr<MultiMessage>>::register_converter();

    py::class_<MultiMessage, std::shared_ptr<MultiMessage>>(m, "MultiMessage")
        .def(py::init<>([](std::shared_ptr<MessageMeta> meta, cudf::size_type mess_offset, cudf::size_type mess_count) {
                 return std::make_shared<MultiMessage>(std::move(meta), mess_offset, mess_count);
             }),
             py::arg("meta"),
             py::arg("mess_offset"),
             py::arg("mess_count"))
        .def_readonly("meta", &MultiMessage::meta)
        .def_readonly("mess_offset", &MultiMessage::mess_offset)
        .def_readonly("mess_count", &MultiMessage::mess_count)
        .def(
            "get_meta",
            [](MultiMessage& self) {
                // Mimic this python code
                // self.meta.df.loc[self.meta.df.index[self.mess_offset:self.mess_offset + self.mess_count], columns] =
                // value
                auto df = self.meta->get_py_table();

                auto index_slice =
                    py::slice(py::int_(self.mess_offset), py::int_(self.mess_offset + self.mess_count), py::none());

                // Must do implicit conversion to py::object here!!!
                py::object df_slice = df.attr("loc")[df.attr("index")[index_slice]];

                return df_slice;
            },
            py::return_value_policy::move)
        .def(
            "get_meta",
            [](MultiMessage& self, py::object col_name) {
                // // Get the column and convert to cudf
                // auto info = self.get_meta(col_name);

                // auto py_table_struct = make_series_from_table_info(info, (PyObject*)info.get_parent_table().ptr());

                // if (!py_table_struct)
                // {
                //     throw py::error_already_set();
                // }

                // py::object py_table = py::reinterpret_steal<py::object>((PyObject*)py_table_struct);

                // return py_table;

                // Mimic this python code
                // self.meta.df.loc[self.meta.df.index[self.mess_offset:self.mess_offset + self.mess_count], columns] =
                // value
                auto df = self.meta->get_py_table();

                auto index_slice =
                    py::slice(py::int_(self.mess_offset), py::int_(self.mess_offset + self.mess_count), py::none());

                // Must do implicit conversion to py::object here!!!
                py::object df_slice = df.attr("loc")[py::make_tuple(df.attr("index")[index_slice], col_name)];

                return df_slice;
            },
            py::return_value_policy::move)
        .def(
            "get_meta",
            [](MultiMessage& self, py::object columns) {
                // // Get the column and convert to cudf
                // auto info = self.get_meta(columns);

                // auto py_table_struct = make_table_from_table_info(info, (PyObject*)info.get_parent_table().ptr());

                // if (!py_table_struct)
                // {
                //     throw py::error_already_set();
                // }

                // py::object py_table = py::reinterpret_steal<py::object>((PyObject*)py_table_struct);

                // return py_table;

                // Mimic this python code
                // self.meta.df.loc[self.meta.df.index[self.mess_offset:self.mess_offset + self.mess_count], columns] =
                // value
                auto df = self.meta->get_py_table();

                auto index_slice =
                    py::slice(py::int_(self.mess_offset), py::int_(self.mess_offset + self.mess_count), py::none());

                // Must do implicit conversion to py::object here!!!
                py::object df_slice = df.attr("loc")[py::make_tuple(df.attr("index")[index_slice], columns)];

                return df_slice;
            },
            py::return_value_policy::move)
        .def(
            "set_meta",
            [](MultiMessage& self, py::object columns, py::object value) {
                if (py::isinstance<py::list>(columns) || py::isinstance<py::tuple>(columns)) {
                    auto columns_vec = columns.cast<std::vector<std::string>>();
                    self.set_meta(columns_vec, value);
                } else if (py::isinstance<py::str>(columns)) {
                    auto column = columns.cast<std::string>();
                    self.set_meta(column, value);
                } else {
                    std::stringstream msg;
                    msg << "Unsupported python object (expected str or a list/tuple of str): " << py::str(columns);
                    throw std::runtime_error(msg.str());
                }
            },
            py::return_value_policy::move)
        .def(
            "get_slice",
            [](MultiMessage& self, size_t start, size_t stop) {
                // Returns shared_ptr
                return self.get_slice(start, stop);
            },
            py::return_value_policy::reference_internal);

    py::class_<InferenceMemory, std::shared_ptr<InferenceMemory>>(m, "InferenceMemory")
        .def_readonly("count", &InferenceMemory::count);

    py::class_<InferenceMemoryNLP, InferenceMemory, std::shared_ptr<InferenceMemoryNLP>>(m, "InferenceMemoryNLP")
        .def(py::init<>([](cudf::size_type count, py::object input_ids, py::object input_mask, py::object seq_ids) {
                 // Conver the cupy arrays to tensors
                 return std::make_shared<InferenceMemoryNLP>(count,
                                                             std::move(cupy_to_tensor(input_ids)),
                                                             std::move(cupy_to_tensor(input_mask)),
                                                             std::move(cupy_to_tensor(seq_ids)));
             }),
             py::arg("count"),
             py::arg("input_ids"),
             py::arg("input_mask"),
             py::arg("seq_ids"))
        .def_readonly("count", &InferenceMemoryNLP::count)
        .def_property(
            "input_ids",
            [m](InferenceMemoryNLP& self) { return tensor_to_cupy(self.get_input_ids(), m); },
            [](InferenceMemoryNLP& self, py::object cupy_value) {
                return self.set_input_ids(cupy_to_tensor(cupy_value));
            })
        .def_property(
            "input_mask",
            [m](InferenceMemoryNLP& self) { return tensor_to_cupy(self.get_input_mask(), m); },
            [](InferenceMemoryNLP& self, py::object cupy_value) {
                return self.set_input_mask(cupy_to_tensor(cupy_value));
            })
        .def_property(
            "seq_ids",
            [m](InferenceMemoryNLP& self) { return tensor_to_cupy(self.get_seq_ids(), m); },
            [](InferenceMemoryNLP& self, py::object cupy_value) {
                return self.set_seq_ids(cupy_to_tensor(cupy_value));
            });

    py::class_<InferenceMemoryFIL, InferenceMemory, std::shared_ptr<InferenceMemoryFIL>>(m, "InferenceMemoryFIL")
        .def(py::init<>([](cudf::size_type count, py::object input__0, py::object seq_ids) {
                 // Conver the cupy arrays to tensors
                 return std::make_shared<InferenceMemoryFIL>(
                     count, std::move(cupy_to_tensor(input__0)), std::move(cupy_to_tensor(seq_ids)));
             }),
             py::arg("count"),
             py::arg("input__0"),
             py::arg("seq_ids"))
        .def_readonly("count", &InferenceMemoryFIL::count)
        .def("get_tensor", [](InferenceMemoryFIL& self, const std::string& name) { return self.inputs[name]; })
        .def_property(
            "input__0",
            [m](InferenceMemoryFIL& self) { return tensor_to_cupy(self.get_input__0(), m); },
            [](InferenceMemoryFIL& self, py::object cupy_value) {
                return self.set_input__0(cupy_to_tensor(cupy_value));
            })
        .def_property(
            "seq_ids",
            [m](InferenceMemoryFIL& self) { return tensor_to_cupy(self.get_seq_ids(), m); },
            [](InferenceMemoryFIL& self, py::object cupy_value) {
                return self.set_seq_ids(cupy_to_tensor(cupy_value));
            });

    neo::EdgeConnector<std::shared_ptr<MultiInferenceMessage>, py::object>::register_converter();
    neo::EdgeConnector<py::object, std::shared_ptr<MultiInferenceMessage>>::register_converter();

    py::class_<MultiInferenceMessage, MultiMessage, std::shared_ptr<MultiInferenceMessage>>(m, "MultiInferenceMessage")
        .def(py::init<>([](std::shared_ptr<MessageMeta> meta,
                           cudf::size_type mess_offset,
                           cudf::size_type mess_count,
                           std::shared_ptr<InferenceMemory> memory,
                           cudf::size_type offset,
                           cudf::size_type count) {
                 return std::make_shared<MultiInferenceMessage>(
                     std::move(meta), mess_offset, mess_count, std::move(memory), offset, count);
             }),
             py::arg("meta"),
             py::arg("mess_offset"),
             py::arg("mess_count"),
             py::arg("memory"),
             py::arg("offset"),
             py::arg("count"))
        .def_readonly("memory", &MultiInferenceMessage::memory)
        .def_readonly("offset", &MultiInferenceMessage::offset)
        .def_readonly("count", &MultiInferenceMessage::count)
        .def("get_input",
             [m](MultiInferenceMessage& self, const std::string& name) {
                 const auto& py_tensor = tensor_to_cupy(self.get_input(name), m);

                 //  //  Need to get just our portion. TODO(MDD): THis should be handled in get_input
                 //  py::object sliced = py_tensor[py::make_tuple(
                 //      py::slice(py::int_(self.offset), py::int_(self.offset + self.count), py::none()),
                 //      py::slice(py::none(), py::none(), py::none()))];

                 return py_tensor;
             })
        .def(
            "get_slice",
            [&, m](MultiInferenceMessage& self, size_t start, size_t stop) {
                // py::object seq_ids = tensor_to_cupy(self.get_input("seq_ids"), m);

                // int mess_start = seq_ids[py::make_tuple(start, 0)].attr("item")().cast<int>();
                // int mess_stop  = seq_ids[py::make_tuple(stop - 1, 0)].attr("item")().cast<int>() + 1;

                // return std::make_shared<MultiInferenceMessage>(
                //     self.meta, mess_start, mess_stop - mess_start, self.memory, start, stop - start);
                return self.get_slice(start, stop);
            },
            py::return_value_policy::reference_internal);

    py::class_<MultiInferenceNLPMessage, MultiInferenceMessage, std::shared_ptr<MultiInferenceNLPMessage>>(
        m, "MultiInferenceNLPMessage")
        .def(py::init<>([](std::shared_ptr<MessageMeta> meta,
                           cudf::size_type mess_offset,
                           cudf::size_type mess_count,
                           std::shared_ptr<InferenceMemory> memory,
                           cudf::size_type offset,
                           cudf::size_type count) {
                 return std::make_shared<MultiInferenceNLPMessage>(
                     std::move(meta), mess_offset, mess_count, std::move(memory), offset, count);
             }),
             py::arg("meta"),
             py::arg("mess_offset"),
             py::arg("mess_count"),
             py::arg("memory"),
             py::arg("offset"),
             py::arg("count"))
        .def_readonly("memory", &MultiInferenceNLPMessage::memory)
        .def_readonly("offset", &MultiInferenceNLPMessage::offset)
        .def_readonly("count", &MultiInferenceNLPMessage::count)
        .def_property_readonly("input_ids",
                               [m](MultiInferenceNLPMessage& self) {
                                   // Get and convert
                                   auto tensor = self.get_input_ids();

                                   return tensor_to_cupy(tensor, m);
                               })
        .def_property_readonly("input_mask",
                               [m](MultiInferenceNLPMessage& self) {
                                   // Get and convert
                                   auto tensor = self.get_input_mask();

                                   return tensor_to_cupy(tensor, m);
                               })
        .def_property_readonly("seq_ids", [m](MultiInferenceNLPMessage& self) {
            // Get and convert
            auto tensor = self.get_seq_ids();

            return tensor_to_cupy(tensor, m);
        });

    py::class_<MultiInferenceFILMessage, MultiInferenceMessage, std::shared_ptr<MultiInferenceFILMessage>>(
        m, "MultiInferenceFILMessage")
        .def(py::init<>([](std::shared_ptr<MessageMeta> meta,
                           cudf::size_type mess_offset,
                           cudf::size_type mess_count,
                           std::shared_ptr<InferenceMemory> memory,
                           cudf::size_type offset,
                           cudf::size_type count) {
                 return std::make_shared<MultiInferenceFILMessage>(
                     std::move(meta), mess_offset, mess_count, std::move(memory), offset, count);
             }),
             py::arg("meta"),
             py::arg("mess_offset"),
             py::arg("mess_count"),
             py::arg("memory"),
             py::arg("offset"),
             py::arg("count"))
        .def_readonly("memory", &MultiInferenceFILMessage::memory)
        .def_readonly("offset", &MultiInferenceFILMessage::offset)
        .def_readonly("count", &MultiInferenceFILMessage::count);

    py::class_<ResponseMemory, std::shared_ptr<ResponseMemory>>(m, "ResponseMemory")
        .def_readonly("count", &ResponseMemory::count)
        .def(
            "get_output",
            [m](ResponseMemory& self, const std::string& name) {
                // Directly return the tensor object
                if (!self.has_output(name))
                {
                    throw py::key_error();
                }

                return tensor_to_cupy(self.outputs[name], m);
            },
            py::return_value_policy::reference_internal)
        .def(
            "get_output_tensor",
            [m](ResponseMemory& self, const std::string& name) {
                // Directly return the tensor object
                if (!self.has_output(name))
                {
                    throw py::key_error();
                }

                return self.outputs[name];
            },
            py::return_value_policy::reference_internal);

    py::class_<ResponseMemoryProbs, ResponseMemory, std::shared_ptr<ResponseMemoryProbs>>(m, "ResponseMemoryProbs")
        .def(py::init<>([](cudf::size_type count, py::object probs) {
                 // Conver the cupy arrays to tensors
                 return std::make_shared<ResponseMemoryProbs>(count, std::move(cupy_to_tensor(probs)));
             }),
             py::arg("count"),
             py::arg("probs"))
        .def_readonly("count", &ResponseMemoryProbs::count)
        .def_property(
            "probs",
            [&, m](ResponseMemoryProbs& self) {
                // Convert to cupy
                return tensor_to_cupy(self.get_probs(), m);
            },
            [](ResponseMemoryProbs& self, py::object cupy_value) {
                return self.set_probs(cupy_to_tensor(cupy_value));
            });

    neo::EdgeConnector<std::shared_ptr<MultiResponseMessage>, py::object>::register_converter();
    neo::EdgeConnector<py::object, std::shared_ptr<MultiResponseMessage>>::register_converter();

    py::class_<MultiResponseMessage, MultiMessage, std::shared_ptr<MultiResponseMessage>>(m, "MultiResponseMessage")
        .def(py::init<>([](std::shared_ptr<MessageMeta> meta,
                           cudf::size_type mess_offset,
                           cudf::size_type mess_count,
                           std::shared_ptr<ResponseMemory> memory,
                           cudf::size_type offset,
                           cudf::size_type count) {
                 return std::make_shared<MultiResponseMessage>(
                     std::move(meta), mess_offset, mess_count, std::move(memory), offset, count);
             }),
             py::arg("meta"),
             py::arg("mess_offset"),
             py::arg("mess_count"),
             py::arg("memory"),
             py::arg("offset"),
             py::arg("count"))
        .def_readonly("memory", &MultiResponseMessage::memory)
        .def_readonly("offset", &MultiResponseMessage::offset)
        .def_readonly("count", &MultiResponseMessage::count)
        .def("get_output", [&, m](MultiResponseMessage& self, const std::string& name) {
            auto tensor = self.get_output(name);

            return tensor_to_cupy(tensor, m);
        });

    py::class_<MultiResponseProbsMessage, MultiResponseMessage, std::shared_ptr<MultiResponseProbsMessage>>(
        m, "MultiResponseProbsMessage")
        .def(py::init<>([](std::shared_ptr<MessageMeta> meta,
                           cudf::size_type mess_offset,
                           cudf::size_type mess_count,
                           std::shared_ptr<ResponseMemory> memory,
                           cudf::size_type offset,
                           cudf::size_type count) {
                 return std::make_shared<MultiResponseProbsMessage>(
                     std::move(meta), mess_offset, mess_count, std::move(memory), offset, count);
             }),
             py::arg("meta"),
             py::arg("mess_offset"),
             py::arg("mess_count"),
             py::arg("memory"),
             py::arg("offset"),
             py::arg("count"))
        .def_readonly("memory", &MultiResponseProbsMessage::memory)
        .def_readonly("offset", &MultiResponseProbsMessage::offset)
        .def_readonly("count", &MultiResponseProbsMessage::count)
        .def_property_readonly("probs", [m](MultiResponseProbsMessage& self) {
            // Get and convert
            auto tensor = self.get_probs();

            return tensor_to_cupy(tensor, m);
        });

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace morpheus
