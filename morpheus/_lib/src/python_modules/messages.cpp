/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/messages/memory/inference_memory.hpp"
#include "morpheus/messages/memory/inference_memory_fil.hpp"
#include "morpheus/messages/memory/inference_memory_nlp.hpp"
#include "morpheus/messages/memory/response_memory.hpp"
#include "morpheus/messages/memory/response_memory_probs.hpp"
#include "morpheus/messages/memory/tensor_memory.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/messages/multi_inference_fil.hpp"
#include "morpheus/messages/multi_inference_nlp.hpp"
#include "morpheus/messages/multi_response.hpp"
#include "morpheus/messages/multi_response_probs.hpp"
#include "morpheus/messages/multi_tensor.hpp"
#include "morpheus/objects/data_table.hpp"
#include "morpheus/objects/mutable_table_ctx_mgr.hpp"
#include "morpheus/types.hpp"  // for TensorIndex
#include "morpheus/utilities/cudf_util.hpp"

#include <boost/fiber/future/future.hpp>
#include <mrc/channel/status.hpp>  // for Status
#include <mrc/edge/edge_connector.hpp>
#include <mrc/node/rx_sink_base.hpp>
#include <mrc/node/rx_source_base.hpp>
#include <mrc/types.hpp>
#include <pybind11/cast.h>
#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep
#include <pymrc/port_builders.hpp>
#include <pymrc/utils.hpp>  // for pymrc::import
#include <rxcpp/rx.hpp>

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace morpheus {

namespace fs = std::filesystem;
namespace py = pybind11;

// Define the pybind11 module m, as 'pipeline'.
PYBIND11_MODULE(messages, m)
{
    m.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: morpheus.messages
        .. autosummary::
           :toctree: _generate

        )pbdoc";

    // Load the cudf helpers
    CudfHelper::load();

    mrc::pymrc::import(m, "cupy");
    mrc::pymrc::import(m, "morpheus._lib.common");

    // Required for SegmentObject
    mrc::pymrc::import(m, "mrc.core.node");

    // Allows python objects to keep DataTable objects alive
    py::class_<IDataTable, std::shared_ptr<IDataTable>>(m, "DataTable");

    mrc::pymrc::PortBuilderUtil::register_port_util<std::shared_ptr<MessageMeta>>();
    mrc::pymrc::PortBuilderUtil::register_port_util<std::shared_ptr<MultiMessage>>();
    mrc::pymrc::PortBuilderUtil::register_port_util<std::shared_ptr<MultiTensorMessage>>();
    mrc::pymrc::PortBuilderUtil::register_port_util<std::shared_ptr<MultiInferenceMessage>>();
    mrc::pymrc::PortBuilderUtil::register_port_util<std::shared_ptr<MultiInferenceFILMessage>>();
    mrc::pymrc::PortBuilderUtil::register_port_util<std::shared_ptr<MultiInferenceNLPMessage>>();
    mrc::pymrc::PortBuilderUtil::register_port_util<std::shared_ptr<MultiResponseMessage>>();
    mrc::pymrc::PortBuilderUtil::register_port_util<std::shared_ptr<MultiResponseProbsMessage>>();

    // EdgeConnectors for derived classes of MultiMessage to MultiMessage
    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiTensorMessage>,
                             std::shared_ptr<morpheus::MultiMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiInferenceMessage>,
                             std::shared_ptr<morpheus::MultiTensorMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiInferenceMessage>,
                             std::shared_ptr<morpheus::MultiMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiInferenceFILMessage>,
                             std::shared_ptr<morpheus::MultiInferenceMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiInferenceFILMessage>,
                             std::shared_ptr<morpheus::MultiTensorMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiInferenceFILMessage>,
                             std::shared_ptr<morpheus::MultiMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiInferenceNLPMessage>,
                             std::shared_ptr<morpheus::MultiInferenceMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiInferenceNLPMessage>,
                             std::shared_ptr<morpheus::MultiTensorMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiInferenceNLPMessage>,
                             std::shared_ptr<morpheus::MultiMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiResponseMessage>,
                             std::shared_ptr<morpheus::MultiMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiResponseMessage>,
                             std::shared_ptr<morpheus::MultiTensorMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiResponseProbsMessage>,
                             std::shared_ptr<morpheus::MultiResponseMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiResponseProbsMessage>,
                             std::shared_ptr<morpheus::MultiTensorMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiResponseProbsMessage>,
                             std::shared_ptr<morpheus::MultiMessage>>::register_converter();

    // Tensor Memory classes
    py::class_<TensorMemory, std::shared_ptr<TensorMemory>>(m, "TensorMemory")
        .def(py::init<>(&TensorMemoryInterfaceProxy::init), py::arg("count"), py::arg("tensors") = py::none())
        .def_readonly("count", &TensorMemory::count)
        .def_property_readonly("tensor_names", &TensorMemoryInterfaceProxy::tensor_names_getter)
        .def("has_tensor", &TensorMemoryInterfaceProxy::has_tensor)
        .def("get_tensors", &TensorMemoryInterfaceProxy::get_tensors, py::return_value_policy::move)
        .def("set_tensors", &TensorMemoryInterfaceProxy::set_tensors, py::arg("tensors"))
        .def("get_tensor", &TensorMemoryInterfaceProxy::get_tensor, py::arg("name"), py::return_value_policy::move)
        .def("set_tensor", &TensorMemoryInterfaceProxy::set_tensor, py::arg("name"), py::arg("tensor"));

    py::class_<InferenceMemory, TensorMemory, std::shared_ptr<InferenceMemory>>(m, "InferenceMemory")
        .def(py::init<>(&InferenceMemoryInterfaceProxy::init), py::arg("count"), py::arg("tensors") = py::none())
        .def("get_input", &InferenceMemoryInterfaceProxy::get_tensor, py::arg("name"), py::return_value_policy::move)
        .def("set_input", &InferenceMemoryInterfaceProxy::set_tensor, py::arg("name"), py::arg("tensor"));

    py::class_<InferenceMemoryFIL, InferenceMemory, std::shared_ptr<InferenceMemoryFIL>>(m, "InferenceMemoryFIL")
        .def(py::init<>(&InferenceMemoryFILInterfaceProxy::init),
             py::arg("count"),
             py::arg("input__0"),
             py::arg("seq_ids"))
        .def_property("input__0",
                      &InferenceMemoryFILInterfaceProxy::get_input__0,
                      &InferenceMemoryFILInterfaceProxy::set_input__0)
        .def_property(
            "seq_ids", &InferenceMemoryFILInterfaceProxy::get_seq_ids, &InferenceMemoryFILInterfaceProxy::set_seq_ids);

    py::class_<InferenceMemoryNLP, InferenceMemory, std::shared_ptr<InferenceMemoryNLP>>(m, "InferenceMemoryNLP")
        .def(py::init<>(&InferenceMemoryNLPInterfaceProxy::init),
             py::arg("count"),
             py::arg("input_ids"),
             py::arg("input_mask"),
             py::arg("seq_ids"))
        .def_property("input_ids",
                      &InferenceMemoryNLPInterfaceProxy::get_input_ids,
                      &InferenceMemoryNLPInterfaceProxy::set_input_ids)
        .def_property("input_mask",
                      &InferenceMemoryNLPInterfaceProxy::get_input_mask,
                      &InferenceMemoryNLPInterfaceProxy::set_input_mask)
        .def_property(
            "seq_ids", &InferenceMemoryNLPInterfaceProxy::get_seq_ids, &InferenceMemoryNLPInterfaceProxy::set_seq_ids);

    py::class_<ResponseMemory, TensorMemory, std::shared_ptr<ResponseMemory>>(m, "ResponseMemory")
        .def(py::init<>(&ResponseMemoryInterfaceProxy::init), py::arg("count"), py::arg("tensors") = py::none())
        .def("get_output", &ResponseMemoryInterfaceProxy::get_tensor, py::arg("name"), py::return_value_policy::move)
        .def("set_output", &ResponseMemoryInterfaceProxy::set_tensor, py::arg("name"), py::arg("tensor"));

    py::class_<ResponseMemoryProbs, ResponseMemory, std::shared_ptr<ResponseMemoryProbs>>(m, "ResponseMemoryProbs")
        .def(py::init<>(&ResponseMemoryProbsInterfaceProxy::init), py::arg("count"), py::arg("probs"))
        .def_property(
            "probs", &ResponseMemoryProbsInterfaceProxy::get_probs, &ResponseMemoryProbsInterfaceProxy::set_probs);

    // Context manager for Mutable Dataframes. Attempting to use it outside of a with block will raise an exception
    py::class_<MutableTableCtxMgr, std::shared_ptr<MutableTableCtxMgr>>(m, "MutableTableCtxMgr")
        .def("__enter__", &MutableTableCtxMgr::enter, py::return_value_policy::reference)
        .def("__exit__", &MutableTableCtxMgr::exit)
        .def("__getattr__", &MutableTableCtxMgr::throw_usage_error)
        .def("__getitem__", &MutableTableCtxMgr::throw_usage_error)
        .def("__setattr__", &MutableTableCtxMgr::throw_usage_error)
        .def("__setitem__", &MutableTableCtxMgr::throw_usage_error);

    // Message classes
    py::class_<MessageMeta, std::shared_ptr<MessageMeta>>(m, "MessageMeta")
        .def(py::init<>(&MessageMetaInterfaceProxy::init_python), py::arg("df"))
        .def_property_readonly("count", &MessageMetaInterfaceProxy::count)
        .def_property_readonly("df", &MessageMetaInterfaceProxy::df_property, py::return_value_policy::move)
        .def("copy_dataframe", &MessageMetaInterfaceProxy::get_data_frame, py::return_value_policy::move)
        .def("mutable_dataframe", &MessageMetaInterfaceProxy::mutable_dataframe, py::return_value_policy::move)
        .def("has_sliceable_index", &MessageMetaInterfaceProxy::has_sliceable_index)
        .def("ensure_sliceable_index", &MessageMetaInterfaceProxy::ensure_sliceable_index)
        .def_static("make_from_file", &MessageMetaInterfaceProxy::init_cpp);

    py::class_<MultiMessage, std::shared_ptr<MultiMessage>>(m, "MultiMessage")
        .def(py::init<>(&MultiMessageInterfaceProxy::init),
             py::kw_only(),
             py::arg("meta"),
             py::arg("mess_offset") = 0,
             py::arg("mess_count")  = -1)
        .def_property_readonly("meta", &MultiMessageInterfaceProxy::meta)
        .def_property_readonly("mess_offset", &MultiMessageInterfaceProxy::mess_offset)
        .def_property_readonly("mess_count", &MultiMessageInterfaceProxy::mess_count)
        .def("get_meta",
             static_cast<pybind11::object (*)(MultiMessage&)>(&MultiMessageInterfaceProxy::get_meta),
             py::return_value_policy::move)
        .def("get_meta",
             static_cast<pybind11::object (*)(MultiMessage&, std::string)>(&MultiMessageInterfaceProxy::get_meta),
             py::return_value_policy::move)
        .def("get_meta",
             static_cast<pybind11::object (*)(MultiMessage&, std::vector<std::string>)>(
                 &MultiMessageInterfaceProxy::get_meta),
             py::return_value_policy::move)
        .def("get_meta",
             static_cast<pybind11::object (*)(MultiMessage&, pybind11::none)>(&MultiMessageInterfaceProxy::get_meta),
             py::return_value_policy::move)
        .def("set_meta", &MultiMessageInterfaceProxy::set_meta, py::return_value_policy::move)
        .def("get_slice", &MultiMessageInterfaceProxy::get_slice, py::return_value_policy::reference_internal)
        .def("copy_ranges",
             &MultiMessageInterfaceProxy::copy_ranges,
             py::arg("ranges"),
             py::arg("num_selected_rows") = py::none(),
             py::return_value_policy::move)
        .def("get_meta_list", &MultiMessageInterfaceProxy::get_meta_list, py::return_value_policy::move);

    py::class_<MultiTensorMessage, MultiMessage, std::shared_ptr<MultiTensorMessage>>(m, "MultiTensorMessage")
        .def(py::init<>(&MultiTensorMessageInterfaceProxy::init),
             py::kw_only(),
             py::arg("meta"),
             py::arg("mess_offset") = 0,
             py::arg("mess_count")  = -1,
             py::arg("memory"),
             py::arg("offset")         = 0,
             py::arg("count")          = -1,
             py::arg("id_tensor_name") = "seq_ids")
        .def_property_readonly("memory", &MultiTensorMessageInterfaceProxy::memory)
        .def_property_readonly("offset", &MultiTensorMessageInterfaceProxy::offset)
        .def_property_readonly("count", &MultiTensorMessageInterfaceProxy::count)
        .def("get_tensor", &MultiTensorMessageInterfaceProxy::get_tensor)
        .def("get_id_tensor", &MultiResponseMessageInterfaceProxy::get_id_tensor);

    py::class_<MultiInferenceMessage, MultiTensorMessage, std::shared_ptr<MultiInferenceMessage>>(
        m, "MultiInferenceMessage")
        .def(py::init<>(&MultiInferenceMessageInterfaceProxy::init),
             py::kw_only(),
             py::arg("meta"),
             py::arg("mess_offset") = 0,
             py::arg("mess_count")  = -1,
             py::arg("memory"),
             py::arg("offset")         = 0,
             py::arg("count")          = -1,
             py::arg("id_tensor_name") = "seq_ids")
        .def("get_input", &MultiInferenceMessageInterfaceProxy::get_tensor);

    py::class_<MultiInferenceNLPMessage, MultiInferenceMessage, std::shared_ptr<MultiInferenceNLPMessage>>(
        m, "MultiInferenceNLPMessage")
        .def(py::init<>(&MultiInferenceNLPMessageInterfaceProxy::init),
             py::kw_only(),
             py::arg("meta"),
             py::arg("mess_offset") = 0,
             py::arg("mess_count")  = -1,
             py::arg("memory"),
             py::arg("offset")         = 0,
             py::arg("count")          = -1,
             py::arg("id_tensor_name") = "seq_ids")
        .def_property_readonly("input_ids", &MultiInferenceNLPMessageInterfaceProxy::input_ids)
        .def_property_readonly("input_mask", &MultiInferenceNLPMessageInterfaceProxy::input_mask)
        .def_property_readonly("seq_ids", &MultiInferenceNLPMessageInterfaceProxy::seq_ids);

    py::class_<MultiInferenceFILMessage, MultiInferenceMessage, std::shared_ptr<MultiInferenceFILMessage>>(
        m, "MultiInferenceFILMessage")
        .def(py::init<>(&MultiInferenceFILMessageInterfaceProxy::init),
             py::kw_only(),
             py::arg("meta"),
             py::arg("mess_offset") = 0,
             py::arg("mess_count")  = -1,
             py::arg("memory"),
             py::arg("offset")         = 0,
             py::arg("count")          = -1,
             py::arg("id_tensor_name") = "seq_ids")
        .def_property_readonly("input__0", &MultiInferenceFILMessageInterfaceProxy::input__0)
        .def_property_readonly("seq_ids", &MultiInferenceFILMessageInterfaceProxy::seq_ids);

    py::class_<MultiResponseMessage, MultiTensorMessage, std::shared_ptr<MultiResponseMessage>>(m,
                                                                                                "MultiResponseMessage")
        .def(py::init<>(&MultiResponseMessageInterfaceProxy::init),
             py::kw_only(),
             py::arg("meta"),
             py::arg("mess_offset") = 0,
             py::arg("mess_count")  = -1,
             py::arg("memory"),
             py::arg("offset")            = 0,
             py::arg("count")             = -1,
             py::arg("id_tensor_name")    = "seq_ids",
             py::arg("probs_tensor_name") = "probs")
        .def_property("probs_tensor_name",
                      &MultiResponseMessageInterfaceProxy::probs_tensor_name_getter,
                      &MultiResponseMessageInterfaceProxy::probs_tensor_name_setter)
        .def("get_output", &MultiResponseMessageInterfaceProxy::get_tensor)
        .def("get_probs_tensor", &MultiResponseMessageInterfaceProxy::get_probs_tensor);

    py::class_<MultiResponseProbsMessage, MultiResponseMessage, std::shared_ptr<MultiResponseProbsMessage>>(
        m, "MultiResponseProbsMessage")
        .def(py::init<>(&MultiResponseProbsMessageInterfaceProxy::init),
             py::kw_only(),
             py::arg("meta"),
             py::arg("mess_offset") = 0,
             py::arg("mess_count")  = -1,
             py::arg("memory"),
             py::arg("offset")            = 0,
             py::arg("count")             = -1,
             py::arg("id_tensor_name")    = "seq_ids",
             py::arg("probs_tensor_name") = "probs")
        .def_property_readonly("probs", &MultiResponseProbsMessageInterfaceProxy::probs);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace morpheus
