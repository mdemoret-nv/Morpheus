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

#include <morpheus/stages/monitor.hpp>

#include <morpheus/utilities/matx_util.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>

namespace morpheus {
// // Component public implementations
// // ************ MonitorStage **************************** //
// MonitorStage::MonitorStage(std::size_t num_class_labels, std::map<std::size_t, std::string> idx2label) :
//   PythonNode(base_t::op_factory_from_sub_fn(build_operator())),
//   m_num_class_labels(num_class_labels),
//   m_idx2label(std::move(idx2label))
// {
//     CHECK(m_idx2label.size() <= m_num_class_labels) << "idx2label should represent a subset of the class_labels";
// }

float MonitorStageBase::get_throughput() const
{
    return 0.0;
}

// MonitorStage::stream_fn_t MonitorStage::build_operator()
// {
//     return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
//         return input.subscribe(rxcpp::make_observer<sink_type_t>(
//             [this, &output](sink_type_t x) {
//                 const auto& probs  = x->get_probs();
//                 const auto& shape  = probs.get_shape();
//                 const auto& stride = probs.get_stride();

//                 CHECK(shape.size() == 2 && shape[1] == m_num_class_labels)
//                     << "Label count does not match output of model. Label count: " << m_num_class_labels
//                     << ", Model output: " << shape[1];

//                 const std::size_t num_rows    = shape[0];
//                 const std::size_t num_columns = shape[1];

//                 auto tmp_buffer = std::make_shared<rmm::device_buffer>(probs.bytes(), rmm::cuda_stream_per_thread);

//                 SRF_CHECK_CUDA(
//                     cudaMemcpy(tmp_buffer->data(), probs.data(), tmp_buffer->size(), cudaMemcpyDeviceToDevice));

//                 // Depending on the input the stride is given in bytes or elements,
//                 // divide the stride elements by the smallest item to ensure tensor_stride is defined in
//                 // terms of elements
//                 std::vector<TensorIndex> tensor_stride(stride.size());
//                 auto min_stride = std::min_element(stride.cbegin(), stride.cend());

//                 std::transform(stride.cbegin(),
//                                stride.cend(),
//                                tensor_stride.begin(),
//                                std::bind(std::divides<>(), std::placeholders::_1, *min_stride));

//                 auto tensor_obj = Tensor::create(
//                     tmp_buffer,
//                     probs.dtype(),
//                     std::vector<TensorIndex>{static_cast<long long>(shape[0]), static_cast<long long>(shape[1])},
//                     tensor_stride);

//                 std::vector<std::string> columns(m_idx2label.size());
//                 std::vector<TensorObject> tensors(m_idx2label.size());

//                 std::size_t i = 0;
//                 for (const auto& [column_num, column_name] : m_idx2label)
//                 {
//                     columns[i] = column_name;
//                     tensors[i] = tensor_obj.slice(std::vector<TensorIndex>{0, static_cast<TensorIndex>(column_num)},
//                                                   std::vector<TensorIndex>{static_cast<TensorIndex>(num_rows),
//                                                                            static_cast<TensorIndex>(column_num +
//                                                                            1)});

//                     ++i;
//                 }

//                 x->set_meta(columns, tensors);

//                 output.on_next(x);
//             },
//             [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
//             [&]() { output.on_completed(); }));
//     };
// }

// ************ MonitorStageInterfaceProxy ************* //
std::shared_ptr<srf::segment::Object<MonitorStage<pybind11::object>>> MonitorStageInterfaceProxy::init(
    srf::segment::Builder& builder,
    const std::string& name,
    pybind11::type input_type,
    std::string description,
    float smoothing,
    std::string unit,
    bool delayed_start)
{
    std::shared_ptr<srf::segment::Object<MonitorStage<pybind11::object>>> stage;

    auto type_info = pybind11::detail::get_type_info((PyTypeObject*)input_type.ptr());

    if (type_info != nullptr)
    {
        // This is a C++ type registered with pybind11
    }
    else
    {
        // This is a plain old python type
        stage =
            builder.construct_object<MonitorStage<pybind11::object>>(name, description, smoothing, unit, delayed_start);
    }

    return stage;
}

}  // namespace morpheus
