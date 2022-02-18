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

#include "morpheus/matx_functions.hpp"

#include <memory>
#include <stdexcept>
#include <type_traits>

#include <matx.h>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include "trtlab/cuda/sync.h"

#include <morpheus/tensor.hpp>
#include <morpheus/type_utils.hpp>

namespace morpheus {

struct matx_cast
{
    size_t element_count;
    rmm::cuda_stream_view stream;

    template <typename InputT,
              typename OutputT,
              std::enable_if_t<!cudf::is_numeric<InputT>() || !cudf::is_numeric<OutputT>()>* = nullptr>
    void operator()(void* input_data, void* output_data)
    {
        throw std::invalid_argument("Unsupported conversion");
    }

    template <typename InputT,
              typename OutputT,
              std::enable_if_t<cudf::is_numeric<InputT>() && cudf::is_numeric<OutputT>()>* = nullptr>
    void operator()(void* input_data, void* output_data)
    {
        matx::tensorShape_t<1> shape({static_cast<matx::index_t>(element_count)});

        matx::tensor_t<InputT, 1> input_tensor(static_cast<InputT*>(input_data), shape);
        matx::tensor_t<OutputT, 1> output_tensor(static_cast<OutputT*>(output_data), shape);

        (output_tensor = input_tensor).run(stream.value());
    }
};

std::shared_ptr<rmm::device_buffer> cast(const DevMemInfo& input, trtlab::neo::TypeId output_type)
{
    auto input_dtype  = DType(input.type_id);
    auto output_dtype = DType(output_type);

    // Create the output
    auto output = std::make_shared<rmm::device_buffer>(
        output_dtype.item_size() * input.element_count, input.buffer->stream(), input.buffer->memory_resource());

    cudf::double_type_dispatcher(cudf::data_type{input_dtype.cudf_type_id()},
                                 cudf::data_type{output_dtype.cudf_type_id()},
                                 matx_cast{input.element_count, output->stream()},
                                 input.data(),
                                 output->data());

    trtlab::cuda_sync<trtlab::standard_threads>::stream_sync(output->stream().value());

    return output;
}

struct matx_logits
{
    size_t element_count;
    rmm::cuda_stream_view stream;

    template <typename InputT, std::enable_if_t<!cudf::is_floating_point<InputT>()>* = nullptr>
    void operator()(void* input_data, void* output_data)
    {
        throw std::invalid_argument("Unsupported conversion");
    }

    template <typename InputT, std::enable_if_t<cudf::is_floating_point<InputT>()>* = nullptr>
    void operator()(void* input_data, void* output_data)
    {
        matx::tensorShape_t<1> shape({static_cast<matx::index_t>(element_count)});

        matx::tensor_t<InputT, 1> input_tensor(static_cast<InputT*>(input_data), shape);

        matx::tensor_t<InputT, 1> output_tensor(static_cast<InputT*>(output_data), shape);

        (output_tensor = (InputT)1 / ((InputT)1 + matx::exp((InputT)-1 * input_tensor))).run(stream.value());
    }
};

std::shared_ptr<rmm::device_buffer> logits(const DevMemInfo& input)
{
    auto input_dtype = DType(input.type_id);

    // Now create the output
    auto output = std::make_shared<rmm::device_buffer>(
        input_dtype.item_size() * input.element_count, input.buffer->stream(), input.buffer->memory_resource());

    cudf::type_dispatcher(cudf::data_type{input_dtype.cudf_type_id()},
                          matx_logits{input.element_count, output->stream()},
                          input.data(),
                          output->data());

    return output;
}

struct matx_transpose
{
    size_t element_count;
    rmm::cuda_stream_view stream;
    size_t rows;
    size_t cols;

    template <typename InputT, std::enable_if_t<!cudf::is_numeric<InputT>()>* = nullptr>
    void operator()(void* input_data, void* output_data)
    {
        throw std::invalid_argument("Unsupported conversion");
    }

    template <typename InputT, std::enable_if_t<cudf::is_numeric<InputT>()>* = nullptr>
    void operator()(void* input_data, void* output_data)
    {
        matx::tensorShape_t<2> input_shape({static_cast<matx::index_t>(rows), static_cast<matx::index_t>(cols)});
        matx::tensorShape_t<2> output_shape({static_cast<matx::index_t>(cols), static_cast<matx::index_t>(rows)});

        matx::tensor_t<InputT, 2> input_tensor(static_cast<InputT*>(input_data), input_shape);

        matx::tensor_t<InputT, 2> output_tensor(static_cast<InputT*>(output_data), output_shape);

        (output_tensor = input_tensor.Permute({1, 0})).run(stream.value());
    }
};

// Perform transpose
std::shared_ptr<rmm::device_buffer> transpose(const DevMemInfo& input, size_t rows, size_t cols)
{
    auto input_dtype = DType(input.type_id);

    // Now create the output
    auto output = std::make_shared<rmm::device_buffer>(
        input_dtype.item_size() * input.element_count, input.buffer->stream(), input.buffer->memory_resource());

    cudf::type_dispatcher(cudf::data_type{input_dtype.cudf_type_id()},
                          matx_transpose{input.element_count, output->stream(), rows, cols},
                          input.data(),
                          output->data());

    return output;
}

struct matx_create_seg_ids
{
    size_t element_count;
    size_t fea_len;
    rmm::cuda_stream_view stream;

    template <typename OutputT, std::enable_if_t<!std::is_integral_v<OutputT>>* = nullptr>
    void operator()(void* output_data)
    {
        throw std::invalid_argument("Unsupported conversion");
    }

    template <typename OutputT, std::enable_if_t<std::is_integral_v<OutputT>>* = nullptr>
    void operator()(void* output_data)
    {
        matx::tensorShape_t<2> shape({static_cast<matx::index_t>(element_count), 3});

        matx::tensor_t<OutputT, 2> output_tensor(static_cast<OutputT*>(output_data), shape);

        auto col0 = output_tensor.template Slice<1>({0, 0}, {matx::matxEnd, matx::matxDropDim});
        auto col2 = output_tensor.template Slice<1>({0, 2}, {matx::matxEnd, matx::matxDropDim});
        auto range_col =
            matx::range_x<OutputT>(matx::tensorShape_t<1>({static_cast<matx::index_t>(element_count)}), 0, 1);

        (col0 = range_col).run(stream.value());
        (col2 = fea_len - 1).run(stream.value());
    }
};

std::shared_ptr<rmm::device_buffer> create_seg_ids(size_t row_count, size_t fea_len, trtlab::neo::TypeId output_type)
{
    auto output_dtype = DType(output_type);

    // Now create the output
    auto output =
        std::make_shared<rmm::device_buffer>(output_dtype.item_size() * row_count * 3, rmm::cuda_stream_per_thread);

    cudf::type_dispatcher(cudf::data_type{output_dtype.cudf_type_id()},
                          matx_create_seg_ids{row_count, fea_len, output->stream()},
                          output->data());

    return output;
}

struct matx_threshold
{
    size_t rows;
    size_t cols;
    rmm::cuda_stream_view stream;

    template <typename InputT, std::enable_if_t<!cudf::is_floating_point<InputT>()>* = nullptr>
    void operator()(void* input_data, void* output_data, double threshold)
    {
        throw std::invalid_argument("Unsupported conversion");
    }

    template <typename InputT, std::enable_if_t<cudf::is_floating_point<InputT>()>* = nullptr>
    void operator()(void* input_data, void* output_data, double threshold)
    {
        matx::tensorShape_t<2> input_shape({static_cast<matx::index_t>(rows), static_cast<matx::index_t>(cols)});

        // Output is always 1 column
        matx::tensorShape_t<1> output_shape({static_cast<matx::index_t>(rows)});

        matx::tensor_t<InputT, 2> input_tensor(static_cast<InputT*>(input_data), input_shape);

        // // Tmp array to hold > threshold value
        // matx::tensor_t<InputT, 1> tmp_tensor(output_shape);

        // // Calc above a threshold
        // (tmp_tensor = input_tensor >= (InputT)threshold).run(stream.value());

        // matx::tensor_t<bool, 1> output_tensor(static_cast<bool*>(output_data), output_shape);

        // // Columnwise reduction
        // matx::any(output_tensor, tmp_tensor, stream.value());

        // Tmp array to hold max value
        matx::tensor_t<InputT, 1> max_tensor(output_shape);

        // row-wise reduction
        matx::rmax(max_tensor, input_tensor, stream.value());

        matx::tensor_t<bool, 1> output_tensor(static_cast<bool*>(output_data), output_shape);

        // Convert max value to bool
        (output_tensor = max_tensor >= (InputT)threshold).run(stream.value());
    }
};

DeviceTensorView threshold(const DeviceTensorView& input, double thresh_val)
{
    auto input_dtype = DType(input.dtype());

    auto output = DeviceTensorView(input.tensor_like(DType::create<bool>(), {input.shape(0)}));

    // Now create the output 1D array of bools
    // auto output = std::make_shared<rmm::device_buffer>(
    //     sizeof(bool) * rows, input.buffer->stream(), input.buffer->memory_resource());

    cudf::type_dispatcher(cudf::data_type{input_dtype.cudf_type_id()},
                          matx_threshold{input.shape(0), input.shape(1), output.stream()},
                          input.data(),
                          output.data(),
                          thresh_val);

    return output;
}

}  // namespace morpheus
