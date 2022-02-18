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

#pragma once

#include <cstddef>
#include <memory>

// #include <cudf/types.hpp>
#include <rmm/device_buffer.hpp>
#include <trtlab/neo/util/type_utils.hpp>

namespace trtlab::neo {
class TensorObject;
}

namespace morpheus {

struct DeviceTensorView;

// Simple object that just holds 4 things: element count, element dtype, device_buffer, and bytes_offset
struct DevMemInfo
{
    // Number of elements in the buffer
    size_t element_count;
    // Type of elements in the buffer
    trtlab::neo::TypeId type_id;
    // Buffer of data
    std::shared_ptr<rmm::device_buffer> buffer;
    // Offset from head of data in bytes
    size_t offset;

    void* data() const
    {
        return static_cast<uint8_t*>(buffer->data()) + offset;
    }
};

// Convert one device_buffer type to another
std::shared_ptr<rmm::device_buffer> cast(const DevMemInfo& input, trtlab::neo::TypeId output_type);

// Calculate logits on device_buffer
std::shared_ptr<rmm::device_buffer> logits(const DevMemInfo& input);

// Perform transpose
std::shared_ptr<rmm::device_buffer> transpose(const DevMemInfo& input, size_t rows, size_t cols);

// Builds an Nx3 segment ID matrix
std::shared_ptr<rmm::device_buffer> create_seg_ids(size_t row_count, size_t fea_len, trtlab::neo::TypeId output_type);

// Return a Nx1 array of boolean where x[i,j] >= thresh_val
std::shared_ptr<rmm::device_buffer> threshold(const DevMemInfo& input, size_t rows, size_t cols, double thresh_val);

DeviceTensorView threshold(const DeviceTensorView& input, double thresh_val);

}  // namespace morpheus
