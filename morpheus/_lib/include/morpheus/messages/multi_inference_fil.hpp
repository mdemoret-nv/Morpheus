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

#pragma once

#include "morpheus/messages/memory/inference_memory.hpp"  // for InferenceMemory
#include "morpheus/messages/meta.hpp"                     // for MessageMeta
#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/objects/tensor_object.hpp"

#include <cudf/types.hpp>

#include <cstddef>  // for size_t
#include <memory>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiInferenceFILMessage****************************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */

/**
 * A stronger typed version of `MultiInferenceMessage` that is used for FIL workloads. Helps ensure the
 * proper inputs are set and eases debugging.
 *
 */
#pragma GCC visibility push(default)
class MultiInferenceFILMessage : public MultiInferenceMessage
{
  public:
    /**
     * @brief Construct a new Multi Inference FIL Message object
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Holds the generic tensor data in cupy arrays that will be used for inference stages
     * @param offset Message offset in inference memory object
     * @param count Message count in inference memory object
     */
    MultiInferenceFILMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                             size_t mess_offset,
                             size_t mess_count,
                             std::shared_ptr<morpheus::InferenceMemory> memory,
                             size_t offset,
                             size_t count);

    /**
     * @brief Returns the 'input__0' tensor, throws a `std::runtime_error` if it does not exist
     *
     * @param name
     * @return const TensorObject
     */
    const TensorObject get_input__0() const;

    /**
     * @brief Sets a tensor named 'input__0'
     *
     * @param input__0
     */
    void set_input__0(const TensorObject& input__0);

    /**
     * @brief Returns the 'seq_ids' tensor, throws a `std::runtime_error` if it does not exist
     *
     * @param name
     * @return const TensorObject
     */
    const TensorObject get_seq_ids() const;

    /**
     * @brief Sets a tensor named 'seq_ids'
     *
     * @param seq_ids
     */
    void set_seq_ids(const TensorObject& seq_ids);
};

/****** MultiInferenceFILMessageInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MultiInferenceFILMessageInterfaceProxy
{
    /**
     * @brief Create and initialize a MultiInferenceFILMessage, and return a shared pointer to the result
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Holds the generic tensor data in cupy arrays that will be used for inference stages
     * @param offset Message offset in inference memory object
     * @param count Message count in inference memory object
     * @return std::shared_ptr<MultiInferenceFILMessage>
     */
    static std::shared_ptr<MultiInferenceFILMessage> init(std::shared_ptr<MessageMeta> meta,
                                                          cudf::size_type mess_offset,
                                                          cudf::size_type mess_count,
                                                          std::shared_ptr<InferenceMemory> memory,
                                                          cudf::size_type offset,
                                                          cudf::size_type count);

    /**
     * @brief Returns a shared pointer of a inference memory object
     *
     * @param self
     * @return std::shared_ptr<morpheus::InferenceMemory>
     */
    static std::shared_ptr<morpheus::InferenceMemory> memory(MultiInferenceFILMessage& self);

    /**
     * @brief Message offset in inference memory object
     *
     * @param self
     * @return std::size_t
     */
    static std::size_t offset(MultiInferenceFILMessage& self);

    /**
     * @brief Message count in inference memory object
     *
     * @param self
     * @return std::size_t
     */
    static std::size_t count(MultiInferenceFILMessage& self);
};
#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
