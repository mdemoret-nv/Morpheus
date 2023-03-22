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

#include "morpheus/messages/memory/tensor_memory.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/messages/multi_tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/types.hpp"  // for TensorIndex

#include <pybind11/pytypes.h>

#include <memory>
#include <string>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiResponseMessage****************************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */

/**
 * This class is used to get or set the inference output from message containers derived
 * from ResponseMemory.
 *
 */
#pragma GCC visibility push(default)
class MultiResponseMessage : public DerivedMultiMessage<MultiResponseMessage, MultiTensorMessage>
{
  public:
    /**
     * @brief Default copy constructor
     */
    MultiResponseMessage(const MultiResponseMessage& other) = default;

    /**
     * @brief Construct a new Multi Response Message object
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Shared pointer of a tensor memory
     * @param offset Message offset in inference memory instance
     * @param count Message count in inference memory instance
     * @param id_tensor_name Name of the tensor that correlates tensor rows to message IDs
     * @param probs_tensor_name Name of the tensor that holds output probabilities
     */
    MultiResponseMessage(std::shared_ptr<MessageMeta> meta,
                         TensorIndex mess_offset              = 0,
                         TensorIndex mess_count               = -1,
                         std::shared_ptr<TensorMemory> memory = nullptr,
                         TensorIndex offset                   = 0,
                         TensorIndex count                    = -1,
                         std::string id_tensor_name           = "seq_ids",
                         std::string probs_tensor_name        = "probs");

    std::string probs_tensor_name;

    /**
     * @brief Returns the output tensor with the given name.
     *
     * @param name
     * @return const TensorObject
     * @throws std::runtime_error If no tensor matching `name` exists
     */
    const TensorObject get_output(const std::string& name) const;

    /**
     * @brief Returns the output tensor with the given name.
     *
     * @param name
     * @return TensorObject
     * @throws std::runtime_error If no tensor matching `name` exists
     */
    TensorObject get_output(const std::string& name);

    /**
     * @brief Update the value of a given output tensor. The tensor must already exist, otherwise this will halt on a
     * fatal error.
     *
     * @param name
     * @param value
     */
    void set_output(const std::string& name, const TensorObject& value);

    /**
     * @brief Get the tensor that holds output probabilities. Equivalent to `get_tensor(probs_tensor_name)`
     *
     * @return const TensorObject
     */
    TensorObject get_probs_tensor() const;
};

/****** MultiResponseMessageInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MultiResponseMessageInterfaceProxy : public MultiTensorMessageInterfaceProxy
{
    /**
     * @brief Create and initialize a MultiResponseMessage, and return a shared pointer to the result
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Shared pointer of a tensor memory
     * @param offset Message offset in inference memory instance
     * @param count Message count in inference memory instance
     * @param id_tensor_name Name of the tensor that correlates tensor rows to message IDs
     * @param probs_tensor_name Name of the tensor that holds output probabilities
     * @return std::shared_ptr<MultiResponseMessage>
     */
    static std::shared_ptr<MultiResponseMessage> init(std::shared_ptr<MessageMeta> meta,
                                                      TensorIndex mess_offset,
                                                      TensorIndex mess_count,
                                                      std::shared_ptr<TensorMemory> memory,
                                                      TensorIndex offset,
                                                      TensorIndex count,
                                                      std::string id_tensor_name,
                                                      std::string probs_tensor_name);

    /**
     * @brief Gets the `probs_tensor_name` property
     *
     * @param self
     * @return std::string Name of `probs_tensor_name`
     */
    static std::string probs_tensor_name_getter(MultiResponseMessage& self);

    /**
     * @brief Sets the `probs_tensor_name` property
     *
     * @param self
     * @param probs_tensor_name New name of `probs_tensor_name` property
     */
    static void probs_tensor_name_setter(MultiResponseMessage& self, std::string probs_tensor_name);

    /**
     * @brief Returns the output tensor for a given name
     *
     * @param self
     * @param name : Tensor name
     * @return pybind11::object
     * @throws pybind11::key_error When no matching tensor exists.
     */
    static pybind11::object get_output(MultiResponseMessage& self, const std::string& name);

    /**
     * @brief Get the tensor that holds output probabilities. Equivalent to `get_tensor(probs_tensor_name)`
     *
     * @param self
     * @return pybind11::object A cupy.ndarray object
     */
    static pybind11::object get_probs_tensor(MultiResponseMessage& self);
};
#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
