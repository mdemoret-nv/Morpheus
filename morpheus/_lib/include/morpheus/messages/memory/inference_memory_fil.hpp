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

#include "morpheus/messages/memory/inference_memory.hpp"
#include "morpheus/objects/tensor_object.hpp"

#include <cudf/types.hpp>      // for size_type
#include <pybind11/pytypes.h>  // for object

#include <cstddef>
#include <memory>
#include <string>

namespace morpheus {
/****** Component public implementations *******************/
/****** InferenceMemoryFIL****************************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */

/**
 * This is a container class for data that needs to be submitted to the inference server for FIL category
 * usecases.
 */
class InferenceMemoryFIL : public InferenceMemory
{
  public:
    /**
     * @brief Construct a new Inference Memory FIL object
     *
     * @param count : Message count in inference memory object
     * @param input__0 : Inference input
     * @param seq_ids : Ids used to index from an inference input to a message. Necessary since there can be more
     * inference inputs than messages (i.e., if some messages get broken into multiple inference requests)
     */
    InferenceMemoryFIL(size_t count, TensorObject input__0, TensorObject seq_ids);

    /**
     * @brief Returns the 'input__0' tensor, throws a `std::runtime_error` if it does not exist
     *
     * @throw std::runtime_error
     * @return const TensorObject&
     */
    const TensorObject& get_input__0() const;

    /**
     * @brief Returns the 'seq_ids' tensor, throws a `std::runtime_error` if it does not exist
     *
     * @throw std::runtime_error
     * @return const TensorObject&
     */
    const TensorObject& get_seq_ids() const;

    /**
     * @brief Sets a tensor named 'input__0'
     *
     * @param input_ids
     * @throw std::runtime_error
     * @throw std::runtime_error
     */
    void set_input__0(TensorObject input_ids);

    /**
     * @brief Sets a tensor named 'seq_ids'
     *
     * @param seq_ids
     * @throw std::runtime_error
     */
    void set_seq_ids(TensorObject seq_ids);
};

/****** InferenceMemoryFILInterfaceProxy *************************/
#pragma GCC visibility push(default)
/**
 * @brief Interface proxy, used to insulate python bindings
 */
struct InferenceMemoryFILInterfaceProxy
{
    /**
     * @brief Create and initialize an InferenceMemoryFIL object, and return a shared pointer to the result
     *
     * @param count : Message count in inference memory object
     * @param input__0 : Inference input
     * @param seq_ids : Ids used to index from an inference input to a message. Necessary since there can be more
     * inference inputs than messages (i.e., if some messages get broken into multiple inference requests)
     * @return std::shared_ptr<InferenceMemoryFIL>
     */
    static std::shared_ptr<InferenceMemoryFIL> init(cudf::size_type count,
                                                    pybind11::object input__0,
                                                    pybind11::object seq_ids);

    /**
     * Get messages count in the inference memory instance
     *
     * @param self
     * @return std::size_t
     */
    static std::size_t count(InferenceMemoryFIL& self);

    /**
     * Return the requested tensor for a given name
     *
     * @param self
     * @param name Tensor name
     * @return TensorObject
     */
    static TensorObject get_tensor(InferenceMemoryFIL& self, const std::string& name);

    /**
     * @brief Returns the 'input__0' as cupy array
     *
     * @param self
     * @return pybind11::object
     */
    static pybind11::object get_input__0(InferenceMemoryFIL& self);

    /**
     * @brief Sets a tensor named 'input__0'
     *
     * @param self
     * @param cupy_values
     */
    static void set_input__0(InferenceMemoryFIL& self, pybind11::object cupy_values);

    /**
     * @brief Returns the 'seq_ids' as a cupy array
     *
     * @param self
     * @return pybind11::object
     */
    static pybind11::object get_seq_ids(InferenceMemoryFIL& self);

    /**
     * @brief Sets a tensor named 'seq_ids'
     *
     * @param self
     * @param cupy_values
     */
    static void set_seq_ids(InferenceMemoryFIL& self, pybind11::object cupy_values);
};
#pragma GCC visibility pop

/** @} */  // end of group
}  // namespace morpheus
