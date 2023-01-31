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
#include "morpheus/objects/tensor_object.hpp"  // for TensorObject

#include <pybind11/pytypes.h>

#include <cstddef>  // for size_t
#include <string>

namespace morpheus {
/****** Component public implementations *******************/
/****** ResponseMemory****************************************/
/**
 * @addtogroup messages
 * @{
 * @file
 */

/**
 * @brief Output memory block holding the results of inference
 *
 */
class ResponseMemory : public TensorMemory
{
  public:
    /**
     * @brief Construct a new Response Memory object
     *
     * @param count
     */
    ResponseMemory(size_t count);
    /**
     * @brief Construct a new Response Memory object
     *
     * @param count
     * @param tensors
     */
    ResponseMemory(size_t count, tensor_map_t &&tensors);

    /**
     * @brief Checks if a tensor named `name` exists in `tensors`
     *
     * @param name
     * @return true
     * @return false
     */
    bool has_output(const std::string &name) const;
};

/****** ResponseMemoryInterfaceProxy *************************/
#pragma GCC visibility push(default)
/**
 * @brief Interface proxy, used to insulate python bindings.
 *
 */
struct ResponseMemoryInterfaceProxy
{
    /**
     * @brief Get the output object
     *
     * @param self
     * @param name
     * @return pybind11::object
     */
    static pybind11::object get_output(ResponseMemory &self, const std::string &name);

    /**
     * @brief Get the output tensor object
     *
     * @param self
     * @param name
     * @return TensorObject
     */
    static TensorObject get_output_tensor(ResponseMemory &self, const std::string &name);
};
#pragma GCC visibility pop

/** @} */  // end of group
}  // namespace morpheus
