/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/export.h"
#include "morpheus/messages/control.hpp"
#include "morpheus/stages/add_scores_stage_base.hpp"

#include <mrc/node/operators/router.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <sys/types.h>

#include <atomic>
#include <cstddef>  // for size_t
#include <map>
#include <memory>
#include <string>

namespace morpheus {

/****** Component public implementations *******************/
/****** AddClassificationStage********************************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

/**
 * @brief Add detected classifications to each message. Classification labels based on probabilities calculated in
 * inference stage. Label indexes will be looked up in the idx2label property.
 */
class MORPHEUS_EXPORT ControlMessageRouter : public mrc::node::Router<size_t, std::shared_ptr<ControlMessage>>
{
    using key_t = size_t;

  public:
    ControlMessageRouter();

  protected:
    key_t determine_key_for_value(const input_data_t& t) override
    {
        // Save off the next key index
        auto next_key = m_next_key++;

        // Save off the keys
        auto saved_keys = this->edge_connection_keys();

        return saved_keys[next_key % saved_keys.size()];
    }

  private:
    std::atomic_size_t m_next_key{0};
};

/****** AddClassificationStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT ControlMessageRouterInterfaceProxy
{
    static std::shared_ptr<mrc::segment::Object<ControlMessageRouter>> init(mrc::segment::Builder& builder,
                                                                            const std::string& name);
};
/** @} */  // end of group
}  // namespace morpheus
