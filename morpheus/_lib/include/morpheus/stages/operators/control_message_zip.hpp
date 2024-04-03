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

#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <pymrc/nodes/zip.hpp>
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

class MORPHEUS_EXPORT ControlMessageDynamicZip
  : public mrc::pymrc::PythonDynamicZipComponent<size_t, std::shared_ptr<ControlMessage>>
{
    using base_t = mrc::pymrc::PythonDynamicZipComponent<size_t, std::shared_ptr<ControlMessage>>;
    using key_t  = size_t;

  public:
    ControlMessageDynamicZip(size_t max_outstanding);
};

/****** AddClassificationStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT ControlMessageDynamicZipInterfaceProxy
{
    static std::shared_ptr<mrc::segment::Object<ControlMessageDynamicZip>> init(mrc::segment::Builder& builder,
                                                                                const std::string& name,
                                                                                size_t max_outstanding);
};
/** @} */  // end of group
}  // namespace morpheus
