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

#include "morpheus/stages/add_scores.hpp"

#include "morpheus/stages/add_scores_stage_base.hpp"

#include <cstddef>  // for size_t
#include <map>
#include <memory>
#include <optional>
#include <utility>  // for move
// IWYU thinks we need __alloc_traits<>::value_type for vector assignments
// IWYU pragma: no_include <ext/alloc_traits.h>

namespace morpheus {

// Component public implementations
// ************ AddScoresStage **************************** //
AddScoresStage::AddScoresStage(std::map<std::size_t, std::string> idx2label) :
  AddScoresStageBase(std::move(idx2label), std::nullopt)
{}

// ************ AddScoresStageInterfaceProxy ************* //
std::shared_ptr<mrc::segment::Object<AddScoresStage>> AddScoresStageInterfaceProxy::init(
    mrc::segment::Builder& builder, const std::string& name, std::map<std::size_t, std::string> idx2label)
{
    return builder.construct_object<AddScoresStage>(name, std::move(idx2label));
}

}  // namespace morpheus
