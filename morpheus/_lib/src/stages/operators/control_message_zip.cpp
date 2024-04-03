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

#include "morpheus/stages/operators/control_message_zip.hpp"

#include "mrc/segment/builder.hpp"
#include "mrc/segment/object.hpp"

#include <memory>

namespace morpheus {

// Component public implementations
// ************ AddClassificationStage **************************** //
ControlMessageDynamicZip::ControlMessageDynamicZip(size_t max_outstanding) : base_t(max_outstanding) {}

// ************ AddClassificationStageInterfaceProxy ************* //
std::shared_ptr<mrc::segment::Object<ControlMessageDynamicZip>> ControlMessageDynamicZipInterfaceProxy::init(
    mrc::segment::Builder& builder, const std::string& name, size_t max_outstanding)
{
    return builder.construct_object<ControlMessageDynamicZip>(name, max_outstanding);
}

}  // namespace morpheus
