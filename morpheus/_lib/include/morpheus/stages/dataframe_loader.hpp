/*
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

#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/types.hpp"  // for TensorIndex

#include <boost/fiber/context.hpp>
#include <boost/fiber/future/future.hpp>
#include <mrc/node/rx_sink_base.hpp>
#include <mrc/node/rx_source_base.hpp>
#include <mrc/node/sink_properties.hpp>
#include <mrc/node/source_properties.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <mrc/types.hpp>
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>
// IWYU pragma: no_include "rxcpp/sources/rx-iterate.hpp"

#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace morpheus {

/**
 * @addtogroup stages
 * @{
 * @file
 */

#pragma GCC visibility push(default)

class DataFrameLoaderStage : public mrc::pymrc::PythonNode<std::string, std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::string, std::shared_ptr<MessageMeta>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    DataFrameLoaderStage();

  private:
    /**
     * TODO(Documentation)
     */
    subscribe_fn_t build_operator();
};

/****** DeserializationStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct DataFrameLoaderStageInterfaceProxy
{
    static std::shared_ptr<mrc::segment::Object<DataFrameLoaderStage>> init(mrc::segment::Builder& builder,
                                                                            const std::string& name);
};
#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
