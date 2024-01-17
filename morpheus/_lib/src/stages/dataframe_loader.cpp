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

#include "morpheus/stages/dataframe_loader.hpp"

#include "mrc/node/rx_sink_base.hpp"
#include "mrc/node/rx_source_base.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/segment/object.hpp"
#include "mrc/types.hpp"

#include "morpheus/io/deserializers.hpp"
#include "morpheus/objects/file_types.hpp"
#include "morpheus/types.hpp"
#include "morpheus/utilities/python_util.hpp"
#include "morpheus/utilities/string_util.hpp"

#include <glog/logging.h>
#include <mrc/segment/builder.hpp>
#include <pyerrors.h>
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>

#include <algorithm>  // for min
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <utility>

namespace morpheus {

namespace fs = std::filesystem;

// Component public implementations
// ************ DeserializationStage **************************** //
DataFrameLoaderStage::DataFrameLoaderStage() : PythonNode(base_t::op_factory_from_sub_fn(build_operator())) {}

DataFrameLoaderStage::subscribe_fn_t DataFrameLoaderStage::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t filename) {
                if (!fs::exists(filename))
                {
                    LOG(ERROR) << MORPHEUS_CONCAT_STR("Cannot load DataFrame. File does not exist: " << filename);
                    return;
                }

                // Load the DataFrame
                auto data_table     = load_table_from_file(filename, FileTypes::Auto, true);
                int index_col_count = prepare_df_index(data_table);

                // Next, create the message metadata. This gets reused for repeats
                // When index_col_count is 0 this will cause a new range index to be created
                auto meta = MessageMeta::create_from_cpp(std::move(data_table), index_col_count);

                output.on_next(std::move(meta));
            },
            [&](std::exception_ptr error_ptr) {
                output.on_error(error_ptr);
            },
            [&]() {
                output.on_completed();
            }));
    };
}

// ************ DeserializationStageInterfaceProxy ************* //
std::shared_ptr<mrc::segment::Object<DataFrameLoaderStage>> DataFrameLoaderStageInterfaceProxy::init(
    mrc::segment::Builder& builder, const std::string& name)
{
    auto stage = builder.construct_object<DataFrameLoaderStage>(name);

    return stage;
}
}  // namespace morpheus
