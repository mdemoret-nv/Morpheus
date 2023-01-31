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

#include "morpheus/stages/file_source.hpp"

#include "morpheus/io/deserializers.hpp"

#include <cudf/column/column.hpp>  // for column
#include <cudf/io/csv.hpp>
#include <cudf/io/json.hpp>
#include <cudf/scalar/scalar.hpp>  // for string_scalar
#include <cudf/strings/replace.hpp>
#include <cudf/strings/strings_column_view.hpp>  // for strings_column_view
#include <cudf/table/table.hpp>                  // for table
#include <cudf/types.hpp>
#include <glog/logging.h>
#include <mrc/segment/builder.hpp>
#include <pybind11/cast.h>  // for object_api::operator()
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // for str_attr_accessor
#include <pybind11/pytypes.h>   // for pybind11::int_

#include <algorithm>  // for find
#include <cstddef>    // for size_t
#include <filesystem>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>  // for runtime_error
#include <utility>
// IWYU thinks we need __alloc_traits<>::value_type for vector assignments
// IWYU pragma: no_include <ext/alloc_traits.h>

namespace morpheus {
// Component public implementations
// ************ FileSourceStage ************* //
FileSourceStage::FileSourceStage(std::string filename, int repeat) :
  PythonSource(build()),
  m_filename(std::move(filename)),
  m_repeat(repeat)
{}

FileSourceStage::subscriber_fn_t FileSourceStage::build()
{
    return [this](rxcpp::subscriber<source_type_t> output) {
        auto data_table     = load_table_from_file(m_filename);
        int index_col_count = get_index_col_count(data_table);

        // Next, create the message metadata. This gets reused for repeats
        // When index_col_count is 0 this will cause a new range index to be created
        auto meta = MessageMeta::create_from_cpp(std::move(data_table), index_col_count);

        // next_meta stores a copy of the upcoming meta
        std::shared_ptr<MessageMeta> next_meta = nullptr;

        for (cudf::size_type repeat_idx = 0; repeat_idx < m_repeat; ++repeat_idx)
        {
            if (!output.is_subscribed())
            {
                // Grab the GIL before disposing, just in case
                pybind11::gil_scoped_acquire gil;

                // Reset meta to allow the DCHECK after the loop to pass
                meta.reset();

                break;
            }

            // Clone the meta object before pushing while we still have access to it
            if (repeat_idx + 1 < m_repeat)
            {
                // GIL must come after get_info
                pybind11::gil_scoped_acquire gil;

                // Use the copy function
                auto df = meta->get_py_table().attr("copy")();

                pybind11::int_ df_len = pybind11::len(df);

                pybind11::object index = df.attr("index");

                df.attr("index") = index + df_len;

                next_meta = MessageMeta::create_from_python(std::move(df));
            }

            DCHECK(meta) << "Cannot push null meta";

            output.on_next(std::move(meta));

            // Move next_meta into meta
            std::swap(meta, next_meta);
        }

        DCHECK(!meta) << "meta was not properly pushed";
        DCHECK(!next_meta) << "next_meta was not properly pushed";

        output.on_completed();
    };
}

// ************ FileSourceStageInterfaceProxy ************ //
std::shared_ptr<mrc::segment::Object<FileSourceStage>> FileSourceStageInterfaceProxy::init(
    mrc::segment::Builder& builder, const std::string& name, std::string filename, int repeat)
{
    auto stage = builder.construct_object<FileSourceStage>(name, filename, repeat);

    return stage;
}
}  // namespace morpheus
