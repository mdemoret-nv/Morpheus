/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pybind11/pytypes.h>
#include <cstddef>
#include <morpheus/messages/multi.hpp>

#include <pysrf/node.hpp>
#include <srf/segment/builder.hpp>

#include <memory>
#include <string>
#include <utility>

namespace morpheus {

#pragma GCC visibility push(default)

/****** Component public implementations *******************/
/****** MonitorStage********************************/
/**
 * TODO(Documentation)
 */

class MonitorStageBase
{
  public:
    MonitorStageBase(std::string description, float smoothing, std::string unit, bool delayed_start);

    float get_throughput() const;

  protected:
    std::string m_description{"Progress"};
    float m_smoothing{0.05f};
    std::string m_unit{"messages"};
    bool m_delayed_start{false};
};

template <typename T>
class MonitorStage : public MonitorStageBase, public srf::pysrf::PythonNode<T, T>
{
  public:
    using base_t = srf::pysrf::PythonNode<T, T>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::stream_fn_t;
    using typename base_t::subscribe_fn_t;
    using count_fn_t = std::function<size_t(const T &)>;

    MonitorStage(std::string description, float smoothing, std::string unit, bool delayed_start, count_fn_t count_fn) :
      MonitorStageBase(description, smoothing, unit, delayed_start),
      base_t(build_operator()),
      m_count_fn(count_fn)
    {}

  private:
    stream_fn_t build_operator()
    {
        return [this](const rxcpp::observable<sink_type_t> &upstream) {
            return upstream.map([this](sink_type_t data_object) {
                size_t message_count = m_count_fn(data_object);

                // Return unchanged
                return data_object;
            });
        };
    }

    count_fn_t m_count_fn;
};

/****** MonitorStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MonitorStageInterfaceProxy
{
    /**
     * @brief Create and initialize a MonitorStage, and return the result.
     */
    static std::shared_ptr<srf::segment::Object<MonitorStageBase>> init(srf::segment::Builder &builder,
                                                                        const std::string &name,
                                                                        pybind11::type input_type,
                                                                        std::string description,
                                                                        float smoothing,
                                                                        std::string unit,
                                                                        bool delayed_start);
};

#pragma GCC visibility pop
}  // namespace morpheus
