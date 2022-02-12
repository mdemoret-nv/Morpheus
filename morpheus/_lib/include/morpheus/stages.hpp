/**
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <cudf/strings/strings_column_view.hpp>
#include <functional>
#include <memory>
#include <numeric>
#include <regex>
#include <utility>

#include "morpheus/common.hpp"
#include "morpheus/messages.hpp"
#include "morpheus/type_utils.hpp"

#include <http_client.h>
#include <pybind11/gil.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <thrust/iterator/constant_iterator.h>
#include <nlohmann/json.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/json.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <nvtext/subword_tokenize.hpp>

#include "pyneo/node.hpp"
#include "trtlab/neo/core/segment_object.hpp"
#include "trtlab/neo/util/type_utils.hpp"

namespace morpheus {

using namespace pybind11::literals;

namespace fs = std::filesystem;
namespace tc = triton::client;
using json   = nlohmann::json;

class FileSourceStage : public pyneo::PythonSource<std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = pyneo::PythonSource<std::shared_ptr<MessageMeta>>;
    using base_t::source_type_t;

    FileSourceStage(const neo::Segment& parent, const std::string& name, std::string filename, int repeat = 1) :
      neo::SegmentObject(parent, name),
      base_t(parent, name),
      m_filename(std::move(filename)),
      m_repeat(repeat)
    {
        this->set_source_observable(neo::Observable<source_type_t>([this](neo::Subscriber<source_type_t>& sub) {
            auto data_table = this->load_table();

            // Using 0 will default to creating a new range index
            int index_col_count = 0;

            // Check if we have a first column with INT64 data type
            if (data_table.metadata.column_names.size() >= 1 &&
                data_table.tbl->get_column(0).type().id() == cudf::type_id::INT64)
            {
                std::regex index_regex(R"((unnamed: 0|id))",
                                       std::regex_constants::ECMAScript | std::regex_constants::icase);

                // Get the column name
                auto col_name = data_table.metadata.column_names[0];

                // Check it against some common terms
                if (std::regex_search(col_name, index_regex))
                {
                    // Also, if its the hideous 'Unnamed: 0', then just use an empty string
                    if (col_name == "Unnamed: 0")
                    {
                        data_table.metadata.column_names[0] = "";
                    }

                    index_col_count = 1;
                }
            }

            // Next, create the message metadata. This gets reused for repeats
            auto meta = MessageMeta::create_from_cpp(std::move(data_table), index_col_count);

            // Always push at least 1
            sub.on_next(meta);

            for (cudf::size_type repeat_idx = 1; repeat_idx < m_repeat; ++repeat_idx)
            {
                // Clone the previous meta object
                {
                    py::gil_scoped_acquire gil;

                    // Use the copy function
                    auto df = meta->get_py_table().attr("copy")();

                    py::int_ df_len = py::len(df);

                    py::object index = df.attr("index");

                    df.attr("index") = index + df_len;

                    meta = MessageMeta::create_from_python(std::move(df));
                }

                sub.on_next(meta);
            }

            sub.on_completed();
        }));
    }

  private:
    cudf::io::table_with_metadata load_table()
    {
        auto file_path = fs::path(m_filename);

        if (file_path.extension() == ".json" || file_path.extension() == ".jsonlines")
        {
            // First, load the file into json
            auto options = cudf::io::json_reader_options::builder(cudf::io::source_info{m_filename}).lines(true);

            auto tbl = cudf::io::read_json(options.build());

            auto found = std::find(tbl.metadata.column_names.begin(), tbl.metadata.column_names.end(), "data");

            if (found == tbl.metadata.column_names.end())
                return tbl;

            // Super ugly but cudf cant handle newlines and add extra escapes. So we need to convert
            // \\n -> \n
            // \\/ -> \/
            auto columns = tbl.tbl->release();

            size_t idx = found - tbl.metadata.column_names.begin();

            auto updated_data = cudf::strings::replace(
                cudf::strings_column_view{columns[idx]->view()}, cudf::string_scalar("\\n"), cudf::string_scalar("\n"));

            updated_data = cudf::strings::replace(
                cudf::strings_column_view{updated_data->view()}, cudf::string_scalar("\\/"), cudf::string_scalar("/"));

            columns[idx] = std::move(updated_data);

            tbl.tbl = std::move(std::make_unique<cudf::table>(std::move(columns)));

            return tbl;
        }
        else if (file_path.extension() == ".csv")
        {
            auto options = cudf::io::csv_reader_options::builder(cudf::io::source_info{m_filename});

            return cudf::io::read_csv(options.build());
        }
        else
        {
            LOG(FATAL) << "Unknown extension for file: " << m_filename;
            throw std::runtime_error("Unknown extension");
        }
    }

    std::string m_filename;
    int m_repeat{1};
};

class DeserializeStage : public pyneo::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<MultiMessage>>
{
  public:
    using base_t = pyneo::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<MultiMessage>>;
    using base_t::operator_fn_t;
    using base_t::reader_type_t;
    using base_t::writer_type_t;

    DeserializeStage(const neo::Segment& parent, const std::string& name, size_t batch_size) :
      neo::SegmentObject(parent, name),
      PythonNode(parent, name, build_operator()),
      m_batch_size(batch_size)
    {}

  private:
    operator_fn_t build_operator()
    {
        return [this](neo::Observable<reader_type_t>& input, neo::Subscriber<writer_type_t>& output) {
            return input.subscribe(neo::make_observer<reader_type_t>(
                [this, &output](reader_type_t&& x) {
                    // Make one large MultiMessage
                    auto full_message = std::make_shared<MultiMessage>(x, 0, x->count());

                    // Loop over the MessageMeta and create sub-batches
                    for (size_t i = 0; i < x->count(); i += this->m_batch_size)
                    {
                        auto next = full_message->get_slice(i, std::min(i + this->m_batch_size, x->count()));

                        output.on_next(std::move(next));
                    }
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
    }

    size_t m_batch_size;
};

class PreprocessNLPStage
  : public pyneo::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>>
{
  public:
    using base_t = pyneo::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>>;
    using base_t::operator_fn_t;
    using base_t::reader_type_t;
    using base_t::writer_type_t;

    PreprocessNLPStage(const neo::Segment& parent,
                       const std::string& name,
                       std::string vocab_hash_file,
                       uint32_t sequence_length,
                       bool truncation,
                       bool do_lower_case,
                       bool add_special_token,
                       int stride = -1) :
      neo::SegmentObject(parent, name),
      PythonNode(parent, name, build_operator()),
      m_vocab_hash_file(std::move(vocab_hash_file)),
      m_sequence_length(sequence_length),
      m_truncation(truncation),
      m_do_lower_case(do_lower_case),
      m_add_special_token(add_special_token),
      m_stride(stride)
    {}

  private:
    operator_fn_t build_operator()
    {
        return [this](neo::Observable<reader_type_t>& input, neo::Subscriber<writer_type_t>& output) {
            uint32_t stride = m_stride;

            // Auto calc stride to be 75% of sequence length
            if (stride < 0)
            {
                stride = m_sequence_length / 2;
                stride = stride + stride / 2;
            }

            return input.subscribe(neo::make_observer<reader_type_t>(
                [this, stride, &output](reader_type_t&& x) {
                    // Convert to string view
                    auto string_col = cudf::strings_column_view{x->get_meta("data").get_column(0)};

                    // Create the hashed vocab
                    thread_local std::unique_ptr<nvtext::hashed_vocabulary> vocab =
                        nvtext::load_vocabulary_file(this->m_vocab_hash_file);

                    // Perform the tokenizer
                    auto token_results = nvtext::subword_tokenize(string_col,
                                                                  *vocab,
                                                                  this->m_sequence_length,
                                                                  stride,
                                                                  this->m_do_lower_case,
                                                                  this->m_truncation,
                                                                  string_col.size() * 2);

                    // Build the results
                    auto memory = std::make_shared<InferenceMemory>(token_results.nrows_tensor);

                    int32_t length = token_results.tensor_token_ids->size() / token_results.sequence_length;
                    auto input_ids_released =
                        cudf::cast(token_results.tensor_token_ids->view(), cudf::data_type(cudf::type_id::INT32))
                            ->release();

                    memory->inputs["input_ids"] = std::move(Tensor::create(
                        std::move(input_ids_released.data),
                        DType::create<int32_t>(),
                        std::vector<neo::TensorIndex>{length, static_cast<int>(token_results.sequence_length)},
                        std::vector<neo::TensorIndex>{},
                        0));

                    length = token_results.tensor_attention_mask->size() / token_results.sequence_length;
                    auto input_mask_released =
                        cudf::cast(token_results.tensor_attention_mask->view(), cudf::data_type(cudf::type_id::INT32))
                            ->release();
                    memory->inputs["input_mask"] = std::move(Tensor::create(
                        std::move(input_mask_released.data),
                        DType::create<int32_t>(),
                        std::vector<neo::TensorIndex>{length, static_cast<int>(token_results.sequence_length)},
                        std::vector<neo::TensorIndex>{},
                        0));

                    length = token_results.tensor_metadata->size() / 3;
                    auto seq_ids_released =
                        cudf::cast(token_results.tensor_metadata->view(), cudf::data_type(cudf::type_id::INT32))
                            ->release();
                    memory->inputs["seq_ids"] =
                        std::move(Tensor::create(std::move(seq_ids_released.data),
                                                 DType::create<int32_t>(),
                                                 std::vector<neo::TensorIndex>{length, static_cast<int32_t>(3)},
                                                 std::vector<neo::TensorIndex>{},
                                                 0));

                    auto next = std::make_shared<MultiInferenceMessage>(
                        x->meta, x->mess_offset, x->mess_count, std::move(memory), 0, memory->count);

                    output.on_next(std::move(next));
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
    }

    std::string m_vocab_hash_file;
    uint32_t m_sequence_length;
    bool m_truncation;
    bool m_do_lower_case;
    bool m_add_special_token;
    int m_stride{-1};
};

class PreprocessFILStage
  : public pyneo::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>>
{
  public:
    using base_t = pyneo::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>>;
    using base_t::operator_fn_t;
    using base_t::reader_type_t;
    using base_t::writer_type_t;

    PreprocessFILStage(const neo::Segment& parent, const std::string& name) :
      neo::SegmentObject(parent, name),
      PythonNode(parent, name, build_operator())
    {}

  private:
    operator_fn_t build_operator()
    {
        return [this](neo::Observable<reader_type_t>& input, neo::Subscriber<writer_type_t>& output) {
            return input.subscribe(neo::make_observer<reader_type_t>(
                [this, &output](reader_type_t&& x) {
                    std::vector<std::string> fea_cols = {
                        "nvidia_smi_log.gpu.pci.tx_util",
                        "nvidia_smi_log.gpu.pci.rx_util",
                        "nvidia_smi_log.gpu.fb_memory_usage.used",
                        "nvidia_smi_log.gpu.fb_memory_usage.free",
                        "nvidia_smi_log.gpu.bar1_memory_usage.total",
                        "nvidia_smi_log.gpu.bar1_memory_usage.used",
                        "nvidia_smi_log.gpu.bar1_memory_usage.free",
                        "nvidia_smi_log.gpu.utilization.gpu_util",
                        "nvidia_smi_log.gpu.utilization.memory_util",
                        "nvidia_smi_log.gpu.temperature.gpu_temp",
                        "nvidia_smi_log.gpu.temperature.gpu_temp_max_threshold",
                        "nvidia_smi_log.gpu.temperature.gpu_temp_slow_threshold",
                        "nvidia_smi_log.gpu.temperature.gpu_temp_max_gpu_threshold",
                        "nvidia_smi_log.gpu.temperature.memory_temp",
                        "nvidia_smi_log.gpu.temperature.gpu_temp_max_mem_threshold",
                        "nvidia_smi_log.gpu.power_readings.power_draw",
                        "nvidia_smi_log.gpu.clocks.graphics_clock",
                        "nvidia_smi_log.gpu.clocks.sm_clock",
                        "nvidia_smi_log.gpu.clocks.mem_clock",
                        "nvidia_smi_log.gpu.clocks.video_clock",
                        "nvidia_smi_log.gpu.applications_clocks.graphics_clock",
                        "nvidia_smi_log.gpu.applications_clocks.mem_clock",
                        "nvidia_smi_log.gpu.default_applications_clocks.graphics_clock",
                        "nvidia_smi_log.gpu.default_applications_clocks.mem_clock",
                        "nvidia_smi_log.gpu.max_clocks.graphics_clock",
                        "nvidia_smi_log.gpu.max_clocks.sm_clock",
                        "nvidia_smi_log.gpu.max_clocks.mem_clock",
                        "nvidia_smi_log.gpu.max_clocks.video_clock",
                        "nvidia_smi_log.gpu.max_customer_boost_clocks.graphics_clock",
                    };
                    // TODO(MDD): Add some sort of lock here to prevent fixing columns after they have been accessed
                    auto df_meta           = x->get_meta(fea_cols);
                    auto df_meta_col_names = df_meta.get_column_names();

                    auto packed_data = std::make_shared<rmm::device_buffer>(
                        fea_cols.size() * x->mess_count * sizeof(float), rmm::cuda_stream_per_thread);

                    std::vector<std::string> bad_cols;

                    auto df_just_features = df_meta.get_view();

                    for (size_t i = 0; i < df_meta.num_columns(); ++i)
                    {
                        if (df_just_features.column(df_meta.num_indices() + i).type().id() == cudf::type_id::STRING)
                        {
                            bad_cols.push_back(df_meta_col_names[i]);
                        }
                    }

                    // Need to ensure all string columns have been converted to numbers. This requires running a regex
                    // which is too difficult to do from C++ at this time. So grab the GIL, make the conversions, and
                    // release. This is horribly inefficient, but so is the JSON lines format for this workflow
                    if (!bad_cols.empty())
                    {
                        py::gil_scoped_acquire gil;

                        py::object df = x->meta->get_py_table();

                        std::string regex = R"((\d+))";

                        for (auto c : bad_cols)
                        {
                            df[py::str(c)] = df[py::str(c)]
                                                 .attr("str")
                                                 .attr("extract")(py::str(regex), "expand"_a = true)
                                                 .attr("astype")(py::str("float32"));
                        }

                        // Now re-get the meta
                        df_meta          = x->get_meta(fea_cols);
                        df_just_features = df_meta.get_view();
                    }

                    for (size_t i = 0; i < df_meta.num_columns(); ++i)
                    {
                        auto curr_col = df_just_features.column(df_meta.num_indices() + i);

                        auto curr_ptr = static_cast<float*>(packed_data->data()) + i * df_just_features.num_rows();

                        // Check if we are something other than float
                        if (curr_col.type().id() != cudf::type_id::FLOAT32)
                        {
                            auto float_data = cudf::cast(curr_col, cudf::data_type(cudf::type_id::FLOAT32))->release();

                            // Do the copy here before it goes out of scipe
                            NEO_CHECK_CUDA(cudaMemcpy(curr_ptr,
                                                      float_data.data->data(),
                                                      df_just_features.num_rows() * sizeof(float),
                                                      cudaMemcpyDeviceToDevice));
                        }
                        else
                        {
                            NEO_CHECK_CUDA(cudaMemcpy(curr_ptr,
                                                      curr_col.data<float>(),
                                                      df_just_features.num_rows() * sizeof(float),
                                                      cudaMemcpyDeviceToDevice));
                        }
                    }

                    // Need to do a transpose here
                    auto transposed_data =
                        transpose(DevMemInfo{x->mess_count * fea_cols.size(), neo::TypeId::FLOAT32, packed_data, 0},
                                  fea_cols.size(),
                                  x->mess_count);

                    auto input__0 = Tensor::create(transposed_data,
                                                   DType::create<float>(),
                                                   std::vector<neo::TensorIndex>{static_cast<long long>(x->mess_count),
                                                                                 static_cast<int>(fea_cols.size())},
                                                   std::vector<neo::TensorIndex>{},
                                                   0);

                    auto seg_ids = Tensor::create(
                        create_seg_ids(x->mess_count, fea_cols.size(), trtlab::neo::TypeId::UINT32),
                        DType::create<uint32_t>(),
                        std::vector<neo::TensorIndex>{static_cast<long long>(x->mess_count), static_cast<int>(3)},
                        std::vector<neo::TensorIndex>{},
                        0);

                    // Build the results
                    auto memory = std::make_shared<InferenceMemoryFIL>(x->mess_count, input__0, seg_ids);

                    auto next = std::make_shared<MultiInferenceMessage>(
                        x->meta, x->mess_offset, x->mess_count, std::move(memory), 0, memory->count);

                    output.on_next(std::move(next));
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
    }

    std::string m_vocab_file;
};

template <typename FuncT, typename SeqT>
auto foreach_map(SeqT seq, FuncT func)
{
    using value_t  = typename SeqT::value_type;
    using return_t = decltype(func(std::declval<value_t>()));

    std::vector<return_t> result{};

    std::transform(seq.begin(), seq.end(), std::back_inserter(result), func);

    return result;
}

void __checkTritonErrors(tc::Error status,
                         const std::string& methodName,
                         const std::string& filename,
                         const int& lineNumber)
{
    if (!status.IsOk())
    {
        std::string err_msg =
            CONCAT_STR("Triton Error while executing '" << methodName << "'. Error: " + status.Message() << "\n"
                                                        << filename << "(" << lineNumber << ")");
        LOG(ERROR) << err_msg;
        throw std::runtime_error(err_msg);
    }
}

#define CHECK_TRITON(method) __checkTritonErrors(method, #method, __FILE__, __LINE__);

class InferenceClientStage
  : public pyneo::PythonNode<std::shared_ptr<MultiInferenceMessage>, std::shared_ptr<MultiResponseMessage>>
{
  public:
    using base_t = pyneo::PythonNode<std::shared_ptr<MultiInferenceMessage>, std::shared_ptr<MultiResponseMessage>>;
    using base_t::operator_fn_t;
    using base_t::reader_type_t;
    using base_t::writer_type_t;

    InferenceClientStage(const neo::Segment& parent,
                         const std::string& name,
                         std::string model_name,
                         std::string server_url,
                         bool force_convert_inputs,
                         bool use_shared_memory,
                         bool needs_logits,
                         std::map<std::string, std::string> inout_mapping = {}) :
      neo::SegmentObject(parent, name),
      PythonNode(parent, name, build_operator()),
      m_model_name(std::move(model_name)),
      m_server_url(std::move(server_url)),
      m_force_convert_inputs(force_convert_inputs),
      m_use_shared_memory(use_shared_memory),
      m_needs_logits(needs_logits),
      m_inout_mapping(std::move(inout_mapping)),
      m_options(m_model_name)
    {
        // Connect with the server to setup the inputs/outputs
        this->connect_with_server();
    }

  private:
    struct TritonInOut
    {
        std::string name;
        size_t bytes;
        DType datatype;
        std::vector<int> shape;
        std::string mapped_name;
        size_t offset;
    };

    bool is_default_grpc_port(std::string& server_url)
    {
        // Check if we are the default gRPC port of 8001 and try 8000 for http client instead
        size_t colon_loc = server_url.find_last_of(':');

        if (colon_loc == -1)
        {
            return false;
        }

        // Check if the port matches 8001
        if (server_url.size() < colon_loc + 1 || server_url.substr(colon_loc + 1) != "8001")
        {
            return false;
        }

        // It matches, change to 8000
        server_url = server_url.substr(0, colon_loc) + ":8000";

        return true;
    }

    void connect_with_server()
    {
        std::string server_url = m_server_url;

        std::unique_ptr<tc::InferenceServerHttpClient> client;

        auto result = tc::InferenceServerHttpClient::Create(&client, server_url, false);

        // Now load the input/outputs for the model
        bool is_server_live = false;

        tc::Error status = client->IsServerLive(&is_server_live);

        if (!status.IsOk() && this->is_default_grpc_port(server_url))
        {
            // We are using the default gRPC port, try the default HTTP
            std::unique_ptr<tc::InferenceServerHttpClient> unique_client;

            auto result = tc::InferenceServerHttpClient::Create(&unique_client, server_url, false);

            client = std::move(unique_client);

            status = client->IsServerLive(&is_server_live);

            if (!status.IsOk())
                throw std::runtime_error(CONCAT_STR("Unable to connect to Triton at '"
                                                    << m_server_url
                                                    << "'. Check the URL and port and ensure the server is running."));
        }

        // Save this for new clients
        m_server_url = server_url;

        if (!is_server_live)
            throw std::runtime_error("Server is not live");

        bool is_server_ready = false;
        CHECK_TRITON(client->IsServerReady(&is_server_ready));

        if (!is_server_ready)
            throw std::runtime_error("Server is not ready");

        bool is_model_ready = false;
        CHECK_TRITON(client->IsModelReady(&is_model_ready, this->m_model_name));

        if (!is_model_ready)
            throw std::runtime_error("Model is not ready");

        std::string model_metadata_json;
        CHECK_TRITON(client->ModelMetadata(&model_metadata_json, this->m_model_name));

        auto model_metadata = json::parse(model_metadata_json);

        std::string model_config_json;
        CHECK_TRITON(client->ModelConfig(&model_config_json, this->m_model_name));

        auto model_config = json::parse(model_config_json);

        if (model_config.contains("max_batch_size"))
        {
            m_max_batch_size = model_config.at("max_batch_size").get<int>();
        }

        for (auto const& input : model_metadata.at("inputs"))
        {
            auto shape = input.at("shape").get<std::vector<int>>();

            auto dtype = DType::from_triton(input.at("datatype").get<std::string>());

            size_t bytes = dtype.item_size();

            for (auto& y : shape)
            {
                if (y == -1)
                {
                    y = m_max_batch_size;
                }

                bytes *= y;
            }

            std::string mapped_name = input.at("name").get<std::string>();

            if (m_inout_mapping.find(mapped_name) != m_inout_mapping.end())
            {
                mapped_name = m_inout_mapping[mapped_name];
            }

            m_model_inputs.push_back(TritonInOut{input.at("name").get<std::string>(),
                                                 bytes,
                                                 DType::from_triton(input.at("datatype").get<std::string>()),
                                                 shape,
                                                 mapped_name,
                                                 0});
        }

        for (auto const& output : model_metadata.at("outputs"))
        {
            auto shape = output.at("shape").get<std::vector<int>>();

            auto dtype = DType::from_triton(output.at("datatype").get<std::string>());

            size_t bytes = dtype.item_size();

            for (auto& y : shape)
            {
                if (y == -1)
                {
                    y = m_max_batch_size;
                }

                bytes *= y;
            }

            std::string mapped_name = output.at("name").get<std::string>();

            if (m_inout_mapping.find(mapped_name) != m_inout_mapping.end())
            {
                mapped_name = m_inout_mapping[mapped_name];
            }

            m_model_outputs.push_back(
                TritonInOut{output.at("name").get<std::string>(), bytes, dtype, shape, mapped_name, 0});
        }
    }

    operator_fn_t build_operator()
    {
        return [this](neo::Observable<reader_type_t>& input, neo::Subscriber<writer_type_t>& output) {
            std::unique_ptr<tc::InferenceServerHttpClient> client;

            CHECK_TRITON(tc::InferenceServerHttpClient::Create(&client, m_server_url, false));

            return input.subscribe(neo::make_observer<reader_type_t>(
                [this, &output, &client](reader_type_t&& x) {
                    auto reponse_memory = std::make_shared<ResponseMemory>(x->count);

                    // Create the output memory blocks
                    for (auto& model_output : m_model_outputs)
                    {
                        auto total_shape = model_output.shape;

                        // First dimension will always end up being the number of rows
                        total_shape[0] = x->count;

                        auto elem_count =
                            std::accumulate(total_shape.begin(), total_shape.end(), 1, std::multiplies<>());

                        // Create the output memory
                        auto output_buffer = std::make_shared<rmm::device_buffer>(
                            elem_count * model_output.datatype.item_size(), rmm::cuda_stream_per_thread);

                        reponse_memory->outputs[model_output.mapped_name] =
                            Tensor::create(std::move(output_buffer),
                                           model_output.datatype,
                                           std::vector<neo::TensorIndex>{static_cast<int>(total_shape[0]),
                                                                         static_cast<int>(total_shape[1])},
                                           std::vector<neo::TensorIndex>{},
                                           0);
                    }

                    // This will be the final output of all mini-batches
                    auto response = std::make_shared<MultiResponseProbsMessage>(
                        x->meta, x->mess_offset, x->mess_count, std::move(reponse_memory), 0, reponse_memory->count);

                    for (size_t i = 0; i < x->count; i += m_max_batch_size)
                    {
                        tc::InferInput* input1;

                        size_t start = i;
                        size_t stop  = std::min(i + m_max_batch_size, x->count);

                        reader_type_t mini_batch_input =
                            std::static_pointer_cast<MultiInferenceMessage>(x->get_slice(start, stop));
                        writer_type_t mini_batch_output =
                            std::static_pointer_cast<MultiResponseProbsMessage>(response->get_slice(start, stop));

                        // Iterate on the model inputs in case the model takes less than what tensors are available
                        std::vector<std::pair<std::shared_ptr<tc::InferInput>, std::vector<uint8_t>>> saved_inputs =
                            foreach_map(m_model_inputs, [this, &mini_batch_input](auto const& model_input) {
                                DCHECK(mini_batch_input->memory->has_input(model_input.mapped_name))
                                    << "Model input '" << model_input.mapped_name << "' not found in InferenceMemory";

                                auto const& inp_tensor = mini_batch_input->get_input(model_input.mapped_name);

                                // Convert to the right type. Make shallow if necessary
                                auto final_tensor = inp_tensor.as_type(model_input.datatype);

                                std::vector<uint8_t> inp_data = final_tensor.get_host_data();

                                // Test
                                tc::InferInput* inp_ptr;

                                tc::InferInput::Create(&inp_ptr,
                                                       model_input.name,
                                                       {inp_tensor.shape(0), inp_tensor.shape(1)},
                                                       model_input.datatype.triton_str());
                                std::shared_ptr<tc::InferInput> inp_shared;
                                inp_shared.reset(inp_ptr);

                                inp_ptr->AppendRaw(inp_data);

                                return std::make_pair(inp_shared, std::move(inp_data));
                            });

                        std::vector<std::shared_ptr<const tc::InferRequestedOutput>> saved_outputs =
                            foreach_map(m_model_outputs, [this](auto const& model_output) {
                                // Generate the outputs to be requested.
                                tc::InferRequestedOutput* out_ptr;

                                tc::InferRequestedOutput::Create(&out_ptr, model_output.name);
                                std::shared_ptr<const tc::InferRequestedOutput> out_shared;
                                out_shared.reset(out_ptr);

                                return out_shared;
                            });

                        std::vector<tc::InferInput*> inputs =
                            foreach_map(saved_inputs, [](auto x) { return x.first.get(); });

                        std::vector<const tc::InferRequestedOutput*> outputs =
                            foreach_map(saved_outputs, [](auto x) { return x.get(); });

                        // this->segment().resources().fiber_pool().enqueue([client, output](){});

                        tc::InferResult* results;

                        CHECK_TRITON(client->Infer(&results, m_options, inputs, outputs));

                        for (auto& model_output : m_model_outputs)
                        {
                            std::vector<int64_t> output_shape;

                            CHECK_TRITON(results->Shape(model_output.name, &output_shape));

                            // Make sure we have at least 2 dims
                            while (output_shape.size() < 2)
                            {
                                output_shape.push_back(1);
                            }

                            const uint8_t* output_ptr = nullptr;
                            size_t output_ptr_size    = 0;
                            CHECK_TRITON(results->RawData(model_output.name, &output_ptr, &output_ptr_size));

                            auto output_buffer =
                                std::make_shared<rmm::device_buffer>(output_ptr_size, rmm::cuda_stream_per_thread);

                            NEO_CHECK_CUDA(
                                cudaMemcpy(output_buffer->data(), output_ptr, output_ptr_size, cudaMemcpyHostToDevice));

                            // If we need to do logits, do that here
                            if (m_needs_logits)
                            {
                                size_t element_count =
                                    std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<>());
                                output_buffer = logits(
                                    DevMemInfo{element_count, model_output.datatype.type_id(), output_buffer, 0});
                            }

                            mini_batch_output->set_output(
                                model_output.mapped_name,
                                Tensor::create(std::move(output_buffer),
                                               model_output.datatype,
                                               std::vector<neo::TensorIndex>{static_cast<int>(output_shape[0]),
                                                                             static_cast<int>(output_shape[1])},
                                               std::vector<neo::TensorIndex>{},
                                               0));
                        }
                    }
                    output.on_next(std::move(response));
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
    }

    std::string m_model_name;
    std::string m_server_url;
    bool m_force_convert_inputs;
    bool m_use_shared_memory;
    bool m_needs_logits{true};
    std::map<std::string, std::string> m_inout_mapping;

    // Below are settings created during handshake with server
    // std::shared_ptr<tc::InferenceServerHttpClient> m_client;
    std::vector<TritonInOut> m_model_inputs;
    std::vector<TritonInOut> m_model_outputs;
    tc::InferOptions m_options;
    int m_max_batch_size{-1};
};

class FilterDetectionsStage : public pyneo::PythonNode<std::shared_ptr<MultiResponseProbsMessage>, std::shared_ptr<MultiResponseProbsMessage>>
{
  public:
    using base_t = pyneo::PythonNode<std::shared_ptr<MultiResponseProbsMessage>, std::shared_ptr<MultiResponseProbsMessage>>;
    using base_t::operator_fn_t;
    using base_t::reader_type_t;
    using base_t::writer_type_t;

    FilterDetectionsStage(const neo::Segment& parent, const std::string& name, float threshold) :
      neo::SegmentObject(parent, name),
      PythonNode(parent, name, build_operator()),
      m_threshold(threshold)
    {}

  private:
    operator_fn_t build_operator()
    {
        return [this](neo::Observable<reader_type_t>& input, neo::Subscriber<writer_type_t>& output) {
            return input.subscribe(neo::make_observer<reader_type_t>(
                [this, &output](reader_type_t&& x) {
                    const auto& probs = x->get_probs();
                    const auto& shape = probs.get_shape();

                    CHECK(probs.rank() == 1 || (shape.size() == 2 && shape[1] == 1)) << "C++ impl of the FilterDetectionsStage currently only supports single dimensional arrays";

                    std::vector<float> values(static_cast<std::size_t>(probs.count()));
                    cudaMemcpy(values.data(), probs.data(), probs.bytes(), cudaMemcpyDeviceToHost);

                    // using size() as our marker for undefined
                    std::size_t slice_start = values.size();
                    for (std::size_t i=0; i < values.size(); ++i) {
                        if (bool above_threshold = values[i] > m_threshold;
                            above_threshold && slice_start == values.size()) {

                            slice_start = i;
                        } else if (!above_threshold && slice_start != values.size()) {
                            output.on_next(x->get_slice(slice_start, i));
                            slice_start = values.size();
                        }
                    }

                    if (slice_start != values.size()) {
                        // Last elem was above the threshold
                        output.on_next(x->get_slice(slice_start, values.size()));
                    }
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
    }

    float m_threshold;
};

class AddClassificationsStage : public pyneo::PythonNode<std::shared_ptr<MultiResponseProbsMessage>, std::shared_ptr<MultiResponseProbsMessage>>
{
  public:
    using base_t = pyneo::PythonNode<std::shared_ptr<MultiResponseProbsMessage>, std::shared_ptr<MultiResponseProbsMessage>>;
    using base_t::operator_fn_t;
    using base_t::reader_type_t;
    using base_t::writer_type_t;

    AddClassificationsStage(const neo::Segment& parent, const std::string& name, float threshold, std::size_t num_class_labels, std::map<std::size_t, std::string> idx2label) :
      neo::SegmentObject(parent, name),
      PythonNode(parent, name, build_operator()),
      m_threshold(threshold),
      m_num_class_labels(num_class_labels),
      m_idx2label(std::move(idx2label))
    {CHECK(m_idx2label.size() <= m_num_class_labels) << "idx2label should represent a subset of the class_labels";}

  private:
    operator_fn_t build_operator()
    {
        return [this](neo::Observable<reader_type_t>& input, neo::Subscriber<writer_type_t>& output) {
            return input.subscribe(neo::make_observer<reader_type_t>(
                [this, &output](reader_type_t&& x) {
                    const auto& probs = x->get_probs();
                    const auto& shape = probs.get_shape();

                    CHECK(shape.size() == 2 && shape[1] == m_num_class_labels)
                        << "Label count does not match output of model. Label count: " << m_num_class_labels
                        << ", Model output: " << shape[1];

                    //CHECK(shape.size() == 2 && shape[1] == 1) << "C++ impl of the AddClassificationsStage currently only supports single dimensional arrays";

                    const std::size_t num_columns = static_cast<std::size_t>(shape[1]);
                    const std::size_t num_rows = static_cast<std::size_t>(shape[0]);

                    std::vector<float> values(probs.count());
                    cudaMemcpy(values.data(), probs.data(), probs.bytes(), cudaMemcpyDeviceToHost);

                    std::vector<bool> probs_np(num_rows);
                    for (const auto& p : m_idx2label)
                    {
                        std::size_t column = p.first;
                        for (std::size_t row=0; row < num_rows; ++row)
                        {
                            probs_np[row] = values[column + row*num_columns] > m_threshold;
                        }

                        {
                            // TODO: Figure out a way to do this without going through python
                            py::gil_scoped_acquire gil;
                            x->set_meta(p.second, py::cast(probs_np));
                        }
                    }

                    output.on_next(x);
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
    }

    float m_threshold;
    std::size_t m_num_class_labels;
    std::map<std::size_t, std::string> m_idx2label;
};

class AddScoresStage : public pyneo::PythonNode<std::shared_ptr<MultiResponseProbsMessage>, std::shared_ptr<MultiResponseProbsMessage>>
{
  public:
    using base_t = pyneo::PythonNode<std::shared_ptr<MultiResponseProbsMessage>, std::shared_ptr<MultiResponseProbsMessage>>;
    using base_t::operator_fn_t;
    using base_t::reader_type_t;
    using base_t::writer_type_t;

    AddScoresStage(const neo::Segment& parent, const std::string& name, std::size_t num_class_labels, std::map<std::size_t, std::string> idx2label) :
      neo::SegmentObject(parent, name),
      PythonNode(parent, name, build_operator()),
      m_num_class_labels(num_class_labels),
      m_idx2label(std::move(idx2label))
    {CHECK(m_idx2label.size() <= m_num_class_labels) << "idx2label should represent a subset of the class_labels";}

  private:
    operator_fn_t build_operator()
    {
        return [this](neo::Observable<reader_type_t>& input, neo::Subscriber<writer_type_t>& output) {
            return input.subscribe(neo::make_observer<reader_type_t>(
                [this, &output](reader_type_t&& x) {
                    const auto& probs = x->get_probs();
                    const auto& shape = probs.get_shape();

                    CHECK(shape.size() > 1 && shape[1] == m_num_class_labels)
                        << "Label count does not match output of model. Label count: " << m_num_class_labels
                        << ", Model output: " << shape[1];

                    CHECK(shape.size() == 2 && shape[1] == 1) << "C++ impl of the AddScoresStage currently only supports single dimensional arrays";

                    std::vector<float> values(static_cast<std::size_t>(probs.count()));
                    cudaMemcpy(values.data(), probs.data(), probs.bytes(), cudaMemcpyDeviceToHost);

                    {
                        // TODO: Figure out a way to do this without going through python
                        py::gil_scoped_acquire gil;
                        x->set_meta(m_idx2label[0], py::cast(values));
                    }

                    output.on_next(x);
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
    }

    std::size_t m_num_class_labels;
    std::map<std::size_t, std::string> m_idx2label;
};


}  // namespace morpheus
