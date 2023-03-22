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

#include "morpheus/stages/preprocess_nlp.hpp"

#include "morpheus/messages/memory/inference_memory.hpp"  // for InferenceMemory
#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/objects/dev_mem_info.hpp"
#include "morpheus/objects/dtype.hpp"
#include "morpheus/objects/table_info.hpp"  // for TableInfo
#include "morpheus/objects/tensor.hpp"
#include "morpheus/types.hpp"  // for TensorIndex, TensorMap
#include "morpheus/utilities/matx_util.hpp"

#include <cudf/column/column.hpp>                // for column, column::contents
#include <cudf/strings/strings_column_view.hpp>  // for strings_column_view
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <mrc/segment/builder.hpp>
#include <nvtext/subword_tokenize.hpp>
#include <pymrc/node.hpp>
#include <rmm/device_buffer.hpp>  // for device_buffer

#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <utility>

namespace morpheus {
// Component public implementations
// ************ PreprocessNLPStage ************************* //
PreprocessNLPStage::PreprocessNLPStage(std::string vocab_hash_file,
                                       uint32_t sequence_length,
                                       bool truncation,
                                       bool do_lower_case,
                                       bool add_special_token,
                                       int stride,
                                       std::string column) :
  PythonNode(base_t::op_factory_from_sub_fn(build_operator())),
  m_vocab_hash_file(std::move(vocab_hash_file)),
  m_sequence_length(sequence_length),
  m_truncation(truncation),
  m_do_lower_case(do_lower_case),
  m_add_special_token(add_special_token),
  m_stride(stride),
  m_column(std::move(column))
{}

PreprocessNLPStage::subscribe_fn_t PreprocessNLPStage::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        uint32_t stride = m_stride;

        // Auto calc stride to be 75% of sequence length
        if (stride < 0)
        {
            stride = m_sequence_length / 2;
            stride = stride + stride / 2;
        }

        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output, stride](sink_type_t x) {
                // Convert to string view
                auto string_col = cudf::strings_column_view{x->get_meta(this->m_column).get_column(0)};

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

                TensorIndex length = token_results.tensor_token_ids->size() / token_results.sequence_length;
                auto input_ids_released =
                    cudf::cast(token_results.tensor_token_ids->view(), cudf::data_type(cudf::type_id::INT32))
                        ->release();

                memory->set_tensor("input_ids",
                                   Tensor::create(std::move(input_ids_released.data),
                                                  DType::create<int32_t>(),
                                                  {length, static_cast<TensorIndex>(token_results.sequence_length)},
                                                  {},
                                                  0));

                length = token_results.tensor_attention_mask->size() / token_results.sequence_length;
                auto input_mask_released =
                    cudf::cast(token_results.tensor_attention_mask->view(), cudf::data_type(cudf::type_id::INT32))
                        ->release();
                memory->set_tensor("input_mask",
                                   Tensor::create(std::move(input_mask_released.data),
                                                  DType::create<int32_t>(),
                                                  {length, static_cast<TensorIndex>(token_results.sequence_length)},
                                                  {},
                                                  0));

                auto tensor_index_dtype = DType::create<TensorIndex>();
                length                  = token_results.tensor_metadata->size() / 3;
                auto seq_ids_released   = cudf::cast(token_results.tensor_metadata->view(),
                                                   cudf::data_type(tensor_index_dtype.cudf_type_id()))
                                            ->release();

                std::shared_ptr<rmm::device_buffer> seq_ids_data = std::move(seq_ids_released.data);

                if (x->mess_offset > 0)
                {
                    // Add an offset to the seq_ids so the message IDs line up
                    MatxUtil::offset_seq_ids(
                        DevMemInfo{seq_ids_data, tensor_index_dtype.type_id(), {length, 3}, {1, 3}}, x->mess_offset);
                }

                memory->set_tensor("seq_ids", Tensor::create(seq_ids_data, tensor_index_dtype, {length, 3}, {}, 0));

                auto next = std::make_shared<MultiInferenceMessage>(
                    x->meta, x->mess_offset, x->mess_count, std::move(memory), 0, memory->count);

                output.on_next(std::move(next));
            },
            [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
            [&]() { output.on_completed(); }));
    };
}

// ************ PreprocessNLPStageInterfaceProxy *********** //
std::shared_ptr<mrc::segment::Object<PreprocessNLPStage>> PreprocessNLPStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    std::string vocab_hash_file,
    uint32_t sequence_length,
    bool truncation,
    bool do_lower_case,
    bool add_special_token,
    int stride,
    std::string column)
{
    auto stage = builder.construct_object<PreprocessNLPStage>(
        name, vocab_hash_file, sequence_length, truncation, do_lower_case, add_special_token, stride, column);

    return stage;
}
}  // namespace morpheus
