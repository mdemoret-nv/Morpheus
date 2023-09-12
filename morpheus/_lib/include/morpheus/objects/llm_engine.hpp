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

#include "morpheus/messages/control.hpp"

#include <pybind11/pytypes.h>

#include <memory>
#include <variant>

namespace morpheus::llm {

class LLMEngine;

struct LLMTask
{};

struct LLMGeneratePrompt
{};

struct LLMGenerateResult
{};

class LLMPromptGenerator
{
  public:
    virtual std::optional<std::variant<LLMGeneratePrompt, LLMGenerateResult>> try_handle(
        LLMEngine& engine, const LLMTask& input_task, std::shared_ptr<ControlMessage> input_message) = 0;
};

class LLMEngine
{
  public:
    LLMEngine() = default;

    virtual void add_prompt_generator(std::shared_ptr<LLMPromptGenerator> prompt_generator)
    {
        m_prompt_generators.push_back(prompt_generator);
    }

    virtual std::vector<std::shared_ptr<ControlMessage>> run(std::shared_ptr<ControlMessage> input_message)
    {
        std::vector<std::shared_ptr<ControlMessage>> output_messages;

        for (auto& prompt_generator : m_prompt_generators)
        {
            auto result = prompt_generator->try_handle(*this, LLMTask{}, input_message);

            if (result)
            {
                if (std::holds_alternative<LLMGeneratePrompt>(*result)) {}

                break;
            }
        }

        return output_messages;
    }

  private:
    std::vector<std::shared_ptr<LLMPromptGenerator>> m_prompt_generators;
};

}  // namespace morpheus::llm
