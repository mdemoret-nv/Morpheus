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
#include <stdexcept>
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

class LLMTaskHandler
{
  public:
    virtual std::optional<std::vector<std::shared_ptr<ControlMessage>>> try_handle(
        LLMEngine& engine,
        const LLMTask& input_task,
        std::shared_ptr<ControlMessage> input_message,
        const LLMGenerateResult& responses) = 0;
};

class LLMEngine
{
  public:
    using prompt_t = std::variant<LLMGeneratePrompt, LLMGenerateResult>;

    LLMEngine() = default;

    virtual void add_prompt_generator(std::shared_ptr<LLMPromptGenerator> prompt_generator)
    {
        m_prompt_generators.push_back(prompt_generator);
    }

    virtual void add_task_handler(std::shared_ptr<LLMTaskHandler> task_handler)
    {
        m_task_handlers.push_back(task_handler);
    }

    virtual std::vector<std::shared_ptr<ControlMessage>> run(std::shared_ptr<ControlMessage> input_message)
    {
        if (!input_message)
        {
            throw std::runtime_error("LLMEngine::run() called with a null message");
        }

        if (!input_message->has_task("llm_engine"))
        {
            throw std::runtime_error("LLMEngine::run() called with a message that does not have the 'llm_engine' task");
        }

        std::vector<std::shared_ptr<ControlMessage>> output_messages;

        while (input_message->has_task("llm_engine"))
        {
            auto current_task = input_message->remove_task("llm_engine");

            // Temp create an instance of LLMTask for type safety
            LLMTask tmp_task;

            auto prompts = this->generate_prompts(tmp_task, input_message);
            LLMGenerateResult results;

            if (std::holds_alternative<LLMGeneratePrompt>(prompts))
            {
                auto gen_prompt = std::get<LLMGeneratePrompt>(prompts);

                results = this->execute_model(gen_prompt);
            }
            else
            {
                results = std::get<LLMGenerateResult>(prompts);
            }

            auto tasks = this->handle_tasks(tmp_task, input_message, results);

            output_messages.insert(output_messages.end(), tasks.begin(), tasks.end());
        }

        return output_messages;
    }

  private:
    prompt_t generate_prompts(const LLMTask& input_task, std::shared_ptr<ControlMessage> input_message)
    {
        for (auto& prompt_generator : m_prompt_generators)
        {
            auto prompt_result = prompt_generator->try_handle(*this, input_task, input_message);

            if (prompt_result.has_value())
            {
                return prompt_result.value();
            }
        }

        throw std::runtime_error("No prompt generator was able to handle the input message");
    }

    LLMGenerateResult execute_model(const LLMGeneratePrompt& prompt)
    {
        return LLMGenerateResult{};
    }

    std::vector<std::shared_ptr<ControlMessage>> handle_tasks(const LLMTask& input_task,
                                                              std::shared_ptr<ControlMessage> input_message,
                                                              const LLMGenerateResult& results)
    {
        for (auto& task_handler : m_task_handlers)
        {
            auto new_tasks = task_handler->try_handle(*this, input_task, input_message, results);

            if (new_tasks.has_value())
            {
                return new_tasks.value();
            }
        }

        throw std::runtime_error("No task handler was able to handle the input message and responses generated");
    }

    std::vector<std::shared_ptr<LLMPromptGenerator>> m_prompt_generators;
    std::vector<std::shared_ptr<LLMTaskHandler>> m_task_handlers;
};

}  // namespace morpheus::llm
