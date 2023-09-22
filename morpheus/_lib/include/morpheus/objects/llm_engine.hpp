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

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace morpheus::llm {

class LLMEngine;

struct LLMTask
{
    LLMTask() = default;
    LLMTask(std::string task_type, nlohmann::json task_dict) :
      task_type(std::move(task_type)),
      task_dict(std::move(task_dict))
    {}

    std::string task_type;

    size_t size() const
    {
        return this->task_dict.size();
    }

    nlohmann::basic_json<>::const_reference get(const std::string& key) const
    {
        return this->task_dict.at(key);
    }

    void set(const std::string& key, nlohmann::basic_json<>::value_type value)
    {
        this->task_dict[key] = std::move(value);
    }

    nlohmann::json task_dict;
};

struct LLMGeneratePrompt
{
    LLMGeneratePrompt() = default;
    LLMGeneratePrompt(std::string model_name, nlohmann::json model_kwargs, std::vector<std::string> prompts) :
      model_name(std::move(model_name)),
      model_kwargs(std::move(model_kwargs)),
      prompts(std::move(prompts))
    {}

    std::string model_name;
    nlohmann::json model_kwargs;
    std::vector<std::string> prompts;
};

struct LLMGenerateResult : public LLMGeneratePrompt
{
    LLMGenerateResult() = default;

    LLMGenerateResult(const LLMGeneratePrompt& other, std::vector<std::string> responses) :
      LLMGeneratePrompt(other),
      responses(std::move(responses))
    {}

    std::vector<std::string> responses;
};

class LLMService
{
  public:
    virtual LLMGenerateResult generate(LLMGeneratePrompt prompt) const = 0;
};

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

    LLMEngine(std::shared_ptr<LLMService> llm_service) : m_llm_service(std::move(llm_service)) {}

    std::shared_ptr<LLMService> get_llm_service() const
    {
        return m_llm_service;
    }

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
            LLMTask tmp_task(current_task["task_type"].get<std::string>(), current_task.at("task_dict"));

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
        return m_llm_service->generate(prompt);
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

    std::shared_ptr<LLMService> m_llm_service;
    std::vector<std::shared_ptr<LLMPromptGenerator>> m_prompt_generators;
    std::vector<std::shared_ptr<LLMTaskHandler>> m_task_handlers;
};

}  // namespace morpheus::llm
