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

#include <mrc/coroutines/task.hpp>
#include <mrc/types.hpp>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <pybind11/pytypes.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace morpheus {
template <typename T>
using Task = mrc::coroutines::Task<T>;
}

namespace morpheus::llm {

// Ordered mapping of input names (current node) to output names (from previous nodes)
using input_map_t = std::vector<std::pair<std::string, std::string>>;

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

struct LLMContextState
{
    LLMTask task;
    std::shared_ptr<ControlMessage> message;
    nlohmann::json outputs;
};

class LLMContext : public std::enable_shared_from_this<LLMContext>
{
  public:
    LLMContext() : m_state(std::make_shared<LLMContextState>())
    {
        m_outputs_future = m_outputs_promise.get_future().share();
    }

    LLMContext(LLMTask task, std::shared_ptr<ControlMessage> message) : LLMContext()
    {
        m_state->task    = std::move(task);
        m_state->message = std::move(message);
    }

    LLMContext(std::shared_ptr<const LLMContext> parent, std::string name, input_map_t inputs) : LLMContext()
    {
        this->m_parent = parent;
        this->m_name   = std::move(name);
        this->m_inputs = std::move(inputs);
        this->m_state  = parent->m_state;
    }

    const std::string& name() const
    {
        return m_name;
    }

    const input_map_t& inputs() const
    {
        return m_inputs;
    }

    const LLMTask& task() const
    {
        return m_state->task;
    }

    std::shared_ptr<ControlMessage> message() const
    {
        return m_state->message;
    }

    nlohmann::json::const_reference all_outputs() const
    {
        return m_state->outputs;
    }

    std::string full_name() const
    {
        std::string full_name = m_name;

        // Determine the full name
        if (m_parent && !m_parent->m_name.empty())
        {
            full_name = m_parent->m_name + "." + full_name;
        }

        return full_name;
    }

    std::shared_ptr<LLMContext> push(std::string name, input_map_t inputs) const
    {
        return std::make_shared<LLMContext>(this->shared_from_this(), std::move(name), std::move(inputs));
    }

    nlohmann::json get_input() const
    {
        if (m_inputs.size() == 1)
        {
            return m_state->outputs[nlohmann::json::json_pointer(m_inputs[0].second)];
        }

        nlohmann::json inputs;

        for (const auto& [input_name, output_name] : m_inputs)
        {
            inputs[input_name] = m_state->outputs[nlohmann::json::json_pointer(output_name)];
        }

        return inputs;
    }

    nlohmann::json::const_reference get_input(const std::string& input_name) const
    {
        // Get the full name of the input
        std::string full_name = this->full_name() + "." + input_name;

        return m_state->outputs[full_name];
    }

    void set_output(nlohmann::json outputs)
    {
        std::string full_name = this->full_name();

        m_state->outputs[full_name] = std::move(outputs);

        // Notify that the outputs are complete
        this->outputs_complete();
    }

    void set_output(std::string output_name, nlohmann::json outputs)
    {
        std::string full_name = this->full_name() + "." + output_name;

        m_state->outputs[full_name] = std::move(outputs);
    }

    void outputs_complete()
    {
        m_outputs_promise.set_value();
    }

    nlohmann::json::const_reference get_outputs() const
    {
        // Wait for the outputs to be available
        m_outputs_future.wait();

        return m_state->outputs[this->full_name()];
    }

  private:
    std::shared_ptr<const LLMContext> m_parent{nullptr};
    std::string m_name;
    input_map_t m_inputs;

    std::shared_ptr<LLMContextState> m_state;

    mrc::Promise<void> m_outputs_promise;
    mrc::SharedFuture<void> m_outputs_future;
};

class LLMNodeBase
{
  public:
    virtual Task<std::shared_ptr<LLMContext>> execute(std::shared_ptr<LLMContext> context) = 0;
};

class LLMNodeRunner
{
  public:
    LLMNodeRunner(std::string name, input_map_t inputs, std::shared_ptr<LLMNodeBase> node) :
      m_name(std::move(name)),
      m_inputs(std::move(inputs)),
      m_node(std::move(node))
    {}

    virtual Task<std::shared_ptr<LLMContext>> execute(std::shared_ptr<LLMContext> context)
    {
        // Create a new context
        auto child_context = context->push(m_name, m_inputs);

        // Also need error handling here
        co_return co_await m_node->execute(child_context);
    }

    const std::string& name() const
    {
        return m_name;
    }

    const input_map_t& inputs() const
    {
        return m_inputs;
    }

  private:
    std::string m_name;
    input_map_t m_inputs;
    std::shared_ptr<LLMNodeBase> m_node;
};

class LLMNode : public LLMNodeBase
{
  public:
    virtual std::shared_ptr<LLMNodeRunner> add_node(std::string name,
                                                    input_map_t inputs,
                                                    std::shared_ptr<LLMNodeBase> node)
    {
        auto node_runner = std::make_shared<LLMNodeRunner>(std::move(name), std::move(inputs), std::move(node));

        // Perform checks that the existing nodes meet the requirements

        m_children.push_back(node_runner);

        return node_runner;
    }

    Task<std::shared_ptr<LLMContext>> execute(std::shared_ptr<LLMContext> context) override
    {
        for (auto& child : m_children)
        {
            // Run the child node
            co_await child->execute(context);

            // Wait for the child node outputs (This will yield if not already available)
            // context->get_outputs();
        }

        co_return context;
    }

  private:
    std::vector<std::shared_ptr<LLMNodeRunner>> m_children;
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
    using return_t = std::optional<std::vector<std::shared_ptr<ControlMessage>>>;

    virtual Task<return_t> try_handle(std::shared_ptr<LLMContext> context) = 0;
};

class LLMTaskHandlerRunner
{
  public:
    LLMTaskHandlerRunner(input_map_t inputs, std::shared_ptr<LLMTaskHandler> handler) :
      m_inputs(std::move(inputs)),
      m_handler(std::move(handler))
    {}

    virtual Task<LLMTaskHandler::return_t> try_handle(std::shared_ptr<LLMContext> context)
    {
        // Create a new context
        auto child_context = context->push("TaskHandler", m_inputs);

        // Also need error handling here
        co_return co_await m_handler->try_handle(child_context);
    }

    const input_map_t& input_names() const
    {
        return m_inputs;
    }

  private:
    input_map_t m_inputs;
    std::shared_ptr<LLMTaskHandler> m_handler;
};

class LLMEngine : public LLMNode
{
  public:
    using prompt_t = std::variant<LLMGeneratePrompt, LLMGenerateResult>;

    LLMEngine() = default;

    virtual void add_prompt_generator(std::shared_ptr<LLMPromptGenerator> prompt_generator)
    {
        m_prompt_generators.push_back(prompt_generator);
    }

    virtual void add_task_handler(input_map_t inputs, std::shared_ptr<LLMTaskHandler> task_handler)
    {
        m_task_handlers.push_back(std::make_shared<LLMTaskHandlerRunner>(inputs, task_handler));
    }

    virtual Task<std::vector<std::shared_ptr<ControlMessage>>> run(std::shared_ptr<ControlMessage> input_message)
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

            // Set the name, task, control_message and inputs on the context
            auto context = std::make_shared<LLMContext>(tmp_task, input_message);

            // Call the base node
            co_await this->execute(context);

            // Pass the outputs into the task generators
            auto tasks = co_await this->handle_tasks(context);

            output_messages.insert(output_messages.end(), tasks.begin(), tasks.end());
        }

        co_return output_messages;
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

    Task<std::vector<std::shared_ptr<ControlMessage>>> handle_tasks(std::shared_ptr<LLMContext> context)
    {
        // Wait for the base node outputs (This will yield if not already available)
        // auto outputs = context->get_outputs();

        for (auto& task_handler : m_task_handlers)
        {
            auto new_tasks = co_await task_handler->try_handle(context);

            if (new_tasks.has_value())
            {
                co_return new_tasks.value();
            }
        }

        throw std::runtime_error("No task handler was able to handle the input message and responses generated");
    }

    std::vector<std::shared_ptr<LLMPromptGenerator>> m_prompt_generators;
    std::vector<std::shared_ptr<LLMTaskHandlerRunner>> m_task_handlers;
};

}  // namespace morpheus::llm
