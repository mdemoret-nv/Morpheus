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

#include "jsoncons_ext/jsonpath/json_query.hpp"

#include "morpheus/messages/control.hpp"
#include "morpheus/utilities/string_util.hpp"

#include <glog/logging.h>
#include <jsoncons/json.hpp>
#include <jsoncons_ext/jsonpath/jsonpath.hpp>
#include <mrc/coroutines/task.hpp>
#include <mrc/coroutines/when_all.hpp>
#include <mrc/types.hpp>
#include <nlohmann/detail/json_pointer.hpp>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <pybind11/pytypes.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace morpheus {
template <typename T>
using Task = mrc::coroutines::Task<T>;
}

namespace morpheus::llm {

struct InputMap
{
    // // Construct using a placeholder for the node name
    // InputMap(std::string input_name) : input_name(std::move(input_name)), node_name("-") {}

    // InputMap(std::string input_name, std::string node_name) :
    //   input_name(std::move(input_name)),
    //   node_name(std::move(node_name))
    // {}

    std::string input_name;      // The name of the upstream node to use as input
    std::string node_name{"-"};  // The name of the input that the upstream node maps to
};

// Ordered mapping of input names (current node) to output names (from previous nodes)
using input_map_t = std::vector<InputMap>;

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
    nlohmann::json values;

    // std::map<std::string, std::unique_ptr<cudf::column>> outputs_columns;
};

class LLMContextBase : public std::enable_shared_from_this<LLMContextBase>
{};

class RootLLMContext : public LLMContextBase
{};

class LLMContext2 : public LLMContextBase
{
  private:
    std::map<std::string, nlohmann::json::const_reference> m_inputs;
    nlohmann::json m_outputs;
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

    LLMContext(std::shared_ptr<LLMContext> parent, std::string name, input_map_t inputs) :
      m_parent(std::move(parent)),
      m_name(std::move(name)),
      m_inputs(std::move(inputs))
    {
        m_outputs_future = m_outputs_promise.get_future().share();

        // this->m_parent = parent;
        // this->m_name   = std::move(name);
        // this->m_inputs = std::move(inputs);
    }

    std::shared_ptr<LLMContext> parent() const
    {
        return m_parent;
    }

    const std::string& name() const
    {
        return m_name;
    }

    const input_map_t& input_map() const
    {
        return m_inputs;
    }

    const LLMTask& task() const
    {
        if (m_parent)
        {
            return m_parent->task();
        }

        return m_state->task;
    }

    std::shared_ptr<ControlMessage>& message() const
    {
        if (m_parent)
        {
            return m_parent->message();
        }

        return m_state->message;
    }

    nlohmann::json::const_reference all_outputs() const
    {
        return m_outputs;
    }

    std::string full_name() const
    {
        // Determine the full name
        if (m_parent)
        {
            return m_parent->full_name() + "/" + m_name;
        }

        // If we dont have a parent, we are the root context. So return nothing
        return "";
    }

    std::shared_ptr<LLMContext> push(std::string name, input_map_t inputs)
    {
        return std::make_shared<LLMContext>(this->shared_from_this(), std::move(name), std::move(inputs));
    }

    void pop()
    {
        // Copy the outputs from the child context to the parent
        if (m_output_names.empty())
        {
            // Use them all by default
            m_parent->set_output(m_name, std::move(m_outputs));
        }
        else if (m_output_names.size() == 1)
        {
            // Treat only a single output as the output
            m_parent->set_output(m_name, std::move(m_outputs[m_output_names[0]]));
        }
        else
        {
            // Build a new json object with only the specified keys
            nlohmann::json new_outputs;

            for (const auto& output_name : m_output_names)
            {
                new_outputs[output_name] = m_outputs[output_name];
            }

            m_parent->set_output(m_name, std::move(new_outputs));
        }
    }

    nlohmann::json::const_reference get_input() const
    {
        if (m_inputs.size() > 1)
        {
            throw std::runtime_error(
                "LLMContext::get_input() called on a context with multiple inputs. Use get_input(input_name) instead.");
        }

        return this->get_input(m_inputs[0].node_name);

        // nlohmann::json inputs;

        // for (const auto& [input_name, output_name] : m_inputs)
        // {
        //     inputs[input_name] = m_state->outputs[nlohmann::json::json_pointer(output_name)];
        // }

        // return inputs;
    }

    nlohmann::json::const_reference get_input(const std::string& node_name) const
    {
        if (node_name[0] == '$')
        {
            // Interpolate it as a json path
            auto outputs_str = m_outputs.dump();

            jsoncons::json tmp_json = jsoncons::json::parse(outputs_str);

            std::ostringstream ss;
            jsoncons::jsonpath::json_query(tmp_json, node_name).dump_pretty(ss);

            LOG(INFO) << ss.str();
        }

        if (node_name[0] == '/')
        {
            nlohmann::json::json_pointer node_json_ptr(node_name);

            if (!m_outputs.contains(node_json_ptr))
            {
                throw std::runtime_error(
                    MORPHEUS_CONCAT_STR("Input '" << node_name << "' not found in the output map"));
            }

            // Get the value from a sibling output
            return m_outputs[node_json_ptr];
        }
        else
        {
            // Must be on the parent, so find the mapping between this namespace and the parent
            auto found = std::find_if(m_inputs.begin(), m_inputs.end(), [&node_name](const auto& map_iterator) {
                return map_iterator.node_name == node_name;
            });

            if (found == m_inputs.end())
            {
                throw std::runtime_error(
                    MORPHEUS_CONCAT_STR("Input '" << node_name << "' not found in the input list"));
            }

            auto& input_name = found->input_name;

            // Get the value from a parent output
            return m_parent->get_input(input_name);
        }
    }

    nlohmann::json get_inputs() const
    {
        nlohmann::json inputs;

        for (const auto& in_map : m_inputs)
        {
            inputs[in_map.node_name] = this->get_input(in_map.node_name);
        }

        return inputs;
    }

    void set_output(nlohmann::json outputs)
    {
        m_outputs = std::move(outputs);
        // auto full_name = nlohmann::json::json_pointer(this->full_name());

        // // if (m_parent)
        // // {
        // //     auto& output = m_parent->get_outputs()
        // // }

        // m_state->values[full_name] = std::move(outputs);

        // // Notify that the outputs are complete
        // this->outputs_complete();
    }

    void set_output(const std::string& output_name, nlohmann::json outputs)
    {
        m_outputs[output_name] = std::move(outputs);
        // std::string full_name = nlohmann::json::json_pointer(this->full_name() + "/" + output_name);

        // m_state->values[full_name] = std::move(outputs);

        // std::vector<int32_t> test(outputs.size(), 0);

        // //           using RepType        = typename ElementTo::rep;
        // //   auto transformer     = fixed_width_type_converter<ElementFrom, RepType>{};
        // //   auto transform_begin = thrust::make_transform_iterator(begin, transformer);
        // //   auto const size      = cudf::distance(begin, end);
        // auto const elements = thrust::host_vector<int32_t>(test.begin(), test.end());
        // auto device_buff =
        //     rmm::device_buffer{elements.data(), test.size() * sizeof(int32_t), cudf::get_default_stream()};

        // // Create a cudf column
        // auto new_column = std::make_unique<cudf::column>(
        //     cudf::data_type{cudf::type_id::INT32}, outputs.size(), std::move(device_buff), rmm::device_buffer{}, 0);

        // m_state->outputs_columns[full_name] = std::move(new_column);
    }

    void set_output_names(std::vector<std::string> output_names)
    {
        m_output_names = std::move(output_names);
    }

    void outputs_complete()
    {
        m_outputs_promise.set_value();
    }

    nlohmann::json::const_reference view_outputs() const
    {
        // // // Wait for the outputs to be available
        // // m_outputs_future.wait();

        // return m_state->values[this->full_name()];
        return m_outputs;
    }

  private:
    std::shared_ptr<LLMContext> m_parent{nullptr};
    std::string m_name;
    input_map_t m_inputs;
    std::vector<std::string> m_output_names;  // Names of keys to be used as the output. Empty means use all keys

    std::shared_ptr<LLMContextState> m_state;

    nlohmann::json m_outputs;

    mrc::Promise<void> m_outputs_promise;
    mrc::SharedFuture<void> m_outputs_future;
};

class LLMNodeBase
{
  public:
    virtual std::vector<std::string> get_input_names() const                               = 0;
    virtual Task<std::shared_ptr<LLMContext>> execute(std::shared_ptr<LLMContext> context) = 0;
};

class LLMNodeRunner
{
  public:
    LLMNodeRunner(std::string name, input_map_t inputs, std::shared_ptr<LLMNodeBase> node) :
      m_name(std::move(name)),
      m_inputs(std::move(inputs)),
      m_node(std::move(node))
    {
        // TODO(MDD): Check that the input map is valid

        // Get the inputs of the current node
        auto input_names = m_node->get_input_names();

        // Replace any placeholders with the real node input name
        for (size_t i = 0; i < m_inputs.size(); ++i)
        {
            const auto& node_name  = m_inputs[i].node_name;
            const auto& input_name = m_inputs[i].input_name;

            // Check that the input name and node names are valid
            CHECK_EQ(node_name.find("*"), std::string::npos) << "Invalid node name '" << node_name << "'";
            CHECK_EQ(input_name.find("*"), std::string::npos) << "Invalid input_name '" << input_name << "'";

            CHECK_EQ(node_name.find("-"), std::string::npos) << "Invalid node name '" << node_name << "'";
            CHECK_EQ(input_name.find("-"), std::string::npos) << "Invalid input_name '" << input_name << "'";

            // Determine if the inputs are coming from a parent node or a sibling node
            if (m_inputs[i].input_name[0] == '/')
            {
                m_sibling_input_names.push_back(m_inputs[i].input_name);
            }
            else
            {
                m_parent_input_names.push_back(m_inputs[i].input_name);
            }
        }
    }

    virtual Task<std::shared_ptr<LLMContext>> execute(std::shared_ptr<LLMContext> context)
    {
        // Create a new context
        auto child_context = context->push(m_name, m_inputs);

        // Also need error handling here
        auto returned_context = co_await m_node->execute(child_context);

        // Call pop to apply the outputs to the parent context
        child_context->pop();

        co_return returned_context;
    }

    const std::string& name() const
    {
        return m_name;
    }

    const input_map_t& inputs() const
    {
        return m_inputs;
    }

    const std::vector<std::string>& sibling_input_names() const
    {
        return m_sibling_input_names;
    }

    const std::vector<std::string>& parent_input_names() const
    {
        return m_parent_input_names;
    }

  private:
    std::string m_name;
    input_map_t m_inputs;
    std::shared_ptr<LLMNodeBase> m_node;

    std::vector<std::string> m_sibling_input_names;
    std::vector<std::string> m_parent_input_names;
};

input_map_t process_input_names(const input_map_t& inputs, const std::vector<std::string>& input_names)
{
    input_map_t final_inputs;
    input_map_t placeholder_inputs;

    // Perform any placeholder replacements
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        const auto& single_input = inputs[i];

        bool found_star_input_name = single_input.input_name.find('*') != std::string::npos;
        bool found_star_node_name  = single_input.node_name == "*";

        if (found_star_input_name != found_star_node_name)
        {
            throw std::runtime_error(
                "LLMNode::add_node() called with a placeholder input name and node name that "
                "do not match");
        }
        else if (found_star_input_name && found_star_node_name)
        {
            // Need to process these after the non-placeholder inputs
            placeholder_inputs.push_back(single_input);
        }
        else
        {
            // No placeholder, so just add the input. If the node_name == "-", then replace it with the input name
            if (single_input.node_name == "-")
            {
                // If we start with a slash, that means we are mapping from another node, not a parent.
                if (single_input.input_name[0] == '/')
                {
                    if (inputs.size() != input_names.size())
                    {
                        throw std::runtime_error(MORPHEUS_CONCAT_STR(
                            "When mapping from a sibling node, the number of siblings must match. Provided: "
                            << inputs.size() << ", Expected: " << input_names.size()));
                    }

                    // Match by index
                    final_inputs.push_back({single_input.input_name, input_names[i]});
                }
                else
                {
                    // Match by name
                    auto found = std::find(input_names.begin(), input_names.end(), single_input.input_name);

                    if (found != input_names.end())
                    {
                        final_inputs.push_back({single_input.input_name, *found});
                    }
                    else if (input_names.size() == 1)
                    {
                        // We can infer that the node name is the only one
                        final_inputs.push_back({single_input.input_name, input_names[0]});
                    }
                    else
                    {
                        throw std::runtime_error(MORPHEUS_CONCAT_STR("Could not find a matching node name for input '"
                                                                     << single_input.input_name << "'"));
                    }
                }
            }
            else
            {
                final_inputs.push_back(single_input);
            }
        }
    }

    if (!placeholder_inputs.empty())
    {
        // TODO(MDD): Support multiple placeholders
        CHECK_EQ(placeholder_inputs.size(), 1) << "Only a single placeholder input is currently supported";

        std::set<std::string> specified_names;

        std::transform(final_inputs.begin(),
                       final_inputs.end(),
                       std::inserter(specified_names, specified_names.begin()),
                       [](const auto& input) {
                           return input.node_name;
                       });

        std::set<std::string> total_names(input_names.begin(), input_names.end());

        std::vector<std::string> remaining_names;

        // Find the remaining names
        std::set_difference(total_names.begin(),
                            total_names.end(),
                            specified_names.begin(),
                            specified_names.end(),
                            std::back_inserter(remaining_names));

        auto star_input_name_loc = placeholder_inputs[0].input_name.find('*');

        // Loop over the remaining names and add them to the final inputs
        for (const auto& remaining_name : remaining_names)
        {
            // Make a copy of the string to avoid modifying the original
            auto replaced = std::string(placeholder_inputs[0].input_name);
            replaced.replace(star_input_name_loc, 1, remaining_name);
            final_inputs.push_back({replaced, remaining_name});
        }
    }

    if (input_names.size() != final_inputs.size())
    {
        throw std::runtime_error(MORPHEUS_CONCAT_STR(
            "The number of inputs provided does not match the number of inputs expected by the node. Provided: "
            << final_inputs.size() << ", Expected: " << input_names.size()));
    }

    return final_inputs;
}

class LLMNode : public LLMNodeBase
{
  public:
    virtual std::shared_ptr<LLMNodeRunner> add_node(std::string name,
                                                    input_map_t inputs,
                                                    std::shared_ptr<LLMNodeBase> node,
                                                    bool is_output = false)
    {
        // Get the inputs of the current node
        auto input_names = node->get_input_names();

        auto final_inputs = process_input_names(inputs, input_names);

        auto node_runner = std::make_shared<LLMNodeRunner>(std::move(name), std::move(final_inputs), std::move(node));

        // Add the child inputs to the current inputs
        for (const auto& parent_input : node_runner->parent_input_names())
        {
            if (std::find(m_input_names.begin(), m_input_names.end(), parent_input) == m_input_names.end())
            {
                m_input_names.push_back(parent_input);
            }
        }

        // Perform checks that the existing nodes meet the requirements

        m_child_runners.push_back(node_runner);

        if (is_output)
        {
            m_output_node_names.push_back(node_runner->name());
        }

        return node_runner;
    }

    std::vector<std::string> get_input_names() const override
    {
        return m_input_names;
    }

    Task<std::shared_ptr<LLMContext>> execute(std::shared_ptr<LLMContext> context) override
    {
        for (auto& runner : m_child_runners)
        {
            // Run the child node
            co_await runner->execute(context);

            // Wait for the child node outputs (This will yield if not already available)
            // context->get_outputs();
        }

        // Before returning, set the output names to only propagate the specified outputs
        context->set_output_names(m_output_node_names);

        co_return context;
    }

  private:
    std::vector<std::shared_ptr<LLMNodeRunner>> m_child_runners;

    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_node_names;  // Names of nodes to be used as the output
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

    virtual std::vector<std::string> get_input_names() const               = 0;
    virtual Task<return_t> try_handle(std::shared_ptr<LLMContext> context) = 0;
};

class LLMTaskHandlerRunner
{
  public:
    LLMTaskHandlerRunner(input_map_t inputs, std::shared_ptr<LLMTaskHandler> handler) :
      m_inputs(std::move(inputs)),
      m_handler(std::move(handler))
    {
        // TODO(MDD): Check that the input map is valid

        // Get the inputs of the current node
        auto input_names = m_handler->get_input_names();

        // Replace any placeholders with the real node input name
        for (size_t i = 0; i < m_inputs.size(); ++i)
        {
            const auto& node_name  = m_inputs[i].node_name;
            const auto& input_name = m_inputs[i].input_name;

            // Check that the input name and node names are valid
            CHECK_EQ(node_name.find("*"), std::string::npos) << "Invalid node name '" << node_name << "'";
            CHECK_EQ(input_name.find("*"), std::string::npos) << "Invalid input_name '" << input_name << "'";

            CHECK_EQ(node_name.find("-"), std::string::npos) << "Invalid node name '" << node_name << "'";
            CHECK_EQ(input_name.find("-"), std::string::npos) << "Invalid input_name '" << input_name << "'";
        }
    }

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
        auto input_names = task_handler->get_input_names();

        auto final_inputs = process_input_names(inputs, input_names);

        m_task_handlers.push_back(std::make_shared<LLMTaskHandlerRunner>(std::move(final_inputs), task_handler));
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
