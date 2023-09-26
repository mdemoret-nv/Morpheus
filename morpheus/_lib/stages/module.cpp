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

#include "mrc/channel/status.hpp"
#include "mrc/coroutines/task.hpp"
#include "mrc/node/rx_sink_base.hpp"
#include "mrc/node/rx_source_base.hpp"
#include "mrc/types.hpp"

#include "morpheus/messages/control.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/objects/file_types.hpp"  // for FileTypes
#include "morpheus/objects/llm_engine.hpp"
#include "morpheus/stages/add_classification.hpp"
#include "morpheus/stages/add_scores.hpp"
#include "morpheus/stages/deserialize.hpp"
#include "morpheus/stages/file_source.hpp"
#include "morpheus/stages/filter_detection.hpp"
#include "morpheus/stages/kafka_source.hpp"
#include "morpheus/stages/preallocate.hpp"
#include "morpheus/stages/preprocess_fil.hpp"
#include "morpheus/stages/preprocess_nlp.hpp"
#include "morpheus/stages/serialize.hpp"
#include "morpheus/stages/triton_inference.hpp"
#include "morpheus/stages/write_to_file.hpp"
#include "morpheus/utilities/cudf_util.hpp"
#include "morpheus/version.hpp"

#include <boost/fiber/future/async.hpp>
#include <boost/fiber/future/future.hpp>
#include <glog/logging.h>
#include <mrc/segment/object.hpp>
#include <mrc/utils/string_utils.hpp>
#include <pybind11/attr.h>  // for multiple_inheritance
#include <pybind11/detail/common.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // for arg, init, class_, module_, str_attr_accessor, PYBIND11_MODULE, pybind11
#include <pybind11/pytypes.h>   // for dict, sequence
#include <pybind11/stl.h>
#include <pymrc/types.hpp>
#include <pymrc/utils.hpp>  // for pymrc::import
#include <rxcpp/rx.hpp>

#include <future>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace morpheus {
namespace py = pybind11;

class StopIteration : public py::stop_iteration
{
  public:
    StopIteration(py::object&& result) : stop_iteration("--"), m_result(std::move(result)){};

    void set_error() const override
    {
        PyErr_SetObject(PyExc_StopIteration, this->m_result.ptr());
    }

  private:
    py::object m_result;
};

class CoroAwaitable : public std::enable_shared_from_this<CoroAwaitable>
{
  public:
    CoroAwaitable() = default;

    CoroAwaitable(mrc::coroutines::Task<pybind11::object>&& task) : m_task(std::move(task)) {}

    std::shared_ptr<CoroAwaitable> iter()
    {
        return this->shared_from_this();
    }

    std::shared_ptr<CoroAwaitable> await()
    {
        return this->shared_from_this();
    }

    void next()
    {
        // Need to release the GIL before  waiting
        py::gil_scoped_release nogil;

        auto status = !m_task.resume();

        if (status)
        {
            // Grab the gil before moving and throwing
            py::gil_scoped_acquire gil;

            // job done -> throw
            auto exception = StopIteration(std::move(m_task.promise().result()));

            throw exception;
        }
    }

  private:
    mrc::coroutines::Task<pybind11::object> m_task;
};

std::function<void()> create_gil_initializer()
{
    bool has_pydevd_trace = false;

    // We check if there is a debugger by looking at sys.gettrace() and seeing if the function contains 'pydevd'
    // somewhere in the module name. Its important to get this right because calling `debugpy.debug_this_thread()`
    // will fail if there is no debugger and can dramatically alter performanc
    auto sys = pybind11::module_::import("sys");

    auto trace_func = sys.attr("gettrace")();

    py::print("Trace func: ", trace_func);

    if (!trace_func.is_none())
    {
        // Convert it to a string to quickly get its module and name
        auto trace_func_str = pybind11::str(trace_func);

        if (!trace_func_str.attr("find")("pydevd").equal(pybind11::int_(-1)))
        {
            VLOG(10) << "Found pydevd trace function. Will attempt to enable debugging for MRC threads.";
            has_pydevd_trace = true;
        }
    }
    else
    {
        VLOG(10) << "Not setting up debugging. No trace function found.";
    }

    return [has_pydevd_trace] {
        pybind11::gil_scoped_acquire gil;

        // // Increment the ref once to prevent creating and destroying the thread state constantly
        // gil.inc_ref();

        try
        {
            // Try to load debugpy only if we found a trace function
            if (has_pydevd_trace)
            {
                auto debugpy = pybind11::module_::import("debugpy");

                auto debug_this_thread = debugpy.attr("debug_this_thread");

                debug_this_thread();

                VLOG(10) << "Debugging enabled from mrc threads";
            }
        } catch (const pybind11::error_already_set& err)
        {
            if (err.matches(PyExc_ImportError))
            {
                VLOG(10) << "Debugging disabled. Breakpoints will not be hit. Could import error on debugpy";
                // Fail silently
            }
            else
            {
                VLOG(10) << "Debugging disabled. Breakpoints will not be hit. Unknown error: " << err.what();
                // Rethrow everything else
                throw;
            }
        }
    };
}

class PyLLMService : public llm::LLMService
{
  public:
    llm::LLMGenerateResult generate(llm::LLMGeneratePrompt prompt) const override
    {
        using return_t = llm::LLMGenerateResult;

        pybind11 ::gil_scoped_acquire gil;

        pybind11 ::function override = pybind11 ::get_override(static_cast<const llm ::LLMService*>(this), "generate");

        if (!override)
        {
            // Problem
            pybind11 ::pybind11_fail(
                "Tried to call pure virtual function \""
                "llm::LLMService"
                "::"
                "generate"
                "\"");
        }

        auto override_result = override(prompt);

        // Now determine if the override result is a coroutine or not
        if (py::module::import("asyncio").attr("iscoroutine")(override_result).cast<bool>())
        {
            py::print("Returned a coroutine");

            // Need to schedule the result to run on the loop
            auto future = py::module::import("asyncio").attr("run_coroutine_threadsafe")(override_result, m_loop);

            // We are a dask future. Quickly check if its done, then release
            while (!future.attr("done")().cast<bool>())
            {
                // Release the GIL and wait for it to be done
                py::gil_scoped_release nogil;

                boost::this_fiber::yield();
            }

            // Completed, move into the returned object
            override_result = future.attr("result")();
        }
        else
        {
            py::print("Did not return a coroutine");
        }

        // Now cast back to the C++ type
        if (pybind11 ::detail ::cast_is_temporary_value_reference<return_t>::value)
        {
            static pybind11 ::detail ::override_caster_t<return_t> caster;
            return pybind11 ::detail ::cast_ref<return_t>(std ::move(override_result), caster);
        }
        return pybind11 ::detail ::cast_safe<return_t>(std ::move(override_result));
    }

  private:
    void set_loop(py::object loop)
    {
        m_loop = std::move(loop);
    }

    mrc::pymrc::PyHolder m_loop;

    friend class PyLLMEngine;
};

class PyLLMPromptGenerator : public llm::LLMPromptGenerator
{
  public:
    using llm::LLMPromptGenerator::LLMPromptGenerator;

    std::optional<std::variant<llm::LLMGeneratePrompt, llm::LLMGenerateResult>> try_handle(
        llm::LLMEngine& engine, const llm::LLMTask& input_task, std::shared_ptr<ControlMessage> input_message) override
    {
        using return_t = std::optional<std::variant<llm::LLMGeneratePrompt, llm::LLMGenerateResult>>;

        pybind11 ::gil_scoped_acquire gil;

        pybind11 ::function override =
            pybind11 ::get_override(static_cast<const llm ::LLMPromptGenerator*>(this), "try_handle");

        if (!override)
        {
            // Problem
            pybind11 ::pybind11_fail(
                "Tried to call pure virtual function \""
                "llm::LLMPromptGenerator"
                "::"
                "try_handle"
                "\"");
        }

        auto override_result = override(engine, input_task, input_message);

        // Now determine if the override result is a coroutine or not
        if (py::module::import("asyncio").attr("iscoroutine")(override_result).cast<bool>())
        {
            py::print("Returned a coroutine");

            // Need to schedule the result to run on the loop
            auto future = py::module::import("asyncio").attr("run_coroutine_threadsafe")(override_result, m_loop);

            // We are a dask future. Quickly check if its done, then release
            while (!future.attr("done")().cast<bool>())
            {
                // Release the GIL and wait for it to be done
                py::gil_scoped_release nogil;

                boost::this_fiber::yield();
            }

            // Completed, move into the returned object
            override_result = future.attr("result")();
        }
        else
        {
            py::print("Did not return a coroutine");
        }

        // Now cast back to the C++ type
        if (pybind11 ::detail ::cast_is_temporary_value_reference<return_t>::value)
        {
            static pybind11 ::detail ::override_caster_t<return_t> caster;
            return pybind11 ::detail ::cast_ref<return_t>(std ::move(override_result), caster);
        }
        return pybind11 ::detail ::cast_safe<return_t>(std ::move(override_result));
    }

  private:
    std::optional<std::variant<llm::LLMGeneratePrompt, llm::LLMGenerateResult>> inner_try_handle(
        llm::LLMEngine& engine, const llm::LLMTask& input_task, std::shared_ptr<ControlMessage> input_message)
    {
        using return_t = std::optional<std::variant<llm::LLMGeneratePrompt, llm::LLMGenerateResult>>;

        PYBIND11_OVERLOAD_PURE(return_t, llm ::LLMPromptGenerator, try_handle, engine, input_task, input_message);
    }

  private:
    void set_loop(py::object loop)
    {
        m_loop = std::move(loop);
    }

    mrc::pymrc::PyHolder m_loop;

    friend class PyLLMEngine;
};

class PyLLMTaskHandler : public llm::LLMTaskHandler
{
  public:
    using llm::LLMTaskHandler::LLMTaskHandler;

    std::optional<std::vector<std::shared_ptr<ControlMessage>>> try_handle(
        llm::LLMEngine& engine,
        const llm::LLMTask& input_task,
        std::shared_ptr<ControlMessage> input_message,
        const llm::LLMGenerateResult& responses) override
    {
        using return_t = std::optional<std::vector<std::shared_ptr<ControlMessage>>>;

        pybind11 ::gil_scoped_acquire gil;

        pybind11 ::function override =
            pybind11 ::get_override(static_cast<const llm ::LLMTaskHandler*>(this), "try_handle");

        if (!override)
        {
            // Problem
            pybind11 ::pybind11_fail(
                "Tried to call pure virtual function \""
                "llm::LLMTaskHandler"
                "::"
                "try_handle"
                "\"");
        }

        auto override_result = override(engine, input_task, input_message, responses);

        // Now determine if the override result is a coroutine or not
        if (py::module::import("asyncio").attr("iscoroutine")(override_result).cast<bool>())
        {
            py::print("Returned a coroutine");

            // Need to schedule the result to run on the loop
            auto future = py::module::import("asyncio").attr("run_coroutine_threadsafe")(override_result, m_loop);

            // We are a dask future. Quickly check if its done, then release
            while (!future.attr("done")().cast<bool>())
            {
                // Release the GIL and wait for it to be done
                py::gil_scoped_release nogil;

                boost::this_fiber::yield();
            }

            // Completed, move into the returned object
            override_result = future.attr("result")();
        }
        else
        {
            py::print("Did not return a coroutine");
        }

        // Now cast back to the C++ type
        if (pybind11 ::detail ::cast_is_temporary_value_reference<return_t>::value)
        {
            static pybind11 ::detail ::override_caster_t<return_t> caster;
            return pybind11 ::detail ::cast_ref<return_t>(std ::move(override_result), caster);
        }
        return pybind11 ::detail ::cast_safe<return_t>(std ::move(override_result));
    }

  private:
    void set_loop(py::object loop)
    {
        m_loop = std::move(loop);
    }

    mrc::pymrc::PyHolder m_loop;

    friend class PyLLMEngine;
};

template <class BaseT = llm::LLMNodeBase>
class PyLLMNodeBase : public BaseT
{
  public:
    using BaseT::BaseT;

    // std::shared_ptr<llm::LLMNodeRunner> add_node(std::string name,
    //                                              std::vector<std::string> input_names,
    //                                              std::shared_ptr<llm::LLMNodeBase> node) override
    // {
    //     // // Try to cast the object to a python object to ensure that we keep it alive
    //     // auto py_node = std::dynamic_pointer_cast<PyLLMNodeBase>(node);

    //     // if (py_node)
    //     // {
    //     // Store the python object to keep it alive
    //     m_py_nodes[node] = py::cast(node);
    //     // }

    //     // Call the base class implementation
    //     return llm::LLMNode::add_node(name, input_names, node);
    // }

    void execute(std::shared_ptr<llm::LLMContext> context) override
    {
        pybind11 ::gil_scoped_acquire gil;

        pybind11 ::function override = pybind11 ::get_override(static_cast<const BaseT*>(this), "execute");

        if (!override)
        {
            // Problem
            pybind11 ::pybind11_fail(
                "Tried to call pure virtual function \""
                "llm::LLMNodeBase"
                "::"
                "execute"
                "\"");
        }

        auto override_result = override(context);

        // Now determine if the override result is a coroutine or not
        if (py::module::import("asyncio").attr("iscoroutine")(override_result).cast<bool>())
        {
            py::print("Returned a coroutine");

            auto loop = py::module::import("asyncio").attr("get_running_loop")();

            // Need to schedule the result to run on the loop
            auto future = py::module::import("asyncio").attr("run_coroutine_threadsafe")(override_result, loop);

            // We are a dask future. Quickly check if its done, then release
            while (!future.attr("done")().cast<bool>())
            {
                // Release the GIL and wait for it to be done
                py::gil_scoped_release nogil;

                boost::this_fiber::yield();
            }

            // // Completed, move into the returned object
            // override_result = future.attr("result")();
        }
        else
        {
            py::print("Did not return a coroutine");
        }

        // // Now cast back to the C++ type
        // if (pybind11 ::detail ::cast_is_temporary_value_reference<return_t>::value)
        // {
        //     static pybind11 ::detail ::override_caster_t<return_t> caster;
        //     return pybind11 ::detail ::cast_ref<return_t>(std ::move(override_result), caster);
        // }
        // return pybind11 ::detail ::cast_safe<return_t>(std ::move(override_result));
    }

  private:
    std::map<std::shared_ptr<llm::LLMNodeBase>, py::object> m_py_nodes;
};

template <class BaseT = llm::LLMNode>
class PyLLMNode : public PyLLMNodeBase<BaseT>
{
  public:
    using PyLLMNodeBase<BaseT>::PyLLMNodeBase;

    std::shared_ptr<llm::LLMNodeRunner> add_node(std::string name,
                                                 std::vector<std::string> input_names,
                                                 std::shared_ptr<llm::LLMNodeBase> node) override
    {
        // // Try to cast the object to a python object to ensure that we keep it alive
        // auto py_node = std::dynamic_pointer_cast<PyLLMNodeBase>(node);

        // if (py_node)
        // {
        // Store the python object to keep it alive
        m_py_nodes[node] = py::cast(node);
        // }

        // Call the base class implementation
        return llm::LLMNode::add_node(name, input_names, node);
    }

    // void execute(std::shared_ptr<llm::LLMContext> context) override
    // {
    //     pybind11 ::gil_scoped_acquire gil;

    //     pybind11 ::function override = pybind11 ::get_override(static_cast<const BaseT*>(this), "execute");

    //     if (!override)
    //     {
    //         // Problem
    //         pybind11 ::pybind11_fail(
    //             "Tried to call pure virtual function \""
    //             "llm::LLMNodeBase"
    //             "::"
    //             "execute"
    //             "\"");
    //     }

    //     auto override_result = override(context);

    //     // Now determine if the override result is a coroutine or not
    //     if (py::module::import("asyncio").attr("iscoroutine")(override_result).cast<bool>())
    //     {
    //         py::print("Returned a coroutine");

    //         auto loop = py::module::import("asyncio").attr("get_running_loop")();

    //         // Need to schedule the result to run on the loop
    //         auto future = py::module::import("asyncio").attr("run_coroutine_threadsafe")(override_result, loop);

    //         // We are a dask future. Quickly check if its done, then release
    //         while (!future.attr("done")().cast<bool>())
    //         {
    //             // Release the GIL and wait for it to be done
    //             py::gil_scoped_release nogil;

    //             boost::this_fiber::yield();
    //         }

    //         // // Completed, move into the returned object
    //         // override_result = future.attr("result")();
    //     }
    //     else
    //     {
    //         py::print("Did not return a coroutine");
    //     }

    //     // // Now cast back to the C++ type
    //     // if (pybind11 ::detail ::cast_is_temporary_value_reference<return_t>::value)
    //     // {
    //     //     static pybind11 ::detail ::override_caster_t<return_t> caster;
    //     //     return pybind11 ::detail ::cast_ref<return_t>(std ::move(override_result), caster);
    //     // }
    //     // return pybind11 ::detail ::cast_safe<return_t>(std ::move(override_result));
    // }

  private:
    std::map<std::shared_ptr<llm::LLMNodeBase>, py::object> m_py_nodes;
};

// class PyLLMNode : public PyLLMNode<llm::LLMNode>
// {
//   public:
//     //     std::shared_ptr<llm::LLMNodeRunner> add_node(std::string name,
//     //                                                  std::vector<std::string> input_names,
//     //                                                  std::shared_ptr<LLMNodeBase> node) override
//     //     {
//     //         // Try to cast the object to a python object to ensure that we keep it alive
//     //         auto py_node = std::dynamic_pointer_cast<PyLLMNodeBase>(node);

//     //         if (py_node)
//     //         {
//     //             // Store the python object to keep it alive
//     //             m_py_nodes[node] = py::cast(node);

//     //             // // Also, set the loop on the service
//     //             // py_node->set_loop(m_loop);
//     //         }

//     //         // Call the base class implementation
//     //         return llm::LLMNode::add_node(name, input_names, node);
//     //     }

//     //   private:
//     //     std::map<std::shared_ptr<llm::LLMNodeBase>, py::object> m_py_nodes;
// };

class PyLLMEngine : public PyLLMNode<llm::LLMEngine>
{
  public:
    PyLLMEngine() : PyLLMNode<llm::LLMEngine>()
    {
        // std::promise<void> loop_ready;

        // auto future = loop_ready.get_future();

        // auto setup_debugging = create_gil_initializer();

        // m_thread = std::thread(
        //     [this](std::promise<void> loop_ready, std::function<void()> setup_debugging) {
        //         // Acquire the GIL (and also initialize the ThreadState)
        //         py::gil_scoped_acquire acquire;

        //         // Initialize the debugger
        //         setup_debugging();

        //         py::print("Creating loop");

        //         // Gets (or more likely, creates) an event loop and runs it forever until stop is called
        //         m_loop = py::module::import("asyncio").attr("new_event_loop")();

        //         py::print("Setting loop current");

        //         // Set the event loop as the current event loop
        //         py::module::import("asyncio").attr("set_event_loop")(m_loop);

        //         py::print("Signaling promise");

        //         // Signal we are ready
        //         loop_ready.set_value();

        //         py::print("Running forever");

        //         m_loop.attr("run_forever")();
        //     },
        //     std::move(loop_ready),
        //     std::move(setup_debugging));

        // py::print("Waiting for startup");
        // {
        //     // Free the GIL otherwise we deadlock
        //     py::gil_scoped_release nogil;

        //     future.get();
        // }

        // // Finally, try and see if our LLM Service is a python object and keep it alive
        // auto py_llm_service = std::dynamic_pointer_cast<PyLLMService>(llm_service);

        // if (py_llm_service)
        // {
        //     // Store the python object to keep it alive
        //     m_py_llm_service = py::cast(llm_service);

        //     // Also, set the loop on the service
        //     py_llm_service->set_loop(m_loop);
        // }

        // py::print("Engine started");
    }

    ~PyLLMEngine()
    {
        // Acquire the GIL on this thread and call stop on the event loop
        py::gil_scoped_acquire acquire;

        m_loop.attr("stop")();

        // Finally, join on the thread
        m_thread.join();
    }

    void add_prompt_generator(std::shared_ptr<llm::LLMPromptGenerator> prompt_generator) override
    {
        // Try to cast the object to a python object to ensure that we keep it alive
        auto py_prompt_generator = std::dynamic_pointer_cast<PyLLMPromptGenerator>(prompt_generator);

        if (py_prompt_generator)
        {
            // Store the python object to keep it alive
            m_py_prompt_generators[prompt_generator] = py::cast(prompt_generator);

            // Also, set the loop on the service
            py_prompt_generator->set_loop(m_loop);
        }

        // Call the base class implementation
        llm::LLMEngine::add_prompt_generator(prompt_generator);
    }

    void add_task_handler(std::shared_ptr<llm::LLMTaskHandler> task_handler) override
    {
        // Try to cast the object to a python object to ensure that we keep it alive
        auto py_task_handler = std::dynamic_pointer_cast<PyLLMTaskHandler>(task_handler);

        if (py_task_handler)
        {
            // Store the python object to keep it alive
            m_py_task_handler[task_handler] = py::cast(task_handler);

            // Also, set the loop on the service
            py_task_handler->set_loop(m_loop);
        }

        // Call the base class implementation
        llm::LLMEngine::add_task_handler(task_handler);
    }

    const py::object& get_loop() const
    {
        return m_loop;
    }

    // std::vector<std::shared_ptr<ControlMessage>> run(std::shared_ptr<ControlMessage> input_message) override
    // {
    //     std::vector<std::shared_ptr<ControlMessage>> output_messages;

    //     return output_messages;
    // }

  private:
    std::thread m_thread;
    py::object m_loop;

    // Keep the python objects alive by saving references in this object
    py::object m_py_llm_service;
    std::map<std::shared_ptr<llm::LLMPromptGenerator>, py::object> m_py_prompt_generators;
    std::map<std::shared_ptr<llm::LLMTaskHandler>, py::object> m_py_task_handler;
};

PYBIND11_MODULE(stages, _module)
{
    _module.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: morpheus.stages
        .. autosummary::
           :toctree: _generate

        )pbdoc";

    // Load the cudf helpers
    CudfHelper::load();

    mrc::pymrc::from_import(_module, "morpheus._lib.common", "FilterSource");

    py::class_<mrc::segment::Object<AddClassificationsStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<AddClassificationsStage>>>(
        _module, "AddClassificationsStage", py::multiple_inheritance())
        .def(py::init<>(&AddClassificationStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("idx2label"),
             py::arg("threshold"));

    py::class_<mrc::segment::Object<AddScoresStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<AddScoresStage>>>(
        _module, "AddScoresStage", py::multiple_inheritance())
        .def(
            py::init<>(&AddScoresStageInterfaceProxy::init), py::arg("builder"), py::arg("name"), py::arg("idx2label"));

    py::class_<mrc::segment::Object<DeserializeStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<DeserializeStage>>>(
        _module, "DeserializeStage", py::multiple_inheritance())
        .def(py::init<>(&DeserializeStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("batch_size"),
             py::arg("ensure_sliceable_index") = true);

    py::class_<mrc::segment::Object<FileSourceStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<FileSourceStage>>>(
        _module, "FileSourceStage", py::multiple_inheritance())
        .def(py::init<>(&FileSourceStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("filename"),
             py::arg("repeat"),
             py::arg("parser_kwargs"));

    py::class_<mrc::segment::Object<FilterDetectionsStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<FilterDetectionsStage>>>(
        _module, "FilterDetectionsStage", py::multiple_inheritance())
        .def(py::init<>(&FilterDetectionStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("threshold"),
             py::arg("copy"),
             py::arg("filter_source"),
             py::arg("field_name") = "probs");

    py::class_<mrc::segment::Object<InferenceClientStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<InferenceClientStage>>>(
        _module, "InferenceClientStage", py::multiple_inheritance())
        .def(py::init<>(&InferenceClientStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("model_name"),
             py::arg("server_url"),
             py::arg("force_convert_inputs"),
             py::arg("use_shared_memory"),
             py::arg("needs_logits"),
             py::arg("inout_mapping") = py::dict());

    py::class_<mrc::segment::Object<KafkaSourceStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<KafkaSourceStage>>>(
        _module, "KafkaSourceStage", py::multiple_inheritance())
        .def(py::init<>(&KafkaSourceStageInterfaceProxy::init_with_single_topic),
             py::arg("builder"),
             py::arg("name"),
             py::arg("max_batch_size"),
             py::arg("topic"),
             py::arg("batch_timeout_ms"),
             py::arg("config"),
             py::arg("disable_commits")       = false,
             py::arg("disable_pre_filtering") = false,
             py::arg("stop_after")            = 0,
             py::arg("async_commits")         = true)
        .def(py::init<>(&KafkaSourceStageInterfaceProxy::init_with_multiple_topics),
             py::arg("builder"),
             py::arg("name"),
             py::arg("max_batch_size"),
             py::arg("topics"),
             py::arg("batch_timeout_ms"),
             py::arg("config"),
             py::arg("disable_commits")       = false,
             py::arg("disable_pre_filtering") = false,
             py::arg("stop_after")            = 0,
             py::arg("async_commits")         = true);

    py::class_<mrc::segment::Object<PreallocateStage<MessageMeta>>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PreallocateStage<MessageMeta>>>>(
        _module, "PreallocateMessageMetaStage", py::multiple_inheritance())
        .def(py::init<>(&PreallocateStageInterfaceProxy<MessageMeta>::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("needed_columns"));

    py::class_<mrc::segment::Object<PreallocateStage<MultiMessage>>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PreallocateStage<MultiMessage>>>>(
        _module, "PreallocateMultiMessageStage", py::multiple_inheritance())
        .def(py::init<>(&PreallocateStageInterfaceProxy<MultiMessage>::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("needed_columns"));

    py::class_<mrc::segment::Object<PreprocessFILStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PreprocessFILStage>>>(
        _module, "PreprocessFILStage", py::multiple_inheritance())
        .def(py::init<>(&PreprocessFILStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("features"));

    py::class_<mrc::segment::Object<PreprocessNLPStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PreprocessNLPStage>>>(
        _module, "PreprocessNLPStage", py::multiple_inheritance())
        .def(py::init<>(&PreprocessNLPStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("vocab_hash_file"),
             py::arg("sequence_length"),
             py::arg("truncation"),
             py::arg("do_lower_case"),
             py::arg("add_special_token"),
             py::arg("stride"),
             py::arg("column"));

    py::class_<mrc::segment::Object<SerializeStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<SerializeStage>>>(
        _module, "SerializeStage", py::multiple_inheritance())
        .def(py::init<>(&SerializeStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("include"),
             py::arg("exclude"),
             py::arg("fixed_columns") = true);

    py::class_<mrc::segment::Object<WriteToFileStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<WriteToFileStage>>>(
        _module, "WriteToFileStage", py::multiple_inheritance())
        .def(py::init<>(&WriteToFileStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("filename"),
             py::arg("mode")              = "w",
             py::arg("file_type")         = FileTypes::Auto,
             py::arg("include_index_col") = true,
             py::arg("flush")             = false);

    py::class_<CoroAwaitable, std::shared_ptr<CoroAwaitable>>(_module, "CoroAwaitable")
        .def(py::init<>())
        .def("__iter__", &CoroAwaitable::iter)
        .def("__await__", &CoroAwaitable::await)
        .def("__next__", &CoroAwaitable::next);

    py::class_<llm::LLMTask>(_module, "LLMTask")
        .def(py::init<>())
        .def(py::init([](std::string task_type, py::dict task_dict) {
            return llm::LLMTask(std::move(task_type), mrc::pymrc::cast_from_pyobject(task_dict));
        }))
        .def_readonly("task_type", &llm::LLMTask::task_type)
        .def("__getitem__",
             [](const llm::LLMTask& self, const std::string& key) {
                 try
                 {
                     return mrc::pymrc::cast_from_json(self.get(key));
                 } catch (const std::out_of_range&)
                 {
                     throw py::key_error("key '" + key + "' does not exist");
                 }
             })
        .def("__setitem__",
             [](llm::LLMTask& self, const std::string& key, py::object value) {
                 try
                 {
                     // Convert to C++ nholman object

                     return self.set(key, mrc::pymrc::cast_from_pyobject(std::move(value)));
                 } catch (const std::out_of_range&)
                 {
                     throw py::key_error("key '" + key + "' does not exist");
                 }
             })
        .def("__len__", &llm::LLMTask::size)
        .def("get", [](const llm::LLMTask& self, const std::string& key, py::object default_value) {
            try
            {
                return mrc::pymrc::cast_from_json(self.get(key));
            } catch (const nlohmann::detail::out_of_range&)
            {
                return default_value;
            }
        });
    // .def(
    //     "__iter__",
    //     [](const StringMap& map) { return py::make_key_iterator(map.begin(), map.end()); },
    //     py::keep_alive<0, 1>())
    // .def(
    //     "items",
    //     [](const StringMap& map) { return py::make_iterator(map.begin(), map.end()); },
    //     py::keep_alive<0, 1>())
    // .def(
    //     "values",
    //     [](const StringMap& map) { return py::make_value_iterator(map.begin(), map.end()); },
    //     py::keep_alive<0, 1>());

    py::class_<llm::LLMGeneratePrompt>(_module, "LLMGeneratePrompt")
        .def(py::init<>())
        .def(py::init([](std::string model_name, py::dict model_kwargs, std::vector<std::string> prompts) {
                 return llm::LLMGeneratePrompt(
                     std::move(model_name), mrc::pymrc::cast_from_pyobject(model_kwargs), std::move(prompts));
             }),
             py::arg("model_name"),
             py::arg("model_kwargs"),
             py::arg("prompts"))
        .def_readwrite("model_name", &llm::LLMGeneratePrompt::model_name)
        .def_property(
            "model_kwargs",
            [](llm::LLMGeneratePrompt& self) { return mrc::pymrc::cast_from_json(self.model_kwargs); },
            [](llm::LLMGeneratePrompt& self, py::dict model_kwargs) {
                self.model_kwargs = mrc::pymrc::cast_from_pyobject(model_kwargs);
            })
        .def_readwrite("prompts", &llm::LLMGeneratePrompt::prompts);

    py::class_<llm::LLMGenerateResult, llm::LLMGeneratePrompt>(_module, "LLMGenerateResult")
        .def(py::init([]() { return llm::LLMGenerateResult(); }))
        .def(py::init([](llm::LLMGeneratePrompt& other, std::vector<std::string> responses) {
            return llm::LLMGenerateResult(other, std::move(responses));
        }))
        .def_readwrite("responses", &llm::LLMGenerateResult::responses);

    py::class_<llm::LLMContext, std::shared_ptr<llm::LLMContext>>(_module, "LLMContext")
        .def("task", [](llm::LLMContext& self) { return self.task(); })
        .def("message", [](llm::LLMContext& self) { return self.message(); })
        .def("get_input",
             [](llm::LLMContext& self) {
                 // Convert the return value
                 return mrc::pymrc::cast_from_json(self.get_input());
             })
        .def("set_output", [](llm::LLMContext& self, py::dict value) {
            // Convert and pass to the base
            self.set_output(mrc::pymrc::cast_from_pyobject(value));
        });

    py::class_<llm::LLMNodeBase, PyLLMNodeBase<>, std::shared_ptr<llm::LLMNodeBase>>(_module, "LLMNodeBase")
        .def(py::init_alias<>())
        .def("execute", &llm::LLMNodeBase::execute);

    py::class_<llm::LLMNodeRunner, std::shared_ptr<llm::LLMNodeRunner>>(_module, "LLMNodeRunner")
        .def_property_readonly("name", &llm::LLMNodeRunner::name)
        .def_property_readonly("input_names", &llm::LLMNodeRunner::input_names)
        .def("execute", &llm::LLMNodeRunner::execute);

    py::class_<llm::LLMNode, llm::LLMNodeBase, PyLLMNode<>, std::shared_ptr<llm::LLMNode>>(_module, "LLMNode")
        .def(py::init_alias<>())
        .def("add_node", &llm::LLMNode::add_node)
        .def("execute", &llm::LLMNode::execute);

    auto LLMService =
        py::class_<llm::LLMService, PyLLMService, std::shared_ptr<llm::LLMService>>(_module, "LLMService");

    auto LLMPromptGenerator =
        py::class_<llm::LLMPromptGenerator, PyLLMPromptGenerator, std::shared_ptr<llm::LLMPromptGenerator>>(
            _module, "LLMPromptGenerator");

    auto LLMTaskHandler = py::class_<llm::LLMTaskHandler, PyLLMTaskHandler, std::shared_ptr<llm::LLMTaskHandler>>(
        _module, "LLMTaskHandler");

    auto LLMEngine = py::class_<llm::LLMEngine, PyLLMEngine, std::shared_ptr<llm::LLMEngine>>(_module, "LLMEngine");

    LLMService.def(py::init<>()).def("generate", &llm::LLMService::generate);

    LLMPromptGenerator.def(py::init<>()).def("try_handle", &llm::LLMPromptGenerator::try_handle);

    LLMTaskHandler.def(py::init<>()).def("try_handle", &llm::LLMTaskHandler::try_handle);

    LLMEngine
        .def(py::init_alias<>())
        // .def(py::init([](std::shared_ptr<llm::LLMService> llm_service) {
        //     return std::make_shared<PyLLMEngine>(std::move(llm_service));
        // }))
        .def("add_prompt_generator", &llm::LLMEngine::add_prompt_generator, py::arg("prompt_generator"))
        .def("add_task_handler", &llm::LLMEngine::add_task_handler, py::arg("task_handler"))
        .def("add_node", &llm::LLMEngine::add_node, py::arg("name"), py::arg("input_names"), py::arg("node"))
        .def("run", &llm::LLMEngine::run, py::arg("input_message"))
        .def(
            "arun",
            [](llm::LLMEngine& self, std::shared_ptr<ControlMessage> message) {
                auto double_task = [](std::uint64_t x) -> mrc::coroutines::Task<std::uint64_t> {
                    {
                        py::gil_scoped_acquire gil;
                        py::print("In double_task");
                    }
                    // Return the value
                    co_return x * 2;
                };

                auto convert_to_object_task([=](std::uint64_t x) -> mrc::coroutines::Task<py::object> {
                    auto doubled = co_await double_task(x);

                    py::gil_scoped_acquire gil;
                    py::print("In convert_to_object_task");

                    // Return the value
                    co_return py::int_(doubled);
                });

                return std::make_shared<CoroAwaitable>(convert_to_object_task(10));
            },
            "async def arun(self, arg0: morpheus._lib.messages.ControlMessage) -> CoroAwaitable");

    _module.attr("__version__") =
        MRC_CONCAT_STR(morpheus_VERSION_MAJOR << "." << morpheus_VERSION_MINOR << "." << morpheus_VERSION_PATCH);
}
}  // namespace morpheus
