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
#include "mrc/node/rx_sink_base.hpp"
#include "mrc/node/rx_source_base.hpp"
#include "mrc/types.hpp"

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

#include <boost/fiber/future/future.hpp>
#include <glog/logging.h>
#include <mrc/segment/object.hpp>
#include <mrc/utils/string_utils.hpp>
#include <pybind11/attr.h>  // for multiple_inheritance
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // for arg, init, class_, module_, str_attr_accessor, PYBIND11_MODULE, pybind11
#include <pybind11/pytypes.h>   // for dict, sequence
#include <pybind11/stl.h>
#include <pymrc/utils.hpp>  // for pymrc::import
#include <rxcpp/rx.hpp>

#include <future>
#include <map>
#include <memory>
#include <sstream>
#include <thread>
#include <vector>

namespace morpheus {
namespace py = pybind11;

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

class PyLLMEngine : public llm::LLMEngine
{
  public:
    PyLLMEngine() : llm::LLMEngine()
    {
        std::promise<void> loop_ready;

        auto future = loop_ready.get_future();

        auto setup_debugging = create_gil_initializer();

        m_thread = std::thread(
            [this](std::promise<void> loop_ready, std::function<void()> setup_debugging) {
                // Acquire the GIL (and also initialize the ThreadState)
                py::gil_scoped_acquire acquire;

                // Initialize the debugger
                setup_debugging();

                py::print("Creating loop");

                // Gets (or more likely, creates) an event loop and runs it forever until stop is called
                m_loop = py::module::import("asyncio").attr("new_event_loop")();

                py::print("Setting loop current");

                // Set the event loop as the current event loop
                py::module::import("asyncio").attr("set_event_loop")(m_loop);

                py::print("Signaling promise");

                // Signal we are ready
                loop_ready.set_value();

                py::print("Running forever");

                m_loop.attr("run_forever")();
            },
            std::move(loop_ready),
            std::move(setup_debugging));

        py::print("Waiting for startup");
        {
            // Free the GIL otherwise we deadlock
            py::gil_scoped_release nogil;

            future.get();
        }

        py::print("Engine started");
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
        auto py_prompt_generator = py::cast(prompt_generator);

        // Store the prompt generator in an array to keep it alive
        m_py_prompt_generators[prompt_generator] = py_prompt_generator;

        // Call the base class implementation
        llm::LLMEngine::add_prompt_generator(prompt_generator);
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

    std::map<std::shared_ptr<llm::LLMPromptGenerator>, py::object> m_py_prompt_generators;
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

            // We need to schedule the coroutine to run on the event loop. Cast the llm engine to get that
            auto& py_engine = dynamic_cast<PyLLMEngine&>(engine);

            // Need to schedule the result to run on the loop
            auto future =
                py::module::import("asyncio").attr("run_coroutine_threadsafe")(override_result, py_engine.get_loop());

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

    py::class_<llm::LLMTask>(_module, "LLMTask").def(py::init<>());

    py::class_<llm::LLMGeneratePrompt>(_module, "LLMGeneratePrompt").def(py::init<>());

    py::class_<llm::LLMGenerateResult>(_module, "LLMGenerateResult").def(py::init<>());

    auto LLMPromptGenerator =
        py::class_<llm::LLMPromptGenerator, PyLLMPromptGenerator, std::shared_ptr<llm::LLMPromptGenerator>>(
            _module, "LLMPromptGenerator");

    auto LLMEngine = py::class_<llm::LLMEngine, PyLLMEngine, std::shared_ptr<llm::LLMEngine>>(_module, "LLMEngine");

    LLMPromptGenerator.def(py::init<>()).def("try_handle", &llm::LLMPromptGenerator::try_handle);

    LLMEngine.def(py::init_alias<>())
        .def("add_prompt_generator", &llm::LLMEngine::add_prompt_generator, py::arg("prompt_generator"))
        .def("run", &llm::LLMEngine::run, py::arg("input_message"));

    _module.attr("__version__") =
        MRC_CONCAT_STR(morpheus_VERSION_MAJOR << "." << morpheus_VERSION_MINOR << "." << morpheus_VERSION_PATCH);
}
}  // namespace morpheus
