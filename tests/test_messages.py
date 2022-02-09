import os
import importlib
import unittest

import cupy as cp

import morpheus._lib.messages as neom
from morpheus.config import Config
from morpheus.pipeline import messages
from tests import BaseMorpheusTest


class TestMessages(BaseMorpheusTest):
    def check_message(self: unittest.TestCase,
                      python_type: type,
                      cpp_type: type,
                      should_be_cpp: bool,
                      no_cpp_class: bool,
                      args: tuple):

        instance = python_type(*args)

        # Check that the C++ type is set in the class
        self.assertIs(python_type._cpp_class, None if no_cpp_class else cpp_type)

        # Check that the isinstance to Python type works
        self.assertIsInstance(instance, python_type)

        # Check that the instantiated class is the right type
        self.assertIs(instance.__class__, cpp_type if should_be_cpp and cpp_type is not None else python_type)

    def check_all_messages(self, should_be_cpp: bool, no_cpp_class: bool):

        self.check_message(messages.MessageMeta, neom.MessageMeta, should_be_cpp, no_cpp_class, (None, ))

        # UserMessageMeta doesn't contain a C++ impl, so we should
        # always received the python impl
        self.check_message(messages.UserMessageMeta, None, should_be_cpp, no_cpp_class, (None, None))

        self.check_message(messages.MultiMessage, neom.MultiMessage, should_be_cpp, no_cpp_class, (None, 0, 1))

        self.assertIs(messages.InferenceMemory._cpp_class, None if no_cpp_class else neom.InferenceMemory)
        # C++ impl for InferenceMemory doesn't have a constructor
        if (should_be_cpp):
            self.assertRaises(TypeError, messages.InferenceMemory, 1)

        cp_array = cp.zeros((1, 2))

        self.check_message(messages.InferenceMemoryNLP,
                           neom.InferenceMemoryNLP,
                           should_be_cpp,
                           no_cpp_class, (1, cp_array, cp_array, cp_array))

        self.check_message(messages.InferenceMemoryFIL,
                           neom.InferenceMemoryFIL,
                           should_be_cpp,
                           no_cpp_class, (1, cp_array, cp_array))

        # No C++ impl, should always get the Python class
        self.check_message(messages.InferenceMemoryAE, None, should_be_cpp, no_cpp_class, (1, cp_array, cp_array))

        self.check_message(messages.MultiInferenceMessage,
                           neom.MultiInferenceMessage,
                           should_be_cpp,
                           no_cpp_class, (None, 0, 1, None, 0, 1))

        self.check_message(messages.MultiInferenceNLPMessage,
                           neom.MultiInferenceNLPMessage,
                           should_be_cpp,
                           no_cpp_class, (None, 0, 1, None, 0, 1))

        self.check_message(messages.MultiInferenceFILMessage,
                           neom.MultiInferenceFILMessage,
                           should_be_cpp,
                           no_cpp_class, (None, 0, 1, None, 0, 1))

        self.assertIs(messages.ResponseMemory._cpp_class, None if no_cpp_class else neom.ResponseMemory)
        # C++ impl doesn't have a constructor
        if (should_be_cpp):
            self.assertRaises(TypeError, messages.ResponseMemory, 1)

        self.check_message(messages.ResponseMemoryProbs,
                           neom.ResponseMemoryProbs,
                           should_be_cpp,
                           no_cpp_class, (1, cp_array))

        # No C++ impl
        self.check_message(messages.ResponseMemoryAE, None, should_be_cpp, no_cpp_class, (1, cp_array))

        self.check_message(messages.MultiResponseMessage,
                           neom.MultiResponseMessage,
                           should_be_cpp,
                           no_cpp_class, (None, 0, 1, None, 0, 1))

        self.check_message(messages.MultiResponseProbsMessage,
                           neom.MultiResponseProbsMessage,
                           should_be_cpp,
                           no_cpp_class, (None, 0, 1, None, 0, 1))

        # No C++ impl
        self.check_message(messages.MultiResponseAEMessage,
                           None,
                           should_be_cpp,
                           no_cpp_class, (None, 0, 1, None, 0, 1, ''))

    def test_constructor_cpp(self):
        config = Config.get()
        config.use_cpp = True

        self.check_all_messages(True, False)

    def test_constructor_no_cpp(self):
        config = Config.get()
        config.use_cpp = False

        self.check_all_messages(False, False)

    def test_constructor_no_cpp_env(self):
        """
        Cleanups are called in reverse order.
        _save_env_vars will add a cleanup handler to remove the `MORPHEUS_NO_CPP` environment variable
        then our own handler will reload the messages module.
        Both cleanups will be called even if the test fails
        """
        self.addCleanup(importlib.reload, messages)
        self._save_env_vars()

        os.environ['MORPHEUS_NO_CPP'] = '1'
        importlib.reload(messages)

        config = Config.get()
        config.use_cpp = True

        self.check_all_messages(False, True)


if __name__ == '__main__':
    unittest.main()
