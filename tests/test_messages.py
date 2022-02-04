import os
import importlib
import unittest
from unittest import mock

import cupy as cp

import morpheus._lib.messages as neom
from morpheus.config import Config
from morpheus.pipeline import messages
from tests import BaseMorpheusTest


class TestMessages(BaseMorpheusTest):
    def test_constructor_cpp(self):
        config = Config.get()
        config.use_cpp = True

        m = messages.MessageMeta(None)
        self.assertIsInstance(m, neom.MessageMeta)

        # UserMessageMeta doesn't contain a C++ impl, so we should
        # always received the python impl
        m = messages.UserMessageMeta(None, None)
        self.assertIsInstance(m, messages.UserMessageMeta)

        m = messages.MultiMessage(None, 0, 1)
        self.assertIsInstance(m, neom.MultiMessage)

        self.assertRaises(TypeError, messages.InferenceMemory, 1)

        cp_array = cp.zeros((1, 2))
        m = messages.InferenceMemoryNLP(1, cp_array, cp_array, cp_array)
        self.assertIsInstance(m, neom.InferenceMemoryNLP)

        m = messages.InferenceMemoryFIL(1, cp_array, cp_array)
        self.assertIsInstance(m, neom.InferenceMemoryFIL)

        # No C++ impl, should always get the Python class
        m = messages.InferenceMemoryAE(1, cp_array, cp_array)
        self.assertIsInstance(m, messages.InferenceMemoryAE)

        m = messages.MultiInferenceMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, neom.MultiInferenceMessage)

        m = messages.MultiInferenceNLPMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, neom.MultiInferenceNLPMessage)

        m = messages.MultiInferenceFILMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, neom.MultiInferenceFILMessage)

        self.assertRaises(TypeError, messages.ResponseMemory, 1)

        m = messages.ResponseMemoryProbs(1, cp_array)
        self.assertIsInstance(m, neom.ResponseMemoryProbs)

        # No C++ impl
        m = messages.ResponseMemoryAE(1, cp_array)
        self.assertIsInstance(m, messages.ResponseMemoryAE)

        m = messages.MultiResponseMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, neom.MultiResponseMessage)

        m = messages.MultiResponseProbsMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, neom.MultiResponseProbsMessage)

        # No C++ impl
        m = messages.MultiResponseAEMessage(None, 0, 1, None, 0, 1, '')
        self.assertIsInstance(m, messages.MultiResponseAEMessage)

    def test_constructor_no_cpp(self):
        config = Config.get()
        config.use_cpp = False

        m = messages.MessageMeta(None)
        self.assertIsInstance(m, messages.MessageMeta)

        m = messages.UserMessageMeta(None, None)
        self.assertIsInstance(m, messages.UserMessageMeta)

        m = messages.MultiMessage(None, 0, 1)
        self.assertIsInstance(m, messages.MultiMessage)

        m = messages.InferenceMemory(1)
        self.assertIsInstance(m, messages.InferenceMemory)

        cp_array = cp.zeros((1, 2))
        m = messages.InferenceMemoryNLP(1, cp_array, cp_array, cp_array)
        self.assertIsInstance(m, messages.InferenceMemoryNLP)

        m = messages.InferenceMemoryFIL(1, cp_array, cp_array)
        self.assertIsInstance(m, messages.InferenceMemoryFIL)

        # No C++ impl, should always get the Python class
        m = messages.InferenceMemoryAE(1, cp_array, cp_array)
        self.assertIsInstance(m, messages.InferenceMemoryAE)

        m = messages.MultiInferenceMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiInferenceMessage)

        m = messages.MultiInferenceNLPMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiInferenceNLPMessage)

        m = messages.MultiInferenceFILMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiInferenceFILMessage)

        m = messages.ResponseMemory(1)
        self.assertIsInstance(m, messages.ResponseMemory)

        m = messages.ResponseMemoryProbs(1, cp_array)
        self.assertIsInstance(m, messages.ResponseMemoryProbs)

        # No C++ impl
        m = messages.ResponseMemoryAE(1, cp_array)
        self.assertIsInstance(m, messages.ResponseMemoryAE)

        m = messages.MultiResponseMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiResponseMessage)

        m = messages.MultiResponseProbsMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiResponseProbsMessage)

        m = messages.MultiResponseAEMessage(None, 0, 1, None, 0, 1, '')
        self.assertIsInstance(m, messages.MultiResponseAEMessage)

    def test_constructor_no_cpp_env(self):
        """
        Cleanups are called in reverse order.
        _save_env_vars will add a cleanup handler to remove the `MORPHEUS_NO_CPP` environment variable
        then our own handler will reload the messages module.
        Both cleanups will be called even if the test fails
        """
        self.addCleanup(importlib.reload, messages)
        self._save_env_vars()

        os.environ['MORPHEUS_NO_CPP']='1'
        importlib.reload(messages)

        config = Config.get()
        config.use_cpp = True

        m = messages.MessageMeta(None)
        self.assertIsInstance(m, messages.MessageMeta)

        m = messages.UserMessageMeta(None, None)
        self.assertIsInstance(m, messages.UserMessageMeta)

        m = messages.MultiMessage(None, 0, 1)
        self.assertIsInstance(m, messages.MultiMessage)

        m = messages.InferenceMemory(1)
        self.assertIsInstance(m, messages.InferenceMemory)

        cp_array = cp.zeros((1, 2))
        m = messages.InferenceMemoryNLP(1, cp_array, cp_array, cp_array)
        self.assertIsInstance(m, messages.InferenceMemoryNLP)

        m = messages.InferenceMemoryFIL(1, cp_array, cp_array)
        self.assertIsInstance(m, messages.InferenceMemoryFIL)

        m = messages.InferenceMemoryAE(1, cp_array, cp_array)
        self.assertIsInstance(m, messages.InferenceMemoryAE)

        m = messages.MultiInferenceMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiInferenceMessage)

        m = messages.MultiInferenceNLPMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiInferenceNLPMessage)

        m = messages.MultiInferenceFILMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiInferenceFILMessage)

        m = messages.ResponseMemory(1)
        self.assertIsInstance(m, messages.ResponseMemory)

        m = messages.ResponseMemoryProbs(1, cp_array)
        self.assertIsInstance(m, messages.ResponseMemoryProbs)

        m = messages.ResponseMemoryAE(1, cp_array)
        self.assertIsInstance(m, messages.ResponseMemoryAE)

        m = messages.MultiResponseMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiResponseMessage)

        m = messages.MultiResponseProbsMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiResponseProbsMessage)

        m = messages.MultiResponseAEMessage(None, 0, 1, None, 0, 1, '')
        self.assertIsInstance(m, messages.MultiResponseAEMessage)


if __name__ == '__main__':
    unittest.main()
