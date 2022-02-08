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

        self.assertIs(messages.MessageMeta.get_impl_class(), neom.MessageMeta)
        m = messages.MessageMeta(None)
        self.assertIsInstance(m, neom.MessageMeta)

        # UserMessageMeta doesn't contain a C++ impl, so we should
        # always received the python impl
        self.assertIs(messages.UserMessageMeta.get_impl_class(), messages.UserMessageMeta)
        m = messages.UserMessageMeta(None, None)
        self.assertIsInstance(m, messages.UserMessageMeta)

        self.assertIs(messages.MultiMessage.get_impl_class(), neom.MultiMessage)
        m = messages.MultiMessage(None, 0, 1)
        self.assertIsInstance(m, neom.MultiMessage)

        self.assertIs(messages.InferenceMemory.get_impl_class(), neom.InferenceMemory)
        # C++ impl for InferenceMemory doesn't have a constructor
        self.assertRaises(TypeError, messages.InferenceMemory, 1)

        cp_array = cp.zeros((1, 2))

        self.assertIs(messages.InferenceMemoryNLP.get_impl_class(), neom.InferenceMemoryNLP)
        m = messages.InferenceMemoryNLP(1, cp_array, cp_array, cp_array)
        self.assertIsInstance(m, neom.InferenceMemoryNLP)

        self.assertIs(messages.InferenceMemoryFIL.get_impl_class(), neom.InferenceMemoryFIL)
        m = messages.InferenceMemoryFIL(1, cp_array, cp_array)
        self.assertIsInstance(m, neom.InferenceMemoryFIL)

        # No C++ impl, should always get the Python class
        self.assertIs(messages.InferenceMemoryAE.get_impl_class(), messages.InferenceMemoryAE)
        m = messages.InferenceMemoryAE(1, cp_array, cp_array)
        self.assertIsInstance(m, messages.InferenceMemoryAE)

        self.assertIs(messages.MultiInferenceMessage.get_impl_class(), neom.MultiInferenceMessage)
        m = messages.MultiInferenceMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, neom.MultiInferenceMessage)

        self.assertIs(messages.MultiInferenceNLPMessage.get_impl_class(), neom.MultiInferenceNLPMessage)
        m = messages.MultiInferenceNLPMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, neom.MultiInferenceNLPMessage)

        self.assertIs(messages.MultiInferenceFILMessage.get_impl_class(), neom.MultiInferenceFILMessage)
        m = messages.MultiInferenceFILMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, neom.MultiInferenceFILMessage)

        self.assertIs(messages.ResponseMemory.get_impl_class(), neom.ResponseMemory)
        # C++ impl doesn't have a constructor
        self.assertRaises(TypeError, messages.ResponseMemory, 1)

        self.assertIs(messages.ResponseMemoryProbs.get_impl_class(), neom.ResponseMemoryProbs)
        m = messages.ResponseMemoryProbs(1, cp_array)
        self.assertIsInstance(m, neom.ResponseMemoryProbs)

        # No C++ impl
        self.assertIs(messages.ResponseMemoryAE.get_impl_class(), messages.ResponseMemoryAE)
        m = messages.ResponseMemoryAE(1, cp_array)
        self.assertIsInstance(m, messages.ResponseMemoryAE)

        self.assertIs(messages.MultiResponseMessage.get_impl_class(), neom.MultiResponseMessage)
        m = messages.MultiResponseMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, neom.MultiResponseMessage)

        self.assertIs(messages.MultiResponseProbsMessage.get_impl_class(), neom.MultiResponseProbsMessage)
        m = messages.MultiResponseProbsMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, neom.MultiResponseProbsMessage)

        # No C++ impl
        self.assertIs(messages.MultiResponseAEMessage.get_impl_class(), messages.MultiResponseAEMessage)
        m = messages.MultiResponseAEMessage(None, 0, 1, None, 0, 1, '')
        self.assertIsInstance(m, messages.MultiResponseAEMessage)

    def test_constructor_no_cpp(self):
        config = Config.get()
        config.use_cpp = False

        self.assertIs(messages.MessageMeta.get_impl_class(), messages.MessageMeta)
        m = messages.MessageMeta(None)
        self.assertIsInstance(m, messages.MessageMeta)

        self.assertIs(messages.UserMessageMeta.get_impl_class(), messages.UserMessageMeta)
        m = messages.UserMessageMeta(None, None)
        self.assertIsInstance(m, messages.UserMessageMeta)

        self.assertIs(messages.MultiMessage.get_impl_class(), messages.MultiMessage)
        m = messages.MultiMessage(None, 0, 1)
        self.assertIsInstance(m, messages.MultiMessage)

        self.assertIs(messages.InferenceMemory.get_impl_class(), messages.InferenceMemory)
        m = messages.InferenceMemory(1)
        self.assertIsInstance(m, messages.InferenceMemory)

        cp_array = cp.zeros((1, 2))
        self.assertIs(messages.InferenceMemoryNLP.get_impl_class(), messages.InferenceMemoryNLP)
        m = messages.InferenceMemoryNLP(1, cp_array, cp_array, cp_array)
        self.assertIsInstance(m, messages.InferenceMemoryNLP)

        self.assertIs(messages.InferenceMemoryFIL.get_impl_class(), messages.InferenceMemoryFIL)
        m = messages.InferenceMemoryFIL(1, cp_array, cp_array)
        self.assertIsInstance(m, messages.InferenceMemoryFIL)

        self.assertIs(messages.InferenceMemoryAE.get_impl_class(), messages.InferenceMemoryAE)
        m = messages.InferenceMemoryAE(1, cp_array, cp_array)
        self.assertIsInstance(m, messages.InferenceMemoryAE)

        self.assertIs(messages.MultiInferenceMessage.get_impl_class(), messages.MultiInferenceMessage)
        m = messages.MultiInferenceMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiInferenceMessage)

        self.assertIs(messages.MultiInferenceNLPMessage.get_impl_class(), messages.MultiInferenceNLPMessage)
        m = messages.MultiInferenceNLPMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiInferenceNLPMessage)

        self.assertIs(messages.MultiInferenceFILMessage.get_impl_class(), messages.MultiInferenceFILMessage)
        m = messages.MultiInferenceFILMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiInferenceFILMessage)

        self.assertIs(messages.ResponseMemory.get_impl_class(), messages.ResponseMemory)
        m = messages.ResponseMemory(1)
        self.assertIsInstance(m, messages.ResponseMemory)

        self.assertIs(messages.ResponseMemoryProbs.get_impl_class(), messages.ResponseMemoryProbs)
        m = messages.ResponseMemoryProbs(1, cp_array)
        self.assertIsInstance(m, messages.ResponseMemoryProbs)

        self.assertIs(messages.ResponseMemoryAE.get_impl_class(), messages.ResponseMemoryAE)
        m = messages.ResponseMemoryAE(1, cp_array)
        self.assertIsInstance(m, messages.ResponseMemoryAE)

        self.assertIs(messages.MultiResponseMessage.get_impl_class(), messages.MultiResponseMessage)
        m = messages.MultiResponseMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiResponseMessage)

        self.assertIs(messages.MultiResponseProbsMessage.get_impl_class(), messages.MultiResponseProbsMessage)
        m = messages.MultiResponseProbsMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiResponseProbsMessage)

        self.assertIs(messages.MultiResponseAEMessage.get_impl_class(), messages.MultiResponseAEMessage)
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

        self.assertIs(messages.MessageMeta.get_impl_class(), messages.MessageMeta)
        m = messages.MessageMeta(None)
        self.assertIsInstance(m, messages.MessageMeta)

        self.assertIs(messages.UserMessageMeta.get_impl_class(), messages.UserMessageMeta)
        m = messages.UserMessageMeta(None, None)
        self.assertIsInstance(m, messages.UserMessageMeta)

        self.assertIs(messages.MultiMessage.get_impl_class(), messages.MultiMessage)
        m = messages.MultiMessage(None, 0, 1)
        self.assertIsInstance(m, messages.MultiMessage)

        self.assertIs(messages.InferenceMemory.get_impl_class(), messages.InferenceMemory)
        m = messages.InferenceMemory(1)
        self.assertIsInstance(m, messages.InferenceMemory)

        cp_array = cp.zeros((1, 2))
        self.assertIs(messages.InferenceMemoryNLP.get_impl_class(), messages.InferenceMemoryNLP)
        m = messages.InferenceMemoryNLP(1, cp_array, cp_array, cp_array)
        self.assertIsInstance(m, messages.InferenceMemoryNLP)

        self.assertIs(messages.InferenceMemoryFIL.get_impl_class(), messages.InferenceMemoryFIL)
        m = messages.InferenceMemoryFIL(1, cp_array, cp_array)
        self.assertIsInstance(m, messages.InferenceMemoryFIL)

        self.assertIs(messages.InferenceMemoryAE.get_impl_class(), messages.InferenceMemoryAE)
        m = messages.InferenceMemoryAE(1, cp_array, cp_array)
        self.assertIsInstance(m, messages.InferenceMemoryAE)

        self.assertIs(messages.MultiInferenceMessage.get_impl_class(), messages.MultiInferenceMessage)
        m = messages.MultiInferenceMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiInferenceMessage)

        self.assertIs(messages.MultiInferenceNLPMessage.get_impl_class(), messages.MultiInferenceNLPMessage)
        m = messages.MultiInferenceNLPMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiInferenceNLPMessage)

        self.assertIs(messages.MultiInferenceFILMessage.get_impl_class(), messages.MultiInferenceFILMessage)
        m = messages.MultiInferenceFILMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiInferenceFILMessage)

        self.assertIs(messages.ResponseMemory.get_impl_class(), messages.ResponseMemory)
        m = messages.ResponseMemory(1)
        self.assertIsInstance(m, messages.ResponseMemory)

        self.assertIs(messages.ResponseMemoryProbs.get_impl_class(), messages.ResponseMemoryProbs)
        m = messages.ResponseMemoryProbs(1, cp_array)
        self.assertIsInstance(m, messages.ResponseMemoryProbs)

        self.assertIs(messages.ResponseMemoryAE.get_impl_class(), messages.ResponseMemoryAE)
        m = messages.ResponseMemoryAE(1, cp_array)
        self.assertIsInstance(m, messages.ResponseMemoryAE)

        self.assertIs(messages.MultiResponseMessage.get_impl_class(), messages.MultiResponseMessage)
        m = messages.MultiResponseMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiResponseMessage)

        self.assertIs(messages.MultiResponseProbsMessage.get_impl_class(), messages.MultiResponseProbsMessage)
        m = messages.MultiResponseProbsMessage(None, 0, 1, None, 0, 1)
        self.assertIsInstance(m, messages.MultiResponseProbsMessage)

        self.assertIs(messages.MultiResponseAEMessage.get_impl_class(), messages.MultiResponseAEMessage)
        m = messages.MultiResponseAEMessage(None, 0, 1, None, 0, 1, '')
        self.assertIsInstance(m, messages.MultiResponseAEMessage)


if __name__ == '__main__':
    unittest.main()
