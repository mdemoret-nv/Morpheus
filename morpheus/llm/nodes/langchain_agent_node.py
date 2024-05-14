# Copyright (c) 2023-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import os
import typing

from langchain_core.callbacks import FileCallbackHandler

from morpheus.llm import LLMContext
from morpheus.llm import LLMNodeBase

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from langchain.agents import AgentExecutor


class LangChainAgentNode(LLMNodeBase):
    """
    Executes a LangChain agent in an LLMEngine

    Parameters
    ----------
    agent_executor : AgentExecutor
        The agent executor to use to execute.
    """

    def __init__(self, agent_executor: "AgentExecutor"):
        super().__init__()

        self._agent_executor = agent_executor

        self._input_names = self._agent_executor.input_keys

        self._semaphore = asyncio.Semaphore(int(os.environ.get("MORPHEUS_CONCURRENCY", 100)))

    def get_input_names(self):
        return self._input_names + ["context"]

    async def _run_single(self, context_vars, **kwargs):

        all_lists = all(isinstance(v, list) for v in kwargs.values())

        # Check if all values are a list
        if all_lists:

            # Transform from dict[str, list[Any]] to list[dict[str, Any]]
            input_list = [dict(zip(kwargs, t)) for t in zip(*kwargs.values())]

            # If the context vars are not a list too, then convert to a list and append the index
            if (not isinstance(context_vars, list)):
                context_vars = [f"{context_vars}/{x}" for x in range(len(input_list))]

            # Run multiple again
            results_async = [self._run_single(context_vars=c, **x) for c, x in zip(context_vars, input_list)]

            results = await asyncio.gather(*results_async, return_exceptions=True)

            # # Transform from list[dict[str, Any]] to dict[str, list[Any]]
            # results = {k: [x[k] for x in results] for k in results[0]}

            return results

        async with self._semaphore:

            # We are not dealing with a list, so run single
            try:

                return await self._agent_executor.arun(metadata={"context": os.path.join(context_vars, "agent.log")},
                                                       **kwargs)
            except Exception as e:
                logger.exception("Error running agent: %s", e)
                return e

    async def execute(self, context: LLMContext) -> LLMContext:  # pylint: disable=invalid-overridden-method

        input_dict: dict = context.get_inputs()

        context_vars = input_dict.pop("context")

        results = await self._run_single(context_vars=context_vars, **input_dict)

        context.set_output(results)

        return context
