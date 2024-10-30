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
import json
import logging
import string
import typing

from morpheus_llm.llm import LLMContext
from morpheus_llm.llm import LLMNodeBase

logger = logging.getLogger(__name__)


class PromptTemplateNode(LLMNodeBase):
    """
    Populates a template string with the values from the upstream node.

    Parameters
    ----------
    template : str
        The template string to populate.
    template_format : str, optional default="f-string"
        The format of the template string. Must be one of: f-string, jinja.
    """

    def __init__(self, template: str, template_format: typing.Literal["f-string", "jinja"] = "f-string") -> None:
        super().__init__()
        self._template_str = template
        self._template_format = template_format

        if (self._template_format == "f-string"):
            formatter = string.Formatter()
            # The parse method is returning an iterable of tuples in the form of:
            # (literal_text, field_name, format_spec, conversion)
            # https://docs.python.org/3.10/library/string.html#string.Formatter.parse
            self._input_names = []
            for (_, field_name, _, _) in formatter.parse(self._template_str):
                if field_name == '':
                    raise ValueError("Unnamed fields in templates are not supported")

                if field_name is not None:
                    self._input_names.append(field_name)

        elif (self._template_format == "jinja"):
            from jinja2 import Template
            from jinja2 import meta

            self._template_jinja = Template(self._template_str, enable_async=True, trim_blocks=True, lstrip_blocks=True)

            self._input_names = list(
                meta.find_undeclared_variables(self._template_jinja.environment.parse(self._template_str)))
        else:
            raise ValueError(f"Invalid template format: {self._template_format}, must be one of: f-string, jinja")

    def get_input_names(self):
        return self._input_names

    async def execute(self, context: LLMContext):  # pylint: disable=invalid-overridden-method

        # Get the keys from the task
        input_dict = context.get_inputs()

        # Transform from dict[str, list[Any]] to list[dict[str, Any]]
        input_list = [dict(zip(input_dict, t)) for t in zip(*input_dict.values())]

        if (self._template_format == "f-string"):
            output_list = [self._template_str.format(**x) for x in input_list]
        elif (self._template_format == "jinja"):
            render_coros = [self._template_jinja.render_async(**inputs) for inputs in input_list]

            output_list = await asyncio.gather(*render_coros)

        context.set_output(output_list)

        return context


class PromptTemplateNode2(LLMNodeBase):
    """
    Populates a template string with the values from the upstream node.

    Parameters
    ----------
    template : str
        The template string to populate.
    template_format : str, optional default="f-string"
        The format of the template string. Must be one of: f-string, jinja.
    """

    def __init__(self,
                 prefix_messages: list[dict],
                 templates: list[dict],
                 template_format: typing.Literal["f-string", "jinja"] = "f-string") -> None:
        super().__init__()
        self._prefix_messages = prefix_messages
        self._templates = templates
        # self._template_str = template
        self._template_format = template_format

        name_set = set()

        if (self._template_format == "f-string"):
            formatter = string.Formatter()
            # The parse method is returning an iterable of tuples in the form of:
            # (literal_text, field_name, format_spec, conversion)
            # https://docs.python.org/3.10/library/string.html#string.Formatter.parse
            self._input_names = []
            for (_, field_name, _, _) in formatter.parse(self._template_str):
                if field_name == '':
                    raise ValueError("Unnamed fields in templates are not supported")

                if field_name is not None:
                    self._input_names.append(field_name)

        elif (self._template_format == "jinja"):
            from jinja2 import Template
            from jinja2 import meta

            self._template_jinja: list[Template] = []

            for template_dict in self._templates:

                template_str = template_dict["content"]

                template_jinja = Template(template_str, enable_async=True, trim_blocks=True, lstrip_blocks=True)

                self._template_jinja.append(template_jinja)

                name_set.update(list(meta.find_undeclared_variables(template_jinja.environment.parse(template_str))))
        else:
            raise ValueError(f"Invalid template format: {self._template_format}, must be one of: f-string, jinja")

        self._input_names = list(name_set)

    def get_input_names(self):
        return self._input_names

    async def execute(self, context: LLMContext):  # pylint: disable=invalid-overridden-method

        # Get the keys from the task
        input_dict = context.get_inputs()

        # Transform from dict[str, list[Any]] to list[dict[str, Any]]
        input_list = [dict(zip(input_dict, t)) for t in zip(*input_dict.values())]

        final_output = []

        if (self._template_format == "f-string"):
            output_list = [self._template_str.format(**x) for x in input_list]
        elif (self._template_format == "jinja"):

            output_list = [[] for _ in range(len(input_list))]

            for template_idx, template_obj in enumerate(self._templates):
                template_jinja = self._template_jinja[template_idx]

                render_coros = [template_jinja.render_async(**inputs) for inputs in input_list]

                rendered_output = await asyncio.gather(*render_coros)

                # Now replace the content with the template content
                for input_idx, o in enumerate(rendered_output):

                    obj = {**template_obj, "content": o}

                    output_list[input_idx].append(obj)

        for o in output_list:

            final_output.append(self._prefix_messages + o)

        context.set_output(final_output)

        return context
