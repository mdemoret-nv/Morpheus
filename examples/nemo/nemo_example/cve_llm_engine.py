import asyncio
import logging
import os
import re
import typing
from typing import List

import aiohttp
import langchain.output_parsers
import openai
import requests
from bs4 import BeautifulSoup
from langchain.agents import AgentExecutor
from langchain.cache import SQLiteCache
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain.chains.base import Chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.llms.openai import OpenAIChat
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
from langchain.vectorstores import Milvus

import cudf

from morpheus._lib.llm import CoroAwaitable
from morpheus._lib.llm import LangChainTemplateNodeCpp
from morpheus._lib.llm import LLMContext
from morpheus._lib.llm import LLMEngine
from morpheus._lib.llm import LLMNode
from morpheus._lib.llm import LLMNodeBase
from morpheus._lib.llm import LLMTaskHandler
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta

from .common import ExtracterNode
from .common import FunctionWrapperNode
from .common import LLMDictTask
from .llm_engine import NeMoLangChain
from .nemo_service import NeMoService

# Setup the cache to avoid repeated calls
langchain_cache_path = "./.cache/langchain"
os.makedirs(langchain_cache_path, exist_ok=True)
langchain.llm_cache = SQLiteCache(database_path=os.path.join(langchain_cache_path, ".langchain.db"))

logger = logging.getLogger(f"morpheus.{__name__}")

MILVUS_COLLECTION_NAME = "Confluence"


# A version of LlamaCpp that supports async calls
class LlamaCppAsync(LlamaCpp):

    _mutex: asyncio.Lock = asyncio.Lock()

    class Config:
        underscore_attrs_are_private = True

    async def _acall(self,
                     prompt: str,
                     stop: List[str] | None = None,
                     run_manager: AsyncCallbackManagerForLLMRun | None = None) -> str:

        # Can only have one running at a time
        async with self._mutex:
            return await asyncio.to_thread(self._call, prompt, stop, None)


class LangChainTemplateNode(LLMNodeBase):

    def __init__(self, template: str, template_format: typing.Literal["f-string", "jinja"] = "f-string") -> None:
        super().__init__()

        self._input_variables = ["question"]
        self._template = template
        self._template_format = template_format

        if (self._template_format == "f-string"):
            self._input_names = []
        elif (self._template_format == "jinja"):
            from jinja2 import Template
            from jinja2 import meta

            jinja_template = Template(self._template)

            self._input_names = list(meta.find_undeclared_variables(jinja_template.environment.parse(self._template)))

    def get_input_names(self):
        return self._input_names

    async def execute(self, context: LLMContext):

        # if ("0" in context.input_map[0].input_name):
        #     context.parent.get_input("$.chat1.choices[*][0].message.content")

        # Get the keys from the task
        input_dict = context.get_inputs()

        # Transform from dict[str, list[Any]] to list[dict[str, Any]]
        input_list = [dict(zip(input_dict, t)) for t in zip(*input_dict.values())]

        if (self._template_format == "f-string"):
            output_list = [self._template.format(**x) for x in input_list]
        elif (self._template_format == "jinja"):

            from jinja2 import Template

            template = Template(self._template, enable_async=True)

            render_coros = [template.render_async(**inputs) for inputs in input_list]

            output_list = await asyncio.gather(*render_coros)

        context.set_output(output_list)

        return context


class LangChainLlmNode(LLMNodeBase):

    def __init__(self, temperature: float) -> None:
        super().__init__()

        self._temperature = temperature

        self._nemo_service = NeMoService.instance()
        self._nemo_client = self._nemo_service.get_client(model_name="gpt-43b-002",
                                                          infer_kwargs={"temperature": temperature})

    def get_input_names(self):
        return ["question"]

    async def execute(self, context: LLMContext):

        # Get the keys from the task
        input_dict = context.get_input("question")

        output_list = await self._nemo_client.query_batch_async(prompt=input_dict)

        context.set_output(output_list)

        return context


class LangChainAgentNode(LLMNodeBase):

    def __init__(self, agent_executor: AgentExecutor):
        super().__init__()

        self._agent_executor = agent_executor

        self._input_names = self._agent_executor.input_keys

    def get_input_names(self):
        return self._input_names

    async def _run_single(self, **kwargs):

        # Transform from dict[str, list[Any]] to list[dict[str, Any]]
        input_list = [dict(zip(kwargs, t)) for t in zip(*kwargs.values())]

        results_async = [self._agent_executor.arun(**x) for x in input_list]

        results = await asyncio.gather(*results_async)

        return results

    async def execute(self, context: LLMContext):

        input_dict = context.get_inputs()

        # Transform from dict[str, list[Any]] to list[dict[str, Any]]
        input_list = [dict(zip(input_dict, t)) for t in zip(*input_dict.values())]

        results_async = [self._run_single(**x) for x in input_list]

        results = await asyncio.gather(*results_async)

        # Transform from list[dict[str, Any]] to dict[str, list[Any]]
        # results = {k: [x[k] for x in results] for k in results[0]}

        context.set_output(results)


# Prompt generator wrapper around a LangChain agent executor
class LangChainChainNode(LLMNode):

    def __init__(self, chain: Chain):
        super().__init__()

        self._chain = chain

        self._chain_dict = chain.dict()

        # template_node = LangChainTemplateNode(template=self._chain_dict["prompt"]["template"])
        template_node = LangChainTemplateNodeCpp(template=self._chain_dict["prompt"]["template"])

        self.add_node("prompt", [("question", "/extract_prompt")], template_node)

        self.add_node("llm", [("input", "/langchain.prompt")],
                      LangChainLlmNode(temperature=self._chain_dict["llm"]["temperature"]))

    # async def execute(self, context: LLMContext):

    #     input_dict: dict = context.get_input()

    #     if (isinstance(input_dict, list)):
    #         input_dict = {"query": input_dict}

    #     # Transform from dict[str, list[Any]] to list[dict[str, Any]]
    #     input_list = [dict(zip(input_dict, t)) for t in zip(*input_dict.values())]

    #     # outputs = []

    #     # for x in input_list:
    #     #     result = await self._chain.acall(inputs=x)

    #     #     outputs.append(result)

    #     output_coros = [self._chain.acall(inputs=x) for x in input_list]

    #     outputs = await asyncio.gather(*output_coros)

    #     # Uncomment to run synchronously
    #     # results = [self._chain(inputs=x) for x in input_list]

    #     # Extract the results from the output
    #     results = [x["result"] for x in outputs]

    #     context.set_output(results)


class CVELookupNode(LLMNodeBase):

    def __init__(self) -> None:
        super().__init__()

    async def _web_scrape(self, session: aiohttp.ClientSession, cve: str):
        '''scrape websites for latest cve info:
        cve description, cvss vector, cwe description, vendor names'''

        # Get CVE description from nist
        nist_url = 'https://nvd.nist.gov/vuln/detail/' + cve
        async with session.get(nist_url) as response:
            soup = BeautifulSoup(await response.text(), 'html.parser')

        cve_description = soup.find("p", attrs={"data-testid": "vuln-description"})

        if cve_description is None:
            print("No data for this CVE. Please check format to match CVE-2023-1234")
        else:
            cve_description = cve_description.text

        # Get CVSS vector from nist
        cvss_vector = soup.find('span', class_='tooltipCvss3NistMetrics')
        if cvss_vector:
            cvss_vector = cvss_vector.text
        else:
            cvss_vector = None

        # Get CWE name
        title_tag = soup.find('title')
        if title_tag:
            title_text = title_tag.string.strip()

        # Get CWE name and description
        link_element = soup.find('td', attrs={'data-testid': 'vuln-CWEs-link-0'}).find('a')
        if link_element:
            cwe_link = link_element['href']
            async with session.get(cwe_link) as response:
                soup = BeautifulSoup(await response.text(), 'html.parser')

        title_tag = soup.find('title')
        if title_tag:
            cwe_name = title_tag.string.strip()
            cwe_name = re.sub(r'^.*?-\s*', '', cwe_name).strip()
            description_div = soup.find('div', id='Description')
            if description_div:
                cwe_description = description_div.find('div', class_='indent').text.strip()
            else:
                cwe_description = None
            extended_description_div = soup.find('div', id='Extended_Description')
            if extended_description_div:
                cwe_extended_description = extended_description_div.find('div', class_='indent').text.strip()
            else:
                cwe_extended_description = None
        else:
            cwe_name = None
            cwe_description = None
            cwe_extended_description = None

        # Get vendor names fom CVE details
        cve_deets_url = 'https://www.cvedetails.com/cve/' + cve

        async with session.get(cve_deets_url) as response:
            soup = BeautifulSoup(await response.text(), 'html.parser')

        # Find all the vendor names within the <a> tags
        vendor_tags = soup.find_all('a', href=re.compile('/vendor/'))
        if vendor_tags:
            # Extract the text from the vendor tags
            vendor_names = [tag.text.strip() for tag in vendor_tags]
            # Get unique
            vendor_names = list(set(vendor_names))
        else:
            vendor_names = None

        return {
            "cve_description": cve_description,
            "cvss_vector": cvss_vector,
            "cwe_name": cwe_name,
            "cwe_description": cwe_description,
            "cwe_extended_description": cwe_extended_description,
            "vendor_names": vendor_names,
        }

    def get_input_names(self):
        return ["cve"]

    async def execute(self, context: LLMContext):
        '''scrape websites for latest cve info:
        cve description, cvss vector, cwe description, vendor names'''

        # Get CVE name from context
        cve_names: list[str] = context.get_input()

        async with aiohttp.ClientSession() as session:

            cve_info_coros = [self._web_scrape(session, cve) for cve in cve_names]

            cve_infos = await asyncio.gather(*cve_info_coros)

        # Convert from list[dict] to dict[list]
        cve_infos = {k: [x[k] for x in cve_infos] for k in cve_infos[0]}

        context.set_output(cve_infos)


class OpenAIChatCompletionNode(LLMNodeBase):

    def __init__(self, model_name: str, set_assistant=False) -> None:
        super().__init__()

        self._model_name = model_name
        self._set_assistant = set_assistant

        self._model = OpenAIChat(model_name=self._model_name, temperature=0, cache=True)

    def get_input_names(self):
        if (self._set_assistant):
            return ["assistant", "user"]

        return ["user"]

    async def _run_one(self, user: str, assistant: str = None):

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=user),
        ]

        if (assistant is not None):
            messages.append(AIMessage(content=assistant))

        output2 = await self._model.apredict_messages(messages=messages)

        return {"message": output2.content}

    async def execute(self, context: LLMContext):

        input_dict = context.get_inputs()

        # Transform from dict[str, list[Any]] to list[dict[str, Any]]
        input_list = [dict(zip(input_dict, t)) for t in zip(*input_dict.values())]

        output_coros = [self._run_one(**inp) for inp in input_list]

        outputs = await asyncio.gather(*output_coros)

        # Convert from list[dict] to dict[list]
        outputs = {k: [x[k] for x in outputs] for k in outputs[0]}

        context.set_output(outputs)


cve_prompt1 = """This is an example of (1) CVE information, and (2) a checklist produced to determine if a given CVE is exploitable in a containerized environment:
(1) CVE Information:
CVE Description: DISPUTED: In Apache Batik 1.x before 1.10, when deserializing subclass of `AbstractDocument`, the class takes a string from the inputStream as the class name which then use it to call the no-arg constructor of the class. Fix was to check the class type before calling newInstance in deserialization.
CVSS Vector: CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H
CWE Name: CWE-502: Deserialization of Untrusted Data (4.11)
CWE Description: The product deserializes untrusted data without sufficiently verifying that the resulting data will be valid. It is often convenient to serialize objects for communication or to save them for later use. However, deserialized data or code can often be modified without using the provided accessor functions if it does not use cryptography to protect itself. Furthermore, any cryptography would still be client-side security -- which is a dangerous security assumption. Data that is untrusted can not be trusted to be well-formed. When developers place no restrictions on gadget chains, or series of instances and method invocations that can self-execute during the deserialization process (i.e., before the object is returned to the caller), it is sometimes possible for attackers to leverage them to perform unauthorized actions, like generating a shell.
Notable Vulnerable Software Vendors: ['Oracle', 'Apache', 'Canonical', 'Debian']
(2) Checklist:
Based on the information available, CVE-2018-8013 is a critical vulnerability that involves the deserialization of untrusted data in Apache Batik before version 1.10. The Common Vulnerability Scoring System (CVSS) 3.x score is 9.8, indicating a critical severity.
Here's an example of preliminary checklist similiar to what you should create to determine if a Docker container is vulnerable to this exploit:
1. Check the Apache Batik version: The vulnerability affects Apache Batik versions 1.0 to 1.9.1. If your Docker container is running Apache Batik, verify the version. If it's between 1.0 and 1.9.1, it's vulnerable and you should update to version 1.10 or later.
2. Check for any software using vulnerable Apache Batik versions: It's important to note that other applications might use Apache Batik and thus be vulnerable. Check all applications within your Docker container for dependencies on vulnerable versions of Apache Batik. Some notable software includes certain versions of Ubuntu Linux, Debian Linux, and multiple Oracle applications.
3. Evaluate the deserialization risk: Since the vulnerability involves deserialization of untrusted data, you should consider whether your applications are performing such operations. If they are, consider if these deserialization operations are using untrusted, user-supplied data. If so, this is a potential attack vector.
4. Network exposure: The vulnerability has a network attack vector. Consider whether your Docker container has network exposure, especially if the Apache Batik service or any service using it is exposed to the internet.
5. Disputed status: CVE-2018-8013 is disputed and may not be a vulnerability.

Given the following information about {{cve}}, make preliminary checklist for a security analyst to follow to determine whether a Docker container is vulnerable to this exploit.
CVE description: {{cve_description}}
{% if cvss_vector %}
CVSS Vector: {{cvss_vector}}
{% endif %}
{% if cwe_name %}
CWE Name: {{cwe_name}}
{% endif %}
{% if cwe_description %}
CWE Description: {{cwe_description}}
{% endif %}
{% if cwe_extended_description %}
{{cwe_extended_description}}
{% endif %}
{% if vendor_names %}
Notable Vulnerable Software Vendors: {{vendor_names}}
{% endif %}
"""

cve_prompt2 = """Parse the following numbered checklist into a python list in the format ['x', 'y', 'z']: {{template}}"""


def build_langchain_agent():

    # from langchain.tools.render import format_tool_to_openai_function
    # from langchain.agents.format_scratchpad import format_to_openai_functions
    # from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
    from langchain.agents import AgentExecutor
    from langchain.agents import AgentType
    from langchain.agents import Tool
    from langchain.agents import initialize_agent
    from langchain.agents import tool
    from langchain.chains import RetrievalQA
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain
    from langchain.chat_models import ChatOpenAI
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.llms import OpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.prompts import MessagesPlaceholder
    from langchain.utilities import SerpAPIWrapper
    from langchain.vectorstores import FAISS

    openai.api_key = os.getenv('OPENAI_API_KEY')
    SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    ## load SBOM vector DB
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    SBOMvectorDB = FAISS.load_local("/home/mdemoret/Repos/gitlab-master/rachela/cve_tool/vectorDBs/morpheus_sbom_faiss",
                                    embeddings)

    sbom_qa_tool = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.9),
                                               chain_type="stuff",
                                               retriever=SBOMvectorDB.as_retriever())
    tools = [
        Tool(
            name="Docker Container Software Bill of Materials QA System",
            func=sbom_qa_tool.run,
            coroutine=sbom_qa_tool.arun,
            description=
            "useful for when you need to search from the Docker container's software bill of materials (SBOM) to get information about the libraries it contains. Input should be a fully formed question.",
        )
    ]

    # # load code vector DB
    # CODEvectorDB = FAISS.load_local("/home/mdemoret/Repos/gitlab-master/rachela/cve_tool/vectorDBs/morpheus_code_faiss", embeddings)
    # code_qa_tool = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.9),
    #                                            chain_type="stuff",
    #                                            retriever=CODEvectorDB.as_retriever())
    # tools.append(
    #     Tool(
    #         name="Docker Container Code QA System",
    #         func=code_qa_tool.run,
    #         # description="useful for when you need to get search from the Docker container's software code to get information about the dependency or usage of a library or a function. Input should be a fully formed question.",
    #         description=
    #         "useful for when you need to check if an application or any dependency within the Docker container uses a function or a component of a library."
    #     ))

    # GUIDEvectorDB = FAISS.load_local("vectorDBs/morpheus_guide_faiss", embeddings)
    # guide_qa_tool = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.9),
    #                                             chain_type="stuff",
    #                                             retriever=GUIDEvectorDB.as_retriever())
    # tools.append(
    #     Tool(
    #         name="Docker Container Developer Guide QA System",
    #         func=guide_qa_tool.run,
    #         # description="useful for when you need to get search from the Docker container's software code to get information about the dependency or usage of a library or a function. Input should be a fully formed question.",
    #         description=
    #         "useful for when you need to ask questions about the purpose and functionality of the Docker container."))

    search = SerpAPIWrapper()
    tools.append(
        Tool(
            name="Internet Search",
            func=search.run,
            coroutine=search.arun,
            description="useful for when you need to answer questions about external libraries",
        ))

    sys_prompt = (
        "You are very powerful assistant who helps investigate the impact of a reported Common Vulnerabilities and Exposures (CVE) on the Docker container."
        " Information about the Docker container under investigation is stored in vector databases available to you via tools. "
    )

    agent = initialize_agent(tools,
                             llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True,
                             handle_parsing_errors=True)
    agent.agent.llm_chain.prompt.template = (sys_prompt + ' ' + agent.agent.llm_chain.prompt.template.replace(
        "Answer the following questions as best you can.",
        "If the input is not a question, formulate it into a question first. Inlcude intermediate thought in the final answer."
    ).replace(
        "Use the following format:",
        "Use the following format (you're required to start each response with one of [Question, Thought, Action, Action Input, Final Answer]):"
    ))

    return agent


async def parse_list(text: list[str]):
    import ast
    import json

    return_val = []

    for x in text:

        current = ast.literal_eval(x)

        for i in range(len(current)):

            if (isinstance(current[i], list) and len(current[i]) == 1):
                current[i] = current[i][0]

        return_val.append(current)

    return return_val


class CVEChecklistNode(LLMNode):

    def __init__(self):
        super().__init__()

        # cve comes from external
        self.add_node("lookup", inputs=["cve"], node=CVELookupNode())

        # # cve comes from the lookup node
        # self.add_node("checklist_prompt",
        #               inputs=[("/lookup/cve_description", "cve_description"), ("/lookup/cvss_vector", "cvss_vector")],
        #               node=LangChainTemplateNode(template=cve_prompt1, template_format="jinja"))

        # Also want to support the following syntax
        self.add_node("checklist_prompt",
                      inputs=[("/lookup/*", "*"), "cve"],
                      node=LangChainTemplateNode(template=cve_prompt1, template_format="jinja"))

        # Input comes from the checklist prompt
        self.add_node("chat1", inputs=["/checklist_prompt"], node=OpenAIChatCompletionNode(model_name="gpt-3.5-turbo"))

        self.add_node("parse_checklist_prompt",
                      inputs=["/chat1/message"],
                      node=LangChainTemplateNode(template=cve_prompt2, template_format="jinja"))

        self.add_node("chat2",
                      inputs=[("/parse_checklist_prompt", "user")],
                      node=OpenAIChatCompletionNode(model_name="gpt-3.5-turbo"))

        self.add_node("output_parser", inputs=["/chat2/message"], node=FunctionWrapperNode(parse_list), is_output=True)

    # async def execute(self, context: LLMContext):
    #     return await super().execute(context)


class SimpleTaskHandler(LLMTaskHandler):

    def get_input_names(self):
        return ["response"]

    async def try_handle(self, context: LLMContext):

        with context.message().payload().mutable_dataframe() as df:
            df["response"] = context.get_input()

        return [context.message()]


def get_hf_embeddings(model):

    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {
        # 'normalize_embeddings': True, # set True to compute cosine similarity
        "batch_size": 100,
    }

    embeddings = HuggingFaceEmbeddings(model_name=model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    return embeddings


def getVDB(db_name, *, embeddings, collection_name, host="localhost", port="19530"):
    if db_name == 'Chroma':
        #vectordb = Chroma("langchain_store",embeddings, persist_directory="./data-chroma")
        vectordb = Chroma(persist_directory="./data-chroma", embedding_function=embeddings)
        print("Chroma")
    elif db_name == 'FAISS':
        print("FAISS")
        vectordb = FAISS.load_local("vdb_chunks", embeddings, index_name="nv-index")
    elif db_name == 'Milvus':

        # Get existing collection from Milvus
        vectordb: Milvus = Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args={
                "host": host, "port": port
            },
            drop_old=True,
        )
        return vectordb

    raise RuntimeError("Unknown db_type: {db_name}}")


def get_modelpath(model):
    MODEL_DIR = "./shared-dir/llama2/"
    if model == 'llama-2-13b-chat.Q4_K_M':
        print(MODEL_DIR + model + ".gguf")
        return (MODEL_DIR + model + ".gguf")


def get_llm(model, n_ctx, n_gpu_layers, n_batch):
    model_path = get_modelpath(model)
    # llm = LlamaCppAsync(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, temperature=0, n_batch=n_batch)
    # llm = NeMoLangChain(model_name="llama-2-70b-hf")
    llm = NeMoLangChain(model_name="gpt-43b-002")
    return llm


async def run_engine():

    agent = build_langchain_agent()

    engine = LLMEngine()

    engine.add_node("extract_prompt", node=ExtracterNode())
    engine.add_node("checklist", inputs=[("/extract_prompt", "cve")], node=CVEChecklistNode())

    engine.add_node("agent", inputs=["/checklist"], node=LangChainAgentNode(agent))

    # # Add our task handler
    engine.add_task_handler(inputs=["/agent"], handler=SimpleTaskHandler())

    # Create a control message with a single task which uses the LangChain agent executor
    message = ControlMessage()

    message.add_task("llm_engine",
                     {
                         "task_type": "template",
                         "task_dict": LLMDictTask(input_keys=["cves"], model_name="gpt-43b-002").dict(),
                     })

    cves = [
        "CVE-2018-8013",  # "CVE-2023-24329",
        # "CVE-2022-45061",
    ]

    payload = cudf.DataFrame({
        "cves": cves,
    })
    message.payload(MessageMeta(payload))

    # Finally, run the engine
    result = await engine.run(message)

    # coros = [chain.acall(inputs=x) for x in questions]

    # result = await asyncio.gather(*coros)

    # result = [chain(inputs=x) for x in questions]

    print(f"Got results: {result}")
