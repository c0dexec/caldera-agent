from langchain_community.agent_toolkits import OpenAPIToolkit, create_openapi_agent
from langchain_community.tools.json.tool import JsonSpec
from langchain_ollama import ChatOllama
from langchain_community.utilities.requests import RequestsWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import json
import os
from dotenv import load_dotenv
from tools import api_call

load_dotenv()

# with open("response_1765136132246.json") as f:
#     data = json.load(f)
# json_spec = JsonSpec(dict_=data, max_value_length=4000)

# requests_wrapper = RequestsWrapper(headers={"KEY": f"{os.getenv('CALDERA_API_TOKEN')}"})

# ALLOW_DANGEROUS_REQUEST = True

# openapi_toolkit = OpenAPIToolkit.from_llm(
#     ChatOllama(model="llama3.1:70b-instruct-q4_K_M", temperature=0, max_tokens=5000, base_url="http://10.0.0.10:11434"), json_spec, requests_wrapper=requests_wrapper, verbose=True, allow_dangerous_requests=ALLOW_DANGEROUS_REQUEST
# )

# openapi_tools = openapi_toolkit.get_tools()

SYSTEM_PROMPT = """
Your are an AI agent designed to interact with the Caldera API.
You will be provided with the OpenAPI specification for the Caldera API.
Use this specification to understand the available endpoints and their functionalities.
When responding to user queries, ensure that you reference the OpenAPI spec to provide accurate and relevant information.
Always prioritize safety and security when making API calls.

Do exactly what the user asks you to do nothing else. If subsequent API calls are needed to fulfill the user's request, make them to finish the task.

If you are having error make use of "tools.api_call" to make the API calls.
"""

llm = ChatOllama(
    model="llama3.1:70b-instruct-q4_K_M",
    temperature=0,
    max_tokens=5000,
    timeout=None,
    # state_schema=CustomAgentState,  
    checkpointer=InMemorySaver(),
    base_url="http://10.0.0.10:11434"  # Replace with your Ollama server URL
)

ollama_agent_executor = create_agent(
    model=llm,
    tools=[openapi_tools, api_call],
    system_prompt=SYSTEM_PROMPT,
)

ollama_agent_executor.run(
    "Make a request to server to check its health."
)