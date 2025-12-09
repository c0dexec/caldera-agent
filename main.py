# pip install -qU langchain "langchain[anthropic]"
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from load_spec import load_caldera_spec
from langchain_community.agent_toolkits.openapi import planner
from langchain_community.utilities.requests import RequestsWrapper
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
import datetime
import tools
# OpenAI imports
from langchain_openai import ChatOpenAI

import logging

logging.basicConfig(level=logging.DEBUG)


# Load environment variables
load_dotenv()

# Define system prompt
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
    max_tokens=4000,
    timeout=None,
    # state_schema=CustomAgentState,  
    checkpointer=InMemorySaver(),
    base_url="http://10.0.0.10:11434"  # Replace with your Ollama server URL
)

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=16384,)

requests_wrapper = RequestsWrapper(headers={"KEY": f"{os.getenv('CALDERA_API_TOKEN')}"})
ALLOW_DANGEROUS_REQUEST = True

caldera_agent = planner.create_openapi_agent(
    llm=llm,
    api_spec=load_caldera_spec(),
    system_prompt=SYSTEM_PROMPT,
    verbose=True,
    allow_dangerous_requests=ALLOW_DANGEROUS_REQUEST,
    requests_wrapper=requests_wrapper,
    allowed_operations=["get", "post", "put", "delete", "patch", "head"],
    # context_schema=Context,
)


# user_query = (
#     "What's the status of the caldera server?"
# )
# caldera_agent.invoke(user_query)


# # Run the agent
import datetime

def chat_loop():

    config = {"configurable": {"thread_id": "1"}}

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    user_query = input("User: ")

    while user_query.lower() not in ["exit", "quit"]:
        response = caldera_agent.invoke(
            {"input": user_query}, 
            config=config,
            tools=[tools.api_call]
        )
        print(f"Caldera Agent({llm.model}): {response['structured_response'].caldera_output}\nCommand exectued: `{response['structured_response'].caldera_command}`\nUser: ", end="")

        # Write to log file
        with open("caldera_agent.log", "a") as log_file:
            log_file.write(f"[{current_time}] User: {user_query}\n")
            log_file.write(f"[{current_time}] Agent({llm.model}): \n```\n{response['structured_response'].caldera_output}\n```\n Command exectued: `{response['structured_response'].caldera_command}`\n\n")
        
        user_query = input("")
    else:
        print("Exiting chat loop.")
        with open("caldera_agent.log", "a") as log_file:
            log_file.write(f"[{current_time}] User exited the chat loop.\n\n")
        exit()

if __name__ == "__main__":
    chat_loop()