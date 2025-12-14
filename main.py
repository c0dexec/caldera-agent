# pip install -qU langchain "langchain[anthropic]"
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities.requests import RequestsWrapper
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from dotenv import load_dotenv
import os
import datetime
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from tools import api_call, retrieve_context
# OpenAI imports
# from langchain_openai import ChatOpenAI

import logging
import readline

# Python debugging logging
# logging.basicConfig(level=logging.DEBUG)


# Load environment variables
load_dotenv()

# Define system prompt
SYSTEM_PROMPT = f"""
# About you
Your are an AI agent designed to interact with the Caldera API. You will be provided with the OpenAPI specification for the Caldera API.

When responding to user queries, ensure that you reference the OpenAPI spec to provide accurate and relevant information.
Always prioritize safety and security when making API calls.

# Tools
Tools available to you:
1. api_call: Can be used to create API calls
2. retrieve_context: Can be used to retrieve context about the Caldera API from the OpenAPI spec. This tool uses to retrieve data stored as JSON and Mardown to find relevant information.

# Actions
When a user query is received, follow these steps:
1. Access `retrieve_context` to gather relevant information from the OpenAPI spec.
2. If a question was asked then respond directly using the retrieved context. If user requests an action to be performed using the Caldera API, proceed to step 3.
3. Based on the retrieved context, determine the appropriate API endpoint and request type needed to fulfill the user's request.
4. Once you are exactly sure about the endpoint and request type, then use the `tools.api_call` tool to make the API call. If you are getting HTTP Error code (4xx or 5xx) after 2 tries, then exit and inform the user about the failure.

# Rules
Do exactly what the user asks you to do nothing else. And limit your request to only one API call per user query for the same request type and endpoint.
If subsequent API calls to different endpoints are needed to fulfill the user's request, let the user know about them.
If you are every stuck in a loop, or unsure about what to do, respond with a message asking the user for clarification or more information.
"""

from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.01,  # 1 request every 60s
    check_every_n_seconds=0.1,  # Check every 100ms whether allowed to make a request
    max_bucket_size=10,  # Controls the maximum burst size.
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    max_tokens=65536,
    timeout=None,
)

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=16384,)

requests_wrapper = RequestsWrapper(headers={"KEY": f"{os.getenv('CALDERA_API_TOKEN')}"})
ALLOW_DANGEROUS_REQUEST = True

api_response_schema = {
    "type": "object",
    "description": "Can represent either a natural language answer or a Caldera API response.",
    "properties": {
        "mode": {
            "type": "string",
            "description": "Defines whether this output is a normal text message or an API response.",
            "enum": ["message", "api_response"]
        },

        # Natural-language response mode
        "message": {
            "type": ["string", "null"],
            "description": "Direct answer to the user when mode=message."
        },

        # API response mode
        "success": {
            "type": ["boolean", "null"],
            "description": "True if the API request succeeded, false otherwise. Null when mode=message."
        },
        "status_code": {
            "type": ["integer", "null"],
            "description": "HTTP status code returned by the Caldera API. Null when mode=message."
        },
        "endpoint": {
            "type": ["string", "null"],
            "description": "The API endpoint that was called. Null when mode=message."
        },
        "data": {
            "description": "API response content. Null when mode=message.",
            "oneOf": [
                {"type": "object"},
                {"type": "array"},
                {"type": "string"},
                {"type": "number"},
                {"type": "boolean"},
                {"type": "null"}
            ]
        },
        "error": {
            "type": ["object", "null"],
            "description": "Error info if success=false. Null when mode=message.",
            "properties": {
                "message": {"type": "string"},
                "details": {"type": "string"},
                "type": {"type": "string"}
            }
        }
    },

    "required": ["mode"]
}       

caldera_agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    
    tools=[api_call, retrieve_context],
    response_format=ToolStrategy(api_response_schema),
    debug=False,
    middleware=[
        SummarizationMiddleware(
            model=llm,
            trigger=("tokens", 4000),
            keep=("messages", 20),
        ),
    ],
)

# user_query = (
#     "What's the status of the caldera server?"
# )
# result = caldera_agent.invoke(
#     {"messages": [{"role": "user", "content": f"{user_query}"}]},
# )

# result["structured_response"]


# # # Run the agent
# import datetime

def chat_loop():
    config = {"recursion_limit": 10 ,"configurable": {"thread_id": "1"}}
    
    # Setup readline for history
    history_file = os.path.expanduser("~/.caldera_agent_history")
    if os.path.exists(history_file):
        readline.read_history_file(history_file)
    
    readline.set_history_length(1000)
    
    print("\n" + "="*60)
    print("🤖 Caldera Agent Chat Interface")
    print("="*60)
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'help' for available commands")
    print("Use ↑↓ arrow keys to navigate command history")
    print("="*60 + "\n")
    
    conversation_history = []
    
    try:
        while True:
            try:
                user_input = input("You: ").strip()
                
                # Handle empty input
                if not user_input:
                    continue
                
                # Handle exit commands
                if user_input.lower() in ["exit", "quit"]:
                    print("\n👋 Thank you for using Caldera Agent. Goodbye!")
                    break
                
                # Handle help command
                if user_input.lower() == "help":
                    print("""
Available commands:
  help     - Show this help message
  clear    - Clear conversation history
  exit     - Exit the chat
  quit     - Exit the chat
                    """)
                    continue
                
                # Handle clear command
                if user_input.lower() == "clear":
                    conversation_history = []
                    print("✓ Conversation history cleared\n")
                    continue
                
                print("\n⏳ Processing your request...\n")
                
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Build context with conversation history
                conversation_context = ""
                if conversation_history:
                    conversation_context = "Previous conversation:\n"
                    for msg in conversation_history[-4:]:  # Keep last 4 exchanges
                        conversation_context += f"- {msg}\n"
                
                formatted_query = f"""{conversation_context}
Current request: {user_input}
Timestamp: {current_time}"""
                
                try:
                    response = caldera_agent.invoke(
                        {"messages": [{"role": "user", "content": f"{formatted_query}"}]},
                        # {"input": formatted_query}, 
                        config=config,
                        tools=[api_call, retrieve_context]
                    )
                    
                    agent_output = response.get('output') or response.get('structured_response')
                    
                    # Store in history
                    conversation_history.append(f"User: {user_input}")
                    conversation_history.append(f"Agent: {agent_output}")
                    
                    # Display response with formatting
                    print("Agent:")
                    print("-" * 40)
                    # print(agent_output)
                    # from pprint import pprint
                    # [pprint(f"{k}: {agent_output[k]}") for k in agent_output]
                    from rich import print as rprint
                    from rich.pretty import Pretty

                    # agent_output is your dictionary
                    # rprint(Pretty(agent_output))
                    # print(agent_output["message"])
                    # print(type(agent_output))
                    print(agent_output.keys())
                    # print(agent_output)

                    for key, title in [
                        ("mode", "Mode"),
                        ("message", "Message"),
                        ("status_code", "Status Code"),
                        ("endpoint", "Endpoint"),
                        ("data", "Data"),
                        ("error", "Error"),
                    ]:
                        from rich.console import Console
                        from rich.markdown import Markdown
                        value = agent_output.get(key)
                        if value is not None:

                            console = Console()
                            markdown = Markdown(f"### {title}:\n```\n{value}\n")
                            console.print(markdown)


                    # pprint(agent_output)
                    print("-" * 40 + "\n")
                    
                    # Log to file
                    with open("caldera_agent.log", "a") as log_file:
                        log_file.write(f"[{current_time}] User: {user_input}\n")
                        log_file.write(f"[{current_time}] Agent: {agent_output}\n\n")
                
                except Exception as e:
                    print(f"❌ Error: {str(e)}\n")
                    with open("caldera_agent.log", "a") as log_file:
                        log_file.write(f"[{current_time}] ERROR: {str(e)}\n\n")
            
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}\n")
    
    finally:
        # Save history before exiting
        readline.write_history_file(history_file)

if __name__ == "__main__":
    chat_loop()