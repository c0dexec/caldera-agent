import requests
import os
from dotenv import load_dotenv
from langchain.tools import tool, ToolRuntime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from langchain_core.messages import HumanMessage, AIMessage


# Load environment variables
load_dotenv()

# Load LIVE Swagger JSON from your Caldera API
print("📥 Loading live Swagger spec...")
try:
    swagger_spec = requests.get("http://12.1.0.15:8888/api/docs/swagger.json", timeout=10).json()
    print(f"✅ Loaded {swagger_spec['info']['title']} v{swagger_spec['info']['version']}")
except Exception as e:
    print(f"⚠️  Swagger fetch failed: {e}")
    # Fallback minimal spec
    swagger_spec = {
        "info": {"title": "Caldera API", "version": "live"},
        "paths": {
            "/api/v2/health": {"get": {"summary": "Health check"}},
            "/api/v2/agents": {"get": {"summary": "List agents"}}
        }
    }

# Swagger context formatter
def format_swagger_context(spec):
    context = f"""🔥 LIVE Caldera API v{spec['info']['version']} - http://12.1.0.15:8888
    
📋 KEY ENDPOINTS ({len(spec.get('paths', {}))} total):
"""
    paths_shown = 0
    for path, methods in list(spec.get('paths', {}).items())[:12]:  # Show top 12
        for method, details in methods.items():
            summary = details.get('summary', 'No description')[:80]
            context += f"  {method.upper()} {path:<25} | {summary}\n"
            paths_shown += 1
            if paths_shown >= 20:  # Max 20 lines
                break
        if paths_shown >= 20:
            break
    
    param_examples = [
        "/api/v2/agents/{paw} → params={'paw': 'abc123'}",
        "/api/v2/operations → params={'limit': 10}",
        "POST /api/v2/objectives → body={'name': 'test'}"
    ]
    context += f"\n💡 EXAMPLES:\n" + "\n".join(param_examples)
    return context

api_context = format_swagger_context(swagger_spec)

# === YOUR EXACT ORIGINAL TOOL ===
@tool
def api_call(runtime: ToolRuntime, api_path: str, req_type: str, params: dict, payload: str, body: dict) -> str:
    """Make an API call to a specified endpoint. Depending on the req_type, it might include a payload or body which is json text.

    Args:
        api_path: Specific API path to call
        req_type: Type of HTTP request (GET, POST, PUT, DELETE, PATCH, HEAD)
        params: Query parameters for the API call
        payload: File path for payload (PUT/POST, e.g., '/tmp/payload.bin')
        body: JSON body for the API call (if applicable)
    """
    req_type = req_type.lower()
    url = "http://12.1.0.15:8888".strip()
    api_path = api_path.strip()
    full_url = f"{url}/{api_path}"

    # Headers with SECRET_KEY (ADDED)
    headers = {"Authorization": f"Bearer {SECRET_KEY}"}

    # File payload handling (YOUR ORIGINAL LOGIC)
    files = {'file': open(payload, 'rb')} if payload else None

    # Body from runtime.state (YOUR ORIGINAL LOGIC)
    body = runtime.state.get("body", body)

    console.print(f"[yellow]🌐 {req_type.upper()} {full_url}[/yellow]")
    console.print(f"[dim]📥 Params: {params} | 📎 Payload: {payload or 'none'} | 📤 Body: {body}[/dim]")

    try:
        if req_type == "get":
            response = requests.get(full_url, params=params, headers=headers, timeout=60)
        elif req_type == "post":
            response = requests.post(full_url, files=files, json=body, params=params, headers=headers, timeout=60)
        elif req_type == "put":
            # ✅ PAYLOAD SUPPORT FOR PUT (your requirement)
            if files:
                response = requests.put(full_url, files=files, params=params, headers=headers, timeout=60)
            else:
                response = requests.put(full_url, json=body, params=params, headers=headers, timeout=60)
        elif req_type == "delete":
            response = requests.delete(full_url, params=params, headers=headers, timeout=60)
        elif req_type == "patch":
            response = requests.patch(full_url, json=body, params=params, headers=headers, timeout=60)
        elif req_type == "head":
            response = requests.head(full_url, params=params, headers=headers, timeout=60)
        else:
            return f"Unsupported request type: {req_type}"
        
        content = response.text[:1200]
        if response.status_code == 200:
            return f"✅ **SUCCESS {response.status_code}**\n```\n{content}\n```"
        else:
            return f"⚠️  **{response.status_code}**\n```\n{content}\n```"
            
    except FileNotFoundError:
        return f"❌ **File not found**: {payload}"
    except requests.exceptions.RequestException as e:
        return f"❌ **HTTP {getattr(e.response, 'status_code', 'ERROR')}**: {str(e)}"
    except Exception as e:
        return f"❌ **Error**: {str(e)}"

# === GEMINI SETUP ===
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("❌ Set GOOGLE_API_KEY in .env!")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# LangChain agent with proper message formatting
system_prompt = f"""{api_context}

🧠 MEMORY: I'll remember your last API path/params!

🎯 HOW TO USE:
• "GET all agents" → api_call(api_path="api/v2/agents")
• "GET agent abc123" → api_call(api_path="api/v2/agents/abc123") 
• "List operations" → api_call(api_path="api/v2/operations")
• "Same as before but different id" → reuse memory!

Commands: 'memory' | 'health' | 'quit'
"""

# Simple agent: bind tools and create executor
llm_with_tools = llm.bind_tools([api_call])

# Make an agent executor compatible with LangChain v1.1.2
agent_executor = initialize_agent(
    tools=[api_call],   # your @tool-decorated function
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

class ChatBot:
    def __init__(self):
        self.sessions = {}
    
    def get_session(self, user_id: str):
        if user_id not in self.sessions:
            self.sessions[user_id] = {"history": []}
        return self.sessions[user_id]
    
    def chat(self, user_id: str, message: str):
        session = self.get_session(user_id)
        
        # Add memory context
        memory_context = ""
        if session["history"]:
            last = session["history"][-1]["ai"]
            if "api_call" in last:
                memory_context = "\n🧠 Last call saved - say 'same endpoint' to reuse!"
        
        full_message = f"{memory_context}\n\n{message}"
        
        try:
            history_msgs = history_to_messages(session["history"][-6:])
            result = agent_executor.run(full_message)

            response = result["output"]
            session["history"].append({"human": message, "ai": response})
            
            return {
                "response": response,
                "history_length": len(session["history"])
            }
        except Exception as e:
            return {"response": f"❌ Error: {str(e)}", "history_length": len(session["history"])}

def history_to_messages(history):
    msgs = []
    for item in history:
        if "human" in item:
            msgs.append(HumanMessage(content=item["human"]))
        if "ai" in item:
            msgs.append(AIMessage(content=item["ai"]))
    return msgs

# === RICH CLI ===
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

def cli_chat():
    user_id = Prompt.ask("👤 User ID", default="demo")
    console.print(f"\n[bold green]🤖 LIVE Caldera Gemini Bot | User: {user_id}[/bold green]")
    console.print(Panel.fit(api_context, title="🔥 LIVE API", border_style="blue"))
    console.print("[dim]Commands: 'memory' | 'health' | 'quit'[/dim]\n")
    
    chatbot = ChatBot()
    
    while True:
        msg = Prompt.ask("💬")
        msg_lower = msg.lower()
        
        if msg_lower in ['quit', 'exit', 'bye']:
            console.print("👋 Bye!")
            break
        
        if msg_lower == 'health':
            result = chatbot.chat(user_id, "Check API health")
            console.print(Panel(result["response"], title="🏥 Health", border_style="green"))
            continue
        
        if msg_lower == 'memory':
            history = chatbot.sessions.get(user_id, {}).get("history", [])
            if history:
                last = history[-1]
                console.print(f"[bold]🧠 Last API:[/bold] {last['ai'][:200]}...")
            else:
                console.print("🧠 No API calls yet!")
            continue
        
        # Check if the message is a GET request or related to agents
        if msg_lower.startswith("get") or "agents" in msg_lower:
            # Call perform_api_call directly for known agent queries
            response = perform_api_call(api_path="api/v2/agents", req_type="get", params={}, payload=None, body={})
            console.print(f"\n🤖 [bold cyan]{response['output']}[/bold cyan]")
            continue

        result = chatbot.chat(user_id, msg)
        console.print(f"\n🤖 [bold cyan]{result['response']}[/bold cyan]")
        console.print(f"[dim]📊 History: {result['history_length']}[/dim]\n")

if __name__ == "__main__":
    print("🚀 Starting LIVE Caldera Bot...")
    cli_chat()

SECRET_KEY = os.getenv("SECRET_KEY", "")