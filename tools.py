from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
import requests

@tool
def api_call(runtime: ToolRuntime, api_path: str, req_type: str, params: dict, payload: str, body: dict) -> str:
    """Make an API call to a specified endpoint. Depending on the req_type, it might include a payload or body which is json text.

    Args:
        url: Base URL of the API
        api_path: Specific API path to call
        req_type: Type of HTTP request (GET, POST, PUT, DELETE, PATCH, HEAD)
        params: Query parameters for the API call
        payload: File path for payload (if applicable)
        body: JSON body for the API call (if applicable)
    """
    req_type = req_type.lower()
    url = "http://12.1.0.15:8888".strip()
    api_path = api_path.strip()
    full_url = f"{url}/{api_path}"
    auth = {"KEY": f"{os.getenv('CALDERA_API_TOKEN')}"}

    payload = {'file': open(f"{payload}", "rb")} if payload else None

    body = runtime.state["body"]

    if req_type == "get":
        response = requests.get(full_url, params=params, headers=auth)
    elif req_type == "post":
        response = requests.post(full_url, files=payload, json=body, params=params, headers=auth)
    elif req_type == "put":
        response = requests.put(full_url, json=body, params=params, headers=auth)
    elif req_type == "delete":
        response = requests.delete(full_url, params=params, headers=auth)
    elif req_type == "patch":
        response = requests.patch(full_url, json=body, params=params, headers=auth)
    elif req_type == "head":
        response = requests.head(full_url, params=params, headers=auth)
    else:
        return f"Unsupported request type: {req_type}"
    
    return response.text

@dataclass
class Context:
    api_path: str
    body: dict
    payload: str
    params: dict
    req_type: str