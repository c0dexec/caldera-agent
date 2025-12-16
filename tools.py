from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
import requests
import os
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
from langchain.tools import tool
from vectorstore import get_vector_store
from langgraph.prebuilt.tool_node import ToolNode

@tool
def api_call(runtime: ToolRuntime, api_path: str, req_type: str, params: dict, file: str, body: dict) -> str:
    """Make an API call to a specified endpoint. Depending on the req_type, it might include a file or body which is json text.

    Always responds back with the response text using (response.text).

    Args:
        url: Base URL of the API
        api_path: Specific API path to call
        req_type: Type of HTTP request (GET, POST, PUT, DELETE, PATCH, HEAD)
        params: Query parameters for the API call
        file: File path for file (if applicable)
        body: JSON body for the API call (if applicable)
    """
    req_type = req_type.lower()
    url = "http://12.1.0.15:8888".strip()
    api_path = api_path.strip()
    full_url = f"{url}/{api_path}"
    auth = {"KEY": f"{os.getenv('CALDERA_API_TOKEN')}"}

    # Setup retries
    retries = Retry(
        total=1,                 # total retry attempts
        backoff_factor=0.3,      # wait 0.3s, 0.6s, 1.2s, etc
        status_forcelist=[502, 503, 504],
        allowed_methods={"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"},
        raise_on_status=True     # <-- this will raise a RetryError after max retries
    )
    
    s = requests.Session()
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)

    file = {'file': open(f"{file}", "rb")} if file else None

    # print(f"This is the run time state: {runtime.state}")

    body = runtime.state.get("body", {}) if not body else body

    # Make request
    try:
        if req_type == "get":
            response = s.get(full_url, params=params, headers=auth)
        elif req_type == "post":
            response = s.post(full_url, files=file, json=body, params=params, headers=auth)
        elif req_type == "put":
            response = s.put(full_url, json=body, params=params, headers=auth)
        elif req_type == "delete":
            response = s.delete(full_url, params=params, headers=auth)
        elif req_type == "patch":
            response = s.patch(full_url, json=body, params=params, headers=auth)
        elif req_type == "head":
            response = s.head(full_url, params=params, headers=auth)
        else:
            return f"Unsupported request type: {req_type}"
    except requests.exceptions.RetryError:
        # This happens after max retries
        return f"API call failed after maximum retries ({retries.total})"
    except requests.exceptions.RequestException as e:
        # Catch other network errors
        return f"API call failed due to network error: {e}"
    
    return response.text

# Error handling and reporting for tool
def handle_errors(e: ValueError) -> str:
    return "Invalid input provided"

tool_node = ToolNode([api_call], handle_tool_errors=handle_errors)
print(tool_node)

@dataclass
class Context:
    api_path: str
    body: dict
    file: str
    params: dict
    req_type: str


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    vector_store = get_vector_store()

    retrieved_docs = vector_store.similarity_search(query, k=2)

    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs