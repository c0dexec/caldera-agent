import os
import json
import requests
from dotenv import load_dotenv
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec

def load_caldera_spec():
    # Reading URL

    # resp = requests.get(
    #     f"{os.getenv('CALDERA_WEB_URL')}/api/docs/swagger.json",
    #     headers={"KEY": os.getenv("CALDERA_API_TOKEN")}
    # )
    # resp.raise_for_status()

    # Reading File

    with open("response_1765136132246.json") as f:
        resp = json.load(f)

    spec = resp  # already a dict

    # If no 'servers' (likely OpenAPI v2), synthesize one
    if "servers" not in spec:
        host = spec.get("host")
        base = spec.get("basePath", "")
        schemes = spec.get("schemes", [])
        if host:
            scheme = schemes[0] if schemes else ("https" if os.getenv("CALDERA_WEB_URL", "").startswith("https") else "http")
            url = f"{scheme}://{host}{base}"
        else:
            url = os.getenv("CALDERA_WEB_URL")
        spec["servers"] = [{"url": url}]

    # after loading spec into `spec`
    for route, ops in spec.get("paths", {}).items():
        for method, docs in ops.items():
            if isinstance(docs, dict):
                docs.setdefault("responses", {})

    caldera_api_spec = reduce_openapi_spec(spec)
    return caldera_api_spec

# print(load_caldera_spec())