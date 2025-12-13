# About project
- Create an AI agent that can perform user's tasks through the use of Caldera's API.

## Pre-requisites
- Before execution of `main.py` run `index_embedding.py` to initialize our local embedding store.

## Work plan
- [x] Implement OpenAPI v2 ingestion and suppliment it as context.
- [x] Implement payload uploading capabilities.
- ~~[ ] Look into adding custom tool in `planner.create_openapi_agent`.~~
    - [x] Improved custom tool instead.
- [x] Need to optimize token usage.
    - [x] RAG JSON parsing.
    - [x] RAG Markdown parsing.
- [x] Implemented local embedding to reduce context windows. (https://docs.langchain.com/oss/python/integrations/vectorstores/faiss)

# Blockers/Limitations
- Sometime endpoint isnt found right away, need to ask AI again.
- Because of constraints of free API calls, I have to self host the model.
    - Paid models do much better.
