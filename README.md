# About project
- Create an AI agent that can perform user's tasks through the use of Caldera's API.

## Pre-requisites
- Before execution of `main.py` run `index_embedding.py` to initialize our local embedding store.
- Make sure to have an `.env` file with the necessary variables,
    - `GOOGLE_API_KEY`
    - `CALDERA_API_TOKEN`
    - `CALDERA_WEB_URL`
- A subscription to an AI agent would be required, preferably Gemini as the code was tested on [Gemini 2.5 Flash-Lite](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash-lite).

## Work plan
- [x] Implement OpenAPI v2 ingestion and suppliment it as context.
- [x] Implement payload uploading capabilities.
- ~~[ ] Look into adding custom tool in `planner.create_openapi_agent`.~~
    - [x] Improved custom tool instead.
- [x] Need to optimize token usage.
    - [x] RAG JSON parsing.
    - [x] RAG Markdown parsing.
- [x] Implemented local embedding to reduce context windows. (https://docs.langchain.com/oss/python/integrations/vectorstores/faiss)
    - [x] Added summarization of context within the agent.
- [ ] Add toggle for debugging.
- [ ] Find a way to parse User's response and make LLM pass it as variable for `api_request` tool.

# Blockers/Limitations
- Sometime endpoint isnt found right away, need to ask AI again.
- Because of constraints of free API calls, I have to self host the model.
    - Paid models do much better.
- Identified that LLM is not forwarding `params` and `body` to `api_request` tool.
