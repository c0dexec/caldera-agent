# About project
- Create an AI agent that can perform user's tasks through the use of Caldera's API.
## Work plan
- [x] Implement OpenAPI v2 ingestion and suppliment it as context.
- [ ] Implement payload uploading capabilities.
- [ ] Look into adding custom tool in `planner.create_openapi_agent`.

# Blockers/Limitations
- Need to optimize token usage.
- Because of constraints of free API calls, I have to self host the model.
    - Paid models do much better.