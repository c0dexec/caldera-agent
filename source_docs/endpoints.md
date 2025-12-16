# API Endpoints and Supported Request Types

This document lists all available API endpoints and the HTTP methods they support.

| Endpoint | Supported Methods |
|----------|-----------------|
| /api/v2/health | GET, HEAD |
| /api/v2/config/{name} | GET, HEAD |
| /api/v2/config/main | PATCH |
| /api/v2/config/agents | PATCH |
| /api/v2/planners | GET, HEAD |
| /api/v2/planners/{planner_id} | GET, HEAD, PATCH |
| /api/v2/abilities | GET, HEAD, POST |
| /api/v2/abilities/{ability_id} | DELETE, GET, HEAD, PATCH, PUT |
| /api/v2/plugins | GET, HEAD |
| /api/v2/plugins/{name} | GET, HEAD |
| /api/v2/sources | GET, HEAD, POST |
| /api/v2/sources/{id} | DELETE, GET, HEAD, PATCH, PUT |
| /api/v2/objectives | GET, HEAD, POST |
| /api/v2/objectives/{id} | GET, HEAD, PATCH, PUT |
| /api/v2/adversaries | GET, HEAD, POST |
| /api/v2/adversaries/{adversary_id} | DELETE, GET, HEAD, PATCH, PUT |
| /api/v2/agents | GET, HEAD, POST |
| /api/v2/agents/{paw} | DELETE, GET, HEAD, PATCH, PUT |
| /api/v2/deploy_commands | GET, HEAD |
| /api/v2/deploy_commands/{ability_id} | GET, HEAD |
| /api/v2/schedules | GET, HEAD, POST |
| /api/v2/schedules/{id} | DELETE, GET, HEAD, PATCH, PUT |
| /api/v2/operations | GET, HEAD, POST |
| /api/v2/operations/summary | GET, HEAD |
| /api/v2/operations/{id} | DELETE, GET, HEAD, PATCH |
| /api/v2/operations/{id}/report | POST |
| /api/v2/operations/{id}/event-logs | POST |
| /api/v2/operations/{id}/links | GET, HEAD |
| /api/v2/operations/{id}/links/{link_id} | GET, HEAD, PATCH |
| /api/v2/operations/{id}/links/{link_id}/result | GET, HEAD |
| /api/v2/operations/{id}/potential-links | GET, HEAD, POST |
| /api/v2/operations/{id}/potential-links/{paw} | GET, HEAD |
| /api/v2/obfuscators | GET, HEAD |
| /api/v2/obfuscators/{name} | GET, HEAD |
| /api/v2/facts | DELETE, GET, HEAD, PATCH, POST |
| /api/v2/relationships | DELETE, GET, HEAD, PATCH, POST |
| /api/v2/facts/{operation_id} | GET, HEAD |
| /api/v2/relationships/{operation_id} | GET, HEAD |
| /api/v2/contacts | GET, HEAD |
| /api/v2/contacts/{name} | GET, HEAD |
| /api/v2/payloads | GET, HEAD, POST |
| /api/v2/payloads/{name} | DELETE |