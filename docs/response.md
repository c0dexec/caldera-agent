# Caldera
## Version: 5.2.0

### /api/v2/health

#### GET
##### Summary:

Health endpoints returns the status of Caldera

##### Description:

Returns the status of Caldera and additional details including versions of system components

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Includes all loaded plugins and system components. |

### /api/v2/config/{name}

#### GET
##### Summary:

Retrieve Config

##### Description:

Retrieves configuration by name, as specified by {name} in the request url.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| name | path | Name of the configuration file to be retrieved. | Yes |  |

### /api/v2/config/main

#### PATCH
##### Summary:

Update Main Config

##### Description:

Use fields from the ConfigUpdateSchema in the request body to update the main configuration file.

### /api/v2/config/agents

#### PATCH
##### Summary:

Update Agent Config

##### Description:

Use fields from the AgentConfigUpdateSchema in the request body to update the Agent Configuration file.

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | The response consists of data from the Agent configuration file dumped in the AgentConfigUpdateSchema format. |

### /api/v2/planners

#### GET
##### Summary:

Retrieve planners

##### Description:

Retrieve Caldera planners by criteria. Supply fields from the `PlannerSchema` to the `include` and `exclude` fields of the `BaseGetAllQuerySchema` in the request body to filter retrieved planners.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| sort | query |  | No | string |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns a list of matching planners in `PlannerSchema` format. |

### /api/v2/planners/{planner_id}

#### GET
##### Summary:

Retrieve a planner by planner id

##### Description:

Retrieve one Caldera planner based on the planner id (String `UUID`). Supply fields from the `PlannerSchema` to the `include` and `exclude` fields of the `BaseGetOneQuerySchema` in the request body to filter retrieved planners.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| planner_id | path | UUID of the Planner object to be retrieved. | Yes |  |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns a planner with the specified id in `PlannerSchema` format. |

#### PATCH
##### Summary:

Updates an existing planner.

##### Description:

Updates a planner based on the `PlannerSchema` value provided in the message body.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| planner_id | path | UUID of the Planner to be updated | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | JSON dictionary representation of the replaced Planner. |

### /api/v2/abilities

#### GET
##### Summary:

Get all abilities.

##### Description:

Provides a list of all available abilities.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| sort | query |  | No | string |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns a list of all abilities. |

#### POST
##### Summary:

Creates a new ability.

##### Description:

Creates a new ability based on the `AbilitySchema`. "name", "tactic", "technique_name", "technique_id" and "executors" are all required fields.

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | JSON dictionary representation of the created Ability. |

### /api/v2/abilities/{ability_id}

#### GET
##### Summary:

Get an ability.

##### Description:

Provides one ability based on its ability id.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| ability_id | path | UUID of the Ability to be retrieved | Yes |  |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | JSON dictionary representation of the existing Ability. |

#### PUT
##### Summary:

Replaces an existing ability.

##### Description:

Replaces an ability based on the `AbilitySchema` values provided in the message body. "name", "tactic", and "executors" are all required fields.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| ability_id | path | UUID of the Ability to be retrieved | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | JSON dictionary representation of the replaced Ability. |

#### DELETE
##### Summary:

Deletes an ability.

##### Description:

Deletes an existing ability.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| ability_id | path | UUID of the Ability to be retrieved | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 204 | HTTP 204 Status Code (No Content) |

#### PATCH
##### Summary:

Updates an existing ability.

##### Description:

Updates an ability based on the `AbilitySchema` values provided in the message body.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| ability_id | path | UUID of the Ability to be retrieved | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | JSON dictionary representation of the replaced Ability. |

### /api/v2/plugins

#### GET
##### Summary:

Retrieve all plugins

##### Description:

Returns a list of all available plugins in the system, including directory, description,and active status. Supply fields from the `PluginSchema` to the include and exclude fields of the `BaseGetAllQuerySchema` in the request body to filter retrieved plugins.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| sort | query |  | No | string |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns a list in `PluginSchema` format of all available plugins in the system. |

### /api/v2/plugins/{name}

#### GET
##### Summary:

Retrieve plugin by name

##### Description:

If plugin was found with a matching name, an object containing information about the plugin is returned.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| name | path | The name of the plugin | Yes |  |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns a plugin in `PluginSchema` format with the requested name, if it exists. |

### /api/v2/sources

#### GET
##### Summary:

Retrieve all Fact Sources.

##### Description:

Returns a list of all Fact Sources, including custom-created ones.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| sort | query |  | No | string |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns a list of all Sources dumped in SourceSchema format. |

#### POST
##### Summary:

Create a Fact Source.

##### Description:

Create a new Fact Source using the format provided in the SourceSchema.

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns a single Source dumped in SourceSchema format. |

### /api/v2/sources/{id}

#### GET
##### Summary:

Retrieve a Fact Source by its id.

##### Description:

Returns a Fact Source, given a source id.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | The id of the Fact Source | Yes |  |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns a single Source dumped in SourceSchema format. |

#### PUT
##### Summary:

Update an existing or create a new Fact Source.

##### Description:

Use fields from the SourceSchema in the request body to replace an existing Fact Source or create a new Fact Source.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | The id of the Fact Source. | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns a single Source dumped in SourceSchema format. |

#### DELETE
##### Summary:

Delete an existing Fact Source.

##### Description:

Delete a Fact Source, given its id.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | The id of the Fact Source to be deleted. | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns DELETE status. |

#### PATCH
##### Summary:

Update an existing Fact Source.

##### Description:

Returns an updated Fact Source. All fields in a Fact Source can be updated, except for "id" and "adjustments".

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | The id of the Fact Source. | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns a single Source dumped in SourceSchema format. |

### /api/v2/objectives

#### GET
##### Summary:

Retrieve objectives

##### Description:

Retrieve all objectives by criteria. Use fields from the `ObjectiveSchema` in the request body to filter retrieved objectives.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| sort | query |  | No | string |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | A list of all objectives dumped in ObjectiveSchema format. |

#### POST
##### Summary:

Create a new objective

##### Description:

Create a new objective using the format provided in the `ObjectiveSchema`.

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | A single objective in ObjectiveSchema format. |

### /api/v2/objectives/{id}

#### GET
##### Summary:

Retrieve objective by ID

##### Description:

Retrieve one objective by ID. Use fields from the `ObjectiveSchema` in the request body to filter retrieved objective.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | UUID of the objective to be retrieved | Yes |  |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns single objective in ObjectiveSchema format. |

#### PUT
##### Summary:

Create or update an objective

##### Description:

Attempt to update an objective using fields from the `ObjectiveSchema` in the request body. If the objective does not already exist, then create a new one using the `ObjectiveSchema` format.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | UUID of the Objective to be created or updated | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | A single Objective, either newly created or updated, in ObjectiveSchema format. |

#### PATCH
##### Summary:

Update an objective

##### Description:

Update an objective using fields from the `ObjectiveSchema` in the request body.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | UUID of the Objective to be updated | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | The updated Objective in ObjectiveSchema format. |

### /api/v2/adversaries

#### GET
##### Summary:

Retrieve all adversaries

##### Description:

Returns a list of all available adversaries in the system, including plugin, name, description, and atomic ordering. Supply fields from the `AdversarySchema` to the include and exclude fields of the `BaseGetAllQuerySchema` in the request body to filter retrieved adversaries.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| sort | query |  | No | string |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns a list in `AdversarySchema` format of all available adversaries in the system. |

#### POST
##### Summary:

Create a new adversary

##### Description:

Create a new adversary using the format provided in the `AdversarySchema`.

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | A single adversary in AdversarySchema format. |

### /api/v2/adversaries/{adversary_id}

#### GET
##### Summary:

Retrieve adversary by ID

##### Description:

Retrieve one adversary by ID. Use fields from the `AdversarySchema` in the request body to filter retrieved adversary.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| adversary_id | path | UUID of the adversary to be retrieved | Yes |  |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns single adversary in AdversarySchema format. |

#### PUT
##### Summary:

Create or update an adversary

##### Description:

Attempt to update an adversaries using fields from the `AdversarySchema` in the request body. If the adversary does not already exist, then create a new one using the `AdversarySchema` format.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| adversary_id | path | UUID of the adversary to be created or updated | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | A single adversary, either newly created or updated, in AdversarySchema format. |

#### DELETE
##### Summary:

Deletes an adversary.

##### Description:

Deletes an existing adversary.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| adversary_id | path | UUID of the adversary to be retrieved | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 204 | HTTP 204 Status Code (No Content) |

#### PATCH
##### Summary:

Update an adversary

##### Description:

Update an adversary using fields from the `AdversarySchema` in the request body.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| adversary_id | path | UUID of the adversary to be updated | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | The updated adversary in AdversarySchema format. |

### /api/v2/agents

#### GET
##### Summary:

Retrieves all agents

##### Description:

Retrieves all stored agents.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| sort | query |  | No | string |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns a list of all agents. |

#### POST
##### Summary:

Create a new agent

##### Description:

Creates a new agent using the format from 'AgentSchema'.

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns a single agent in 'AgentSchema' format |

### /api/v2/agents/{paw}

#### GET
##### Summary:

Retrieve Agent by paw

##### Description:

Retrieve information about a specific agent using its ID (paw). Use the paw field in the URL to specify matching criteria for the agent to obtain information about.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| paw | path | ID of the Agent to retrieve information about | Yes |  |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns JSON response with specified Agent |

#### PUT
##### Summary:

Create or Update an Agent

##### Description:

Update an agent, or if a existing agent match cannot be found, create one. Use the paw field in the URL to specify matching criteria and the fields from the AgentSchema in the request body to specify new field values.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| paw | path | paw of the Agent to be retrieved | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Json dictionary representation of the created or updated Agent |

#### DELETE
##### Summary:

Delete an Agent

##### Description:

Delete an agent. Use the paw field in the URL to specify matching criteria for the agent(s) to delete.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| paw | path | paw of the Agent to be deleted | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns HTTP 200 |

#### PATCH
##### Summary:

Update an Agent

##### Description:

Update the attributes of a specific Agent using its ID (paw). Use the paw field in the URL to specify matching criteria and the fields from the AgentSchema in the request body to specify updated field values.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| paw | path | ID of the Agent to update | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns JSON response with updated Agent fields |

### /api/v2/deploy_commands

#### GET
##### Summary:

Retrieve deploy commands

##### Description:

Retrieve the deploy commands currently configured within Caldera.

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Json dictionary representation of deploy commands, sorted by Ability ID |

### /api/v2/deploy_commands/{ability_id}

#### GET
##### Summary:

Retrieve deploy commands for an Ability

##### Description:

Retrieve the deploy commands associated with a given ability ID. Use the 'ability_id' field in the URL specify which ability to retrieve deploy commands for.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| ability_id | path | ID of the ability to retrieve deploy commands for | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Json dictionary representation of deploy commands for the specified Ability ID |

### /api/v2/schedules

#### GET
##### Summary:

Retrieve Schedules

##### Description:

Returns all stored schedules.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| sort | query |  | No | string |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | The response is a list of all scheduled operations. |

#### POST
##### Summary:

Create Schedule

##### Description:

Use fields from the ScheduleSchema in the request body to create a new Schedule.

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | The response is a dump of the newly created Schedule object. |

### /api/v2/schedules/{id}

#### GET
##### Summary:

Retrieve Schedule

##### Description:

Retrieves Schedule by UUID, as specified by {id} in the request url.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | UUID of the Schedule to be retrieved. | Yes |  |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | The response is a single dumped Scheduled object. |

#### PUT
##### Summary:

Replace Schedule

##### Description:

Use fields from the ScheduleSchema in the request body to replace an existing Schedule or create a new Schedule.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | UUID of the Schedule to be retrieved. | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | The response is a dump of the newly replaced Schedule object. |

#### DELETE
##### Summary:

Delete Schedule

##### Description:

Deletes a Schedule object from the data service.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | UUID of the Schedule to be retrieved. | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns HTTP 204 No Content status code if Schedule is deleted successfully. |

#### PATCH
##### Summary:

Update Schedule

##### Description:

Use fields from the ScheduleSchema in the request body to update an existing Schedule.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | UUID of the Schedule to be retrieved. | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | The response is a dump of the newly updated Schedule object. |

### /api/v2/operations

#### GET
##### Summary:

Retrieve operations

##### Description:

Retrieve all Caldera operations from memory.  Use fields from the `BaseGetAllQuerySchema` in the request body to filter.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| sort | query |  | No | string |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | The response is a list of all operations. |

#### POST
##### Summary:

Create a new Caldera operation record

##### Description:

Create a new Caldera operation using the format provided in the `OperationSchema`. Required schema fields are as follows: "name", "adversary.adversary_id", "planner.id", and "source.id"

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | The response is the newly-created operation report. |

### /api/v2/operations/summary

#### GET
##### Summary:

Retrieve operations (alternate)

##### Description:

Retrieve all Caldera operations from memory, with an alternate selection of properties. Use fields from the `BaseGetAllQuerySchema` in the request body to filter.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| sort | query |  | No | string |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | The response is a list of all operations. |

### /api/v2/operations/{id}

#### GET
##### Summary:

Retrieve an operation by operation id

##### Description:

Retrieve one Caldera operation from memory based on the operation id (String UUID).  Use fields from the `BaseGetOneQuerySchema` in the request body to add `include` and `exclude` filters.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | UUID of the Operation object to be retrieved. | Yes |  |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | The response is the operation with the specified id, if any. |

#### DELETE
##### Summary:

Delete an operation by operation id

##### Description:

Delete one Caldera operation from memory based on the operation id (String UUID).

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | UUID of the Operation object to be retrieved. | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | There is an empty response from a successful delete request. |

#### PATCH
##### Summary:

Update fields within an operation

##### Description:

Update one Caldera operation in memory based on the operation id (String UUID). The `state`, `autonomous` and `obfuscator` fields in the operation object may be edited in the request body using the `OperationSchema`.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | UUID of the Operation object to be retrieved. | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | The response is the updated operation, including user modifications. |

### /api/v2/operations/{id}/report

#### POST
##### Summary:

Get Operation Report

##### Description:

Retrieves the report for a given operation_id.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path |  | Yes |  |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 |  |

### /api/v2/operations/{id}/event-logs

#### POST
##### Summary:

Get Operation Event Logs

##### Description:

Retrieves the event logs for a given operation_id.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path |  | Yes |  |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 |  |

### /api/v2/operations/{id}/links

#### GET
##### Summary:

Get Links from Operation

##### Description:

Retrieves all links for a given operation_id.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path |  | Yes |  |
| sort | query |  | No | string |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | All links contained in operation with the given `id` (String UUID). |

### /api/v2/operations/{id}/links/{link_id}

#### GET
##### Summary:

Retrieve a specified link from an operation

##### Description:

Retrieve the link with the provided `link_id` (String UUID) from the operation with the given operation `id` (String UUID). Use fields from the `BaseGetOneQuerySchema` in the request body to add `include` and `exclude` filters.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | String UUID of the Operation containing desired link. | Yes |  |
| link_id | path | String UUID of the Link with the above operation. | Yes |  |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | The link matching the provided `link_id` within the operation matching `id`. Use fields from the `BaseGetOneQuerySchema` in the request body to add `include` and `exclude` filters. |

#### PATCH
##### Summary:

Update the specified link within an operation

##### Description:

Update the `command` (String) or `status` (Integer) field within the link with the provided  `link_id` (String UUID) from the operation with the given operation `id` (String UUID).

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | String UUID of the Operation containing desired link. | Yes |  |
| link_id | path | String UUID of the Link with the above operation. | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | The updated link after a successful `PATCH` request. |

### /api/v2/operations/{id}/links/{link_id}/result

#### GET
##### Summary:

Retrieve the result of a link

##### Description:

Retrieve a dictionary containing a link and its results dictionary based on the operation id (String UUID) and link id (String UUID).  Use fields from the `BaseGetOneQuerySchema` in the request body to add `include` and `exclude` filters.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | UUID of the operation object to be retrieved. | Yes |  |
| link_id | path | UUID of the link object to retrieve results of. | Yes |  |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Contains a dictionary with the requested link and its results dictionary. |

### /api/v2/operations/{id}/potential-links

#### GET
##### Summary:

Retrieve potential links for an operation.

##### Description:

Retrieve all potential links for an operation based on the operation id (String UUID).  Use fields from the `BaseGetAllQuerySchema` in the request body to add `include`, `exclude`, and `sort` filters.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | UUID of the operation object to retrieve links for. | Yes |  |
| sort | query |  | No | string |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Response contains a list of link objects for the requested id. |

#### POST
##### Summary:

Creates a potential Link

##### Description:

Creates a potential link to be executed by an agent. Create a potential Link using the format provided in the `LinkSchema`. The request body requires `paw`, `executor`, and `ability`.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | UUID of the operation object for the link to be created on. | Yes |  |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Response contains the newly assigned Link object. |

### /api/v2/operations/{id}/potential-links/{paw}

#### GET
##### Summary:

Retrieve potential links for an operation filterd by agent paw (id)

##### Description:

Retrieve all potential links for an operation-agent pair based on the operation id (String UUID) and the agent paw (id) (String).  Use fields from the `BaseGetAllQuerySchema` in the request body to add `include`, `exclude`, and `sort` filters.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| id | path | String UUID of the Operation containing desired links. | Yes |  |
| paw | path | Agent paw for the specified operation. | Yes |  |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | All potential links for operation and the specified agent paw. |

### /api/v2/obfuscators

#### GET
##### Summary:

Retrieve obfuscators

##### Description:

Retrieves all stored obfuscators.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| sort | query |  | No | string |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns a list of all obfuscators in ObfuscatorSchema format. |

### /api/v2/obfuscators/{name}

#### GET
##### Summary:

Retrieve an obfuscator by name

##### Description:

Retrieve an obfuscator by name, as specified by {name} in the request url.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| name | path | Name of the Obfuscator | Yes |  |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns the specified obfuscator in ObfuscatorSchema format. |

### /api/v2/facts

#### GET
##### Summary:

Retrieve Facts

##### Description:

Retrieve facts by criteria. Use fields from the `FactSchema` in the request body to filter retrieved facts.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| sort | query |  | No | string |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns a list of matching facts, dumped in `FactSchema` format. |

#### POST
##### Summary:

Create a Fact

##### Description:

Create a new fact using the format provided in the `FactSchema`.

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns the created fact, dumped in `FactSchema` format. |

#### DELETE
##### Summary:

Delete One or More Facts

##### Description:

Delete facts using the format provided in the `FactSchema`. This will delete all facts that match the criteria specified in the payload.

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns the deleted fact(s), dumped in `FactSchema` format. |

#### PATCH
##### Summary:

Update One or More Facts

##### Description:

Update existing facts using the format provided in the `FactSchema`. This will update all facts that match the criteria specified in the payload.

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns the updated fact(s), dumped in `FactSchema` format. |

### /api/v2/relationships

#### GET
##### Summary:

Retrieve Relationships

##### Description:

Retrieve relationships by criteria. Use fields from the `RelationshipSchema` in the request body to filter retrieved relationships.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| sort | query |  | No | string |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns a list of matching relationships, dumped in `RelationshipSchema` format. |

#### POST
##### Summary:

Create a Relationship

##### Description:

Create a new relationship using the format provided in the `RelationshipSchema`.

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns the created relationship, dumped in `RelationshipSchema` format. |

#### DELETE
##### Summary:

Delete One or More Relationships

##### Description:

Delete relationships using the format provided in the RelationshipSchema. This will delete all relationships that match the criteria specified in the payload.

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns the deleted relationship(s), dumped in RelationshipSchema format. |

#### PATCH
##### Summary:

Update One or More Relationships

##### Description:

Update existing relationships using the format provided in the `RelationshipSchema`. This will update all relationships that match the criteria specified in the payload.

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns the updated relationship(s), dumped in `RelationshipSchema` format. |

### /api/v2/facts/{operation_id}

#### GET
##### Summary:

Retrieve Facts by operation id

##### Description:

Retrieves facts associated with an operation. Returned facts will either be user-generated facts or learned facts.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| sort | query |  | No | string |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |
| operation_id | path |  | Yes | string |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns a list of facts associated with operation_id, dumped in `FactSchema` format. |

### /api/v2/relationships/{operation_id}

#### GET
##### Summary:

Retrieve Relationships by operation id

##### Description:

Retrieve relationships associated with an operation. Returned relationships will be either user-generated relationships or learned relationships.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| sort | query |  | No | string |
| include | query |  | No | [ string ] |
| exclude | query |  | No | [ string ] |
| operation_id | path |  | Yes | string |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns a list of matching relationships, dumped in `RelationshipSchema` format. |

### /api/v2/contacts/{name}

#### GET
##### Summary:

Retrieve a List of Beacons made by Agents to the specified Contact

##### Description:

Returns a list of beacons made by agents to the specified contact. The response is formatted as a list of dictionaries. The dictionaries have the keys `paw`, `instructions`, and `date`. `paw` being the paw of the agent that made the beacon. `instructions` being a list of strings (commands) executed by the agent since its last beacon. `date` being a UTC date/time string.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| name | path | Name of the contact to get beacons for, e.g. HTTP, TCP, et cetera. | Yes |  |

### /api/v2/contacts

#### GET
##### Summary:

Retrieve a List of all available Contact reports

##### Description:

Returns a list of contacts that at least one agent has beaconed to. As soon as any agent beacons over a given contact, the contact will be returned here.

### /api/v2/payloads

#### GET
##### Summary:

Retrieve payloads

##### Description:

Retrieves all stored payloads.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| sort | query |  | No | boolean |
| exclude_plugins | query |  | No | boolean |
| add_path | query |  | No | boolean |

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Returns a list of all payloads in PayloadSchema format. |

#### POST
##### Summary:

Create a payload

##### Description:

Uploads a payload.

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | The created payload in a list in PayloadSchema format (with name changed in case of a duplicate). |

### /api/v2/payloads/{name}

#### DELETE
##### Summary:

Delete a payload

##### Description:

Deletes a given payload.

##### Parameters

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| name | path |  | Yes | string |

##### Responses

| Code | Description |
| ---- | ----------- |
| 204 | Payload has been properly deleted. |
| 404 | Payload not found. |
