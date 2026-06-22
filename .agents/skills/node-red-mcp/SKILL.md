---
name: node-red-mcp
description: Safe Node-RED flow creation and replacement for home automation using MQTT sensors and MCP integration. Use when creating new flows, replacing existing flows, or troubleshooting Node-RED deployments.
---

# Node-RED Flow Management

Compact guide for safely creating and replacing Node-RED flows through the official Admin API.

This skill is for Claude Code and should stay aligned with the live MCP toolset and runtime behavior.

## System Context

- Node-RED: http://localhost:1880
- MQTT Broker: 192.168.50.104:1883 (HiveMQHome)
- Main use: home automation with Zigbee sensors via Home Assistant/zigbee2mqtt
- MCP server: live sensor data at `/mcp`
- Historian: separate MCP server for historical data

**Topic structure**

- Pattern: `Home/{Location}/{Metric}`
- Examples: `Home/Kitchen/Temperature`, `Home/Bedroom/Humidity`
- Sensor IDs: lowercase only (`kitchen`, `bedroom`, `living_room`)
- Temperature: store in Celsius; convert to Fahrenheit only for display

**Data format**

```json
{
  "name": "Temperature",
  "value": 17.22,
  "timestamp": 1764346999117,
  "properties": {
    "Quality": {
      "type": "Int32",
      "value": 192
    }
  }
}
```

**Quality codes**

- 192 = Good (linkquality > 75)
- 128 = Uncertain (linkquality 50-75)
- 0 = Bad (linkquality < 50)

## Node-RED Admin API

The MCP layer wraps the official Admin API with normalization, dependency analysis, graph summaries, and compact context packs.

Relevant endpoints:

- `GET /auth/login`, `POST /auth/token`, `POST /auth/revoke`
- `GET /settings`, `GET /diagnostics`
- `GET /flows`, `GET /flows/state`, `POST /flows`, `POST /flows/state`
- `POST /flow`, `GET /flow/:id`, `PUT /flow/:id`, `DELETE /flow/:id`
- `GET /nodes`, `POST /nodes`, `GET /nodes/:module`, `PUT /nodes/:module`, `DELETE /nodes/:module`, `GET /nodes/:module/:set`, `PUT /nodes/:module/:set`

## Available MCP Tools

**Primary read tools**

- `node-red-auth-get-scheme`, `node-red-auth-login`, `node-red-auth-revoke`
- `node-red-runtime-get-settings`, `node-red-runtime-get-diagnostics`
- `node-red-flows-list`, `node-red-flows-get`, `node-red-flows-export`
- `node-red-flows-summary`, `node-red-flows-analyze`
- `node-red-graph-dependencies`, `node-red-graph-query`, `node-red-graph-visualize`, `node-red-graph-pack`
- `node-red-nodes-list`, `node-red-nodes-get-module`, `node-red-nodes-get-set`

**Primary edit tools**

- `node-red-flows-create`, `node-red-flows-delete`
- `node-red-flows-clone`, `node-red-flows-rollback`

**Default edit rule**

- Do not rely on update or patch workflows for routine flow changes.
- If a flow must change, inspect it first, then replace the whole tab when needed.

**Backward-compatible aliases**

- `get-flows`, `get-flow`, `list-tabs`
- `create-flow`, `delete-flow`
- `get-flows-formatted`, `find-nodes-by-type`, `search-nodes`, `visualize-flows`
- `inject` (best-effort helper; not part of the official API)

## Orchestration Rules

- For large flows, start with `node-red-flows-summary` or `node-red-graph-pack`.
- Use `node-red-graph-pack` for the semantically relevant nodes, neighbors, and compact edges.
- Always read the target tab first with `node-red-flows-get` or `list-tabs`.
- Treat `/flows` as revision-aware when a `rev` field is present.
- Prefer `node-red-flows-clone` for branch work.
- Prefer delete-and-recreate for whole-tab replacement instead of update/patch workflows.
- Use `node-red-flows-analyze` and `node-red-graph-dependencies` before destructive actions.
- Use `node-red-graph-visualize` for topology review.
- Use `node-red-nodes-*` only for module-level changes.

## Semantic Search Strategy

1. Start with `node-red-flows-summary`.
2. Use `node-red-graph-pack` with a topic, label, node type, or action phrase.
3. Read only the returned subgraph slice and work from those node ids.
4. Re-run `node-red-graph-pack` after edits to verify the local neighborhood.
5. Use `node-red-graph-query` only for broader ranked matches.

## Output Discipline

- Prefer compact summaries, ranked matches, and 1-hop context slices over full flow dumps.
- Reserve `node-red-flows-export` and `node-red-graph-visualize` for narrow debugging or explicit review.
- For large flows, describe changes by node groups and semantic roles, not raw JSON.

## MCP Resources And Prompts

Resources cover runtime settings, diagnostics, flows, nodes, and graph snapshots. Prompts cover `analyze-flow`, `repair-flow`, and `refactor-flow`.

## Reliability Notes

- `GET /flows` may return a plain array or `{ rev, flows }`; normalize both.
- `GET /flow/:id` returns one tab with `id`, `label`, `nodes`, and usually `configs` / `subflows`.
- `POST /flow` needs a full flow object with a top-level `nodes` array.
- When creating or cloning, Node-RED assigns the final flow id; treat local ids as provisional.
- Every node in a submitted flow must have a unique `id`.
- If a tool returns a generic `Node-RED API error`, inspect the raw admin endpoint and auth state.

## Critical Safety Rules

**Always**

1. Describe changes before deploying.
2. Read the existing flow with `get-flow` before replacing it.
3. Use descriptive node names.
4. Add error handling where appropriate.
5. Explain what the flow does and how to test it.
6. Use the official Node-RED Admin API via MCP tools.

**Never**

1. Delete or replace flows without explicit user confirmation.
2. Hardcode credentials.
3. Deploy external communications without asking first.
4. Assume global context keys.
5. Create infinite loops or excessive-rate flows.

**Ask confirmation before**

- Send emails, SMS, webhooks, or external API calls
- Write to files or databases
- Control physical devices
- Use complex function node logic
- Modify or replace existing flows

## Context-Efficient Communication

1. Simple flows (<5 nodes): show the complete flow JSON.
2. Medium flows (5-15 nodes): describe changes in detail; show JSON only if asked.
3. Complex flows (>15 nodes): describe changes; never dump full JSON unless requested.

When describing changes:

- List each node with type and name.
- Explain wiring.
- Show function code or important config only when needed.
- Provide a clear summary of what the flow will do.

## Flow Replacement Workflow

### 1. Understand requirements

- What triggers the flow?
- What data is needed?
- What should happen?
- Any conditions or thresholds?

### 2. Read existing context

```javascript
list-tabs
get-flow(<id>)
visualize-flows
```

### 3. Design and explain

- For simple flows (<5 nodes): present complete flow JSON.
- For medium/complex flows (>5 nodes): describe the design instead:
  - list all nodes that will be added/modified (type and name)
  - describe how they will be wired
  - show any function node code or critical configuration
  - explain data flow
  - note assumptions or dependencies

### 4. Make the change

- **Primary workflow:** delete the target tab, then create the replacement with `create-flow`.
- Provide the label and nodes array for `create-flow`.
- Do not depend on update/patch for routine flow changes.
- Always ask confirmation first before delete/recreate.
- Use `visualize-flows` if a diagram helps verify the result.

### 5. Deploy and verify

- Confirm success.
- Explain how to test.
- Point to debug nodes.
- Provide troubleshooting tips.

## Node Standards

**Naming**

- Descriptive and specific: `Parse Kitchen Temperature` not `function 1`.
- Action-based: `Store Sensor Data`, `Check Temperature Threshold`.
- Include location/context when relevant.

**Positioning**

- Use consistent grid spacing (x: multiples of ~150-200, y: multiples of ~60-80).
- Left-to-right flow direction.
- Group related nodes vertically aligned.

**Color**

- Use Node-RED defaults.
- Rely on node types for visual distinction.

## Common Patterns

### MQTT Data Storage

```javascript
const parts = msg.topic.split('/');
const location = parts[1].toLowerCase();
const metric = parts[2].toLowerCase();

let data = msg.payload;
if (typeof data === 'string') {
  data = JSON.parse(data);
}

let sensorData = global.get('sensorData') || {};
if (!sensorData[location]) sensorData[location] = {};
sensorData[location][metric] = data.value;
sensorData[location].lastUpdate = data.timestamp || Date.now();
sensorData[location].quality = data.properties?.Quality?.value;
global.set('sensorData', sensorData);

return msg;
```

### Temperature Alert

Flow:

1. MQTT In: subscribe to `Home/+/Temperature`
2. Switch: check threshold (> 30 hot, < 15 cold)
3. Delay: rate limit to 1 msg / 5 min
4. Function: format alert message
5. Debug or notification node

Recommended:

- Prefer a Switch node with hot/cold/otherwise rules.
- Add RBE or Delay to prevent repeated alerts.
- Put a Debug node on the alert path during testing.

### MCP Endpoint Handler

Purpose: expose the latest in-memory sensor values from `global.sensorData` without dumping full flow JSON.

Flow:

1. HTTP In: `GET /mcp/sensors`
2. Function: validate query params and shape output
3. HTTP Response: return JSON
4. Catch: capture runtime errors
5. Debug: log errors during testing

Read-only function example:

```javascript
const sensorData = global.get('sensorData') || {};
const q = msg.req?.query || {};
const location = (q.location || '').toString().trim().toLowerCase();
const metric = (q.metric || '').toString().trim().toLowerCase();

let result;

if (!location) {
  result = sensorData;
} else if (!sensorData[location]) {
  msg.statusCode = 404;
  result = { error: `Unknown location: ${location}` };
} else if (!metric) {
  result = sensorData[location];
} else if (sensorData[location][metric] === undefined) {
  msg.statusCode = 404;
  result = { error: `Unknown metric for ${location}: ${metric}` };
} else {
  result = {
    location,
    metric,
    value: sensorData[location][metric],
    lastUpdate: sensorData[location].lastUpdate,
    quality: sensorData[location].quality
  };
}

msg.payload = result;
msg.headers = { 'Content-Type': 'application/json' };
return msg;
```

- Keep this endpoint read-only. If adding writes, ask for confirmation and add auth.
- If exposing beyond localhost, add security and confirm first.

## Natural Language Flow Ops

When the user asks to create/replace a flow:

1. Read first:
   - Use `list-tabs` to find the relevant tab(s)
   - Use `get-flow` with the flow ID
   - `GET /flow/:id` returns the tab plus all nodes

2. Plan and explain:
   - Summarize nodes, wiring, and purpose
   - Follow the context-efficient rules
   - For replacements, explain what will be recreated

3. Make the change:
   - **Primary workflow:** delete the target tab, then create the replacement with `create-flow`
   - Provide the label and nodes array for `create-flow`
   - Do not depend on update/patch for routine flow changes
   - Always ask confirmation first before delete/recreate
   - Use `visualize-flows` if a diagram helps verify the result

4. Test:
   - Click Inject in Node-RED UI when available
   - If MQTT-based, provide a test topic and sample payload
   - Add temporary Debug nodes during validation
   - Direct user to http://localhost:1881

5. Confirm risky actions:
   - Ask before external communications, files/databases, physical devices, or delete/recreate changes

## Troubleshooting Guidelines

- Verify MQTT broker host/port match the environment.
- Confirm topic format matches `Home/{Location}/{Metric}`.
- Use a Debug node on MQTT In output to inspect raw payload and topic.
- If parsing fails, handle string and object payloads.
- Check linkquality/Quality codes and filter bad readings if needed.
- Use Delay and/or RBE to avoid floods.
- Never wire outputs back into inputs without deliberate rate limiting.

## Skill Maintenance

If a pattern or tool behavior changes:

- Add a new common pattern, safety rule, or node configuration note.
- Keep it short and consistent with this file.
- Preserve the read -> plan -> replace -> verify flow.

## Best-Practice Reminders

- Prefer small, single-purpose Function nodes.
- Use Debug nodes early, then disable them in production.
- Keep broker config nodes centralized and reused.
- Avoid global context bloat.
- Document thresholds and topic patterns in node names and comments.
