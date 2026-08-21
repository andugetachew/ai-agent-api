# AI Agent API — API Documentation

Base URL (local): `http://localhost:8010`
All endpoints are versioned under `/v1`, except `/health`.

---

## Authentication

None currently implemented. Rate limiting (`slowapi`, Redis-backed) applies per client IP on run-creation endpoints, default `5/minute` (configurable via `RATE_LIMIT`).

---

## Agent Runs

### `POST /v1/agent/run`

Create and enqueue a new agent run. Returns immediately; execution happens asynchronously via Celery.

**Request body:**
```json
{
  "task": "Find recent papers on RAG systems and summarize the key findings",
  "session_id": null,
  "provider": "anthropic",
  "multi_agent": false
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `task` | string | yes | 1–4000 characters |
| `session_id` | string \| null | no | Continue an existing conversation session; omit or `null` to start a new session |
| `provider` | string | no | `"anthropic"` (default), `"openai"`, or `"ollama"` |
| `multi_agent` | boolean | no | Route via the `Supervisor` to a specialist agent instead of the default single agent |

**Response `202 Accepted`:**
```json
{ "run_id": "1fadc000-86eb-4322-a743-aefd94b6225c", "session_id": "d9c952...", "status": "pending" }
```

**Errors:**
- `422` — validation failure (empty/oversized task, missing field)
- `400` — invalid `provider` value
- `429` — rate limit exceeded

---

### `GET /v1/agent/run/{run_id}`

Full detail of a run: status, plan, pending approval action (if any), final report, and every step.

**Response `200`:**
```json
{
  "run_id": "...",
  "session_id": "...",
  "provider": "anthropic",
  "specialist": null,
  "task": "...",
  "status": "done",
  "plan": "- search for recent papers\n- summarize findings",
  "pending_action": null,
  "final_report": {
    "answer": "...",
    "details": ["..."],
    "sources": ["..."],
    "confidence": "high"
  },
  "created_at": "2026-08-09T01:49:09Z",
  "steps": [
    {
      "step_number": 0,
      "phase": "planning",
      "tool_called": null,
      "tool_input": null,
      "tool_output": null
    },
    {
      "step_number": 1,
      "phase": "action",
      "tool_called": "web_search",
      "tool_input": {"query": "..."},
      "tool_output": "..."
    }
  ]
}
```

`status` values: `pending`, `running`, `done`, `failed`, `max_steps_exceeded`, `awaiting_approval`.
`phase` values (per step): `planning`, `action`, `reflection`.

**Errors:** `404` if `run_id` doesn't exist.

---

### `GET /v1/agent/runs`

Paginated, filterable run history.

**Query params:**
| Param | Type | Default | Notes |
|---|---|---|---|
| `status` | string | none | Filter by exact status |
| `session_id` | string | none | Filter by session |
| `limit` | int | 20 | 1–100 |
| `offset` | int | 0 | |

**Response `200`:**
```json
{
  "total": 6,
  "limit": 20,
  "offset": 0,
  "runs": [
    {
      "run_id": "...",
      "session_id": "...",
      "task": "...",
      "status": "done",
      "created_at": "...",
      "total_input_tokens": 1200,
      "total_output_tokens": 340
    }
  ]
}
```

**Errors:** `400` if `status` is not a recognized value.

---

### `GET /v1/agent/run/{run_id}/stream`

Server-Sent Events. Polls Postgres for new steps and streams them live as the Celery worker writes them, closing on terminal status.

**Response:** `text/event-stream`

event: start
data: {"run_id": "...", "status": "pending"}

event: step
data: {"step_number": 1, "phase": "action", "tool_called": "calculator", "tool_output": "4", "duration_ms": 120, "is_error": false}

event: done
data: {"run_id": "...", "status": "done", "final_report": {...}}


If the run doesn't reach a terminal status within 5 minutes: `event: error` with a timeout message.
If `run_id` doesn't exist: single `event: error` with a not-found message (no `start` event).

---

## Human Approval

### `POST /v1/agent/run/{run_id}/approve`

Resume a run currently paused at `status: "awaiting_approval"` (currently triggered only by the `execute_code` tool).

**Request body:**
```json
{ "approved": true, "reason": "looks safe" }
```

| Field | Type | Required |
|---|---|---|
| `approved` | boolean | yes |
| `reason` | string \| null | no, max 500 chars |

**Response `202 Accepted`:**
```json
{ "run_id": "...", "status": "running", "approved": true }
```

**Errors:**
- `404` — run not found
- `400` — run is not currently `awaiting_approval`

---

## Observability

### `GET /v1/agent/run/{run_id}/observability`

Per-run metrics summary (not the raw step trace — see `/trace` for that).

**Response `200`:**
```json
{
  "run_id": "...",
  "status": "done",
  "duration_ms": 4200,
  "step_count": 4,
  "phase_counts": {"planning": 1, "action": 3, "reflection": 0},
  "tool_usage": {"web_search": 1, "calculator": 2},
  "tool_errors": {},
  "total_input_tokens": 1200,
  "total_output_tokens": 340,
  "estimated_cost_usd": 0.0087
}
```

### `GET /v1/observability/summary`

Aggregate metrics across all runs.

**Response `200`:**
```json
{
  "total_runs": 42,
  "status_counts": {"done": 30, "failed": 10, "max_steps_exceeded": 2},
  "success_rate": 0.7143,
  "total_input_tokens": 50000,
  "total_output_tokens": 12000,
  "estimated_total_cost_usd": 0.33,
  "average_duration_ms": 3800.5,
  "tool_usage": {"web_search": 20, "calculator": 15},
  "tool_errors": {"calculator": 1}
}
```

### `GET /v1/agent/run/{run_id}/trace`

Clean, presentation-friendly narrative: Task → Planner → Tool → Tool → Final Answer.

**Response `200`:**
```json
{
  "run_id": "...",
  "task": "...",
  "status": "done",
  "plan": "...",
  "trace": [
    {"stage": "Planner", "detail": "- search for X\n- summarize"},
    {"stage": "Tool: web_search", "input": {"query": "..."}, "output": "...", "is_error": false, "duration_ms": 340}
  ],
  "final_answer": "...",
  "rating": 5
}
```

---

## Evaluation

### `POST /v1/agent/run/{run_id}/rate`

Submit a rating for a finished run. Stored in a dedicated `run_evaluations` table (one run can have multiple ratings over time).

**Request body:**
```json
{ "rating": 5, "feedback": "Optional text feedback" }
```

| Field | Type | Required |
|---|---|---|
| `rating` | int | yes, 1–5 |
| `feedback` | string \| null | no, max 1000 chars |

**Errors:**
- `404` — run not found
- `400` — run has not finished yet (not in `done`/`failed`/`max_steps_exceeded`)

### `GET /v1/evaluation/summary`

Aggregate rating statistics across all evaluated runs.

---

## File Upload

### `POST /v1/files/upload`

Multipart upload. Supported types: `application/pdf`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document` (`.docx`), `text/plain`. Max size 10 MB. Extracted text capped at 50,000 characters.

**Response `201`:**
```json
{
  "file_id": "...",
  "filename": "report.pdf",
  "content_type": "application/pdf",
  "size_bytes": 48213,
  "extracted_char_count": 3200,
  "preview": "First 300 characters of extracted text..."
}
```

**Errors:**
- `413` — file exceeds 10 MB
- `415` — unsupported content type
- `422` — extraction failed (e.g. corrupt/password-protected PDF)

### `GET /v1/files/{file_id}`

Retrieve full extracted text of a previously uploaded file. Used by the agent's `read_file` tool mid-run.

**Errors:** `404` if not found.

---

## Sessions

### `GET /v1/sessions/{session_id}`

**Response `200`:**
```json
{
  "session_id": "...",
  "summary": "- Asked: X | Answered: Y\n- Asked: ... ",
  "created_at": "...",
  "updated_at": "...",
  "run_count": 7
}
```

`summary` is `null` until a session exceeds 5 completed turns (older turns then fold into this rolling summary automatically; the 5 most recent stay verbatim in the agent's context).

**Errors:** `404` if not found.

---

## Health

### `GET /health`

Checks live connectivity to Postgres and Redis (not a static "OK").

**Response `200` (healthy):**
```json
{ "status": "ok", "database": "connected", "redis": "connected" }
```

**Response `503` (degraded):**
```json
{ "status": "degraded", "database": "unreachable", "redis": "connected" }
```

---

## Tools available to the agent

| Tool | Description | Approval required |
|---|---|---|
| `web_search` | Search the web via Serper, returns summarized results | no |
| `fetch_url` | Fetch and return text content of a URL (truncated to 5000 chars) | no |
| `calculator` | AST-restricted arithmetic evaluation (no `eval` on raw input) | no |
| `execute_code` | Sandboxed Python execution, restricted builtins, enforced timeout | **yes** |
| `read_file` | Read extracted text of a previously uploaded file | no |
| `submit_result` | Terminal tool — submits the structured final answer | no |

## Multi-agent specialists

| Specialist | Tools | Use case |
|---|---|---|
| `research` | `web_search`, `fetch_url`, `read_file` | Finding information |
| `coding` | `execute_code`, `read_file` | Writing/running code |
| `data` | `calculator`, `execute_code`, `read_file` | Numeric analysis |
| `rag` | `read_file`, `web_search` | Answering from uploaded documents |

Routing is performed by a `Supervisor` via a single tool-free classification call to the selected provider; an unrecognized/ambiguous response falls back to `research`.