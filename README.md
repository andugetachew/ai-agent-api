markdown
# AI Agent API

## AI Agent API — Multi-Provider Autonomous Agent Platform

![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)
![Tests](https://img.shields.io/badge/tests-128%20passed-brightgreen)
![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-enabled-teal)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-blue)
![Redis](https://img.shields.io/badge/Redis-7-red)
![Celery](https://img.shields.io/badge/Celery-enabled-green)
![Docker](https://img.shields.io/badge/Docker-enabled-blue)
![License](https://img.shields.io/badge/license-MIT-yellow)

- 128 automated tests
- 90% code coverage
- Unit + integration test layers, including real generated PDF/DOCX file round-trips
- 0 failed tests


A general-purpose autonomous agent platform that completes multi-step tasks using tools — web search, URL fetching, calculation, sandboxed code execution, and document reading. Built on the ReAct pattern (Reason → Act → Observe) with an explicit planning phase and periodic self-reflection, async execution via Celery, persistent conversation memory, human-in-the-loop approval for risky actions, real-time streaming, multi-provider model support, and a supervisor-coordinated multi-agent mode.

Demonstrated via a research use case — give it a task like *"Find recent developments in RAG systems and summarize the key findings"* — but the same agent loop, tool registry, and infrastructure handle any task the registered tools support.

## Architecture

Client → FastAPI (rate-limited, versioned, streams progress) → Redis queue → Celery worker → Model Provider (Anthropic / OpenAI / Ollama)
↓
PostgreSQL (runs, steps, sessions, evaluations)


The API never blocks on the agent loop. `POST /v1/agent/run` writes a `pending` row to Postgres, enqueues a Celery job, and returns a `run_id` instantly. The client polls `GET /v1/agent/run/{run_id}` or opens `GET /v1/agent/run/{run_id}/stream` (Server-Sent Events) to watch it progress live as the worker executes the loop in the background.

## Core capabilities

**Agent loop**
- ReAct reasoning with tool use, backed by a pluggable `ToolRegistry`
- Upfront **planning** step before execution begins, injected into the system prompt for the rest of the run
- Periodic **reflection** nudges (every 3rd step) that reuse the existing turn rather than spending a separate API call — cost-neutral
- Hard `max_steps`/`max_seconds` limits with **forced synthesis**: when a run would otherwise fail open-ended, the agent is required to submit its best answer from what it has, rather than erroring out
- **Retry policy**: transient provider failures (rate limits, timeouts, connection blips) get one automatic retry; non-transient errors (bad key, invalid request) fail immediately rather than retrying into the same wall

**Memory**
- Persistent conversation sessions (`session_id`) spanning multiple runs
- Automatic rolling summarization once a session exceeds 5 completed turns, so context sent to the model stays bounded no matter how long a conversation runs

**Human Approval**
- Specific tools (currently `execute_code`) pause the run and wait for explicit approval before executing
- Full run state (message history, plan, token counts, pending tool calls) is serialized to Postgres at the pause point and reconstructed on `POST /v1/agent/run/{run_id}/approve` — the agent genuinely resumes mid-conversation, not from scratch

**Multi-Agent Support**
- A `Supervisor` routes each task to one of four specialists (Research, Coding, Data Analysis, RAG), each scoped to a relevant subset of tools
- Routing itself is a lightweight, tool-free classification call through the same provider abstraction as the main loop
- Enable via `"multi_agent": true` on `POST /v1/agent/run`

**Multiple Model Providers**
- Anthropic, OpenAI, and Ollama (local) are interchangeable via a normalized provider interface — every provider's response is translated into the same shape the agent loop expects, so switching providers requires zero changes to agent logic
- Select per-run via `"provider": "anthropic" | "openai" | "ollama"`

**Observability & Evaluation**
- Per-run and aggregate metrics: duration, token counts, estimated cost, tool usage breakdown, error rates, phase counts (planning/action/reflection)
- Clean presentation-friendly **traces**: Task → Planner → Tool → Tool → Final Answer
- User **ratings** (1-5 + feedback) stored per run, aggregated into success-rate and satisfaction summaries

**File Upload**
- PDF, DOCX, and TXT upload with text extraction, verified against real generated files of each format
- Extracted content is retrievable by the agent mid-run via the `read_file` tool

**Streaming**
- Server-Sent Events endpoint polls Postgres for new steps as they're written by the worker and streams them live, closing cleanly on terminal status

## Why these design choices

- **A genuinely sandboxed code executor.** `execute_code` runs in a restricted namespace (no `import`, no file/network access, whitelisted builtins only) with a real enforced timeout on a worker thread. Output is captured via a custom `print()` injected into the sandbox rather than `contextlib.redirect_stdout` — `redirect_stdout` swaps `sys.stdout` process-globally, which is unsafe here: if a snippet times out mid-execution, the thread never exits its `with` block and stdout stays hijacked for the rest of the process. Caught and fixed via testing, with a regression test locking it in.
- **Structured output via tool call, not text parsing.** The agent's final answer is submitted through a `submit_result` tool call with a strict JSON schema, far more reliable than regex-parsing free-form text.
- **Resume state is always JSON-safe.** Message history is serialized to plain dicts throughout the loop (not just at the pause point), so a pause can happen at any turn without breaking Postgres storage — verified by round-tripping actual paused state through real JSON serialization in tests.
- **Alembic-managed schema**, migrated deliberately against both the production (Neon) and local test databases on every change — no manual `ALTER TABLE` drift.

## Tech stack

- **FastAPI** — async, versioned, rate-limited HTTP API
- **Anthropic / OpenAI / Ollama** — interchangeable model providers, tool use / function calling
- **Celery + Redis** — background job queue, also backs rate limiting
- **PostgreSQL + SQLAlchemy + Alembic** — persistent storage with managed migrations
- **Docker Compose** — multi-service orchestration (api, worker, redis)
- **pypdf / python-docx** — file text extraction
- **pytest** — unit and integration test suite (100+ tests)
- **GitHub Actions** — CI running the full suite with live Postgres + Redis service containers on every push

## Tools available to the agent

| Tool | Purpose |
|---|---|
| `web_search` | Search the web for a query, returns summarized results |
| `fetch_url` | Fetch and return the text content of a given URL |
| `calculator` | Safely evaluate arithmetic expressions (AST-restricted, no `eval` on raw input) |
| `execute_code` | Run a Python snippet in a sandboxed namespace with an enforced timeout — **requires human approval** |
| `read_file` | Read the extracted text of a previously uploaded file |
| `submit_result` | Terminal tool — submits the structured final answer |

## API

All endpoints are versioned under `/v1`, except `/health`.

### `POST /v1/agent/run`
Enqueue a new agent task.

**Request:**
```json
{
  "task": "Find recent papers on RAG systems and summarize the key findings",
  "session_id": null,
  "provider": "anthropic",
  "multi_agent": false
}
```

**Response (202 Accepted):**
```json
{ "run_id": "...", "session_id": "...", "status": "pending" }
```

### `GET /v1/agent/run/{run_id}`
Full run detail: status, plan, pending approval action (if any), final report, and every step.

### `GET /v1/agent/run/{run_id}/stream`
Server-Sent Events — live `step` events as they happen, then a final `done` event.

### `POST /v1/agent/run/{run_id}/approve`
Resume a run paused for human approval. Body: `{ "approved": true, "reason": "optional" }`.

### `GET /v1/agent/runs`
Paginated run history, filterable by `status` and `session_id`.

### `GET /v1/agent/run/{run_id}/observability`
Per-run metrics: duration, cost, tool usage, error breakdown, phase counts.

### `GET /v1/observability/summary`
Aggregate metrics across all runs: success rate, total cost, tool usage/error rates, average duration.

### `GET /v1/agent/run/{run_id}/trace`
Clean narrative trace: Task → Planner → Tool calls → Final Answer.

### `POST /v1/agent/run/{run_id}/rate`
Submit a 1-5 rating + optional feedback for a completed run.

### `GET /v1/evaluation/summary`
Aggregate rating statistics.

### `POST /v1/files/upload`
Multipart upload (PDF/DOCX/TXT), returns `file_id` and extracted text preview.

### `GET /v1/files/{file_id}`
Retrieve full extracted text of an uploaded file.

### `GET /v1/sessions/{session_id}`
Session summary and rolling memory state.

### `GET /health`
Checks live connectivity to Postgres and Redis; returns `503` if either is down.

## Running locally

```bash
# 1. Configure environment
cp .env.example .env
# fill in ANTHROPIC_API_KEY, DATABASE_URL, CELERY_BROKER_URL
# optional: SERPER_API_KEY, OPENAI_API_KEY, OLLAMA_BASE_URL

# 2. Build and start all services
docker-compose build --no-cache
docker-compose up -d

# 3. Verify
curl http://localhost:8010/health
```

`SERPER_API_KEY` is optional — without it, `web_search` returns a graceful `ERROR:` string the agent can see and route around, rather than crashing the app. `OPENAI_API_KEY`/`OLLAMA_BASE_URL` are only needed if a run actually requests that provider.

## Database migrations

Schema changes are managed with Alembic, applied to both the primary and test databases on every change:

```bash
alembic revision --autogenerate -m "description of the change"
alembic upgrade head              # migrates DATABASE_URL (production)
alembic -x db=test upgrade head   # migrates TEST_DATABASE_URL (local test db)
```

## Running tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

Most test files run fully offline (mocked model provider, stubbed search, in-memory SQLite). `test_db.py` requires a real Postgres connection (`TEST_DATABASE_URL` in `.env`) since it exercises real ORM persistence. All tests run automatically in CI on every push.

## Project structure

ai-agent-api/
├── app/
│ ├── main.py # FastAPI app, all routes
│ ├── agent.py # ReAct loop, planning, reflection, retry, human approval
│ ├── multi_agent.py # Supervisor + specialist routing
│ ├── providers.py # Anthropic/OpenAI/Ollama provider abstraction
│ ├── tools.py # Tool registry + implementations
│ ├── memory.py # Session memory + rolling summarization
│ ├── streaming.py # SSE generator
│ ├── observability.py # Metrics, traces, evaluation aggregation
│ ├── file_processing.py # PDF/DOCX/TXT text extraction
│ ├── models.py # In-memory result dataclasses
│ ├── db.py # SQLAlchemy models + session
│ ├── tasks.py # Celery tasks (run, resume, multi-agent)
│ ├── celery_app.py # Celery configuration
│ ├── logging_config.py # Structured logging setup
│ └── config.py # Typed settings (pydantic-settings)
├── alembic/ # Schema migrations
├── tests/ # 100+ tests across all modules
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
└── requirements.txt

---

## 📄 License

MIT License

---

### 👨‍💻 Author

**Andualem Getachew**

[![GitHub](https://img.shields.io/badge/GitHub-andugetachew-black?logo=github)](https://github.com/andugetachew)
[![Email](https://img.shields.io/badge/Email-andugeta41%40gmail.com-red?logo=gmail)](mailto:andugeta41@gmail.com)