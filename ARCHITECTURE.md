# AI Agent API — Architecture

## Overview

An autonomous agent platform built on the ReAct pattern (Reason → Act → Observe), extended with explicit planning, periodic self-reflection, human-in-the-loop approval, persistent conversation memory, multi-provider model support, and supervisor-coordinated specialist routing. The API layer never blocks on agent execution — every run is dispatched to a Celery worker and tracked through Postgres.

## System diagram

Client
│
▼
FastAPI (main.py) -- rate-limited, versioned routes, SSE streaming
│ writes "pending" row
▼
PostgreSQL (db.py) -- agent_runs, agent_steps, agent_sessions,
│ uploaded_files, run_evaluations
│ enqueues job
▼
Redis -- Celery broker + rate-limit storage
│
▼
Celery worker (tasks.py) -- run_agent_task / run_multi_agent_task / resume_agent_task
│ builds Agent + ModelProvider
▼
Agent (agent.py) -- ReAct loop, planning, reflection, retry, approval pause
│ calls provider.create()
▼
ModelProvider (providers.py) -- Anthropic (passthrough) / OpenAI / Ollama (normalized)
│
▼
Anthropic / OpenAI / Ollama API


## Core modules

### `agent.py` — the ReAct loop

`Agent.run(task, history=None)`:
1. **Planning**: one dedicated, tool-free provider call outlining a short plan. Not counted against `max_steps`. Injected into the system prompt for the rest of the run.
2. **Action loop** (`_action_loop`): iterates up to `max_steps`, each turn:
   - Injects a reflection nudge on every 3rd iteration (`REFLECTION_INTERVAL = 3`), reusing the existing turn rather than spending a separate API call — cost-neutral.
   - Calls the provider via `_call_provider()`, which wraps `provider.create()` with a single retry on transient errors (rate limit, timeout, connection, `429`/`503`) via keyword-matching on the exception message (`_is_transient_error`). Non-transient errors propagate immediately.
   - If the model calls a tool in `APPROVAL_REQUIRED_TOOLS` (currently `{"execute_code"}`), the run **pauses**: current state (message history, plan, running token counts, pending tool blocks) is serialized to plain JSON-safe dicts via `_serialize_block()`/`_serialize_step()` and returned as `resume_state`. No further processing happens until `Agent.resume()` is called.
   - Otherwise executes each tool call via the `ToolRegistry`, appends results, continues.
   - `submit_result` tool call → terminal `DONE` status with the structured final report.
3. **Forced synthesis** (`_force_synthesis`): if `max_steps` or `max_seconds` is exceeded, one final call restricted to only the `submit_result` tool forces the model to submit its best answer from what it has, rather than the run erroring out silently.

`Agent.resume(run_id, task, resume_state, approved, denial_reason)`: reconstructs loop state from `resume_state`, executes the previously-pending tool call if `approved` (or injects a `"REJECTED: ..."` tool result if not), then continues via `_action_loop` from where it left off. Total run duration correctly accumulates pre-pause and post-resume elapsed time (`elapsed_before_ms`).

### `tools.py` — the tool registry

`ToolRegistry` is a plain dict-backed registry (`register()` / `get()` / `run()` / `anthropic_tools()`), independent of any specific provider's schema format — provider-specific translation happens in `providers.py`.

Every tool handler catches its own exceptions and returns an `"ERROR: ..."` string rather than raising — the agent sees the failure as a normal tool result and can adapt, instead of the run crashing.

**`execute_code`** is the one tool with real safety surface:
- Runs via `exec()` against a restricted `__builtins__` dict (no `import`, no `open`, whitelisted names only)
- Output is captured via a custom `_sandboxed_print()` injected into the restricted namespace — **not** `contextlib.redirect_stdout`, which was found (via testing) to swap `sys.stdout` process-globally; a timed-out thread never exiting its `with` block would permanently hijack stdout for the rest of the process. Verified fixed with a dedicated regression test.
- Timeout enforced via `threading.Thread(...).join(timeout=...)`; a thread that doesn't finish in time returns an `ERROR: ... timeout` result (the thread itself cannot be forcibly killed — documented as an accepted limitation for a portfolio-scale deployment, not production-hardened).

**`calculator`** evaluates expressions via `ast.parse()` + a whitelist of allowed node types (`_ALLOWED_AST_NODES`), rejecting anything containing function calls, attribute access, or imports — no raw `eval()` on untrusted input.

### `providers.py` — the model-provider abstraction

Every provider's `.create(system, messages, tools, max_tokens)` returns a `SimpleNamespace` shaped exactly like the Anthropic SDK's response (`.content` list of blocks with `.type`/`.text` or `.name`/`.input`/`.id`, `.usage.input_tokens`/`.output_tokens`). This is the load-bearing design decision that lets `Agent` stay provider-agnostic: it was written once against Anthropic's shape and never needed to change when OpenAI/Ollama support was added.

- **`AnthropicProvider`**: pure passthrough to `client.messages.create()` — zero translation needed.
- **`OpenAIProvider`**: translates Anthropic-shaped tool schemas (`input_schema`) to OpenAI's function-calling format (`parameters`), and Anthropic-shaped message history (with `tool_use`/`tool_result` content blocks) to OpenAI's `chat.completions` message format (`tool_calls`, `role: "tool"`). Response tool calls are parsed back (`json.loads(tc.function.arguments)`) into the normalized block shape.
- **`OllamaProvider`**: subclasses `OpenAIProvider`, pointed at Ollama's local OpenAI-compatible endpoint (`http://localhost:11434/v1` by default) with a dummy API key (Ollama doesn't check it, but the OpenAI client requires a non-empty value).
- **`build_provider(provider_name, ...)`**: factory used by `tasks.py`; unrecognized names default to Anthropic rather than failing, so a typo in a request can't hard-crash a run.

### `multi_agent.py` — supervisor + specialists

`SPECIALISTS` is a plain dict mapping specialist name → `{description, tools}`. `build_specialist_registry(name)` builds a `ToolRegistry` scoped to that specialist's tools plus `submit_result` (every specialist needs a way to finish), pulled from the same `build_default_registry()` used everywhere else — no duplicated tool implementations.

`Supervisor.route(task)` makes one tool-free classification call through the same `ModelProvider` interface as the main agent loop (so routing works with any configured provider), asking the model to respond with only a specialist name. The response is matched case-insensitively against known specialist names; anything unrecognized (empty response, hedging text, garbage) falls back to `DEFAULT_SPECIALIST = "research"`.

`Supervisor.frame_task(specialist_name, task)` prepends a short role-framing string to the task before handing it to a normal `Agent` — this was a deliberate simplification: rather than modifying `Agent`'s core system prompt per specialist (a larger refactor), specialist context is injected via the task text itself, and only the tool registry is actually scoped differently.

### `memory.py` — session persistence and summarization

Sessions (`AgentSessionORM`) persist across multiple runs via `session_id`. `build_context_messages(db, session)` constructs the prior-conversation context for a new run: the rolling `summary` field (if present) as a synthetic user/assistant exchange, followed by up to `RECENT_TURNS_KEPT = 5` most recent completed turns verbatim.

`update_session_memory(db, session_id, task, final_report)` is called after each successful run in a session. If the session now has more than 5 completed turns, the oldest excess turns are folded into `summary` via `_naive_summarize()` — currently simple truncation/concatenation, explicitly documented as a placeholder swappable for a real LLM-based summarization call later. This keeps context sent to the model bounded regardless of how long a session runs (verified by a test that runs 15 turns and asserts the context size never exceeds `2 + RECENT_TURNS_KEPT*2` messages).

### `observability.py` — metrics, traces, evaluation

Three distinct read-only views over the same underlying data, kept separate because they serve different consumers:
- `build_run_observability(run)` — per-run metrics dict (duration, cost, tool usage/error counts, phase counts) for dashboards.
- `build_aggregate_observability(db)` — cross-run aggregates (success rate, total cost, tool usage) via SQL `GROUP BY`/`func.avg`/`func.sum`, not Python-side aggregation.
- `build_run_trace(db, run)` — a narrative, presentation-ready sequence (Task → Planner → Tool → ... → Final Answer) plus the latest rating from `RunEvaluationORM`, intended for display/export rather than metrics.

`estimate_cost_usd()` uses named pricing constants (`INPUT_COST_PER_MILLION`, `OUTPUT_COST_PER_MILLION`) kept in one place specifically so a pricing change is a one-line edit.

### `streaming.py` — Server-Sent Events

`stream_run_events(run_id)` is a plain generator, not a websocket — polls Postgres every `POLL_INTERVAL_SECONDS` (default 1.0s) for new steps since the last check, using `db.expire_all()` to force fresh reads rather than cached ORM state. Bounded by `MAX_WAIT_SECONDS` (default 300s) so a stuck run can't hold a connection open indefinitely. Uses its own DB session (`SessionLocal()`), not FastAPI's request-scoped one, since the generator outlives a single request/response cycle.

### `file_processing.py` — upload extraction

`extract_text(filename, content_type, raw_bytes)` dispatches to format-specific extractors (`pypdf` for PDF, `python-docx` for DOCX, plain UTF-8 decode for TXT), all wrapped in a single `try/except` that raises a normalized `ExtractionError` regardless of which underlying library failed. `MAX_EXTRACTED_CHARS = 50_000` caps stored/returned text so one large upload can't blow out a DB row or flood the agent's context if read via `read_file`. Verified against real generated files (not just trusted library behavior) — a real PDF built with `reportlab`, a real DOCX built with `python-docx`.

### `db.py` / Alembic — schema and migrations

Five tables: `agent_sessions`, `agent_runs`, `agent_steps`, `uploaded_files`, `run_evaluations`. `agent_runs.session_id`, `agent_steps.run_id`, and `run_evaluations.run_id` are foreign keys; `agent_steps` cascade-deletes with its parent run.

Schema changes are managed with Alembic against **two independently-migrated databases** — the production database (Neon) and a local test database — via a custom `-x db=test` flag in `alembic/env.py` that switches the target connection string without ever editing `.env`. `env.py` explicitly calls `load_dotenv()` before reading `os.environ`, since `pydantic-settings` (used elsewhere in the app) only populates its own `Settings` object, not the real process environment.

### `main.py` — API layer

FastAPI app with `slowapi` rate limiting (Redis-backed, same broker as Celery), structured logging (`logging_config.py`), and a `/health` endpoint that actively pings both Postgres and Redis rather than returning a static `200`.

`run_task` (the `POST /v1/agent/run` handler) branches on `body.multi_agent` to dispatch either `run_agent_task` or `run_multi_agent_task` — the request/response contract is identical either way; only the Celery task and internal routing differ.

## Testing strategy

- Nearly all tests run fully offline: a mocked `ModelProvider`/Anthropic client, stubbed search, in-memory SQLite (via `StaticPool` to keep a single shared in-memory DB across connections within one test) for anything touching the ORM.
- `test_db.py` is the deliberate exception — it requires a real Postgres connection (`TEST_DATABASE_URL`) because it's specifically testing real ORM persistence, cascade deletes, and query behavior that SQLite can mask differences in.
- File-processing tests generate **real** PDF/DOCX files at test time (via `reportlab`/`python-docx`) rather than trusting library behavior on synthetic byte strings.
- Human Approval's `resume_state` is tested by actually round-tripping it through `json.dumps()`/`json.loads()`, not just asserting it "should" be serializable.
- 128 tests, 90% coverage as of the last measured run.

## Known limitations / explicitly deferred

- `execute_code`'s sandbox is in-process (restricted builtins + thread timeout), not a subprocess/container — documented in the code as acceptable for a portfolio-scale deployment, not a production security boundary.
- Memory summarization (`_naive_summarize`) is a truncation heuristic, not a real LLM summarization call — isolated in one function specifically so upgrading it later is a one-function change.
- Running the Celery worker inside the same container as the API (for free-tier deployment) trades independent scaling/restart isolation for zero hosting cost — noted as a deliberate deployment-context trade-off, not the intended production topology.