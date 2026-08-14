from __future__ import annotations
import json
import time
from typing import Generator

from db import SessionLocal, AgentRunORM

TERMINAL_STATUSES = {"done", "failed", "max_steps_exceeded"}

# Tunables kept as module-level constants (not buried in the function) so
# tests can override them for fast, deterministic polling instead of
# waiting on real wall-clock time.
POLL_INTERVAL_SECONDS = 1.0
MAX_WAIT_SECONDS = 300.0  # 5 minutes -- bounds how long a connection stays open


def _format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_run_events(run_id: str) -> Generator[str, None, None]:
    """
    Polls Postgres for a run's steps and yields Server-Sent Events as new
    steps appear, closing the stream once the run reaches a terminal status.
    Uses its own DB session (not FastAPI's request-scoped one) since this
    generator outlives a single request/response cycle.
    """
    db = SessionLocal()
    seen_step_count = 0
    waited = 0.0

    try:
        run_row = db.query(AgentRunORM).filter(AgentRunORM.id == run_id).first()
        if run_row is None:
            yield _format_sse("error", {"message": f"run '{run_id}' not found"})
            return

        yield _format_sse("start", {"run_id": run_id, "status": run_row.status})

        while waited < MAX_WAIT_SECONDS:
            db.expire_all()  # force fresh reads instead of stale cached objects
            run_row = db.query(AgentRunORM).filter(AgentRunORM.id == run_id).first()
            if run_row is None:
                yield _format_sse("error", {"message": "run disappeared"})
                return

            new_steps = run_row.steps[seen_step_count:]
            for step in new_steps:
                yield _format_sse("step", {
                    "step_number": step.step_number,
                    "phase": step.phase,
                    "tool_called": step.tool_called,
                    "tool_output": step.tool_output,
                    "duration_ms": step.duration_ms,
                    "is_error": step.is_error,
                })
            seen_step_count = len(run_row.steps)

            if run_row.status in TERMINAL_STATUSES:
                yield _format_sse("done", {
                    "run_id": run_id,
                    "status": run_row.status,
                    "final_report": run_row.final_report,
                })
                return

            time.sleep(POLL_INTERVAL_SECONDS)
            waited += POLL_INTERVAL_SECONDS

        yield _format_sse("error", {"message": "stream timed out waiting for run to complete"})
    finally:
        db.close()