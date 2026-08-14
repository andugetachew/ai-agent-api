import uuid

from fastapi import FastAPI, Depends, Request, HTTPException, Query, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text, func
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis

from config import settings
from db import init_db, get_db, AgentRunORM, AgentSessionORM, UploadedFileORM, engine
from file_processing import extract_text, UnsupportedFileTypeError, ExtractionError
from memory import get_or_create_session
from observability import build_run_observability, build_aggregate_observability
from streaming import stream_run_events
from logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

limiter = Limiter(key_func=get_remote_address, storage_uri=settings.celery_broker_url)

app = FastAPI(title="AI Agent API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    logger.info("Application startup complete")


VALID_PROVIDERS = {"anthropic", "openai", "ollama"}


class RunTaskRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, description="Continue an existing conversation session")
    provider: str = Field(default="anthropic", description="Model backend: anthropic, openai, or ollama")
    multi_agent: bool = Field(default=False, description="Route via the supervisor to a specialist agent")


class ApproveRequest(BaseModel):
    approved: bool
    reason: str | None = Field(default=None, max_length=500)


class RunTaskAcceptedResponse(BaseModel):
    run_id: str
    session_id: str
    status: str


@app.post("/v1/agent/run", response_model=RunTaskAcceptedResponse, status_code=202)
@limiter.limit(settings.rate_limit)
def run_task(request: Request, body: RunTaskRequest, db: Session = Depends(get_db)) -> RunTaskAcceptedResponse:
    from tasks import run_agent_task  # imported here to avoid circular import at module load

    if body.provider not in VALID_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"invalid provider '{body.provider}', must be one of {sorted(VALID_PROVIDERS)}",
        )

    session = get_or_create_session(db, body.session_id)

    run_id = str(uuid.uuid4())
    logger.info(f"run_id={run_id} session_id={session.id} provider={body.provider} accepted task (len={len(body.task)})")

    run_row = AgentRunORM(
        id=run_id,
        session_id=session.id,
        provider=body.provider,
        task=body.task,
        status="pending",
    )
    db.add(run_row)
    db.commit()

    if body.multi_agent:
            from tasks import run_multi_agent_task
            run_multi_agent_task.delay(run_id, body.task)
    else:
        run_agent_task.delay(run_id, body.task)

    return RunTaskAcceptedResponse(run_id=run_id, session_id=session.id, status="pending")


@app.get("/v1/agent/run/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db)) -> dict:
    run_row = db.query(AgentRunORM).filter(AgentRunORM.id == run_id).first()
    if not run_row:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    return {
        "run_id": run_row.id,
        "session_id": run_row.session_id,
        "provider": run_row.provider,
        "task": run_row.task,
        "status": run_row.status,
        "plan": run_row.plan,
        "pending_action": run_row.pending_action,
        "final_report": run_row.final_report,
        "created_at": run_row.created_at.isoformat(),
        "steps": [
            {
                "step_number": s.step_number,
                "phase": s.phase,
                "tool_called": s.tool_called,
                "tool_input": s.tool_input,
                "tool_output": s.tool_output,
            }
            for s in run_row.steps
        ],
    }


VALID_STATUSES = {"pending", "running", "done", "failed", "max_steps_exceeded", "awaiting_approval"}


@app.get("/v1/agent/runs")
def list_runs(
    status: str | None = Query(default=None, description="Filter by run status"),
    session_id: str | None = Query(default=None, description="Filter by session"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> dict:
    if status is not None and status not in VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"invalid status '{status}', must be one of {sorted(VALID_STATUSES)}",
        )

    query = db.query(AgentRunORM)
    if status is not None:
        query = query.filter(AgentRunORM.status == status)
    if session_id is not None:
        query = query.filter(AgentRunORM.session_id == session_id)

    total = query.with_entities(func.count(AgentRunORM.id)).scalar()

    rows = (
        query.order_by(AgentRunORM.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "runs": [
            {
                "run_id": r.id,
                "session_id": r.session_id,
                "task": r.task,
                "status": r.status,
                "created_at": r.created_at.isoformat(),
                "total_input_tokens": r.total_input_tokens,
                "total_output_tokens": r.total_output_tokens,
            }
            for r in rows
        ],
    }


@app.get("/v1/sessions/{session_id}")
def get_session(session_id: str, db: Session = Depends(get_db)) -> dict:
    session = db.query(AgentSessionORM).filter(AgentSessionORM.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail=f"session '{session_id}' not found")
    return {
        "session_id": session.id,
        "summary": session.summary,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "run_count": len(session.runs),
    }


@app.post("/v1/agent/run/{run_id}/approve", status_code=202)
def approve_run(run_id: str, body: ApproveRequest, db: Session = Depends(get_db)) -> dict:
    from tasks import resume_agent_task

    run_row = db.query(AgentRunORM).filter(AgentRunORM.id == run_id).first()
    if not run_row:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    if run_row.status != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"run '{run_id}' is not awaiting approval (current status: {run_row.status})",
        )

    run_row.status = "running"
    db.commit()

    resume_agent_task.delay(run_id, body.approved, body.reason)

    return {"run_id": run_id, "status": "running", "approved": body.approved}


@app.get("/v1/agent/run/{run_id}/observability")
def get_run_observability(run_id: str, db: Session = Depends(get_db)) -> dict:
    run_row = db.query(AgentRunORM).filter(AgentRunORM.id == run_id).first()
    if not run_row:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    return build_run_observability(run_row)


@app.get("/v1/observability/summary")
def get_observability_summary(db: Session = Depends(get_db)) -> dict:
    return build_aggregate_observability(db)


@app.get("/v1/agent/run/{run_id}/stream")
def stream_run(run_id: str, db: Session = Depends(get_db)) -> StreamingResponse:
    run_row = db.query(AgentRunORM).filter(AgentRunORM.id == run_id).first()
    if not run_row:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    return StreamingResponse(
        stream_run_events(run_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@app.post("/v1/files/upload", status_code=201)
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict:
    raw_bytes = await file.read()

    if len(raw_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"file exceeds max size of {MAX_UPLOAD_SIZE_BYTES} bytes")

    content_type = file.content_type or "application/octet-stream"

    try:
        extracted_text = extract_text(file.filename, content_type, raw_bytes)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    except ExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    file_row = UploadedFileORM(
        filename=file.filename,
        content_type=content_type,
        extracted_text=extracted_text,
        size_bytes=len(raw_bytes),
    )
    db.add(file_row)
    db.commit()
    db.refresh(file_row)

    return {
        "file_id": file_row.id,
        "filename": file_row.filename,
        "content_type": file_row.content_type,
        "size_bytes": file_row.size_bytes,
        "extracted_char_count": len(extracted_text) if extracted_text else 0,
        "preview": (extracted_text or "")[:300],
    }


@app.get("/v1/files/{file_id}")
def get_file(file_id: str, db: Session = Depends(get_db)) -> dict:
    file_row = db.query(UploadedFileORM).filter(UploadedFileORM.id == file_id).first()
    if not file_row:
        raise HTTPException(status_code=404, detail=f"file '{file_id}' not found")
    return {
        "file_id": file_row.id,
        "filename": file_row.filename,
        "content_type": file_row.content_type,
        "size_bytes": file_row.size_bytes,
        "extracted_text": file_row.extracted_text,
        "created_at": file_row.created_at.isoformat(),
    }


@app.get("/health")
def health() -> dict:
    checks = {"database": "unreachable", "redis": "unreachable"}

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "connected"
    except Exception as exc:
        logger.warning(f"health check: database unreachable: {exc}")

    try:
        r = redis.from_url(settings.celery_broker_url, socket_connect_timeout=2)
        r.ping()
        checks["redis"] = "connected"
    except Exception as exc:
        logger.warning(f"health check: redis unreachable: {exc}")

    all_ok = all(v == "connected" for v in checks.values())
    status_code = 200 if all_ok else 503
    return JSONResponse(status_code=status_code, content={"status": "ok" if all_ok else "degraded", **checks})