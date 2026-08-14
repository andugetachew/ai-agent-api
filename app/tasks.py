from anthropic import Anthropic

from celery_app import celery_app
from config import settings
from agent import Agent
from tools import build_default_registry
from db import SessionLocal, AgentRunORM, AgentStepORM, AgentSessionORM
from memory import build_context_messages, update_session_memory
from providers import build_provider
from multi_agent import Supervisor, build_specialist_registry

_anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
_registry = build_default_registry()

# Which model name each provider uses by default -- kept in one place so
# adding a fourth provider later means touching one dict, not scattered ifs.
_PROVIDER_MODELS = {
    "anthropic": settings.agent_model,
    "openai": settings.openai_model,
    "ollama": settings.ollama_model,
}


def _build_agent(provider_name: str) -> Agent:
    model = _PROVIDER_MODELS.get(provider_name, settings.agent_model)
    provider = build_provider(
        provider_name,
        model=model,
        anthropic_client=_anthropic_client,
        openai_api_key=settings.openai_api_key,
        ollama_base_url=settings.ollama_base_url,
    )
    return Agent(
        provider=provider,
        registry=_registry,
        model=model,
        max_steps=settings.agent_max_steps,
        max_seconds=settings.agent_max_seconds,
    )


def _persist_result(db, run_row, result) -> None:
    """Shared persistence logic for a completed (or paused) AgentRunResult,
    used by both a fresh run and a resumed one."""
    run_row.status = result.status.value
    run_row.final_report = result.final_report
    run_row.plan = result.plan
    run_row.total_input_tokens = result.total_input_tokens
    run_row.total_output_tokens = result.total_output_tokens
    run_row.duration_ms = result.duration_ms
    run_row.pending_action = result.pending_action
    run_row.resume_state = result.resume_state

    existing_step_count = db.query(AgentStepORM).filter(AgentStepORM.run_id == run_row.id).count()
    for step in result.steps[existing_step_count:]:
        db.add(AgentStepORM(
            run_id=run_row.id,
            step_number=step.step_number,
            thought=step.thought,
            phase=step.phase,
            duration_ms=step.duration_ms,
            is_error=step.is_error,
            tool_called=step.tool_called,
            tool_input=step.tool_input,
            tool_output=step.tool_output,
        ))
    db.commit()


@celery_app.task(name="run_agent_task")
def run_agent_task(run_id: str, task: str) -> None:
    db = SessionLocal()
    try:
        run_row = db.query(AgentRunORM).filter(AgentRunORM.id == run_id).first()
        if run_row is None:
            return
        run_row.status = "running"
        db.commit()

        history: list[dict] = []
        if run_row.session_id:
            session = db.query(AgentSessionORM).filter(AgentSessionORM.id == run_row.session_id).first()
            if session is not None:
                history = build_context_messages(db, session)

        agent = _build_agent(run_row.provider or "anthropic")
        result = agent.run(task, history=history)

        _persist_result(db, run_row, result)

        if run_row.session_id and result.status.value == "done":
            update_session_memory(db, run_row.session_id, task, result.final_report)
    except Exception as exc:
        # Never let an unhandled exception leave a run stuck in "running" forever.
        run_row = db.query(AgentRunORM).filter(AgentRunORM.id == run_id).first()
        if run_row is not None:
            run_row.status = "failed"
            run_row.final_report = {"error": str(exc)}
            db.commit()
    finally:
        db.close()


@celery_app.task(name="run_multi_agent_task")
def run_multi_agent_task(run_id: str, task: str) -> None:
    db = SessionLocal()
    try:
        run_row = db.query(AgentRunORM).filter(AgentRunORM.id == run_id).first()
        if run_row is None:
            return
        run_row.status = "running"
        db.commit()

        provider_name = run_row.provider or "anthropic"
        model = _PROVIDER_MODELS.get(provider_name, settings.agent_model)
        provider = build_provider(
            provider_name, model=model, anthropic_client=_anthropic_client,
            openai_api_key=settings.openai_api_key, ollama_base_url=settings.ollama_base_url,
        )

        supervisor = Supervisor(provider=provider)
        specialist_name = supervisor.route(task)
        run_row.specialist = specialist_name
        db.commit()

        specialist_registry = build_specialist_registry(specialist_name)
        framed_task = supervisor.frame_task(specialist_name, task)

        agent = Agent(
            provider=provider, registry=specialist_registry, model=model,
            max_steps=settings.agent_max_steps, max_seconds=settings.agent_max_seconds,
        )
        result = agent.run(framed_task)

        _persist_result(db, run_row, result)
    except Exception as exc:
        run_row = db.query(AgentRunORM).filter(AgentRunORM.id == run_id).first()
        if run_row is not None:
            run_row.status = "failed"
            run_row.final_report = {"error": str(exc)}
            db.commit()
    finally:
        db.close()


@celery_app.task(name="resume_agent_task")
def resume_agent_task(run_id: str, approved: bool, denial_reason: str | None = None) -> None:
    db = SessionLocal()
    try:
        run_row = db.query(AgentRunORM).filter(AgentRunORM.id == run_id).first()
        if run_row is None or run_row.resume_state is None:
            return
        run_row.status = "running"
        db.commit()

        agent = _build_agent(run_row.provider or "anthropic")
        result = agent.resume(run_id, run_row.task, run_row.resume_state, approved, denial_reason)

        _persist_result(db, run_row, result)

        if run_row.session_id and result.status.value == "done":
            update_session_memory(db, run_row.session_id, run_row.task, result.final_report)
    except Exception as exc:
        run_row = db.query(AgentRunORM).filter(AgentRunORM.id == run_id).first()
        if run_row is not None:
            run_row.status = "failed"
            run_row.final_report = {"error": str(exc)}
            db.commit()
    finally:
        db.close()