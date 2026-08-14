from __future__ import annotations
from datetime import datetime, timezone

from sqlalchemy.orm import Session as DBSession

from db import AgentSessionORM, AgentRunORM

# How many of the most recent completed turns get passed to the agent verbatim.
# Anything older than this gets folded into the rolling `summary` field instead,
# so context sent to Claude stays bounded regardless of how long a session runs.
RECENT_TURNS_KEPT = 5

# Rough character budget before a turn's answer gets truncated when folded into
# the summary -- keeps the summary itself from growing unbounded too.
SUMMARY_ANSWER_TRUNCATE = 300


def get_or_create_session(db: DBSession, session_id: str | None) -> AgentSessionORM:
    if session_id:
        existing = db.query(AgentSessionORM).filter(AgentSessionORM.id == session_id).first()
        if existing is not None:
            return existing
    session_row = AgentSessionORM(id=session_id) if session_id else AgentSessionORM()
    db.add(session_row)
    db.commit()
    db.refresh(session_row)
    return session_row


def build_context_messages(db: DBSession, session: AgentSessionORM) -> list[dict]:
    """
    Builds the prior-conversation context to prepend to a new agent run:
    the rolling summary (if any) as a system-style note, plus the most recent
    completed turns verbatim as user/assistant message pairs.
    """
    messages: list[dict] = []

    if session.summary:
        messages.append({
            "role": "user",
            "content": f"[Summary of earlier conversation in this session]\n{session.summary}",
        })
        messages.append({
            "role": "assistant",
            "content": "Understood, I have the context from our earlier conversation.",
        })

    recent_runs = (
        db.query(AgentRunORM)
        .filter(AgentRunORM.session_id == session.id, AgentRunORM.status == "done")
        .order_by(AgentRunORM.created_at.desc())
        .limit(RECENT_TURNS_KEPT)
        .all()
    )
    for run in reversed(recent_runs):  # oldest-first, so conversation order is correct
        messages.append({"role": "user", "content": run.task})
        answer = (run.final_report or {}).get("answer", "")
        messages.append({"role": "assistant", "content": answer})

    return messages


def _naive_summarize(existing_summary: str | None, task: str, answer: str) -> str:
    """
    Folds one older turn into the rolling summary using simple truncation.
    NOTE: this is a placeholder heuristic, not real summarization -- swap this
    for a Claude call (e.g. "summarize this conversation turn in one sentence")
    once credits are available. Kept structurally isolated in this one function
    specifically so that upgrade is a one-function change, not a refactor.
    """
    truncated_answer = answer[:SUMMARY_ANSWER_TRUNCATE]
    new_line = f"- Asked: {task[:200]} | Answered: {truncated_answer}"
    if existing_summary:
        return f"{existing_summary}\n{new_line}"
    return new_line


def update_session_memory(db: DBSession, session_id: str, task: str, final_report: dict | None) -> None:
    """
    Called after a run completes. If the session now has more than
    RECENT_TURNS_KEPT completed runs, the oldest excess turn is folded into
    the rolling summary so `build_context_messages` keeps returning a bounded
    amount of context, however long the session goes on for.
    """
    session = db.query(AgentSessionORM).filter(AgentSessionORM.id == session_id).first()
    if session is None:
        return

    done_runs = (
        db.query(AgentRunORM)
        .filter(AgentRunORM.session_id == session_id, AgentRunORM.status == "done")
        .order_by(AgentRunORM.created_at.asc())
        .all()
    )

    if len(done_runs) > RECENT_TURNS_KEPT:
        oldest_excess = done_runs[: len(done_runs) - RECENT_TURNS_KEPT]
        for run in oldest_excess:
            answer = (run.final_report or {}).get("answer", "")
            session.summary = _naive_summarize(session.summary, run.task, answer)

    session.updated_at = datetime.now(timezone.utc)
    db.commit()