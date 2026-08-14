from __future__ import annotations
from sqlalchemy.orm import Session
from sqlalchemy import func

from db import AgentRunORM, AgentStepORM

# Claude Sonnet pricing (USD per million tokens). Update if pricing changes --
# kept as named constants in one place specifically so that update is easy.
INPUT_COST_PER_MILLION = 3.00
OUTPUT_COST_PER_MILLION = 15.00


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
    return round(input_cost + output_cost, 6)


def build_run_observability(run: AgentRunORM) -> dict:
    """
    A structured summary view for one run: durations, cost, tool usage
    breakdown, and failure counts -- deliberately NOT the raw chain-of-thought
    text (that lives in the full run detail endpoint if someone wants it).
    """
    tool_usage: dict[str, int] = {}
    tool_errors: dict[str, int] = {}
    phase_counts: dict[str, int] = {"planning": 0, "action": 0, "reflection": 0}
    total_step_duration_ms = 0

    for step in run.steps:
        phase_counts[step.phase] = phase_counts.get(step.phase, 0) + 1
        total_step_duration_ms += step.duration_ms or 0
        if step.tool_called:
            tool_usage[step.tool_called] = tool_usage.get(step.tool_called, 0) + 1
            if step.is_error:
                tool_errors[step.tool_called] = tool_errors.get(step.tool_called, 0) + 1

    return {
        "run_id": run.id,
        "status": run.status,
        "duration_ms": run.duration_ms,
        "step_count": len(run.steps),
        "phase_counts": phase_counts,
        "tool_usage": tool_usage,
        "tool_errors": tool_errors,
        "total_input_tokens": run.total_input_tokens,
        "total_output_tokens": run.total_output_tokens,
        "estimated_cost_usd": estimate_cost_usd(run.total_input_tokens, run.total_output_tokens),
    }


def build_aggregate_observability(db: Session) -> dict:
    """
    Cross-run aggregate stats -- the data a real observability dashboard
    would chart: total cost, success rate, tool usage across all runs,
    average duration.
    """
    total_runs = db.query(func.count(AgentRunORM.id)).scalar() or 0

    status_counts_raw = (
        db.query(AgentRunORM.status, func.count(AgentRunORM.id))
        .group_by(AgentRunORM.status)
        .all()
    )
    status_counts = {status: count for status, count in status_counts_raw}

    token_totals = db.query(
        func.coalesce(func.sum(AgentRunORM.total_input_tokens), 0),
        func.coalesce(func.sum(AgentRunORM.total_output_tokens), 0),
    ).first()
    total_input_tokens, total_output_tokens = token_totals

    avg_duration = db.query(func.avg(AgentRunORM.duration_ms)).filter(
        AgentRunORM.duration_ms > 0
    ).scalar()

    tool_usage_raw = (
        db.query(AgentStepORM.tool_called, func.count(AgentStepORM.id))
        .filter(AgentStepORM.tool_called.isnot(None))
        .group_by(AgentStepORM.tool_called)
        .all()
    )
    tool_usage = {tool: count for tool, count in tool_usage_raw}

    tool_errors_raw = (
        db.query(AgentStepORM.tool_called, func.count(AgentStepORM.id))
        .filter(AgentStepORM.tool_called.isnot(None), AgentStepORM.is_error.is_(True))
        .group_by(AgentStepORM.tool_called)
        .all()
    )
    tool_errors = {tool: count for tool, count in tool_errors_raw}

    done_count = status_counts.get("done", 0)
    success_rate = round(done_count / total_runs, 4) if total_runs > 0 else None

    return {
        "total_runs": total_runs,
        "status_counts": status_counts,
        "success_rate": success_rate,
        "total_input_tokens": int(total_input_tokens or 0),
        "total_output_tokens": int(total_output_tokens or 0),
        "estimated_total_cost_usd": estimate_cost_usd(
            int(total_input_tokens or 0), int(total_output_tokens or 0)
        ),
        "average_duration_ms": round(avg_duration, 1) if avg_duration else None,
        "tool_usage": tool_usage,
        "tool_errors": tool_errors,
    }