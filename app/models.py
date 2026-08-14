from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    AWAITING_APPROVAL = "awaiting_approval"


@dataclass
class StepRecord:
    step_number: int
    thought: str | None
    tool_called: str | None
    tool_input: dict[str, Any] | None
    tool_output: str | None
    phase: str = "action"  # "planning" | "action" | "reflection"
    duration_ms: int = 0  # wall-clock time for this step's API call + tool execution
    is_error: bool = False  # True if tool_output starts with "ERROR"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentRunResult:
    run_id: str
    task: str
    status: RunStatus
    steps: list[StepRecord]
    final_report: dict[str, Any] | None
    plan: str | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    duration_ms: int = 0  # total wall-clock time for the whole run
    pending_action: dict[str, Any] | None = None  # set when status == AWAITING_APPROVAL
    resume_state: dict[str, Any] | None = None  # serialized state needed to continue after approval