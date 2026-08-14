from __future__ import annotations
import time
import uuid
from anthropic import Anthropic

from models import AgentRunResult, RunStatus, StepRecord
from tools import ToolRegistry, build_default_registry
from providers import ModelProvider, AnthropicProvider

SYSTEM_PROMPT = """You are an autonomous agent that completes tasks using the tools \
available to you. Reason step by step about what to do next. Use tools as needed \
to gather information, compute values, or run code. When the task is complete, \
call submit_result with your final answer. Do not call submit_result until you \
have taken at least one action toward the task, unless the task requires no tools \
at all."""

PLANNING_PROMPT = """Given the task below and the tools available to you, write a \
short plan (2-4 bullet points) of the approach you intend to take. Do not solve \
the task yet -- just outline the plan. Keep it brief."""

REFLECTION_NUDGE = """Before continuing: briefly check whether your current approach \
is still on track given your plan. If it needs adjusting, note the adjustment in one \
sentence, then continue toward the task."""

# How often (in loop iterations) a reflection nudge is injected. Reuses the
# existing loop turn rather than spending an extra API call, so this is
# cost-neutral -- it just changes what's asked for on that turn.
REFLECTION_INTERVAL = 3

# Tools that pause the run for human approval before executing, instead of
# running automatically. Kept as a plain set here (not per-tool config) so
# it's easy to see and change in one place.
APPROVAL_REQUIRED_TOOLS = {"execute_code"}

# Simple substring match against the exception message to decide whether a
# provider failure is worth retrying. Deliberately conservative: only retry
# errors that are clearly transient (rate limits, timeouts, overload,
# connection blips) -- anything else (bad API key, invalid request) should
# fail immediately rather than retry into the same wall repeatedly.
TRANSIENT_ERROR_KEYWORDS = (
    "rate_limit", "timeout", "overloaded", "connection",
    "503", "429", "temporarily unavailable",
)
MAX_PROVIDER_RETRIES = 1


def _is_transient_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(keyword in message for keyword in TRANSIENT_ERROR_KEYWORDS)


def _is_error_output(output: str | None) -> bool:
    return bool(output) and output.startswith("ERROR")


def _serialize_block(block) -> dict:
    """Converts a response content block (SDK object) into a plain JSON-safe
    dict, so message history can be persisted to Postgres and reconstructed
    later for resume."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return {"type": block.type}


def _serialize_step(step: StepRecord) -> dict:
    return {
        "step_number": step.step_number,
        "thought": step.thought,
        "tool_called": step.tool_called,
        "tool_input": step.tool_input,
        "tool_output": step.tool_output,
        "phase": step.phase,
        "duration_ms": step.duration_ms,
        "is_error": step.is_error,
    }


def _deserialize_step(d: dict) -> StepRecord:
    return StepRecord(
        step_number=d["step_number"],
        thought=d["thought"],
        tool_called=d["tool_called"],
        tool_input=d["tool_input"],
        tool_output=d["tool_output"],
        phase=d.get("phase", "action"),
        duration_ms=d.get("duration_ms", 0),
        is_error=d.get("is_error", False),
    )


class Agent:
    def __init__(
        self,
        client: Anthropic | None = None,
        registry: ToolRegistry | None = None,
        model: str = "claude-sonnet-4-6",
        max_steps: int = 10,
        max_seconds: float = 90.0,
        provider: ModelProvider | None = None,
    ) -> None:
        self.client = client or Anthropic()
        self.registry = registry or build_default_registry()
        self.model = model
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        # Defaults to wrapping self.client so existing Anthropic-only usage
        # (and every test that constructs Agent(client=...)) keeps working
        # completely unchanged. Only pass provider= explicitly for OpenAI/Ollama.
        self.provider = provider or AnthropicProvider(self.client, model)

    def _call_provider(self, system, messages, tools, max_tokens):
        """Wraps every provider.create() call with a single retry on
        transient failures (rate limits, timeouts, connection blips).
        Non-transient errors (bad API key, invalid request) propagate
        immediately -- retrying those would just fail the same way again."""
        attempt = 0
        while True:
            try:
                return self.provider.create(
                    system=system, messages=messages, tools=tools, max_tokens=max_tokens,
                )
            except Exception as exc:
                if attempt < MAX_PROVIDER_RETRIES and _is_transient_error(exc):
                    attempt += 1
                    continue
                raise

    def _generate_plan(self, task: str) -> tuple[str, int, int, int]:
        """One dedicated, tool-free call that asks the model to outline its
        approach before acting. Returns (plan_text, input_tokens, output_tokens,
        duration_ms). This call does not count against max_steps."""
        step_start = time.monotonic()
        response = self._call_provider(
            system=PLANNING_PROMPT,
            messages=[{"role": "user", "content": task}],
            tools=None,
            max_tokens=300,
        )
        duration_ms = int((time.monotonic() - step_start) * 1000)
        text_blocks = [b.text for b in response.content if b.type == "text"]
        plan_text = " ".join(text_blocks) if text_blocks else "(no plan generated)"
        return plan_text, response.usage.input_tokens, response.usage.output_tokens, duration_ms

    def run(self, task: str, history: list[dict] | None = None) -> AgentRunResult:
        run_id = str(uuid.uuid4())
        steps: list[StepRecord] = []
        run_start = time.monotonic()
        total_in = total_out = 0

        plan_text, plan_in, plan_out, plan_duration = self._generate_plan(task)
        total_in += plan_in
        total_out += plan_out
        steps.append(StepRecord(
            0, plan_text, None, None, None,
            phase="planning", duration_ms=plan_duration,
        ))

        system_prompt = f"{SYSTEM_PROMPT}\n\nYour plan for this task:\n{plan_text}"

        messages = list(history) if history else []
        messages.append({"role": "user", "content": task})

        return self._action_loop(
            run_id, task, steps, messages, total_in, total_out,
            plan_text, system_prompt, run_start, start_step=1, elapsed_before_ms=0,
        )

    def resume(
        self,
        run_id: str,
        task: str,
        resume_state: dict,
        approved: bool,
        denial_reason: str | None = None,
    ) -> AgentRunResult:
        """Continues a run that was paused for human approval. Executes (or
        skips, if denied) the pending tool call(s), tells the model the
        outcome, then continues the normal action loop from where it left off."""
        messages = list(resume_state["messages"])
        system_prompt = resume_state["system_prompt"]
        plan_text = resume_state["plan_text"]
        total_in = resume_state["total_input_tokens"]
        total_out = resume_state["total_output_tokens"]
        steps = [_deserialize_step(s) for s in resume_state["steps"]]
        step_number = resume_state["next_step_number"]
        phase = resume_state.get("pending_phase", "action")
        elapsed_before_ms = resume_state.get("elapsed_ms_before_pause", 0)
        run_start = time.monotonic()

        tool_results = []
        for block in resume_state["pending_tool_blocks"]:
            if block["name"] in APPROVAL_REQUIRED_TOOLS:
                if approved:
                    output = self.registry.run(block["name"], block["input"])
                    is_err = _is_error_output(output)
                else:
                    output = f"REJECTED: user denied this action. Reason: {denial_reason or 'not specified'}"
                    is_err = False
            else:
                output = self.registry.run(block["name"], block["input"])
                is_err = _is_error_output(output)

            steps.append(StepRecord(
                step_number, None, block["name"], block["input"], output,
                phase=phase, duration_ms=0, is_error=is_err,
            ))
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": output,
            })

            if block["name"] == "submit_result":
                return AgentRunResult(
                    run_id, task, RunStatus.DONE, steps,
                    final_report=block["input"],
                    plan=plan_text,
                    total_input_tokens=total_in, total_output_tokens=total_out,
                    duration_ms=elapsed_before_ms + int((time.monotonic() - run_start) * 1000),
                )

        messages.append({"role": "user", "content": tool_results})

        return self._action_loop(
            run_id, task, steps, messages, total_in, total_out,
            plan_text, system_prompt, run_start, start_step=step_number + 1,
            elapsed_before_ms=elapsed_before_ms,
        )

    def _action_loop(
        self, run_id, task, steps, messages, total_in, total_out,
        plan_text, system_prompt, run_start, start_step, elapsed_before_ms,
    ) -> AgentRunResult:
        for step_number in range(start_step, self.max_steps + 1):
            if time.monotonic() - run_start > self.max_seconds:
                return self._force_synthesis(
                    run_id, task, steps, messages, total_in, total_out, plan_text,
                    run_start, elapsed_before_ms, reason="time limit exceeded",
                )

            step_start = time.monotonic()
            is_reflection_turn = step_number > 1 and (step_number - 1) % REFLECTION_INTERVAL == 0
            if is_reflection_turn:
                messages.append({"role": "user", "content": REFLECTION_NUDGE})

            response = self._call_provider(
                system=system_prompt,
                messages=messages,
                tools=self.registry.anthropic_tools(),
                max_tokens=1500,
            )
            total_in += response.usage.input_tokens
            total_out += response.usage.output_tokens

            text_blocks = [b.text for b in response.content if b.type == "text"]
            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            thought = " ".join(text_blocks) if text_blocks else None
            phase = "reflection" if is_reflection_turn else "action"

            if not tool_blocks:
                duration_ms = int((time.monotonic() - step_start) * 1000)
                steps.append(StepRecord(
                    step_number, thought, None, None, None,
                    phase=phase, duration_ms=duration_ms,
                ))
                return AgentRunResult(
                    run_id, task, RunStatus.DONE, steps,
                    final_report={
                        "answer": thought, "details": [], "sources": [], "confidence": "low",
                    },
                    plan=plan_text,
                    total_input_tokens=total_in, total_output_tokens=total_out,
                    duration_ms=elapsed_before_ms + int((time.monotonic() - run_start) * 1000),
                )

            approval_needed = [b for b in tool_blocks if b.name in APPROVAL_REQUIRED_TOOLS]
            if approval_needed:
                messages.append({
                    "role": "assistant",
                    "content": [_serialize_block(b) for b in response.content],
                })
                pending_block = approval_needed[0]
                return AgentRunResult(
                    run_id, task, RunStatus.AWAITING_APPROVAL, steps,
                    final_report=None,
                    plan=plan_text,
                    total_input_tokens=total_in, total_output_tokens=total_out,
                    duration_ms=elapsed_before_ms + int((time.monotonic() - run_start) * 1000),
                    pending_action={
                        "tool_name": pending_block.name,
                        "tool_input": pending_block.input,
                        "step_number": step_number,
                    },
                    resume_state={
                        "messages": messages,
                        "system_prompt": system_prompt,
                        "plan_text": plan_text,
                        "total_input_tokens": total_in,
                        "total_output_tokens": total_out,
                        "steps": [_serialize_step(s) for s in steps],
                        "next_step_number": step_number,
                        "pending_phase": phase,
                        "pending_tool_blocks": [_serialize_block(b) for b in tool_blocks],
                        "elapsed_ms_before_pause": elapsed_before_ms + int((time.monotonic() - run_start) * 1000),
                    },
                )

            messages.append({
                "role": "assistant",
                "content": [_serialize_block(b) for b in response.content],
            })
            tool_results = []
            for block in tool_blocks:
                tool_start = time.monotonic()
                output = self.registry.run(block.name, block.input)
                tool_duration = int((time.monotonic() - tool_start) * 1000)
                steps.append(StepRecord(
                    step_number, thought, block.name, block.input, output,
                    phase=phase, duration_ms=tool_duration, is_error=_is_error_output(output),
                ))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

                if block.name == "submit_result":
                    return AgentRunResult(
                        run_id, task, RunStatus.DONE, steps,
                        final_report=block.input,
                        plan=plan_text,
                        total_input_tokens=total_in, total_output_tokens=total_out,
                        duration_ms=elapsed_before_ms + int((time.monotonic() - run_start) * 1000),
                    )

            messages.append({"role": "user", "content": tool_results})

        return self._force_synthesis(
            run_id, task, steps, messages, total_in, total_out, plan_text,
            run_start, elapsed_before_ms, reason="max steps exceeded",
        )

    def _force_synthesis(self, run_id, task, steps, messages, total_in, total_out, plan_text, run_start, elapsed_before_ms, reason: str):
        messages.append({
            "role": "user",
            "content": (
                f"You have run out of steps/time ({reason}). "
                "Call submit_result now with whatever you've found so far, "
                "marking confidence as 'low' if incomplete."
            ),
        })
        response = self._call_provider(
            system=f"{SYSTEM_PROMPT}\n\nYour plan for this task:\n{plan_text}",
            messages=messages,
            tools=[t for t in self.registry.anthropic_tools() if t["name"] == "submit_result"],
            max_tokens=1000,
        )
        total_in += response.usage.input_tokens
        total_out += response.usage.output_tokens
        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        report = tool_blocks[0].input if tool_blocks else {
            "answer": "Agent could not complete the task in time.",
            "details": [], "sources": [], "confidence": "low",
        }
        return AgentRunResult(
            run_id, task, RunStatus.MAX_STEPS_EXCEEDED, steps,
            final_report=report, plan=plan_text,
            total_input_tokens=total_in, total_output_tokens=total_out,
            duration_ms=elapsed_before_ms + int((time.monotonic() - run_start) * 1000),
        )