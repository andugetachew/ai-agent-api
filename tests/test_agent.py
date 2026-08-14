import sys
import os
from unittest.mock import MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from agent import Agent


def _plan_response(make_response, make_text_block, plan_text="- Step 1\n- Step 2"):
    """Every Agent.run() call now starts with a dedicated planning call.
    This helper builds that mocked response so it can be prepended to every
    test's side_effect list."""
    return make_response([make_text_block(plan_text)])


class TestAgentLoop:
    def test_calculator_task_completes(self, fake_client, registry, make_response, make_text_block, make_tool_use_block):
        fake_client.messages.create.side_effect = [
            _plan_response(make_response, make_text_block),
            make_response([
                make_text_block("I need to compute this."),
                make_tool_use_block("calculator", {"expression": "(245 * 12) + 897"}, "tool_1"),
            ]),
            make_response([
                make_tool_use_block("submit_result", {
                    "answer": "3837",
                    "details": ["Computed via calculator"],
                    "confidence": "high",
                }, "tool_2"),
            ]),
        ]
        agent = Agent(client=fake_client, registry=registry, max_steps=6, max_seconds=30)
        result = agent.run("What is (245 * 12) + 897?")

        assert result.status.value == "done"
        assert result.plan is not None
        assert result.steps[0].phase == "planning"
        assert len(result.steps) == 3
        assert result.steps[1].tool_called == "calculator"
        assert result.steps[1].tool_output == "3837"
        assert result.steps[2].tool_called == "submit_result"
        assert result.final_report["answer"] == "3837"

    def test_direct_answer_without_tool_call(self, fake_client, registry, make_response, make_text_block):
        fake_client.messages.create.side_effect = [
            _plan_response(make_response, make_text_block),
            make_response([make_text_block("The capital of France is Paris.")]),
        ]
        agent = Agent(client=fake_client, registry=registry, max_steps=6, max_seconds=30)
        result = agent.run("What is the capital of France?")

        assert result.status.value == "done"
        assert result.steps[0].phase == "planning"
        assert len(result.steps) == 2
        assert result.steps[1].tool_called is None
        assert "Paris" in result.final_report["answer"]

    def test_max_steps_triggers_forced_synthesis(self, fake_client, registry, make_response, make_text_block, make_tool_use_block):
        def looping_response(*args, **kwargs):
            return make_response([
                make_text_block("Still working..."),
                make_tool_use_block("calculator", {"expression": "1 + 1"}, "tool_x"),
            ])

        fake_client.messages.create.side_effect = [
            _plan_response(make_response, make_text_block),
        ] + [
            looping_response() for _ in range(3)
        ] + [
            make_response([
                make_tool_use_block("submit_result", {
                    "answer": "Could not fully converge",
                    "details": ["Ran out of steps"],
                    "confidence": "low",
                }, "tool_final"),
            ]),
        ]
        agent = Agent(client=fake_client, registry=registry, max_steps=3, max_seconds=30)
        result = agent.run("Keep calculating forever")

        assert result.status.value == "max_steps_exceeded"
        assert result.final_report["answer"] == "Could not fully converge"
        assert result.final_report["confidence"] == "low"
        assert result.plan is not None

    def test_forced_synthesis_with_no_tool_response_returns_fallback(self, fake_client, registry, make_response, make_text_block, make_tool_use_block):
        def looping_response(*args, **kwargs):
            return make_response([
                make_tool_use_block("calculator", {"expression": "1 + 1"}, "tool_x"),
            ])

        fake_client.messages.create.side_effect = [
            _plan_response(make_response, make_text_block),
        ] + [
            looping_response() for _ in range(2)
        ] + [
            make_response([make_text_block("I give up.")]),
        ]
        agent = Agent(client=fake_client, registry=registry, max_steps=2, max_seconds=30)
        result = agent.run("Keep calculating forever")

        assert result.status.value == "max_steps_exceeded"
        assert result.final_report["confidence"] == "low"
        assert "could not complete" in result.final_report["answer"].lower()

    def test_tool_error_does_not_crash_loop(self, fake_client, registry, make_response, make_tool_use_block, make_text_block):
        fake_client.messages.create.side_effect = [
            _plan_response(make_response, make_text_block),
            make_response([
                make_tool_use_block("calculator", {"expression": "not valid ("}, "tool_1"),
            ]),
            make_response([
                make_tool_use_block("submit_result", {
                    "answer": "Could not evaluate expression",
                    "details": ["Calculator returned an error"],
                    "confidence": "low",
                }, "tool_2"),
            ]),
        ]
        agent = Agent(client=fake_client, registry=registry, max_steps=6, max_seconds=30)
        result = agent.run("Compute an invalid expression")

        assert result.status.value == "done"
        assert result.steps[1].tool_output.startswith("ERROR")

    def test_reflection_nudge_injected_every_interval(self, fake_client, registry, make_response, make_text_block, make_tool_use_block):
        """Steps 1, 2, 3 are plain action turns; step 4 (the 4th loop
        iteration, right after 3 completed) should be tagged as reflection."""
        def calc_response(*args, **kwargs):
            return make_response([
                make_tool_use_block("calculator", {"expression": "1 + 1"}, "tool_x"),
            ])

        fake_client.messages.create.side_effect = [
            _plan_response(make_response, make_text_block),
        ] + [
            calc_response() for _ in range(3)
        ] + [
            make_response([
                make_tool_use_block("submit_result", {
                    "answer": "done", "details": [], "confidence": "high",
                }, "tool_final"),
            ]),
        ]
        agent = Agent(client=fake_client, registry=registry, max_steps=6, max_seconds=30)
        result = agent.run("Do several calculator steps")

        assert result.steps[0].phase == "planning"
        assert result.steps[1].phase == "action"
        assert result.steps[2].phase == "action"
        assert result.steps[3].phase == "action"
        assert result.steps[4].phase == "reflection"

class TestHumanApproval:
    def test_risky_tool_pauses_run_for_approval(self, fake_client, registry, make_response, make_text_block, make_tool_use_block):
        fake_client.messages.create.side_effect = [
            _plan_response(make_response, make_text_block),
            make_response([
                make_text_block("I need to run some code."),
                make_tool_use_block("execute_code", {"code": "print(1+1)"}, "tool_risky"),
            ]),
        ]
        agent = Agent(client=fake_client, registry=registry, max_steps=6, max_seconds=30)
        result = agent.run("run some code")

        assert result.status.value == "awaiting_approval"
        assert result.pending_action["tool_name"] == "execute_code"
        assert result.final_report is None
        assert result.resume_state is not None
        assert not any(s.tool_called == "execute_code" for s in result.steps)

    def test_resume_state_is_json_serializable(self, fake_client, registry, make_response, make_text_block, make_tool_use_block):
        import json
        fake_client.messages.create.side_effect = [
            _plan_response(make_response, make_text_block),
            make_response([make_tool_use_block("execute_code", {"code": "x"}, "tool_risky")]),
        ]
        agent = Agent(client=fake_client, registry=registry, max_steps=6, max_seconds=30)
        result = agent.run("run some code")

        serialized = json.dumps(result.resume_state)
        assert len(serialized) > 0

    def test_resume_approved_executes_tool_and_continues(self, fake_client, registry, make_response, make_text_block, make_tool_use_block):
        import json
        fake_client.messages.create.side_effect = [
            _plan_response(make_response, make_text_block),
            make_response([make_tool_use_block("execute_code", {"code": "print(99)"}, "tool_risky")]),
        ]
        agent = Agent(client=fake_client, registry=registry, max_steps=6, max_seconds=30)
        paused = agent.run("run some code")
        resume_state = json.loads(json.dumps(paused.resume_state))

        fake_client2 = MagicMock()
        fake_client2.messages.create.side_effect = [
            make_response([make_tool_use_block("submit_result", {
                "answer": "99", "details": [], "confidence": "high",
            }, "final")]),
        ]
        agent2 = Agent(client=fake_client2, registry=registry, max_steps=6, max_seconds=30)
        resumed = agent2.resume(paused.run_id, "run some code", resume_state, approved=True)

        assert resumed.status.value == "done"
        assert resumed.final_report["answer"] == "99"
        executed_step = next(s for s in resumed.steps if s.tool_called == "execute_code")
        assert executed_step.tool_output == "99"
        assert executed_step.is_error is False

    def test_resume_denied_skips_execution_and_continues(self, fake_client, registry, make_response, make_text_block, make_tool_use_block):
        import json
        fake_client.messages.create.side_effect = [
            _plan_response(make_response, make_text_block),
            make_response([make_tool_use_block("execute_code", {"code": "print(99)"}, "tool_risky")]),
        ]
        agent = Agent(client=fake_client, registry=registry, max_steps=6, max_seconds=30)
        paused = agent.run("run some code")
        resume_state = json.loads(json.dumps(paused.resume_state))

        fake_client2 = MagicMock()
        fake_client2.messages.create.side_effect = [
            make_response([make_tool_use_block("submit_result", {
                "answer": "could not run, denied", "details": [], "confidence": "low",
            }, "final")]),
        ]
        agent2 = Agent(client=fake_client2, registry=registry, max_steps=6, max_seconds=30)
        resumed = agent2.resume(paused.run_id, "run some code", resume_state, approved=False, denial_reason="too risky")

        assert resumed.status.value == "done"
        skipped_step = next(s for s in resumed.steps if s.tool_called == "execute_code")
        assert "REJECTED" in skipped_step.tool_output
        assert "too risky" in skipped_step.tool_output
        assert skipped_step.tool_output != "99"