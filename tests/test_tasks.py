import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))


def _make_fake_db(first_side_effect, count_return=0):
    """Builds a MagicMock DB session that correctly answers both the
    run/session .first() lookups AND the .count() call _persist_result uses
    to figure out which steps are already persisted."""
    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.side_effect = first_side_effect
    fake_db.query.return_value.filter.return_value.count.return_value = count_return
    return fake_db


class TestRunAgentTask:
    def test_successful_run_updates_status_and_persists_steps(self):
        from models import AgentRunResult, RunStatus, StepRecord

        fake_run_row = MagicMock()
        fake_run_row.id = "run-1"
        fake_run_row.session_id = None  # no session -> memory branch skipped

        fake_db = _make_fake_db(first_side_effect=[fake_run_row])

        fake_agent_result = AgentRunResult(
            run_id="run-1",
            task="test task",
            status=RunStatus.DONE,
            steps=[StepRecord(1, "thought", "calculator", {"expression": "1+1"}, "2")],
            final_report={"answer": "2", "details": [], "confidence": "high"},
            total_input_tokens=10,
            total_output_tokens=5,
        )

        with patch("tasks.SessionLocal", return_value=fake_db), \
             patch("tasks.Agent") as MockAgent:
            MockAgent.return_value.run.return_value = fake_agent_result

            import tasks
            tasks.run_agent_task.run("run-1", "test task")

        assert fake_run_row.status == "done"
        assert fake_run_row.final_report == {"answer": "2", "details": [], "confidence": "high"}
        assert fake_run_row.total_input_tokens == 10
        fake_db.add.assert_called_once()  # one step persisted
        fake_db.commit.assert_called()
        fake_db.close.assert_called_once()
        MockAgent.return_value.run.assert_called_once_with("test task", history=[])

    def test_missing_run_row_returns_without_error(self):
        fake_db = _make_fake_db(first_side_effect=[None])

        with patch("tasks.SessionLocal", return_value=fake_db), \
             patch("tasks.Agent") as MockAgent:
            import tasks
            tasks.run_agent_task.run("does-not-exist", "test task")

        MockAgent.return_value.run.assert_not_called()
        fake_db.close.assert_called_once()

    def test_agent_exception_marks_run_failed(self):
        fake_run_row = MagicMock()
        fake_run_row.id = "run-2"
        fake_run_row.session_id = None

        fake_db = _make_fake_db(first_side_effect=[fake_run_row, fake_run_row])

        with patch("tasks.SessionLocal", return_value=fake_db), \
             patch("tasks.Agent") as MockAgent:
            MockAgent.return_value.run.side_effect = RuntimeError("simulated Claude API failure")

            import tasks
            tasks.run_agent_task.run("run-2", "test task")

        assert fake_run_row.status == "failed"
        assert "simulated Claude API failure" in fake_run_row.final_report["error"]
        fake_db.close.assert_called_once()

    def test_session_run_loads_history_and_updates_memory(self):
        """When a run belongs to a session, prior context should be loaded and
        passed to Agent.run(), and update_session_memory should be called after
        a successful completion."""
        from models import AgentRunResult, RunStatus, StepRecord

        fake_run_row = MagicMock()
        fake_run_row.id = "run-3"
        fake_run_row.session_id = "session-abc"

        fake_session_row = MagicMock()
        fake_session_row.id = "session-abc"

        fake_db = _make_fake_db(first_side_effect=[fake_run_row, fake_session_row])

        fake_agent_result = AgentRunResult(
            run_id="run-3",
            task="follow-up question",
            status=RunStatus.DONE,
            steps=[],
            final_report={"answer": "follow-up answer", "details": [], "confidence": "high"},
            total_input_tokens=5,
            total_output_tokens=5,
        )

        with patch("tasks.SessionLocal", return_value=fake_db), \
             patch("tasks.Agent") as MockAgent, \
             patch("tasks.build_context_messages", return_value=[{"role": "user", "content": "earlier"}]) as mock_build_ctx, \
             patch("tasks.update_session_memory") as mock_update_memory:
            MockAgent.return_value.run.return_value = fake_agent_result

            import tasks
            tasks.run_agent_task.run("run-3", "follow-up question")

        mock_build_ctx.assert_called_once()
        MockAgent.return_value.run.assert_called_once_with(
            "follow-up question", history=[{"role": "user", "content": "earlier"}]
        )
        mock_update_memory.assert_called_once_with(
            fake_db, "session-abc", "follow-up question", fake_agent_result.final_report
        )

    def test_run_pausing_for_approval_persists_pending_state(self):
        """When Agent.run() returns AWAITING_APPROVAL, the task should persist
        that status plus pending_action/resume_state, and NOT call
        update_session_memory (the run isn't done yet)."""
        from models import AgentRunResult, RunStatus, StepRecord

        fake_run_row = MagicMock()
        fake_run_row.id = "run-4"
        fake_run_row.session_id = None

        fake_db = _make_fake_db(first_side_effect=[fake_run_row])

        pending_result = AgentRunResult(
            run_id="run-4",
            task="run risky code",
            status=RunStatus.AWAITING_APPROVAL,
            steps=[StepRecord(0, "plan", None, None, None, phase="planning")],
            final_report=None,
            pending_action={"tool_name": "execute_code", "tool_input": {"code": "x"}, "step_number": 1},
            resume_state={"messages": [], "next_step_number": 1},
        )

        with patch("tasks.SessionLocal", return_value=fake_db), \
             patch("tasks.Agent") as MockAgent, \
             patch("tasks.update_session_memory") as mock_update_memory:
            MockAgent.return_value.run.return_value = pending_result

            import tasks
            tasks.run_agent_task.run("run-4", "run risky code")

        assert fake_run_row.status == "awaiting_approval"
        assert fake_run_row.pending_action["tool_name"] == "execute_code"
        assert fake_run_row.resume_state is not None
        mock_update_memory.assert_not_called()


class TestResumeAgentTask:
    def test_resume_approved_completes_run(self):
        from models import AgentRunResult, RunStatus

        fake_run_row = MagicMock()
        fake_run_row.id = "run-5"
        fake_run_row.session_id = None
        fake_run_row.task = "run risky code"
        fake_run_row.resume_state = {"messages": [], "next_step_number": 1}

        fake_db = _make_fake_db(first_side_effect=[fake_run_row])

        resumed_result = AgentRunResult(
            run_id="run-5", task="run risky code", status=RunStatus.DONE,
            steps=[], final_report={"answer": "done", "details": [], "confidence": "high"},
        )

        with patch("tasks.SessionLocal", return_value=fake_db), \
             patch("tasks.Agent") as MockAgent:
            MockAgent.return_value.resume.return_value = resumed_result

            import tasks
            tasks.resume_agent_task.run("run-5", True, None)

        MockAgent.return_value.resume.assert_called_once_with(
            "run-5", "run risky code", {"messages": [], "next_step_number": 1}, True, None
        )
        assert fake_run_row.status == "done"

    def test_resume_missing_resume_state_is_a_no_op(self):
        fake_run_row = MagicMock()
        fake_run_row.id = "run-6"
        fake_run_row.resume_state = None

        fake_db = _make_fake_db(first_side_effect=[fake_run_row])

        with patch("tasks.SessionLocal", return_value=fake_db), \
             patch("tasks.Agent") as MockAgent:
            import tasks
            tasks.resume_agent_task.run("run-6", True, None)

        MockAgent.return_value.resume.assert_not_called()
        fake_db.close.assert_called_once()