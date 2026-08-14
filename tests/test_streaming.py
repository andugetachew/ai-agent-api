import sys
import os
import threading
import time
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from db import Base, AgentRunORM, AgentStepORM
import streaming


@pytest.fixture
def test_session_factory(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)

    monkeypatch.setattr(streaming, "SessionLocal", TestSession)
    monkeypatch.setattr(streaming, "POLL_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(streaming, "MAX_WAIT_SECONDS", 2.0)

    return TestSession


def _parse_events(raw_events: list[str]) -> list[tuple[str, dict]]:
    import json
    parsed = []
    for raw in raw_events:
        lines = raw.strip().split("\n")
        event_type = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        parsed.append((event_type, data))
    return parsed


class TestStreamRunEvents:
    def test_missing_run_yields_error_immediately(self, test_session_factory):
        events = _parse_events(list(streaming.stream_run_events("does-not-exist")))
        assert len(events) == 1
        assert events[0][0] == "error"
        assert "not found" in events[0][1]["message"]

    def test_already_completed_run_yields_start_and_done_only(self, test_session_factory):
        run_id = str(uuid.uuid4())
        db = test_session_factory()
        db.add(AgentRunORM(
            id=run_id, task="t", status="done",
            final_report={"answer": "42", "confidence": "high"},
        ))
        db.commit()

        events = _parse_events(list(streaming.stream_run_events(run_id)))
        event_types = [e[0] for e in events]
        assert event_types == ["start", "done"]
        assert events[-1][1]["final_report"]["answer"] == "42"

    def test_steps_added_concurrently_are_streamed_in_order(self, test_session_factory):
        run_id = str(uuid.uuid4())
        db = test_session_factory()
        db.add(AgentRunORM(id=run_id, task="t", status="pending"))
        db.commit()

        def simulate_progress():
            time.sleep(0.05)
            w = test_session_factory()
            run = w.query(AgentRunORM).filter(AgentRunORM.id == run_id).first()
            run.status = "running"
            w.add(AgentStepORM(run_id=run_id, step_number=0, phase="planning", tool_called=None))
            w.commit()

            time.sleep(0.08)
            w.add(AgentStepORM(run_id=run_id, step_number=1, phase="action", tool_called="calculator", tool_output="42"))
            w.commit()

            time.sleep(0.08)
            run.status = "done"
            run.final_report = {"answer": "42", "confidence": "high"}
            w.commit()

        t = threading.Thread(target=simulate_progress)
        t.start()
        events = _parse_events(list(streaming.stream_run_events(run_id)))
        t.join()

        event_types = [e[0] for e in events]
        assert event_types == ["start", "step", "step", "done"]
        assert events[1][1]["tool_called"] is None
        assert events[2][1]["tool_called"] == "calculator"
        assert events[2][1]["tool_output"] == "42"
        assert events[3][1]["status"] == "done"

    def test_stuck_run_times_out(self, test_session_factory):
        run_id = str(uuid.uuid4())
        db = test_session_factory()
        db.add(AgentRunORM(id=run_id, task="t", status="pending"))
        db.commit()

        events = _parse_events(list(streaming.stream_run_events(run_id)))
        event_types = [e[0] for e in events]
        assert event_types == ["start", "error"]
        assert "timed out" in events[-1][1]["message"]

    def test_error_step_flagged_correctly(self, test_session_factory):
        run_id = str(uuid.uuid4())
        db = test_session_factory()
        db.add(AgentRunORM(id=run_id, task="t", status="done", final_report={"answer": "x"}))
        db.add(AgentStepORM(run_id=run_id, step_number=1, phase="action", tool_called="calculator", tool_output="ERROR: bad expr", is_error=True))
        db.commit()

        events = _parse_events(list(streaming.stream_run_events(run_id)))
        step_events = [e for e in events if e[0] == "step"]
        assert len(step_events) == 1
        assert step_events[0][1]["is_error"] is True