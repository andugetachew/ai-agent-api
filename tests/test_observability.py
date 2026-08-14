import sys
import os
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from db import Base, AgentRunORM, AgentStepORM
import observability


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestEstimateCost:
    def test_zero_tokens_costs_nothing(self):
        assert observability.estimate_cost_usd(0, 0) == 0.0

    def test_known_token_counts_produce_expected_cost(self):
        # 1,000,000 input tokens @ $3/M + 1,000,000 output tokens @ $15/M = $18
        cost = observability.estimate_cost_usd(1_000_000, 1_000_000)
        assert cost == 18.0

    def test_output_tokens_cost_more_than_input(self):
        input_only = observability.estimate_cost_usd(1_000_000, 0)
        output_only = observability.estimate_cost_usd(0, 1_000_000)
        assert output_only > input_only


class TestBuildRunObservability:
    def test_summarizes_single_run_correctly(self, db_session):
        run_id = str(uuid.uuid4())
        run = AgentRunORM(
            id=run_id, task="t", status="done", duration_ms=2500,
            total_input_tokens=1000, total_output_tokens=200,
        )
        db_session.add(run)
        db_session.add(AgentStepORM(run_id=run_id, step_number=0, phase="planning", duration_ms=800, is_error=False))
        db_session.add(AgentStepORM(run_id=run_id, step_number=1, phase="action", duration_ms=1200, tool_called="calculator", is_error=False))
        db_session.add(AgentStepORM(run_id=run_id, step_number=2, phase="action", duration_ms=500, tool_called="submit_result", is_error=False))
        db_session.commit()
        db_session.refresh(run)

        obs = observability.build_run_observability(run)

        assert obs["run_id"] == run_id
        assert obs["status"] == "done"
        assert obs["duration_ms"] == 2500
        assert obs["step_count"] == 3
        assert obs["phase_counts"] == {"planning": 1, "action": 2, "reflection": 0}
        assert obs["tool_usage"] == {"calculator": 1, "submit_result": 1}
        assert obs["tool_errors"] == {}
        assert obs["estimated_cost_usd"] > 0

    def test_tracks_tool_errors_separately_from_usage(self, db_session):
        run_id = str(uuid.uuid4())
        run = AgentRunORM(id=run_id, task="t", status="failed", duration_ms=1000)
        db_session.add(run)
        db_session.add(AgentStepORM(run_id=run_id, step_number=1, phase="action", tool_called="calculator", is_error=True))
        db_session.add(AgentStepORM(run_id=run_id, step_number=2, phase="action", tool_called="calculator", is_error=False))
        db_session.commit()
        db_session.refresh(run)

        obs = observability.build_run_observability(run)

        assert obs["tool_usage"]["calculator"] == 2
        assert obs["tool_errors"]["calculator"] == 1  # only 1 of the 2 calls errored

    def test_run_with_no_steps_returns_empty_breakdown(self, db_session):
        run_id = str(uuid.uuid4())
        run = AgentRunORM(id=run_id, task="t", status="pending")
        db_session.add(run)
        db_session.commit()
        db_session.refresh(run)

        obs = observability.build_run_observability(run)

        assert obs["step_count"] == 0
        assert obs["tool_usage"] == {}
        assert obs["phase_counts"] == {"planning": 0, "action": 0, "reflection": 0}


class TestBuildAggregateObservability:
    def test_aggregates_across_multiple_runs(self, db_session):
        run1_id, run2_id = str(uuid.uuid4()), str(uuid.uuid4())
        db_session.add(AgentRunORM(id=run1_id, task="t1", status="done", duration_ms=2000, total_input_tokens=1000, total_output_tokens=200))
        db_session.add(AgentRunORM(id=run2_id, task="t2", status="failed", duration_ms=1000, total_input_tokens=500, total_output_tokens=100))
        db_session.add(AgentStepORM(run_id=run1_id, step_number=1, tool_called="calculator", is_error=False))
        db_session.add(AgentStepORM(run_id=run2_id, step_number=1, tool_called="calculator", is_error=True))
        db_session.commit()

        agg = observability.build_aggregate_observability(db_session)

        assert agg["total_runs"] == 2
        assert agg["status_counts"] == {"done": 1, "failed": 1}
        assert agg["success_rate"] == 0.5
        assert agg["total_input_tokens"] == 1500
        assert agg["total_output_tokens"] == 300
        assert agg["average_duration_ms"] == 1500.0
        assert agg["tool_usage"]["calculator"] == 2
        assert agg["tool_errors"]["calculator"] == 1

    def test_empty_database_returns_sane_defaults(self, db_session):
        agg = observability.build_aggregate_observability(db_session)

        assert agg["total_runs"] == 0
        assert agg["success_rate"] is None
        assert agg["average_duration_ms"] is None
        assert agg["tool_usage"] == {}