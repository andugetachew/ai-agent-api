import sys
import os
import uuid

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
load_dotenv()

from db import Base, AgentRunORM, AgentStepORM

# Reads from .env so this test works against whatever Postgres you're using
# (local Docker, Neon, etc.) without editing this file. Use a SEPARATE database
# from your real one so tests never touch production data.
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL not set in .env -- skipping DB persistence tests",
)


@pytest.fixture(scope="module")
def db_engine():
    engine = create_engine(TEST_DATABASE_URL)
    Base.metadata.create_all(bind=engine)  # no-op if tables already exist
    yield engine


@pytest.fixture
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def run_id():
    """A unique id per test so parallel/repeated runs never collide."""
    return f"test-{uuid.uuid4()}"


class TestAgentRunPersistence:
    def test_insert_and_fetch_run(self, db_session, run_id):
        run_row = AgentRunORM(
            id=run_id,
            task="Test task: verify DB persistence",
            status="done",
            final_report={"answer": "42", "details": ["test"], "confidence": "high"},
            total_input_tokens=100,
            total_output_tokens=50,
        )
        db_session.add(run_row)
        db_session.commit()

        fetched = db_session.query(AgentRunORM).filter(AgentRunORM.id == run_id).first()
        assert fetched is not None
        assert fetched.status == "done"
        assert fetched.final_report["answer"] == "42"
        assert fetched.total_input_tokens == 100

        # cleanup
        db_session.delete(fetched)
        db_session.commit()

    def test_insert_run_with_steps_and_relationship(self, db_session, run_id):
        run_row = AgentRunORM(id=run_id, task="Task with steps", status="done")
        db_session.add(run_row)
        db_session.add(AgentStepORM(
            run_id=run_id,
            step_number=1,
            thought="Testing thought",
            tool_called="calculator",
            tool_input={"expression": "40 + 2"},
            tool_output="42",
        ))
        db_session.commit()

        fetched_run = db_session.query(AgentRunORM).filter(AgentRunORM.id == run_id).first()
        assert len(fetched_run.steps) == 1
        assert fetched_run.steps[0].tool_called == "calculator"
        assert fetched_run.steps[0].tool_output == "42"

        # cleanup (steps cascade-delete via relationship config)
        db_session.delete(fetched_run)
        db_session.commit()

    def test_cascade_delete_removes_steps(self, db_session, run_id):
        run_row = AgentRunORM(id=run_id, task="Cascade test", status="done")
        db_session.add(run_row)
        db_session.add(AgentStepORM(run_id=run_id, step_number=1, thought=None,
                                     tool_called=None, tool_input=None, tool_output=None))
        db_session.commit()

        db_session.delete(run_row)
        db_session.commit()

        remaining_steps = db_session.query(AgentStepORM).filter(AgentStepORM.run_id == run_id).all()
        assert remaining_steps == []

    def test_missing_run_returns_none(self, db_session):
        result = db_session.query(AgentRunORM).filter(AgentRunORM.id == "does-not-exist").first()
        assert result is None


class TestListRunsQuery:
    """Verifies the query logic behind GET /v1/agent/runs directly against the DB,
    independent of FastAPI wiring (which needs a running app + TestClient)."""

    def test_filter_by_status_and_count(self, db_session):
        ids = [f"test-list-{uuid.uuid4()}" for _ in range(3)]
        statuses = ["done", "done", "failed"]
        for rid, status in zip(ids, statuses):
            db_session.add(AgentRunORM(id=rid, task="list test", status=status))
        db_session.commit()

        from sqlalchemy import func
        done_count = (
            db_session.query(AgentRunORM)
            .filter(AgentRunORM.status == "done", AgentRunORM.id.in_(ids))
            .with_entities(func.count(AgentRunORM.id))
            .scalar()
        )
        assert done_count == 2

        # cleanup
        db_session.query(AgentRunORM).filter(AgentRunORM.id.in_(ids)).delete(synchronize_session=False)
        db_session.commit()

    def test_pagination_limit_and_offset(self, db_session):
        ids = [f"test-page-{uuid.uuid4()}" for _ in range(5)]
        for rid in ids:
            db_session.add(AgentRunORM(id=rid, task="page test", status="done"))
        db_session.commit()

        page = (
            db_session.query(AgentRunORM)
            .filter(AgentRunORM.id.in_(ids))
            .order_by(AgentRunORM.created_at.desc())
            .limit(2)
            .offset(0)
            .all()
        )
        assert len(page) == 2

        # cleanup
        db_session.query(AgentRunORM).filter(AgentRunORM.id.in_(ids)).delete(synchronize_session=False)
        db_session.commit()