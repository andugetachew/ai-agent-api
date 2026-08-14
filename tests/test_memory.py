import sys
import os
import time
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from db import Base, AgentRunORM, AgentSessionORM
import memory


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


def _add_done_run(db, session_id, task, answer):
    db.add(AgentRunORM(
        id=str(uuid.uuid4()),
        session_id=session_id,
        task=task,
        status="done",
        final_report={"answer": answer, "details": [], "confidence": "high"},
    ))
    db.commit()
    time.sleep(0.005)  # ensure distinct created_at ordering, matching real request spacing


class TestGetOrCreateSession:
    def test_creates_new_session_when_none_given(self, db_session):
        session = memory.get_or_create_session(db_session, None)
        assert session.id is not None
        assert session.summary is None

    def test_returns_existing_session(self, db_session):
        created = memory.get_or_create_session(db_session, None)
        fetched = memory.get_or_create_session(db_session, created.id)
        assert fetched.id == created.id

    def test_unknown_session_id_creates_new_row_with_that_id(self, db_session):
        custom_id = str(uuid.uuid4())
        session = memory.get_or_create_session(db_session, custom_id)
        assert session.id == custom_id


class TestBuildContextMessages:
    def test_empty_session_returns_no_messages(self, db_session):
        session = memory.get_or_create_session(db_session, None)
        ctx = memory.build_context_messages(db_session, session)
        assert ctx == []

    def test_recent_turns_included_in_chronological_order(self, db_session):
        session = memory.get_or_create_session(db_session, None)
        for i in range(3):
            _add_done_run(db_session, session.id, f"question {i}", f"answer {i}")

        ctx = memory.build_context_messages(db_session, session)
        # 3 turns * 2 messages each (user + assistant), no summary yet
        assert len(ctx) == 6
        assert ctx[0]["content"] == "question 0"
        assert ctx[1]["content"] == "answer 0"
        assert ctx[-2]["content"] == "question 2"
        assert ctx[-1]["content"] == "answer 2"

    def test_only_recent_turns_kept_verbatim(self, db_session):
        session = memory.get_or_create_session(db_session, None)
        for i in range(memory.RECENT_TURNS_KEPT + 3):
            _add_done_run(db_session, session.id, f"question {i}", f"answer {i}")

        ctx = memory.build_context_messages(db_session, session)
        # only the last RECENT_TURNS_KEPT turns come back verbatim (no summary
        # folded yet, since build_context_messages doesn't summarize itself)
        assert len(ctx) == memory.RECENT_TURNS_KEPT * 2

    def test_summary_prepended_when_present(self, db_session):
        session = memory.get_or_create_session(db_session, None)
        session.summary = "earlier discussion summary"
        db_session.commit()

        ctx = memory.build_context_messages(db_session, session)
        assert ctx[0]["role"] == "user"
        assert "earlier discussion summary" in ctx[0]["content"]
        assert ctx[1]["role"] == "assistant"


class TestUpdateSessionMemory:
    def test_no_summarization_when_within_limit(self, db_session):
        session = memory.get_or_create_session(db_session, None)
        for i in range(memory.RECENT_TURNS_KEPT):
            _add_done_run(db_session, session.id, f"q{i}", f"a{i}")

        memory.update_session_memory(db_session, session.id, "latest", {"answer": "latest answer"})
        db_session.refresh(session)
        assert session.summary is None

    def test_summarizes_oldest_excess_turns(self, db_session):
        session = memory.get_or_create_session(db_session, None)
        for i in range(memory.RECENT_TURNS_KEPT + 2):
            _add_done_run(db_session, session.id, f"q{i}", f"a{i}")

        memory.update_session_memory(db_session, session.id, "latest", {"answer": "latest"})
        db_session.refresh(session)

        assert session.summary is not None
        assert "q0" in session.summary
        assert "q1" in session.summary
        assert "q2" not in session.summary  # kept verbatim, not summarized

    def test_missing_session_is_a_no_op(self, db_session):
        # should not raise even though the session doesn't exist
        memory.update_session_memory(db_session, "does-not-exist", "task", {"answer": "x"})

    def test_full_cycle_keeps_context_bounded(self, db_session):
        """After many turns, context sent to the agent should stay bounded --
        this is the actual point of summarization."""
        session = memory.get_or_create_session(db_session, None)
        for i in range(15):
            _add_done_run(db_session, session.id, f"q{i}", f"a{i}")
            memory.update_session_memory(db_session, session.id, f"q{i}", {"answer": f"a{i}"})

        ctx = memory.build_context_messages(db_session, session)
        # summary (2 msgs) + RECENT_TURNS_KEPT recent turns (2 msgs each)
        assert len(ctx) == 2 + (memory.RECENT_TURNS_KEPT * 2)