from __future__ import annotations
import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, String, Integer, DateTime, ForeignKey, JSON, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class AgentSessionORM(Base):
    __tablename__ = "agent_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    summary = Column(Text, nullable=True)  # rolling summary of older turns
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    runs = relationship("AgentRunORM", back_populates="session")


class AgentRunORM(Base):
    __tablename__ = "agent_runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("agent_sessions.id"), nullable=True)
    task = Column(Text, nullable=False)
    status = Column(String, nullable=False)
    final_report = Column(JSON, nullable=True)
    plan = Column(Text, nullable=True)
    total_input_tokens = Column(Integer, default=0)
    total_output_tokens = Column(Integer, default=0)
    duration_ms = Column(Integer, default=0)
    pending_action = Column(JSON, nullable=True)
    resume_state = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    finished_at = Column(DateTime, nullable=True)

    steps = relationship("AgentStepORM", back_populates="run", cascade="all, delete-orphan")
    session = relationship("AgentSessionORM", back_populates="runs")
    provider = Column(String, default="anthropic")
    specialist = Column(String, nullable=True)


class AgentStepORM(Base):
    __tablename__ = "agent_steps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("agent_runs.id"), nullable=False)
    step_number = Column(Integer, nullable=False)
    thought = Column(Text, nullable=True)
    phase = Column(String, default="action")
    duration_ms = Column(Integer, default=0)
    is_error = Column(Boolean, default=False)
    tool_called = Column(String, nullable=True)
    tool_input = Column(JSON, nullable=True)
    tool_output = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    run = relationship("AgentRunORM", back_populates="steps")


class UploadedFileORM(Base):
    __tablename__ = "uploaded_files"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    extracted_text = Column(Text, nullable=True)
    size_bytes = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()