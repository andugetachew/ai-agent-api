import sys
import os
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))


@pytest.fixture
def test_client():
    """
    A TestClient wired to an in-memory SQLite DB instead of the real Postgres,
    with init_db() and the Celery task dispatch patched out, and the rate
    limiter pointed at in-memory storage instead of Redis, so tests never
    touch real infrastructure.
    """
    import config
    config.settings.celery_broker_url = "memory://"

    with patch("db.init_db"):
        import main
        from db import Base, get_db

        # StaticPool forces every connection to share the SAME in-memory DB --
        # without it, each new connection gets its own empty in-memory database,
        # so tables created via create_all() would be invisible to later sessions.
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=engine)
        TestSession = sessionmaker(bind=engine)

        def override_get_db():
            db = TestSession()
            try:
                yield db
            finally:
                db.close()

        main.app.dependency_overrides[get_db] = override_get_db
        main.limiter.reset()  # clear counts from prior tests sharing this in-memory limiter

        with patch("tasks.run_agent_task.delay") as mock_delay:
            from fastapi.testclient import TestClient
            client = TestClient(main.app)
            client.mock_delay = mock_delay
            yield client

        main.app.dependency_overrides.clear()


class TestRunEndpoint:
    def test_creates_pending_run(self, test_client):
        resp = test_client.post("/v1/agent/run", json={"task": "test task"})
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "pending"
        assert "run_id" in body
        test_client.mock_delay.assert_called_once()

    def test_rejects_empty_task(self, test_client):
        resp = test_client.post("/v1/agent/run", json={"task": ""})
        assert resp.status_code == 422

    def test_rejects_oversized_task(self, test_client):
        resp = test_client.post("/v1/agent/run", json={"task": "x" * 5000})
        assert resp.status_code == 422

    def test_rejects_missing_task_field(self, test_client):
        resp = test_client.post("/v1/agent/run", json={})
        assert resp.status_code == 422


class TestGetRunEndpoint:
    def test_returns_404_for_missing_run(self, test_client):
        resp = test_client.get("/v1/agent/run/does-not-exist")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_returns_created_run(self, test_client):
        create_resp = test_client.post("/v1/agent/run", json={"task": "fetch me back"})
        run_id = create_resp.json()["run_id"]

        get_resp = test_client.get(f"/v1/agent/run/{run_id}")
        assert get_resp.status_code == 200
        body = get_resp.json()
        assert body["run_id"] == run_id
        assert body["task"] == "fetch me back"
        assert body["status"] == "pending"
        assert body["steps"] == []


class TestListRunsEndpoint:
    def test_lists_created_runs(self, test_client):
        test_client.post("/v1/agent/run", json={"task": "task one"})
        test_client.post("/v1/agent/run", json={"task": "task two"})

        resp = test_client.get("/v1/agent/runs")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 2
        assert len(body["runs"]) >= 2

    def test_rejects_invalid_status_filter(self, test_client):
        resp = test_client.get("/v1/agent/runs?status=not_a_real_status")
        assert resp.status_code == 400

    def test_respects_limit(self, test_client):
        for i in range(5):
            test_client.post("/v1/agent/run", json={"task": f"task {i}"})

        resp = test_client.get("/v1/agent/runs?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()["runs"]) == 2
        
class TestFileUploadEndpoint:
    def test_uploads_txt_file_successfully(self, test_client):
        content = b"This is a test document about revenue figures."
        resp = test_client.post(
            "/v1/files/upload",
            files={"file": ("report.txt", content, "text/plain")},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["filename"] == "report.txt"
        assert body["extracted_char_count"] == len(content)
        assert "revenue figures" in body["preview"]

    def test_rejects_unsupported_file_type(self, test_client):
        resp = test_client.post(
            "/v1/files/upload",
            files={"file": ("virus.exe", b"fake", "application/x-msdownload")},
        )
        assert resp.status_code == 415

    def test_get_uploaded_file_returns_full_text(self, test_client):
        content = b"Full content for retrieval test."
        upload_resp = test_client.post(
            "/v1/files/upload",
            files={"file": ("doc.txt", content, "text/plain")},
        )
        file_id = upload_resp.json()["file_id"]

        get_resp = test_client.get(f"/v1/files/{file_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["extracted_text"] == "Full content for retrieval test."

    def test_get_missing_file_returns_404(self, test_client):
        resp = test_client.get("/v1/files/does-not-exist")
        assert resp.status_code == 404

class TestHealthEndpoint:
    def test_health_degraded_when_dependencies_unreachable(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code in (200, 503)
        assert "database" in resp.json()
        assert "redis" in resp.json()