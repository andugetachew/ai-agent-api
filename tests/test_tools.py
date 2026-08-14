import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from tools import calculator_handler, code_executor_handler, ToolRegistry, Tool, web_search_handler, url_fetch_handler

from unittest.mock import patch, MagicMock
class TestCalculator:
    def test_basic_arithmetic(self):
        assert calculator_handler("2 + 2") == "4"

    def test_operator_precedence(self):
        assert calculator_handler("(245 * 12) + 897") == "3837"

    def test_division(self):
        assert calculator_handler("10 / 4") == "2.5"

    def test_rejects_function_calls(self):
        result = calculator_handler("__import__('os').system('echo hacked')")
        assert result.startswith("ERROR")

    def test_rejects_attribute_access(self):
        result = calculator_handler("().__class__")
        assert result.startswith("ERROR")

    def test_invalid_expression_returns_error(self):
        result = calculator_handler("2 +")
        assert result.startswith("ERROR")


class TestCodeExecutor:
    def test_print_output_captured(self):
        result = code_executor_handler("print(sum(i**2 for i in range(1, 11)))")
        assert result.strip() == "385"

    def test_no_output_message(self):
        result = code_executor_handler("x = 5")
        assert "no printed output" in result

    def test_restricted_builtins_blocks_file_access(self):
        result = code_executor_handler("open('/etc/passwd')")
        assert result.startswith("ERROR")

    def test_restricted_builtins_blocks_import(self):
        result = code_executor_handler("import os")
        assert result.startswith("ERROR")

    def test_syntax_error_returns_error_not_crash(self):
        result = code_executor_handler("this is not valid python (")
        assert result.startswith("ERROR")

    def test_timeout_is_enforced(self):
        result = code_executor_handler(
            "total = 0\nfor i in range(10**9):\n    total += i\nprint(total)",
            timeout_seconds=0.3,
        )
        assert "timeout" in result.lower()

    def test_global_stdout_unaffected_after_timeout(self, capsys):
        code_executor_handler(
            "total = 0\nfor i in range(10**9):\n    total += i\nprint(total)",
            timeout_seconds=0.2,
        )
        print("this should print normally")
        captured = capsys.readouterr()
        assert "this should print normally" in captured.out


class TestToolRegistry:
    def test_unknown_tool_returns_error(self):
        reg = ToolRegistry()
        result = reg.run("nonexistent_tool", {})
        assert "unknown tool" in result

    def test_tool_exception_does_not_propagate(self):
        reg = ToolRegistry()

        def broken_handler(**kwargs):
            raise ValueError("simulated failure")

        reg.register(Tool(
            name="broken",
            description="test",
            input_schema={"type": "object", "properties": {}},
            handler=broken_handler,
        ))
        result = reg.run("broken", {})
        assert result.startswith("ERROR")
        assert "simulated failure" in result
class TestWebSearch:
    def test_returns_error_when_no_key_configured(self):
        with patch("tools.settings") as mock_settings:
            mock_settings.serper_api_key = None
            result = web_search_handler("test query")
            assert result.startswith("ERROR")
            assert "not configured" in result

    def test_formats_results_correctly(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "organic": [
                {"title": "Result One", "link": "https://example.com/1", "snippet": "First snippet"},
                {"title": "Result Two", "link": "https://example.com/2", "snippet": "Second snippet"},
            ]
        }
        mock_response.raise_for_status.return_value = None

        with patch("tools.settings") as mock_settings, patch("tools.httpx.post", return_value=mock_response) as mock_post:
            mock_settings.serper_api_key = "fake-key"
            result = web_search_handler("test query")

        assert "Result One" in result
        assert "https://example.com/1" in result
        assert "First snippet" in result
        assert "Result Two" in result
        mock_post.assert_called_once()

    def test_no_results_found(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"organic": []}
        mock_response.raise_for_status.return_value = None

        with patch("tools.settings") as mock_settings, patch("tools.httpx.post", return_value=mock_response):
            mock_settings.serper_api_key = "fake-key"
            result = web_search_handler("obscure query with no results")

        assert "No search results found" in result

    def test_http_error_returns_error_string(self):
        with patch("tools.settings") as mock_settings, patch("tools.httpx.post", side_effect=Exception("connection failed")):
            mock_settings.serper_api_key = "fake-key"
            result = web_search_handler("test query")

        assert result.startswith("ERROR")
        assert "connection failed" in result


class TestUrlFetch:
    def test_successful_fetch_returns_truncated_text(self):
        mock_response = MagicMock()
        mock_response.text = "x" * 10000
        mock_response.raise_for_status.return_value = None

        with patch("tools.httpx.get", return_value=mock_response):
            result = url_fetch_handler("https://example.com")

        assert len(result) == 5000

    def test_http_error_returns_error_string(self):
        import httpx as real_httpx

        with patch("tools.httpx.get", side_effect=real_httpx.HTTPError("404 not found")):
            result = url_fetch_handler("https://example.com/missing")

        assert result.startswith("ERROR fetching")

class TestReadFile:
    def test_reads_existing_uploaded_file(self, monkeypatch):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool
        from db import Base, UploadedFileORM
        import db as db_module
        import uuid

        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(bind=engine)
        TestSession = sessionmaker(bind=engine)
        monkeypatch.setattr(db_module, "SessionLocal", TestSession)

        file_id = str(uuid.uuid4())
        s = TestSession()
        s.add(UploadedFileORM(id=file_id, filename="notes.txt", content_type="text/plain", extracted_text="Important notes here."))
        s.commit()

        from tools import read_file_handler
        result = read_file_handler(file_id)
        assert "notes.txt" in result
        assert "Important notes here." in result

    def test_missing_file_returns_error(self, monkeypatch):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool
        from db import Base
        import db as db_module

        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(bind=engine)
        TestSession = sessionmaker(bind=engine)
        monkeypatch.setattr(db_module, "SessionLocal", TestSession)

        from tools import read_file_handler
        result = read_file_handler("does-not-exist")
        assert result.startswith("ERROR")
    