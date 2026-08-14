import sys
import os
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from tools import build_default_registry  # noqa: E402


@pytest.fixture
def registry():
    """A tool registry with web_search stubbed so tests never hit the network."""
    reg = build_default_registry()
    reg.get("web_search").handler = lambda query: f"Stub result for: {query}"
    return reg


@pytest.fixture
def make_tool_use_block():
    def _make(name, tool_input, block_id="tool_1"):
        return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)
    return _make


@pytest.fixture
def make_text_block():
    def _make(text):
        return SimpleNamespace(type="text", text=text)
    return _make


@pytest.fixture
def make_response():
    def _make(content_blocks, in_tokens=50, out_tokens=30):
        return SimpleNamespace(
            content=content_blocks,
            usage=SimpleNamespace(input_tokens=in_tokens, output_tokens=out_tokens),
        )
    return _make


@pytest.fixture
def fake_client():
    return MagicMock()