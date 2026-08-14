import sys
import os
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import providers


class TestAnthropicProvider:
    def test_passes_through_response_unchanged(self):
        fake_client = MagicMock()
        fake_client.messages.create.return_value = "raw_response"
        provider = providers.AnthropicProvider(client=fake_client, model="claude-sonnet-4-6")

        result = provider.create(system="sys", messages=[], tools=[], max_tokens=100)

        assert result == "raw_response"
        fake_client.messages.create.assert_called_once_with(
            model="claude-sonnet-4-6", max_tokens=100, system="sys", tools=[], messages=[],
        )

    def test_none_tools_becomes_empty_list(self):
        fake_client = MagicMock()
        provider = providers.AnthropicProvider(client=fake_client, model="claude-sonnet-4-6")
        provider.create(system="sys", messages=[], tools=None, max_tokens=100)
        _, kwargs = fake_client.messages.create.call_args
        assert kwargs["tools"] == []


class TestToolSchemaTranslation:
    def test_converts_anthropic_schema_to_openai_format(self):
        anthropic_tools = [{
            "name": "calculator",
            "description": "does math",
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        }]
        result = providers._anthropic_tools_to_openai(anthropic_tools)

        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "calculator"
        assert result[0]["function"]["description"] == "does math"
        assert result[0]["function"]["parameters"]["properties"]["expression"]["type"] == "string"

    def test_handles_multiple_tools(self):
        tools = [
            {"name": "a", "description": "d1", "input_schema": {"type": "object", "properties": {}}},
            {"name": "b", "description": "d2", "input_schema": {"type": "object", "properties": {}}},
        ]
        result = providers._anthropic_tools_to_openai(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"


class TestMessageHistoryTranslation:
    def test_simple_text_messages_pass_through(self):
        messages = [{"role": "user", "content": "hello"}]
        result = providers._anthropic_messages_to_openai("system prompt", messages)
        assert result[0] == {"role": "system", "content": "system prompt"}
        assert result[1] == {"role": "user", "content": "hello"}

    def test_assistant_tool_use_becomes_tool_calls(self):
        messages = [
            {"role": "user", "content": "what is 2+2"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "computing"},
                {"type": "tool_use", "id": "t1", "name": "calculator", "input": {"expression": "2+2"}},
            ]},
        ]
        result = providers._anthropic_messages_to_openai("sys", messages)
        assistant_msg = result[2]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "computing"
        assert assistant_msg["tool_calls"][0]["id"] == "t1"
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "calculator"
        assert json.loads(assistant_msg["tool_calls"][0]["function"]["arguments"]) == {"expression": "2+2"}

    def test_tool_result_becomes_tool_role_message(self):
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "4"},
            ]},
        ]
        result = providers._anthropic_messages_to_openai("sys", messages)
        tool_msg = result[1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "t1"
        assert tool_msg["content"] == "4"

    def test_assistant_message_with_only_tool_use_has_null_content(self):
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "calculator", "input": {}},
            ]},
        ]
        result = providers._anthropic_messages_to_openai("sys", messages)
        assert result[1]["content"] is None

    def test_full_round_trip_conversation(self):
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me compute that."},
                {"type": "tool_use", "id": "tool_1", "name": "calculator", "input": {"expression": "2+2"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tool_1", "content": "4"},
            ]},
        ]
        result = providers._anthropic_messages_to_openai("You are an agent.", messages)

        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[2]["tool_calls"][0]["id"] == "tool_1"
        assert result[3]["role"] == "tool"
        assert result[3]["tool_call_id"] == "tool_1"
        assert result[3]["content"] == "4"


class TestOpenAIProvider:
    def test_normalizes_tool_call_response(self):
        fake_client = MagicMock()
        fake_message = SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                id="call_abc",
                function=SimpleNamespace(name="calculator", arguments=json.dumps({"expression": "2+2"})),
            )],
        )
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=fake_message)],
            usage=SimpleNamespace(prompt_tokens=42, completion_tokens=17),
        )
        fake_client.chat.completions.create.return_value = fake_response

        with patch("openai.OpenAI", return_value=fake_client):
            provider = providers.OpenAIProvider(api_key="fake-key", model="gpt-4o")

        result = provider.create(
            system="sys",
            messages=[{"role": "user", "content": "what is 2+2"}],
            tools=[{"name": "calculator", "description": "math", "input_schema": {"type": "object", "properties": {}}}],
            max_tokens=1000,
        )

        assert result.content[0].type == "tool_use"
        assert result.content[0].name == "calculator"
        assert result.content[0].input == {"expression": "2+2"}
        assert result.content[0].id == "call_abc"
        assert result.usage.input_tokens == 42
        assert result.usage.output_tokens == 17

    def test_normalizes_text_only_response(self):
        fake_client = MagicMock()
        fake_message = SimpleNamespace(content="The answer is 4.", tool_calls=None)
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=fake_message)],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        fake_client.chat.completions.create.return_value = fake_response

        with patch("openai.OpenAI", return_value=fake_client):
            provider = providers.OpenAIProvider(api_key="fake-key", model="gpt-4o")

        result = provider.create(system="sys", messages=[{"role": "user", "content": "hi"}], tools=None, max_tokens=100)

        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "The answer is 4."

    def test_missing_usage_defaults_to_zero(self):
        fake_client = MagicMock()
        fake_message = SimpleNamespace(content="hi", tool_calls=None)
        fake_response = SimpleNamespace(choices=[SimpleNamespace(message=fake_message)], usage=None)
        fake_client.chat.completions.create.return_value = fake_response

        with patch("openai.OpenAI", return_value=fake_client):
            provider = providers.OpenAIProvider(api_key="fake-key", model="gpt-4o")

        result = provider.create(system="sys", messages=[], tools=None, max_tokens=100)
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0


class TestOllamaProvider:
    def test_uses_local_base_url_by_default(self):
        with patch("openai.OpenAI") as MockOpenAI:
            providers.OllamaProvider(model="llama3.1")
            _, kwargs = MockOpenAI.call_args
            assert kwargs["base_url"] == "http://localhost:11434/v1"
            assert kwargs["api_key"] == "ollama"

    def test_accepts_custom_base_url(self):
        with patch("openai.OpenAI") as MockOpenAI:
            providers.OllamaProvider(model="llama3.1", base_url="http://custom-host:11434/v1")
            _, kwargs = MockOpenAI.call_args
            assert kwargs["base_url"] == "http://custom-host:11434/v1"


class TestBuildProvider:
    def test_defaults_to_anthropic_for_unknown_name(self):
        fake_client = MagicMock()
        provider = providers.build_provider("nonsense", model="claude-sonnet-4-6", anthropic_client=fake_client)
        assert isinstance(provider, providers.AnthropicProvider)

    def test_builds_anthropic_provider(self):
        fake_client = MagicMock()
        provider = providers.build_provider("anthropic", model="claude-sonnet-4-6", anthropic_client=fake_client)
        assert isinstance(provider, providers.AnthropicProvider)

    def test_builds_openai_provider(self):
        with patch("openai.OpenAI"):
            provider = providers.build_provider("openai", model="gpt-4o", openai_api_key="fake-key")
        assert isinstance(provider, providers.OpenAIProvider)

    def test_openai_without_key_raises(self):
        try:
            providers.build_provider("openai", model="gpt-4o", openai_api_key=None)
            assert False, "should have raised ValueError"
        except ValueError as exc:
            assert "OPENAI_API_KEY" in str(exc)

    def test_builds_ollama_provider(self):
        with patch("openai.OpenAI"):
            provider = providers.build_provider("ollama", model="llama3.1")
        assert isinstance(provider, providers.OllamaProvider)