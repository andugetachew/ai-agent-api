from __future__ import annotations
import json
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any


def _make_block(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def _make_response(content: list, input_tokens: int, output_tokens: int) -> SimpleNamespace:
    """Builds a response object shaped exactly like the Anthropic SDK's
    Message response: .content is a list of blocks with .type (+ .text or
    .name/.input/.id), and .usage.input_tokens/.output_tokens. Agent.py is
    written against this exact shape, so every provider normalizes to it --
    that's what lets Agent stay provider-agnostic without a rewrite."""
    return SimpleNamespace(
        content=content,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class ModelProvider(ABC):
    """Common interface every provider implements. The model name is fixed
    at construction time (one provider instance = one model), so create()
    only takes the per-call arguments. Must return a response shaped like
    _make_response() above, regardless of the underlying API's native shape."""

    @abstractmethod
    def create(self, system: str, messages: list[dict], tools: list[dict] | None, max_tokens: int) -> SimpleNamespace:
        ...


class AnthropicProvider(ModelProvider):
    """Thin pass-through -- the Anthropic SDK's response already matches the
    normalized shape Agent expects, so no translation is needed here."""

    def __init__(self, client, model: str) -> None:
        self.client = client
        self.model = model

    def create(self, system, messages, tools, max_tokens) -> SimpleNamespace:
        return self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            tools=tools or [],
            messages=messages,
        )


def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Anthropic tool schema: {name, description, input_schema}
    OpenAI tool schema: {type: "function", function: {name, description, parameters}}"""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _anthropic_messages_to_openai(system: str, messages: list[dict]) -> list[dict]:
    """
    Translates Anthropic-shaped message history (with tool_use/tool_result
    content blocks) into OpenAI's chat.completions message format (system
    role message, assistant tool_calls, tool-role responses).
    """
    openai_messages: list[dict] = [{"role": "system", "content": system}]

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue

        # content is a list of blocks (text / tool_use / tool_result)
        if role == "assistant":
            text_parts = [b["text"] for b in content if b.get("type") == "text"]
            tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]
            entry: dict[str, Any] = {"role": "assistant"}
            entry["content"] = " ".join(text_parts) if text_parts else None
            if tool_use_blocks:
                entry["tool_calls"] = [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                    }
                    for b in tool_use_blocks
                ]
            openai_messages.append(entry)
        else:  # role == "user" -- may contain tool_result blocks
            tool_result_blocks = [b for b in content if b.get("type") == "tool_result"]
            if tool_result_blocks:
                for b in tool_result_blocks:
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": b["tool_use_id"],
                        "content": str(b["content"]),
                    })
            else:
                text_parts = [b.get("text", "") for b in content]
                openai_messages.append({"role": "user", "content": " ".join(text_parts)})

    return openai_messages


class OpenAIProvider(ModelProvider):
    """Translates Anthropic-shaped requests/responses to/from OpenAI's
    chat.completions + function-calling API."""

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str | None = None) -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def create(self, system, messages, tools, max_tokens) -> SimpleNamespace:
        openai_messages = _anthropic_messages_to_openai(system, messages)
        openai_tools = _anthropic_tools_to_openai(tools) if tools else None

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        blocks = []
        if message.content:
            blocks.append(_make_block(type="text", text=message.content))
        if message.tool_calls:
            for tc in message.tool_calls:
                blocks.append(_make_block(
                    type="tool_use",
                    id=tc.id,
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments),
                ))

        usage = response.usage
        return _make_response(
            blocks,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


class OllamaProvider(OpenAIProvider):
    """Ollama exposes an OpenAI-compatible endpoint locally, so this reuses
    OpenAIProvider entirely -- just pointed at a local base_url with a dummy
    API key (Ollama doesn't check it, but the OpenAI client requires one)."""

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434/v1") -> None:
        super().__init__(api_key="ollama", model=model, base_url=base_url)


def build_provider(
    provider_name: str,
    model: str,
    anthropic_client=None,
    openai_api_key: str | None = None,
    ollama_base_url: str = "http://localhost:11434/v1",
) -> ModelProvider:
    """Factory: picks the right provider by name. Defaults to Anthropic if
    the name is unrecognized, so a typo in a request doesn't hard-fail."""
    if provider_name == "openai":
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
        return OpenAIProvider(api_key=openai_api_key, model=model)
    if provider_name == "ollama":
        return OllamaProvider(model=model, base_url=ollama_base_url)
    return AnthropicProvider(client=anthropic_client, model=model)