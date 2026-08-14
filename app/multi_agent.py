from __future__ import annotations

from providers import ModelProvider
from tools import ToolRegistry, build_default_registry

# Each specialist gets a scoped subset of tools plus submit_result, and a
# short description used both for routing and to frame the task. Not
# separate agent classes -- same Agent/loop, different tool scope + framing.
SPECIALISTS: dict[str, dict] = {
    "research": {
        "description": "Finds information via web search and reading URLs or documents.",
        "tools": ["web_search", "fetch_url", "read_file"],
    },
    "coding": {
        "description": "Writes and runs Python code to compute or transform something.",
        "tools": ["execute_code", "read_file"],
    },
    "data": {
        "description": "Performs calculations, aggregations, or numeric analysis.",
        "tools": ["calculator", "execute_code", "read_file"],
    },
    "rag": {
        "description": "Answers questions by reading previously uploaded documents.",
        "tools": ["read_file", "web_search"],
    },
}

DEFAULT_SPECIALIST = "research"

ROUTING_PROMPT_TEMPLATE = """You are a routing supervisor. Given the task below, pick exactly \
one specialist from this list to handle it, and respond with ONLY that specialist's name \
(nothing else -- no punctuation, no explanation):

{specialist_descriptions}

Task: {task}"""


def build_specialist_registry(specialist_name: str) -> ToolRegistry:
    """Builds a ToolRegistry scoped to one specialist's allowed tools, plus
    submit_result (every specialist needs a way to finish)."""
    full_registry = build_default_registry()
    spec = SPECIALISTS.get(specialist_name, SPECIALISTS[DEFAULT_SPECIALIST])
    allowed = set(spec["tools"]) | {"submit_result"}

    scoped = ToolRegistry()
    for tool_name in allowed:
        tool = full_registry.get(tool_name)
        if tool is not None:
            scoped.register(tool)
    return scoped


def _format_specialist_descriptions() -> str:
    return "\n".join(f"- {name}: {spec['description']}" for name, spec in SPECIALISTS.items())


class Supervisor:
    """Routes a task to the best-fit specialist by asking the model directly
    (reuses the same provider abstraction as Agent, so this works with any
    of Anthropic/OpenAI/Ollama), then builds a specialist-scoped registry
    for the actual run to use."""

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    def route(self, task: str) -> str:
        prompt = ROUTING_PROMPT_TEMPLATE.format(
            specialist_descriptions=_format_specialist_descriptions(),
            task=task,
        )
        response = self.provider.create(
            system="You are a precise routing function. Respond with only a specialist name.",
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            max_tokens=20,
        )
        text_blocks = [b.text for b in response.content if b.type == "text"]
        raw_answer = " ".join(text_blocks).strip().lower() if text_blocks else ""

        for name in SPECIALISTS:
            if name in raw_answer:
                return name
        return DEFAULT_SPECIALIST

    def frame_task(self, specialist_name: str, task: str) -> str:
        """Prepends a short role framing to the task so the underlying Agent
        (which uses a generic system prompt) still gets specialist context."""
        spec = SPECIALISTS.get(specialist_name, SPECIALISTS[DEFAULT_SPECIALIST])
        return f"[Acting as the {specialist_name} specialist: {spec['description']}]\n\n{task}"