from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import ast
import contextlib
import io
import httpx

from config import settings

_SERPER_URL = "https://google.serper.dev/search"


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., str]  # returns a string result to feed back to the model

    def to_anthropic_format(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def anthropic_tools(self) -> list[dict[str, Any]]:
        return [t.to_anthropic_format() for t in self._tools.values()]

    def run(self, name: str, tool_input: dict[str, Any]) -> str:
        tool = self.get(name)
        if tool is None:
            return f"ERROR: unknown tool '{name}'"
        try:
            return tool.handler(**tool_input)
        except Exception as exc:  # tool failures must not crash the agent loop
            return f"ERROR: tool '{name}' failed: {exc}"


# ---- Tool implementations ----

def web_search_handler(query: str) -> str:
    if not settings.serper_api_key:
        return "ERROR: web search is not configured (missing SERPER_API_KEY)"
    try:
        resp = httpx.post(
            _SERPER_URL,
            headers={
                "X-API-KEY": settings.serper_api_key,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": 5},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"ERROR: web search failed: {exc}"

    results = data.get("organic", [])
    if not results:
        return f"No search results found for: {query}"

    lines = [f"Search results for: {query}"]
    for i, item in enumerate(results, start=1):
        title = item.get("title", "Untitled")
        url = item.get("link", "")
        snippet = (item.get("snippet", "") or "")[:300]
        lines.append(f"{i}. {title} ({url})\n   {snippet}")
    return "\n".join(lines)


def url_fetch_handler(url: str) -> str:
    try:
        resp = httpx.get(url, timeout=10.0, follow_redirects=True)
        resp.raise_for_status()
        text = resp.text
        # crude truncation so one page can't blow the context budget
        return text[:5000]
    except httpx.HTTPError as exc:
        return f"ERROR fetching {url}: {exc}"


# Only these node types are allowed inside calculator expressions.
_ALLOWED_AST_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd,
)


def calculator_handler(expression: str) -> str:
    """Safely evaluate a numeric expression without exec/eval on raw input."""
    try:
        parsed = ast.parse(expression, mode="eval")
        for node in ast.walk(parsed):
            if not isinstance(node, _ALLOWED_AST_NODES):
                return f"ERROR: disallowed expression element: {type(node).__name__}"
        result = eval(compile(parsed, "<calculator>", "eval"))
        return str(result)
    except Exception as exc:
        return f"ERROR: could not evaluate expression: {exc}"

def code_executor_handler(code: str, timeout_seconds: float = 5.0) -> str:
    """
    Execute a short Python snippet in a restricted namespace and capture printed
    output. Enforces a soft timeout using a worker thread: if execution exceeds
    timeout_seconds, an error is returned immediately.

    IMPORTANT: output capture is done via a custom print() injected into the
    restricted builtins, NOT via contextlib.redirect_stdout -- redirect_stdout
    swaps sys.stdout process-globally, which is unsafe here: if the thread times
    out without finishing, it never exits the `with` block and sys.stdout stays
    hijacked for the entire process afterward, silently swallowing all future
    output. A dedicated print() avoids touching global state entirely.

    Note this does not forcibly kill the thread (Python threads cannot be
    hard-killed), so a genuinely stuck snippet keeps consuming a worker thread
    in the background -- acceptable for a portfolio demo, but for real
    production use, run this in a subprocess or container with OS-level
    resource limits instead.
    """
    import threading

    output_lines: list[str] = []

    def _sandboxed_print(*args, **kwargs) -> None:
        sep = kwargs.get("sep", " ")
        output_lines.append(sep.join(str(a) for a in args))

    restricted_globals = {"__builtins__": {
        "print": _sandboxed_print, "range": range, "len": len, "sum": sum,
        "min": min, "max": max, "abs": abs, "round": round,
        "str": str, "int": int, "float": float, "list": list,
        "dict": dict, "set": set, "tuple": tuple, "enumerate": enumerate,
        "sorted": sorted, "zip": zip,
    }}
    result_holder: dict[str, str] = {}

    def _run() -> None:
        try:
            exec(compile(code, "<agent_code>", "exec"), restricted_globals, {})
            output = "\n".join(output_lines)
            result_holder["output"] = output if output else "(code ran with no printed output)"
        except Exception as exc:
            result_holder["output"] = f"ERROR: code execution failed: {exc}"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        return f"ERROR: code execution exceeded {timeout_seconds}s timeout"
    return result_holder.get("output", "ERROR: code execution failed with no output")

def read_file_handler(file_id: str) -> str:
    from db import SessionLocal, UploadedFileORM

    db = SessionLocal()
    try:
        file_row = db.query(UploadedFileORM).filter(UploadedFileORM.id == file_id).first()
        if file_row is None:
            return f"ERROR: no uploaded file found with id '{file_id}'"
        if not file_row.extracted_text:
            return f"ERROR: file '{file_row.filename}' has no extracted text available"
        return f"Content of '{file_row.filename}':\n\n{file_row.extracted_text}"
    finally:
        db.close()

def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="web_search",
            description="Search the web for a query and return a summary of top results.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=web_search_handler,
        )
    )
    registry.register(
        Tool(
            name="fetch_url",
            description="Fetch the raw text content of a given URL.",
            input_schema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            handler=url_fetch_handler,
        )
    )
    registry.register(
        Tool(
            name="calculator",
            description="Evaluate a numeric arithmetic expression, e.g. '(3 + 5) * 2 / 4'.",
            input_schema={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            handler=calculator_handler,
        )
    )
    registry.register(
        Tool(
            name="execute_code",
            description=(
                "Execute a short Python snippet for computation or data processing. "
                "Use print() to return results. No file, network, or system access is available."
            ),
            input_schema={
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
            handler=code_executor_handler,
        )
    )
    registry.register(
        Tool(
            name="read_file",
            description="Read the extracted text content of a previously uploaded file, given its file_id.",
            input_schema={
                "type": "object",
                "properties": {"file_id": {"type": "string"}},
                "required": ["file_id"],
            },
            handler=read_file_handler,
        )
    )
    registry.register(
        Tool(
            name="submit_result",
            description=(
                "Submit the final result of the task. Call this as the last step "
                "once the task is complete or you have gathered enough information."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer or deliverable for the task.",
                    },
                    "details": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Supporting findings, steps, or reasoning.",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "URLs or references used, if any.",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
                "required": ["answer", "details", "confidence"],
            },
            handler=lambda **kwargs: "RESULT_SUBMITTED",
        )
    )
    return registry