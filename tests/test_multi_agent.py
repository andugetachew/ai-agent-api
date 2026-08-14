import sys
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import multi_agent


def _make_response(text: str):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=2),
    )


class TestBuildSpecialistRegistry:
    def test_coding_specialist_gets_scoped_tools(self):
        reg = multi_agent.build_specialist_registry("coding")
        assert set(reg._tools.keys()) == {"execute_code", "read_file", "submit_result"}

    def test_research_specialist_gets_scoped_tools(self):
        reg = multi_agent.build_specialist_registry("research")
        assert set(reg._tools.keys()) == {"web_search", "fetch_url", "read_file", "submit_result"}

    def test_data_specialist_gets_scoped_tools(self):
        reg = multi_agent.build_specialist_registry("data")
        assert set(reg._tools.keys()) == {"calculator", "execute_code", "read_file", "submit_result"}

    def test_rag_specialist_gets_scoped_tools(self):
        reg = multi_agent.build_specialist_registry("rag")
        assert set(reg._tools.keys()) == {"read_file", "web_search", "submit_result"}

    def test_every_specialist_always_gets_submit_result(self):
        for name in multi_agent.SPECIALISTS:
            reg = multi_agent.build_specialist_registry(name)
            assert "submit_result" in reg._tools

    def test_unknown_specialist_falls_back_to_default(self):
        reg = multi_agent.build_specialist_registry("not_a_real_specialist")
        default_reg = multi_agent.build_specialist_registry(multi_agent.DEFAULT_SPECIALIST)
        assert set(reg._tools.keys()) == set(default_reg._tools.keys())

    def test_specialist_tools_are_a_strict_subset_of_full_registry(self):
        from tools import build_default_registry
        full_tool_names = set(build_default_registry()._tools.keys())
        for name in multi_agent.SPECIALISTS:
            reg = multi_agent.build_specialist_registry(name)
            assert set(reg._tools.keys()).issubset(full_tool_names)


class TestSupervisorRouting:
    def test_routes_to_matching_specialist(self):
        fake_provider = MagicMock()
        fake_provider.create.return_value = _make_response("coding")
        supervisor = multi_agent.Supervisor(provider=fake_provider)

        result = supervisor.route("write a python script to sort a list")
        assert result == "coding"

    def test_routes_case_insensitively(self):
        fake_provider = MagicMock()
        fake_provider.create.return_value = _make_response("RESEARCH")
        supervisor = multi_agent.Supervisor(provider=fake_provider)

        result = supervisor.route("find recent news about AI")
        assert result == "research"

    def test_handles_extra_whitespace_and_punctuation(self):
        fake_provider = MagicMock()
        fake_provider.create.return_value = _make_response("  data.  ")
        supervisor = multi_agent.Supervisor(provider=fake_provider)

        result = supervisor.route("calculate the average of these numbers")
        assert result == "data"

    def test_unrecognized_response_falls_back_to_default(self):
        fake_provider = MagicMock()
        fake_provider.create.return_value = _make_response("I'm not sure what to pick here")
        supervisor = multi_agent.Supervisor(provider=fake_provider)

        result = supervisor.route("an ambiguous task")
        assert result == multi_agent.DEFAULT_SPECIALIST

    def test_empty_response_falls_back_to_default(self):
        fake_provider = MagicMock()
        fake_provider.create.return_value = SimpleNamespace(
            content=[], usage=SimpleNamespace(input_tokens=5, output_tokens=0),
        )
        supervisor = multi_agent.Supervisor(provider=fake_provider)

        result = supervisor.route("some task")
        assert result == multi_agent.DEFAULT_SPECIALIST

    def test_route_calls_provider_with_no_tools(self):
        """Routing itself should never expose tools -- it's a pure
        classification call, not an action step."""
        fake_provider = MagicMock()
        fake_provider.create.return_value = _make_response("coding")
        supervisor = multi_agent.Supervisor(provider=fake_provider)

        supervisor.route("write some code")

        _, kwargs = fake_provider.create.call_args
        assert kwargs["tools"] is None

    def test_route_includes_task_in_prompt(self):
        fake_provider = MagicMock()
        fake_provider.create.return_value = _make_response("coding")
        supervisor = multi_agent.Supervisor(provider=fake_provider)

        supervisor.route("a very specific unique task description xyz123")

        _, kwargs = fake_provider.create.call_args
        prompt_content = kwargs["messages"][0]["content"]
        assert "a very specific unique task description xyz123" in prompt_content


class TestTaskFraming:
    def test_frame_task_includes_specialist_name_and_original_task(self):
        fake_provider = MagicMock()
        supervisor = multi_agent.Supervisor(provider=fake_provider)

        framed = supervisor.frame_task("coding", "sort this list")

        assert "coding specialist" in framed
        assert "sort this list" in framed

    def test_frame_task_includes_specialist_description(self):
        fake_provider = MagicMock()
        supervisor = multi_agent.Supervisor(provider=fake_provider)

        framed = supervisor.frame_task("research", "find something")

        assert multi_agent.SPECIALISTS["research"]["description"] in framed

    def test_frame_task_with_unknown_specialist_uses_default_description(self):
        fake_provider = MagicMock()
        supervisor = multi_agent.Supervisor(provider=fake_provider)

        framed = supervisor.frame_task("not_real", "do something")

        default_description = multi_agent.SPECIALISTS[multi_agent.DEFAULT_SPECIALIST]["description"]
        assert default_description in framed
        assert "do something" in framed


class TestSpecialistDescriptions:
    def test_all_specialists_have_description_and_tools(self):
        for name, spec in multi_agent.SPECIALISTS.items():
            assert "description" in spec
            assert isinstance(spec["description"], str) and len(spec["description"]) > 0
            assert "tools" in spec
            assert isinstance(spec["tools"], list) and len(spec["tools"]) > 0

    def test_format_specialist_descriptions_includes_every_specialist(self):
        formatted = multi_agent._format_specialist_descriptions()
        for name in multi_agent.SPECIALISTS:
            assert name in formatted