"""Unit tests for the runner, registry, and interceptors.

These tests mock the Anthropic API so they run without credentials.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

from agent_runner.runner import AgentRunner, RunResult
from agent_runner.tools.registry import ToolRegistry
from agent_runner.tools.builtins import register_builtins
from agent_runner.interceptors.base import InterceptAction, Interceptor
from agent_runner.interceptors.logging import LoggingInterceptor
from agent_runner.interceptors.shadow import ShadowInterceptor


# ---- helpers ----

def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name: str, input_: dict, id_: str = "tool_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=id_)


def _api_response(content: list, stop_reason: str = "end_turn", input_tokens: int = 100, output_tokens: int = 50):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


# ---- ToolRegistry tests ----

class TestToolRegistry:
    def test_register_and_get(self):
        r = ToolRegistry()
        r.register("echo", "Echo input", {"type": "object", "properties": {}}, lambda p: p)
        assert r.get("echo") is not None
        assert r.get("missing") is None

    def test_decorator_registration(self):
        r = ToolRegistry()

        @r.tool(name="greet", description="Greet", input_schema={"type": "object", "properties": {}})
        def greet(params):
            return f"Hello {params.get('name', 'world')}"

        assert "greet" in r.list_names()
        assert r.get("greet")({"name": "Alice"}) == "Hello Alice"

    def test_to_api_schema(self):
        r = ToolRegistry()
        r.register("t1", "Tool 1", {"type": "object"}, lambda p: None)
        schema = r.to_api_schema()
        assert len(schema) == 1
        assert schema[0]["name"] == "t1"

    def test_builtins_register(self):
        r = ToolRegistry()
        register_builtins(r)
        names = r.list_names()
        assert "calculator" in names
        assert "shell" in names
        assert "read_file" in names
        assert "write_file" in names
        assert "list_files" in names

    def test_list_files_tool(self, tmp_path):
        r = ToolRegistry()
        register_builtins(r)
        handler = r.get("list_files")

        # Create test files
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world")
        (tmp_path / "subdir").mkdir()

        result = handler({"path": str(tmp_path)})
        assert "a.txt" in result
        assert "b.txt" in result
        assert "subdir/" in result

    def test_list_files_recursive(self, tmp_path):
        r = ToolRegistry()
        register_builtins(r)
        handler = r.get("list_files")

        (tmp_path / "dir1").mkdir()
        (tmp_path / "dir1" / "nested.txt").write_text("x")

        result = handler({"path": str(tmp_path), "recursive": True})
        assert "dir1/" in result
        assert "nested.txt" in result


# ---- Interceptor tests ----

class TestInterceptors:
    def test_base_allows(self):
        i = Interceptor()
        action, data = i.intercept("any_tool", {"key": "val"})
        assert action == InterceptAction.ALLOW

    def test_logging_interceptor(self):
        li = LoggingInterceptor()
        action, _ = li.intercept("calc", {"expr": "1+1"})
        assert action == InterceptAction.ALLOW
        assert len(li.log) == 1
        assert li.log[0]["tool"] == "calc"

    def test_shadow_allows_reads(self):
        si = ShadowInterceptor()
        action, _ = si.intercept("read_file", {"path": "x.txt"})
        assert action == InterceptAction.ALLOW

    def test_shadow_captures_writes(self):
        si = ShadowInterceptor()
        action, result = si.intercept("write_file", {"path": "x.txt", "content": "hi"})
        assert action == InterceptAction.MOCK
        assert len(si.captured_writes) == 1
        assert "[shadow]" in str(result)

    def test_shadow_replay(self):
        si = ShadowInterceptor()
        si.intercept("write_file", {"path": "x.txt", "content": "hi"})

        registry = ToolRegistry()
        registry.register("write_file", "w", {}, lambda p: f"wrote {p['path']}")
        results = si.replay(registry)
        assert len(results) == 1
        assert "wrote x.txt" in results[0]["output"]


# ---- Custom deny interceptor ----

class DenyAllInterceptor(Interceptor):
    def intercept(self, tool_name, tool_input):
        return InterceptAction.DENY, None


# ---- Runner tests (mocked API) ----

class TestAgentRunner:
    def _make_runner(self, registry=None, interceptors=None):
        if registry is None:
            registry = ToolRegistry()
            registry.register(
                "echo", "Echo", {"type": "object", "properties": {}},
                lambda p: f"echoed: {p.get('msg', '')}"
            )
        return AgentRunner(
            system_prompt="Test",
            tools=registry,
            interceptors=interceptors or [],
            api_key="fake-key",
        )

    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_simple_text_response(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _api_response(
            [_text_block("Hello!")], "end_turn"
        )

        runner = self._make_runner()
        result = runner.run("Hi")
        assert result.final_text == "Hello!"
        assert result.turns_used == 1
        assert len(result.tool_call_log) == 0

    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_tool_call_loop(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # First response: tool call.  Second response: text with the result.
        mock_client.messages.create.side_effect = [
            _api_response([_tool_use_block("echo", {"msg": "test"})], "tool_use"),
            _api_response([_text_block("Done: echoed: test")], "end_turn"),
        ]

        runner = self._make_runner()
        result = runner.run("Echo test")
        assert "echoed: test" in result.final_text
        assert result.turns_used == 2
        assert len(result.tool_call_log) == 1
        assert result.tool_call_log[0]["action"] == "allow"

    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_denied_tool_call(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_client.messages.create.side_effect = [
            _api_response([_tool_use_block("echo", {"msg": "test"})], "tool_use"),
            _api_response([_text_block("Tool was denied.")], "end_turn"),
        ]

        runner = self._make_runner(interceptors=[DenyAllInterceptor()])
        result = runner.run("Echo test")
        assert result.tool_call_log[0]["action"] == "deny"

    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_max_turns_respected(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # Always return tool calls, never end_turn
        mock_client.messages.create.return_value = _api_response(
            [_tool_use_block("echo", {"msg": "loop"})], "tool_use"
        )

        runner = self._make_runner()
        runner.max_turns = 3
        result = runner.run("Loop forever")
        assert result.turns_used == 3

    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_multi_turn_conversation(self, mock_cls):
        """Verify that passing conversation history carries context forward."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_client.messages.create.return_value = _api_response(
            [_text_block("Turn 1 reply")], "end_turn"
        )

        runner = self._make_runner()
        r1 = runner.run("Hello")
        assert r1.final_text == "Turn 1 reply"

        # Second turn carries the conversation forward
        mock_client.messages.create.return_value = _api_response(
            [_text_block("Turn 2 reply")], "end_turn"
        )
        r2 = runner.run("Follow up", conversation=r1.messages)
        assert r2.final_text == "Turn 2 reply"
        # Conversation should have: user1, assistant1, user2, assistant2
        assert len(r2.messages) == 4
        assert r2.messages[0]["content"] == "Hello"
        assert r2.messages[2]["content"] == "Follow up"

    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_usage_tracking(self, mock_cls):
        """Verify token usage is accumulated across turns."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_client.messages.create.side_effect = [
            _api_response([_tool_use_block("echo", {"msg": "x"})], "tool_use", 200, 80),
            _api_response([_text_block("Done")], "end_turn", 300, 40),
        ]

        runner = self._make_runner()
        result = runner.run("test")
        assert result.usage.input_tokens == 500
        assert result.usage.output_tokens == 120
        assert result.usage.total_tokens == 620
        assert result.elapsed_seconds > 0
