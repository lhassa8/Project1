"""Tests for usability improvements: quick_run, Conversation, simple_tool,
tool metadata, approval callbacks, retry logic, config env var."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest
import anthropic as anthropic_module

from agent_runner.runner import AgentRunner, Conversation, RunResult, TokenUsage
from agent_runner.tools.registry import ToolRegistry, ToolDef
from agent_runner.tools.builtins import register_builtins
from agent_runner.interceptors.approval import ApprovalInterceptor
from agent_runner.interceptors.shadow import ShadowInterceptor
from agent_runner.config import AgentConfig


# ---- helpers ----

def _text_block(text):
    return SimpleNamespace(type="text", text=text)

def _tool_use_block(name, input_, id_="t1"):
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=id_)

def _api_response(content, stop_reason="end_turn", input_tokens=10, output_tokens=5):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


# ---- simple_tool decorator ----

class TestSimpleTool:
    def test_simple_tool_generates_schema(self):
        r = ToolRegistry()

        @r.simple_tool("greet", "Greet someone", person=str)
        def greet(params):
            return f"Hello {params['person']}"

        assert r.has("greet")
        schema = r.to_api_schema()
        assert schema[0]["input_schema"]["properties"]["person"]["type"] == "string"
        assert schema[0]["input_schema"]["required"] == ["person"]
        assert r.get("greet")({"person": "Alice"}) == "Hello Alice"

    def test_simple_tool_multiple_types(self):
        r = ToolRegistry()

        @r.simple_tool("calc", "Add", a=float, b=int, verbose=bool)
        def calc(params):
            return params["a"] + params["b"]

        schema = r.to_api_schema()[0]["input_schema"]
        assert schema["properties"]["a"]["type"] == "number"
        assert schema["properties"]["b"]["type"] == "integer"
        assert schema["properties"]["verbose"]["type"] == "boolean"

    def test_simple_tool_with_metadata(self):
        r = ToolRegistry()

        @r.simple_tool("rm", "Remove file", category="dangerous", is_write=True, path=str)
        def rm(params):
            pass

        defn = r.get_def("rm")
        assert defn.category == "dangerous"
        assert defn.is_write is True


# ---- Tool metadata ----

class TestToolMetadata:
    def test_write_and_read_tools(self):
        r = ToolRegistry()
        register_builtins(r)

        writes = r.write_tools()
        reads = r.read_tools()

        assert "shell" in writes
        assert "write_file" in writes
        assert "calculator" in reads
        assert "read_file" in reads
        assert "list_files" in reads

    def test_list_by_category(self):
        r = ToolRegistry()
        register_builtins(r)

        file_tools = r.list_by_category("file")
        assert "read_file" in file_tools
        assert "write_file" in file_tools
        assert "list_files" in file_tools

    def test_has(self):
        r = ToolRegistry()
        r.register("test", "test", {}, lambda p: None)
        assert r.has("test")
        assert not r.has("nope")

    def test_get_def(self):
        r = ToolRegistry()
        r.register("test", "desc", {"type": "object"}, lambda p: None, category="custom", is_write=True)
        defn = r.get_def("test")
        assert isinstance(defn, ToolDef)
        assert defn.category == "custom"
        assert defn.is_write is True
        assert r.get_def("nope") is None


# ---- Shadow from_registry ----

class TestShadowFromRegistry:
    def test_from_registry(self):
        r = ToolRegistry()
        register_builtins(r)
        shadow = ShadowInterceptor.from_registry(r)
        assert "shell" in shadow.write_tools
        assert "write_file" in shadow.write_tools
        assert "read_file" in shadow.read_tools
        assert "calculator" in shadow.read_tools


# ---- Approval callback ----

class TestApprovalCallback:
    def test_custom_callback_approve(self):
        approver = ApprovalInterceptor(
            require_approval_for={"shell"},
            on_approval=lambda name, inp: True,
        )
        from agent_runner.interceptors.base import InterceptAction
        action, _ = approver.intercept("shell", {"command": "ls"})
        assert action == InterceptAction.ALLOW

    def test_custom_callback_deny(self):
        approver = ApprovalInterceptor(
            require_approval_for={"shell"},
            on_approval=lambda name, inp: False,
        )
        from agent_runner.interceptors.base import InterceptAction
        action, _ = approver.intercept("shell", {"command": "rm -rf /"})
        assert action == InterceptAction.DENY

    def test_non_approved_tools_pass_through(self):
        approver = ApprovalInterceptor(
            require_approval_for={"shell"},
            on_approval=lambda name, inp: False,  # would deny
        )
        from agent_runner.interceptors.base import InterceptAction
        action, _ = approver.intercept("calculator", {"expression": "1+1"})
        assert action == InterceptAction.ALLOW


# ---- Conversation ----

class TestConversation:
    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_conversation_carries_history(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_client.messages.create.side_effect = [
            _api_response([_text_block("Reply 1")]),
            _api_response([_text_block("Reply 2")]),
        ]

        r = ToolRegistry()
        runner = AgentRunner(system_prompt="Test", tools=r, api_key="fake")
        conv = runner.conversation()

        r1 = conv.ask("Hello")
        assert r1.final_text == "Reply 1"

        r2 = conv.ask("Follow up")
        assert r2.final_text == "Reply 2"
        assert len(conv.results) == 2
        assert conv.total_usage.total_tokens == 30  # 10+5 per call, 2 calls

    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_conversation_reset(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _api_response([_text_block("Hi")])

        r = ToolRegistry()
        runner = AgentRunner(system_prompt="Test", tools=r, api_key="fake")
        conv = runner.conversation()
        conv.ask("Hello")
        conv.reset()
        assert len(conv.results) == 0
        assert conv.total_usage.total_tokens == 0


# ---- RunResult.failed_tools ----

class TestRunResultFailedTools:
    def test_no_failures(self):
        r = RunResult()
        r.tool_call_log = [{"tool": "echo", "action": "allow", "error": None}]
        assert r.failed_tools() == []

    def test_with_failures(self):
        r = RunResult()
        r.tool_call_log = [
            {"tool": "ok", "action": "allow", "error": None},
            {"tool": "bad", "action": "allow", "error": "TypeError: boom"},
        ]
        failed = r.failed_tools()
        assert len(failed) == 1
        assert failed[0]["tool"] == "bad"


# ---- Retry logic ----

class TestRetryLogic:
    @patch("agent_runner.runner.anthropic.Anthropic")
    @patch("agent_runner.runner.time.sleep")
    def test_retry_on_rate_limit(self, mock_sleep, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_client.messages.create.side_effect = [
            anthropic_module.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429, headers={}),
                body=None,
            ),
            _api_response([_text_block("OK")]),
        ]

        r = ToolRegistry()
        runner = AgentRunner(system_prompt="Test", tools=r, api_key="fake", retries=2, retry_delay=0.01)
        result = runner.run("test")
        assert result.final_text == "OK"
        mock_sleep.assert_called_once()

    @patch("agent_runner.runner.anthropic.Anthropic")
    @patch("agent_runner.runner.time.sleep")
    def test_retry_exhausted_raises(self, mock_sleep, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_client.messages.create.side_effect = anthropic_module.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )

        r = ToolRegistry()
        runner = AgentRunner(system_prompt="Test", tools=r, api_key="fake", retries=1, retry_delay=0.01)
        with pytest.raises(anthropic_module.RateLimitError):
            runner.run("test")


# ---- Config env var ----

class TestConfigEnvVar:
    def test_discover_from_env_var(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "custom.json"
        cfg_file.write_text(json.dumps({"model": "from-env"}))
        monkeypatch.setenv("AGENT_CONFIG", str(cfg_file))
        monkeypatch.chdir(tmp_path)

        cfg = AgentConfig.discover()
        assert cfg is not None
        assert cfg.model == "from-env"

    def test_config_max_tokens(self, tmp_path):
        cfg_file = tmp_path / "agent.json"
        cfg_file.write_text(json.dumps({"max_tokens": 8192}))
        cfg = AgentConfig.from_file(cfg_file)
        assert cfg.max_tokens == 8192

    def test_config_bad_json_gives_clear_error(self, tmp_path):
        cfg_file = tmp_path / "bad.json"
        cfg_file.write_text("{ invalid json }")
        with pytest.raises(ValueError, match="Failed to parse"):
            AgentConfig.from_file(cfg_file)


# ---- max_tokens passthrough ----

class TestMaxTokens:
    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_custom_max_tokens(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _api_response([_text_block("ok")])

        r = ToolRegistry()
        runner = AgentRunner(system_prompt="Test", tools=r, api_key="fake", max_tokens=8192)
        runner.run("test")

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs.get("max_tokens") == 8192 or call_kwargs[1].get("max_tokens") == 8192
