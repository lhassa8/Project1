"""Tests for config file loading and CLI merging."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent_runner.config import AgentConfig, MCPConfig


class TestAgentConfig:
    def test_from_json_file(self, tmp_path):
        cfg_file = tmp_path / "agent.json"
        cfg_file.write_text(json.dumps({
            "model": "claude-opus-4-20250514",
            "max_turns": 10,
            "shadow": True,
            "stream": True,
            "approve": ["shell"],
            "mcp": {
                "command": "npx server",
                "write_tools": ["write"],
            },
        }))

        cfg = AgentConfig.from_file(cfg_file)
        assert cfg.model == "claude-opus-4-20250514"
        assert cfg.max_turns == 10
        assert cfg.shadow is True
        assert cfg.stream is True
        assert cfg.approve == ["shell"]
        assert cfg.mcp.command == "npx server"
        assert cfg.mcp.write_tools == ["write"]

    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.model is None
        assert cfg.max_turns == 25
        assert cfg.shadow is False
        assert cfg.stream is False
        assert cfg.approve == []
        assert cfg.mcp is None

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AgentConfig.from_file(tmp_path / "nonexistent.json")

    def test_merge_cli_overrides(self):
        cfg = AgentConfig(model="original", max_turns=10)
        args = SimpleNamespace(
            model="new-model",
            max_turns=50,
            shadow=True,
            share=False,
            stream=True,
            approve="shell,write_file",
            mcp="npx server",
            mcp_writes="write",
        )
        cfg.merge_cli(args)
        assert cfg.model == "new-model"
        assert cfg.max_turns == 50
        assert cfg.shadow is True
        assert cfg.stream is True
        assert cfg.approve == ["shell", "write_file"]
        assert cfg.mcp.command == "npx server"
        assert cfg.mcp.write_tools == ["write"]

    def test_merge_cli_no_overrides(self):
        cfg = AgentConfig(model="original", max_turns=10, shadow=True)
        args = SimpleNamespace(
            model=None,
            max_turns=25,  # default, should not override
            shadow=False,
            share=False,
            stream=False,
            approve=None,
            mcp=None,
        )
        cfg.merge_cli(args)
        assert cfg.model == "original"
        assert cfg.max_turns == 10
        assert cfg.shadow is True  # CLI false doesn't override config true

    def test_discover_finds_agent_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.json").write_text(json.dumps({"model": "discovered"}))
        cfg = AgentConfig.discover()
        assert cfg is not None
        assert cfg.model == "discovered"

    def test_discover_returns_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert AgentConfig.discover() is None


class TestTokenUsage:
    def test_usage_tracking(self):
        from agent_runner.runner import TokenUsage

        usage = TokenUsage()
        usage.add(100, 50)
        usage.add(200, 100)
        assert usage.input_tokens == 300
        assert usage.output_tokens == 150
        assert usage.total_tokens == 450

    def test_cost_estimation(self):
        from agent_runner.runner import TokenUsage

        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = usage.estimated_cost("claude-sonnet-4-20250514")
        assert cost == 18.0  # 3.0 + 15.0

    def test_cost_unknown_model_uses_default(self):
        from agent_runner.runner import TokenUsage

        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = usage.estimated_cost("unknown-model")
        assert cost == 18.0  # defaults to sonnet rates


class TestRunResultExtended:
    def test_elapsed_seconds_default(self):
        from agent_runner.runner import RunResult

        r = RunResult()
        assert r.elapsed_seconds == 0.0
        assert r.usage.total_tokens == 0
