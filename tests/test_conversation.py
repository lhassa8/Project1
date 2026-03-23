"""Tests for Phase 5: Conversation Intelligence.

Covers ContextManager, cost estimation, and Conversation save/load.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent_runner.runner import AgentRunner, ContextManager, Conversation, TokenUsage
from agent_runner.tools.registry import ToolRegistry


# ---- helpers ----

def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _api_response(content, stop_reason="end_turn", input_tokens=100, output_tokens=50):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _make_runner(**kwargs):
    registry = ToolRegistry()
    registry.register(
        "echo", "Echo", {"type": "object", "properties": {}},
        lambda p: f"echoed: {p.get('msg', '')}",
    )
    defaults = dict(
        system_prompt="Test",
        tools=registry,
        api_key="fake-key",
    )
    defaults.update(kwargs)
    return AgentRunner(**defaults)


# ---- ContextManager tests ----

class TestContextManagerSlidingWindow:
    def test_trims_old_messages(self):
        """Sliding window should keep only the last messages that fit."""
        cm = ContextManager(max_context_tokens=50, strategy="sliding_window", reserve_tokens=0)
        # Each message ~50 chars -> ~12 tokens
        messages = [
            {"role": "user", "content": "A" * 200},      # ~50 tokens
            {"role": "assistant", "content": "B" * 200},  # ~50 tokens
            {"role": "user", "content": "C" * 80},        # ~20 tokens
        ]
        trimmed = cm.trim(messages)
        # Budget is 50 tokens. Last message is 20 tokens, second-to-last is 50.
        # 20 + 50 = 70 > 50, so only the last message should fit.
        assert len(trimmed) < len(messages)
        assert trimmed[-1]["content"] == "C" * 80

    def test_no_trim_when_under_budget(self):
        """Messages under budget should pass through unchanged."""
        cm = ContextManager(max_context_tokens=100_000, strategy="sliding_window", reserve_tokens=4096)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        trimmed = cm.trim(messages)
        assert trimmed == messages
        assert len(trimmed) == 2


class TestContextManagerKeepEnds:
    def test_keeps_first_and_last(self):
        """keep_ends should preserve the first message and last messages that fit."""
        cm = ContextManager(max_context_tokens=60, strategy="keep_ends", reserve_tokens=0)
        messages = [
            {"role": "user", "content": "First"},                # ~1 token
            {"role": "assistant", "content": "X" * 400},         # ~100 tokens — gets dropped
            {"role": "user", "content": "Y" * 400},              # ~100 tokens — gets dropped
            {"role": "assistant", "content": "Z" * 80},          # ~20 tokens
            {"role": "user", "content": "Last"},                 # ~1 token
        ]
        trimmed = cm.trim(messages)
        # First message kept. From the end: "Last" (~1) + "Z*80" (~20) = 21 <= ~59 remaining.
        assert trimmed[0]["content"] == "First"
        assert trimmed[-1]["content"] == "Last"
        # Middle messages with 400 chars should be dropped
        assert len(trimmed) < len(messages)

    def test_no_trim_when_under_budget(self):
        """Messages under budget should pass through unchanged."""
        cm = ContextManager(max_context_tokens=100_000, strategy="keep_ends", reserve_tokens=0)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
        ]
        trimmed = cm.trim(messages)
        assert trimmed == messages


class TestContextManagerNoBudget:
    def test_no_trim_when_under_budget(self):
        """With a large budget, no trimming should occur."""
        cm = ContextManager(max_context_tokens=1_000_000, reserve_tokens=4096)
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        trimmed = cm.trim(messages)
        assert len(trimmed) == 20


# ---- Cost estimation tests ----

class TestEstimateCost:
    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_estimate_cost(self, mock_cls):
        runner = _make_runner()
        estimate = runner.estimate_cost("What is 2+2?", expected_turns=3)

        assert "min_usd" in estimate
        assert "max_usd" in estimate
        assert "model" in estimate
        assert estimate["min_usd"] > 0
        assert estimate["max_usd"] > estimate["min_usd"]
        assert estimate["model"] == runner.model

    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_estimate_cost_single_turn(self, mock_cls):
        runner = _make_runner()
        estimate = runner.estimate_cost("Hi", expected_turns=1)
        assert estimate["min_usd"] > 0
        assert estimate["max_usd"] >= estimate["min_usd"]


# ---- Conversation save/load tests ----

class TestConversationSaveLoad:
    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_roundtrip(self, mock_cls, tmp_path):
        """Save and load should produce identical state."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _api_response(
            [_text_block("Reply 1")], "end_turn"
        )

        runner = _make_runner()
        conv = runner.conversation()
        conv.ask("Hello")

        path = str(tmp_path / "conv.json")
        conv.save(path)

        # Load into a fresh conversation
        conv2 = runner.conversation()
        conv2.load(path)

        assert len(conv2._messages) == len(conv._messages)
        assert conv2.total_usage.input_tokens == conv.total_usage.input_tokens
        assert conv2.total_usage.output_tokens == conv.total_usage.output_tokens

    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_load_resumes_history(self, mock_cls, tmp_path):
        """After loading, conversation should carry the restored history."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _api_response(
            [_text_block("Reply 1")], "end_turn"
        )

        runner = _make_runner()
        conv = runner.conversation()
        conv.ask("Hello")

        path = str(tmp_path / "conv.json")
        conv.save(path)

        # Verify saved file content
        data = json.loads(open(path).read())
        assert "messages" in data
        assert "total_usage" in data
        assert "metadata" in data
        assert data["metadata"]["model"] == runner.model
        assert len(data["messages"]) == 2  # user + assistant

        # Load and continue
        conv2 = runner.conversation()
        conv2.load(path)

        mock_client.messages.create.return_value = _api_response(
            [_text_block("Reply 2")], "end_turn"
        )
        result = conv2.ask("Follow up")
        assert result.final_text == "Reply 2"
        # Should have: user1, assistant1, user2, assistant2
        assert len(conv2._messages) == 4

    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_save_format(self, mock_cls, tmp_path):
        """Saved JSON should have the expected structure."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _api_response(
            [_text_block("Hi")], "end_turn", 150, 30
        )

        runner = _make_runner()
        conv = runner.conversation()
        conv.ask("Hello")

        path = str(tmp_path / "conv.json")
        conv.save(path)

        data = json.loads(open(path).read())
        assert data["total_usage"]["input_tokens"] == 150
        assert data["total_usage"]["output_tokens"] == 30
        assert isinstance(data["metadata"]["timestamp"], float)
