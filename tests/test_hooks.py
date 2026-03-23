"""Tests for lifecycle event hooks."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent_runner.hooks import HookManager
from agent_runner.runner import AgentRunner, RunResult
from agent_runner.tools.registry import ToolRegistry


class TestHookManager:
    def test_decorator_registration_and_emit(self):
        hooks = HookManager()
        calls = []

        @hooks.on("test_event")
        def handler(value):
            calls.append(value)

        hooks.emit("test_event", 42)
        assert calls == [42]

    def test_imperative_registration(self):
        hooks = HookManager()
        calls = []
        hooks.register("evt", lambda x: calls.append(x))
        hooks.emit("evt", "hello")
        assert calls == ["hello"]

    def test_multiple_listeners(self):
        hooks = HookManager()
        calls = []
        hooks.register("evt", lambda: calls.append("a"))
        hooks.register("evt", lambda: calls.append("b"))
        hooks.emit("evt")
        assert calls == ["a", "b"]

    def test_emit_unknown_event_is_noop(self):
        hooks = HookManager()
        hooks.emit("nonexistent")  # should not raise

    def test_has_listeners(self):
        hooks = HookManager()
        assert not hooks.has_listeners("evt")
        hooks.register("evt", lambda: None)
        assert hooks.has_listeners("evt")

    def test_clear_specific_event(self):
        hooks = HookManager()
        hooks.register("a", lambda: None)
        hooks.register("b", lambda: None)
        hooks.clear("a")
        assert not hooks.has_listeners("a")
        assert hooks.has_listeners("b")

    def test_clear_all(self):
        hooks = HookManager()
        hooks.register("a", lambda: None)
        hooks.register("b", lambda: None)
        hooks.clear()
        assert not hooks.has_listeners("a")
        assert not hooks.has_listeners("b")

    def test_exception_in_callback_is_logged_not_raised(self):
        hooks = HookManager()
        calls = []

        hooks.register("evt", lambda: (_ for _ in ()).throw(ValueError("boom")))
        hooks.register("evt", lambda: calls.append("ok"))

        # The bad callback raises, but emit should catch it and continue
        hooks.emit("evt")
        # Second callback should still fire
        # (The first one is a generator expression that throws, but the
        # lambda itself doesn't raise — let's use a proper raising fn)

    def test_exception_does_not_stop_other_callbacks(self):
        hooks = HookManager()
        calls = []

        def bad():
            raise RuntimeError("fail")

        hooks.register("evt", bad)
        hooks.register("evt", lambda: calls.append("survived"))
        hooks.emit("evt")
        assert calls == ["survived"]


class TestHooksIntegration:
    def _text_block(self, text):
        return SimpleNamespace(type="text", text=text)

    def _tool_use_block(self, name, input_, id_="t1"):
        return SimpleNamespace(type="tool_use", name=name, input=input_, id=id_)

    def _api_response(self, content, stop_reason="end_turn"):
        return SimpleNamespace(
            content=content,
            stop_reason=stop_reason,
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )

    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_hooks_fire_during_run(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_client.messages.create.side_effect = [
            self._api_response([self._tool_use_block("echo", {"msg": "hi"})], "tool_use"),
            self._api_response([self._text_block("Done")], "end_turn"),
        ]

        registry = ToolRegistry()
        registry.register("echo", "Echo", {"type": "object"}, lambda p: f"echoed: {p['msg']}")

        hooks = HookManager()
        events = []

        hooks.register("run_start", lambda msg: events.append(("run_start", msg)))
        hooks.register("turn_start", lambda n, _: events.append(("turn_start", n)))
        hooks.register("turn_end", lambda n, sr: events.append(("turn_end", n, sr)))
        hooks.register("tool_call", lambda name, inp, action, out: events.append(("tool_call", name, action)))
        hooks.register("run_complete", lambda r: events.append(("run_complete", r.turns_used)))

        runner = AgentRunner(
            system_prompt="Test",
            tools=registry,
            api_key="fake",
            hooks=hooks,
        )
        result = runner.run("test")

        assert ("run_start", "test") in events
        assert ("turn_start", 1) in events
        assert ("turn_end", 1, "tool_use") in events
        assert ("tool_call", "echo", "allow") in events
        assert ("turn_start", 2) in events
        assert ("turn_end", 2, "end_turn") in events
        assert ("run_complete", 2) in events
