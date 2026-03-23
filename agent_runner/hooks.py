"""Lifecycle event hooks for observing and reacting to runner events.

Hooks are simple callback registrations — no framework, no metaclasses.
Register callbacks to be notified when the runner starts a turn, invokes
a tool, or completes a run.

Usage::

    from agent_runner.hooks import HookManager

    hooks = HookManager()

    @hooks.on("turn_start")
    def log_turn(turn_number, messages):
        print(f"Starting turn {turn_number}")

    @hooks.on("tool_call")
    def log_tool(tool_name, tool_input, action):
        print(f"[{action}] {tool_name}")

    @hooks.on("run_complete")
    def log_done(result):
        print(f"Done in {result.turns_used} turns")

    runner = AgentRunner(..., hooks=hooks)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Known event names and their signatures for documentation and validation.
KNOWN_EVENTS: dict[str, str] = {
    "run_start":    "(user_message: str)",
    "turn_start":   "(turn_number: int, messages: list)",
    "turn_end":     "(turn_number: int, stop_reason: str)",
    "tool_call":    "(tool_name: str, tool_input: dict, action: str, output: Any)",
    "run_complete": "(result: RunResult)",
}


class HookManager:
    """Registry for lifecycle callbacks.

    By default, registering a hook on an unknown event name logs a warning
    to help catch typos.  Set ``strict=True`` to raise instead::

        hooks = HookManager(strict=True)
        hooks.on("trun_start")  # raises ValueError — did you mean turn_start?
    """

    def __init__(self, strict: bool = False) -> None:
        self._listeners: dict[str, list[Callable[..., Any]]] = defaultdict(list)
        self.strict = strict

    def on(self, event: str) -> Callable:
        """Decorator to register a callback for *event*."""
        self._check_event(event)

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._listeners[event].append(fn)
            return fn
        return decorator

    def register(self, event: str, callback: Callable[..., Any]) -> None:
        """Imperatively register a callback (alternative to decorator)."""
        self._check_event(event)
        self._listeners[event].append(callback)

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Fire all callbacks for *event*.  Exceptions are logged, not raised."""
        for cb in self._listeners.get(event, []):
            try:
                cb(*args, **kwargs)
            except Exception:
                logger.exception("Hook %s raised an error in callback %s", event, cb.__name__)

    def has_listeners(self, event: str) -> bool:
        return bool(self._listeners.get(event))

    def clear(self, event: str | None = None) -> None:
        """Remove listeners.  If *event* is None, remove all."""
        if event:
            self._listeners.pop(event, None)
        else:
            self._listeners.clear()

    @staticmethod
    def list_events() -> dict[str, str]:
        """Return known event names with their callback signatures."""
        return dict(KNOWN_EVENTS)

    def _check_event(self, event: str) -> None:
        """Warn or raise on unrecognized event names."""
        if event not in KNOWN_EVENTS:
            msg = f"Unknown hook event '{event}'. Known events: {', '.join(sorted(KNOWN_EVENTS))}"
            if self.strict:
                raise ValueError(msg)
            logger.warning(msg)
