"""Shadow-mode interceptor — captures write operations for review/rollback.

This is the foundation for Phase 2's "shadow mode" where the agent runs
against real systems but all write-side effects are captured and can be
rolled back or replayed after stakeholder approval.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from agent_runner.interceptors.base import InterceptAction, Interceptor


class ShadowInterceptor(Interceptor):
    """Intercepts write tools, records what *would* happen, and returns a
    mock success so the agent continues planning as if the write succeeded.

    Parameters
    ----------
    write_tools : set[str]
        Tool names considered "write" operations (e.g. ``write_file``, ``shell``).
    read_tools : set[str]
        Tool names considered safe/read-only — always allowed.
    """

    def __init__(
        self,
        write_tools: set[str] | None = None,
        read_tools: set[str] | None = None,
    ) -> None:
        self.write_tools = write_tools or {"write_file", "shell"}
        self.read_tools = read_tools or {"read_file", "list_files", "calculator"}
        self.captured_writes: list[dict[str, Any]] = []

    def intercept(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> tuple[InterceptAction, Any]:
        if tool_name in self.read_tools:
            return InterceptAction.ALLOW, tool_input

        if tool_name in self.write_tools:
            record = {
                "tool": tool_name,
                "input": copy.deepcopy(tool_input),
            }
            self.captured_writes.append(record)
            # Return a mock so the agent thinks the write succeeded
            return InterceptAction.MOCK, f"[shadow] {tool_name} captured (not executed)"

        # Unknown tools default to deny in shadow mode
        return InterceptAction.DENY, None

    def get_captured_writes(self) -> list[dict[str, Any]]:
        """Return all captured write operations for review."""
        return list(self.captured_writes)

    def replay(self, tool_registry: Any) -> list[dict[str, Any]]:
        """Re-execute all captured writes.  Returns results."""
        results = []
        for record in self.captured_writes:
            handler = tool_registry.get(record["tool"])
            if handler:
                output = handler(record["input"])
                results.append({"tool": record["tool"], "output": output})
        return results
