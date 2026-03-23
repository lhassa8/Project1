"""Shadow-mode interceptor — virtual filesystem overlay for safe agent runs.

Intercepts write operations, stores them in a virtual filesystem layer, and
routes subsequent reads through the overlay so the agent sees a consistent
view of the world.  No real files are modified until explicit replay.

This solves the fundamental shadow mode problem: an agent that writes a file
and then reads it back will see the written content, not a "file not found"
error.

Usage::

    from agent_runner.interceptors.shadow import ShadowInterceptor
    from agent_runner.shadow import ShadowDiff, TransactionReplay

    shadow = ShadowInterceptor.from_registry(registry)
    runner = AgentRunner(system_prompt="...", tools=registry, interceptors=[shadow])
    result = runner.run("Create config.yaml and then read it back to verify")

    # Review what would change
    diff = ShadowDiff(shadow.state)
    print(diff.summary())
    print(diff.full_diff())

    # Apply changes with rollback protection
    replay = TransactionReplay(shadow.state)
    outcome = replay.execute()
"""

from __future__ import annotations

import copy
from typing import Any

from agent_runner.interceptors.base import InterceptAction, Interceptor
from agent_runner.shadow.state import ShadowState


class ShadowInterceptor(Interceptor):
    """Intercepts write tools via a virtual filesystem overlay.

    Write operations are captured in a ``ShadowState`` that also serves
    subsequent reads, so the agent sees a consistent world view.

    Parameters
    ----------
    write_tools : set[str]
        Tool names considered "write" operations.
    read_tools : set[str]
        Tool names considered safe/read-only.
    state : ShadowState | None
        Pre-existing virtual state (for chaining multiple runs).
    """

    def __init__(
        self,
        write_tools: set[str] | None = None,
        read_tools: set[str] | None = None,
        state: ShadowState | None = None,
    ) -> None:
        self.write_tools = write_tools or {"write_file", "shell"}
        self.read_tools = read_tools or {"read_file", "list_files", "calculator"}
        self.state = state or ShadowState()
        self.captured_writes: list[dict[str, Any]] = []

    @classmethod
    def from_registry(cls, registry: Any) -> ShadowInterceptor:
        """Auto-configure from a ToolRegistry's ``is_write`` metadata.

        Uses ``registry.write_tools()`` and ``registry.read_tools()``
        so you don't have to maintain separate sets::

            shadow = ShadowInterceptor.from_registry(registry)
        """
        return cls(
            write_tools=registry.write_tools(),
            read_tools=registry.read_tools(),
        )

    def intercept(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> tuple[InterceptAction, Any]:
        # --- Write tools: capture in virtual state ---
        if tool_name in self.write_tools:
            record = {
                "tool": tool_name,
                "input": copy.deepcopy(tool_input),
            }
            mock_output = self._handle_write(tool_name, tool_input)
            record["mock_output"] = mock_output
            self.captured_writes.append(record)
            return InterceptAction.MOCK, mock_output

        # --- Read tools: route through virtual FS overlay ---
        if tool_name in self.read_tools:
            virtual_output = self._handle_read(tool_name, tool_input)
            if virtual_output is not None:
                # Serve from virtual layer
                return InterceptAction.MOCK, virtual_output
            # No virtual data — let the real tool handle it
            return InterceptAction.ALLOW, tool_input

        # Unknown tools default to deny in shadow mode
        return InterceptAction.DENY, None

    def _handle_write(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Route a write tool call through the virtual state."""
        if tool_name == "write_file":
            path = tool_input.get("path", "")
            content = tool_input.get("content", "")
            return self.state.write_file(path, content)

        # For shell and other write tools, capture but return a realistic mock
        return f"[shadow] {tool_name} captured (not executed). Input: {_truncate(str(tool_input), 200)}"

    def _handle_read(self, tool_name: str, tool_input: dict[str, Any]) -> str | None:
        """Route a read tool call through the virtual state if relevant.

        Returns the virtual result, or None to fall through to the real tool.
        """
        if tool_name == "read_file":
            path = tool_input.get("path", "")
            # Check if this file has been virtually written
            if self.state.file_exists(path) and self._is_virtual(path):
                return self.state.read_file(path)
            return None  # fall through to real FS

        if tool_name == "list_files":
            path = tool_input.get("path", ".")
            recursive = tool_input.get("recursive", False)
            # Always merge virtual + real for list_files
            if self.state.has_changes():
                return self.state.list_files(path, recursive)
            return None  # no virtual changes, use real tool

        return None  # tool not handled, fall through

    def _is_virtual(self, path: str) -> bool:
        """Check if a path has been written in the virtual layer."""
        import os
        abspath = os.path.abspath(path)
        return abspath in self.state._files

    # ------------------------------------------------------------------
    # Introspection (backwards-compatible)
    # ------------------------------------------------------------------

    def get_captured_writes(self) -> list[dict[str, Any]]:
        """Return all captured write operations for review."""
        return list(self.captured_writes)

    def replay(self, tool_registry: Any) -> list[dict[str, Any]]:
        """Re-execute all captured writes (legacy API).

        For new code, prefer ``TransactionReplay(shadow.state).execute()``
        which provides rollback on failure.
        """
        results = []
        for record in self.captured_writes:
            handler = tool_registry.get(record["tool"])
            if handler:
                output = handler(record["input"])
                results.append({"tool": record["tool"], "output": output})
        return results


def _truncate(s: str, maxlen: int) -> str:
    return s if len(s) <= maxlen else s[:maxlen] + "..."
