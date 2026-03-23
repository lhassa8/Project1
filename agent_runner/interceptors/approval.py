"""Interceptor that requires human approval for specified tools."""

from __future__ import annotations

import json
from typing import Any, Callable

from agent_runner.interceptors.base import InterceptAction, Interceptor

# Default callback: interactive stdin prompt
def _stdin_approval(tool_name: str, tool_input: dict[str, Any]) -> bool:
    """Prompt the operator via stdin.  Returns True if approved."""
    print(f"\n--- Approval Required ---")
    print(f"Tool:  {tool_name}")
    print(f"Input: {json.dumps(tool_input, indent=2, default=str)}")
    answer = input("Allow? [y/N] ").strip().lower()
    return answer in ("y", "yes")


class ApprovalInterceptor(Interceptor):
    """Prompts the operator for approval before executing sensitive tools.

    Parameters
    ----------
    require_approval_for : set[str] | None
        Tool names that need approval.  If ``None``, ALL tools require it.
    auto_deny : set[str] | None
        Tool names that are always denied (overrides approval prompt).
    on_approval : Callable[[str, dict], bool] | None
        Custom approval callback.  Receives ``(tool_name, tool_input)``
        and returns ``True`` to approve, ``False`` to deny.
        Defaults to an interactive stdin prompt.  Override this for
        non-interactive use (CI, web UI, Slack bot, etc.)::

            def slack_approval(tool_name, tool_input):
                return post_to_slack_and_wait(tool_name, tool_input)

            ApprovalInterceptor(
                require_approval_for={"shell"},
                on_approval=slack_approval,
            )
    """

    def __init__(
        self,
        require_approval_for: set[str] | None = None,
        auto_deny: set[str] | None = None,
        on_approval: Callable[[str, dict[str, Any]], bool] | None = None,
    ) -> None:
        self.require_approval_for = require_approval_for
        self.auto_deny = auto_deny or set()
        self.on_approval = on_approval or _stdin_approval

    def intercept(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> tuple[InterceptAction, Any]:
        if tool_name in self.auto_deny:
            return InterceptAction.DENY, None

        needs_approval = (
            self.require_approval_for is None or tool_name in self.require_approval_for
        )
        if not needs_approval:
            return InterceptAction.ALLOW, tool_input

        if self.on_approval(tool_name, tool_input):
            return InterceptAction.ALLOW, tool_input
        return InterceptAction.DENY, None
