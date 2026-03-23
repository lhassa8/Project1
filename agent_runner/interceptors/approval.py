"""Interceptor that requires human approval for specified tools."""

from __future__ import annotations

import json
from typing import Any

from agent_runner.interceptors.base import InterceptAction, Interceptor


class ApprovalInterceptor(Interceptor):
    """Prompts the operator for approval before executing sensitive tools.

    Parameters
    ----------
    require_approval_for : set[str] | None
        Tool names that need approval.  If ``None``, ALL tools require it.
    auto_deny : set[str] | None
        Tool names that are always denied (overrides approval prompt).
    """

    def __init__(
        self,
        require_approval_for: set[str] | None = None,
        auto_deny: set[str] | None = None,
    ) -> None:
        self.require_approval_for = require_approval_for
        self.auto_deny = auto_deny or set()

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

        print(f"\n--- Approval Required ---")
        print(f"Tool:  {tool_name}")
        print(f"Input: {json.dumps(tool_input, indent=2, default=str)}")
        answer = input("Allow? [y/N] ").strip().lower()
        if answer in ("y", "yes"):
            return InterceptAction.ALLOW, tool_input
        return InterceptAction.DENY, None
