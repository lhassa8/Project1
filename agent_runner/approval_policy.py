"""Declarative approval-policy rules.

A policy is a list of :class:`ApprovalRule` instances evaluated in order.
The first matching rule wins; if nothing matches, *default_action* is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ApprovalRule:
    """A single policy rule.

    Parameters
    ----------
    tool : str | None
        Tool name to match.  ``None`` matches every tool.
    action : str
        ``"auto_approve"``, ``"require_review"``, or ``"auto_deny"``.
    condition : str
        ``"always"`` — rule matches unconditionally.
        ``"if_write"`` — rule matches only when *is_write* is ``True``.
        ``"if_pattern"`` — rule matches when *pattern* matches any
        **string** value inside *tool_input*.
    pattern : str | None
        Regular-expression pattern used when ``condition="if_pattern"``.
    """

    tool: str | None = None
    action: str = "require_review"
    condition: str = "always"
    pattern: str | None = None


@dataclass
class ApprovalPolicy:
    """An ordered list of approval rules with a fallback default."""

    rules: list[ApprovalRule] = field(default_factory=list)
    default_action: str = "require_review"

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, data: list[dict[str, Any]]) -> ApprovalPolicy:
        """Build a policy from a list of plain dicts (e.g. loaded from JSON).

        The last element may optionally contain a ``"default_action"`` key
        with no ``"action"`` key — it sets the fallback instead of being
        treated as a rule.
        """
        rules: list[ApprovalRule] = []
        default_action = "require_review"

        for item in data:
            # Allow a bare {"default_action": "..."} entry to set the
            # fallback.
            if "default_action" in item and "action" not in item:
                default_action = item["default_action"]
                continue
            rules.append(
                ApprovalRule(
                    tool=item.get("tool"),
                    action=item.get("action", "require_review"),
                    condition=item.get("condition", "always"),
                    pattern=item.get("pattern"),
                )
            )

        return cls(rules=rules, default_action=default_action)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        is_write: bool = False,
    ) -> str:
        """Return the action for the given tool call.

        Returns one of ``"auto_approve"``, ``"require_review"``, or
        ``"auto_deny"``.
        """
        for rule in self.rules:
            if self._matches(rule, tool_name, tool_input, is_write):
                return rule.action
        return self.default_action

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _matches(
        rule: ApprovalRule,
        tool_name: str,
        tool_input: dict[str, Any],
        is_write: bool,
    ) -> bool:
        # Tool filter — None means "any tool".
        if rule.tool is not None and rule.tool != tool_name:
            return False

        if rule.condition == "always":
            return True

        if rule.condition == "if_write":
            return is_write

        if rule.condition == "if_pattern":
            if rule.pattern is None:
                return False
            compiled = re.compile(rule.pattern)
            return _any_value_matches(compiled, tool_input)

        # Unknown condition — don't match.
        return False


def _any_value_matches(compiled: re.Pattern[str], data: Any) -> bool:
    """Recursively check whether *compiled* matches any string value in *data*."""
    if isinstance(data, str):
        return compiled.search(data) is not None
    if isinstance(data, dict):
        return any(_any_value_matches(compiled, v) for v in data.values())
    if isinstance(data, (list, tuple)):
        return any(_any_value_matches(compiled, v) for v in data)
    return False
