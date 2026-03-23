"""Tests for declarative approval policies."""

from __future__ import annotations

import pytest

from agent_runner.approval_policy import ApprovalPolicy, ApprovalRule


class TestApprovalPolicyEvaluate:
    """Core evaluate() behaviour."""

    def test_default_action_when_no_rules(self):
        policy = ApprovalPolicy(rules=[], default_action="require_review")
        assert policy.evaluate("shell", {}) == "require_review"

    def test_default_action_auto_approve(self):
        policy = ApprovalPolicy(rules=[], default_action="auto_approve")
        assert policy.evaluate("anything", {"x": 1}) == "auto_approve"

    def test_auto_approve_rule_matches(self):
        rule = ApprovalRule(tool="read_file", action="auto_approve")
        policy = ApprovalPolicy(rules=[rule])
        assert policy.evaluate("read_file", {"path": "/tmp/a"}) == "auto_approve"

    def test_auto_deny_rule_matches(self):
        rule = ApprovalRule(tool="shell", action="auto_deny")
        policy = ApprovalPolicy(rules=[rule])
        assert policy.evaluate("shell", {"command": "rm -rf /"}) == "auto_deny"

    def test_require_review_rule(self):
        rule = ApprovalRule(tool="write_file", action="require_review")
        policy = ApprovalPolicy(rules=[rule])
        assert policy.evaluate("write_file", {"path": "x.txt"}) == "require_review"

    def test_first_matching_rule_wins(self):
        rules = [
            ApprovalRule(tool="shell", action="auto_deny"),
            ApprovalRule(tool="shell", action="auto_approve"),  # never reached
        ]
        policy = ApprovalPolicy(rules=rules)
        assert policy.evaluate("shell", {}) == "auto_deny"

    def test_non_matching_tool_falls_through(self):
        rules = [
            ApprovalRule(tool="shell", action="auto_deny"),
        ]
        policy = ApprovalPolicy(rules=rules, default_action="auto_approve")
        assert policy.evaluate("read_file", {}) == "auto_approve"

    def test_wildcard_tool_none_matches_everything(self):
        rule = ApprovalRule(tool=None, action="auto_approve")
        policy = ApprovalPolicy(rules=[rule])
        assert policy.evaluate("shell", {}) == "auto_approve"
        assert policy.evaluate("read_file", {"path": "/x"}) == "auto_approve"

    def test_if_write_condition_matches_when_write(self):
        rule = ApprovalRule(tool=None, action="require_review", condition="if_write")
        policy = ApprovalPolicy(rules=[rule], default_action="auto_approve")
        assert policy.evaluate("write_file", {"path": "x"}, is_write=True) == "require_review"

    def test_if_write_condition_skipped_when_not_write(self):
        rule = ApprovalRule(tool=None, action="require_review", condition="if_write")
        policy = ApprovalPolicy(rules=[rule], default_action="auto_approve")
        assert policy.evaluate("read_file", {"path": "x"}, is_write=False) == "auto_approve"

    def test_if_pattern_matches_string_value(self):
        rule = ApprovalRule(
            tool="shell",
            action="auto_deny",
            condition="if_pattern",
            pattern=r"rm\s+-rf",
        )
        policy = ApprovalPolicy(rules=[rule], default_action="auto_approve")
        assert policy.evaluate("shell", {"command": "rm -rf /tmp"}) == "auto_deny"

    def test_if_pattern_no_match(self):
        rule = ApprovalRule(
            tool="shell",
            action="auto_deny",
            condition="if_pattern",
            pattern=r"rm\s+-rf",
        )
        policy = ApprovalPolicy(rules=[rule], default_action="auto_approve")
        assert policy.evaluate("shell", {"command": "ls -la"}) == "auto_approve"

    def test_if_pattern_matches_nested_value(self):
        rule = ApprovalRule(
            tool=None,
            action="auto_deny",
            condition="if_pattern",
            pattern=r"secret",
        )
        policy = ApprovalPolicy(rules=[rule], default_action="auto_approve")
        assert policy.evaluate("write_file", {"content": {"data": "my secret key"}}) == "auto_deny"

    def test_if_pattern_matches_list_value(self):
        rule = ApprovalRule(
            tool=None,
            action="auto_deny",
            condition="if_pattern",
            pattern=r"DROP TABLE",
        )
        policy = ApprovalPolicy(rules=[rule], default_action="auto_approve")
        assert policy.evaluate("sql", {"queries": ["SELECT 1", "DROP TABLE users"]}) == "auto_deny"

    def test_if_pattern_none_pattern_does_not_match(self):
        rule = ApprovalRule(tool=None, action="auto_deny", condition="if_pattern", pattern=None)
        policy = ApprovalPolicy(rules=[rule], default_action="auto_approve")
        assert policy.evaluate("shell", {"cmd": "ls"}) == "auto_approve"

    def test_unknown_condition_does_not_match(self):
        rule = ApprovalRule(tool=None, action="auto_deny", condition="if_unknown")
        policy = ApprovalPolicy(rules=[rule], default_action="auto_approve")
        assert policy.evaluate("shell", {}) == "auto_approve"

    def test_complex_multi_rule_policy(self):
        """Several rules combined: deny dangerous shell, review writes, approve reads."""
        policy = ApprovalPolicy(
            rules=[
                ApprovalRule(tool="shell", action="auto_deny", condition="if_pattern", pattern=r"rm\s+-rf"),
                ApprovalRule(tool=None, action="require_review", condition="if_write"),
                ApprovalRule(tool="read_file", action="auto_approve"),
            ],
            default_action="require_review",
        )
        # Dangerous shell command
        assert policy.evaluate("shell", {"command": "rm -rf /"}, is_write=True) == "auto_deny"
        # Safe shell write
        assert policy.evaluate("shell", {"command": "echo hi"}, is_write=True) == "require_review"
        # Read (not a write, so if_write skipped; read_file matched)
        assert policy.evaluate("read_file", {"path": "/etc/hosts"}) == "auto_approve"
        # Unknown tool, not a write
        assert policy.evaluate("calculator", {"expr": "1+1"}) == "require_review"


class TestFromConfig:
    def test_basic_parsing(self):
        data = [
            {"tool": "shell", "action": "auto_deny"},
            {"tool": "read_file", "action": "auto_approve", "condition": "always"},
        ]
        policy = ApprovalPolicy.from_config(data)
        assert len(policy.rules) == 2
        assert policy.rules[0].tool == "shell"
        assert policy.rules[0].action == "auto_deny"
        assert policy.rules[1].condition == "always"
        assert policy.default_action == "require_review"

    def test_default_action_from_config(self):
        data = [
            {"tool": "shell", "action": "auto_deny"},
            {"default_action": "auto_approve"},
        ]
        policy = ApprovalPolicy.from_config(data)
        assert len(policy.rules) == 1
        assert policy.default_action == "auto_approve"

    def test_pattern_rule_from_config(self):
        data = [
            {"tool": "shell", "action": "auto_deny", "condition": "if_pattern", "pattern": r"rm\s+-rf"},
        ]
        policy = ApprovalPolicy.from_config(data)
        assert policy.rules[0].pattern == r"rm\s+-rf"

    def test_empty_config(self):
        policy = ApprovalPolicy.from_config([])
        assert len(policy.rules) == 0
        assert policy.default_action == "require_review"

    def test_wildcard_rule_from_config(self):
        data = [{"action": "require_review", "condition": "if_write"}]
        policy = ApprovalPolicy.from_config(data)
        assert policy.rules[0].tool is None
