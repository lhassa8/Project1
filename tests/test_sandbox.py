"""Tests for tool sandboxing — path restrictions, shell policy, resource limits."""

from __future__ import annotations

import os
import pytest

from agent_runner.sandbox import (
    PathSandbox,
    ShellPolicy,
    ResourceLimits,
    SandboxViolation,
    ShellPolicyViolation,
)
from agent_runner.interceptors.logging import LoggingInterceptor
from agent_runner.tools.registry import ToolRegistry
from agent_runner.tools.builtins import register_builtins


class TestPathSandbox:
    def test_allows_file_in_root(self, tmp_path):
        sandbox = PathSandbox(allowed_roots=[str(tmp_path)])
        result = sandbox.validate(str(tmp_path / "safe.txt"))
        assert result == os.path.realpath(str(tmp_path / "safe.txt"))

    def test_rejects_file_outside_root(self, tmp_path):
        sandbox = PathSandbox(allowed_roots=[str(tmp_path)])
        with pytest.raises(SandboxViolation, match="outside allowed roots"):
            sandbox.validate("/etc/passwd")

    def test_rejects_dotdot_escape(self, tmp_path):
        sandbox = PathSandbox(allowed_roots=[str(tmp_path)])
        with pytest.raises(SandboxViolation, match="outside allowed roots"):
            sandbox.validate(str(tmp_path / ".." / ".." / "etc" / "passwd"))

    def test_denied_pattern_blocks_file(self, tmp_path):
        sandbox = PathSandbox(
            allowed_roots=[str(tmp_path)],
            denied_patterns=["*.key", "*.pem"],
        )
        with pytest.raises(SandboxViolation, match="denied pattern"):
            sandbox.validate(str(tmp_path / "server.key"))

    def test_denied_pattern_env(self, tmp_path):
        sandbox = PathSandbox(
            allowed_roots=[str(tmp_path)],
            denied_patterns=[".env*"],
        )
        with pytest.raises(SandboxViolation, match="denied pattern"):
            sandbox.validate(str(tmp_path / ".env.production"))

    def test_multiple_roots(self, tmp_path):
        root1 = tmp_path / "src"
        root2 = tmp_path / "config"
        root1.mkdir()
        root2.mkdir()

        sandbox = PathSandbox(allowed_roots=[str(root1), str(root2)])
        sandbox.validate(str(root1 / "main.py"))
        sandbox.validate(str(root2 / "app.yaml"))
        with pytest.raises(SandboxViolation):
            sandbox.validate("/etc/shadow")

    def test_default_root_is_cwd(self):
        sandbox = PathSandbox()
        cwd = os.path.realpath(os.getcwd())
        # File in CWD should be allowed
        result = sandbox.validate(os.path.join(os.getcwd(), "test.txt"))
        assert result.startswith(cwd)


class TestShellPolicy:
    def test_allow_listed_command(self):
        policy = ShellPolicy(allow=["ls", "cat", "grep"])
        assert policy.check("ls -la") == "ls -la"

    def test_deny_listed_command(self):
        policy = ShellPolicy(deny=["rm", "sudo"])
        with pytest.raises(ShellPolicyViolation, match="denied"):
            policy.check("rm -rf /")

    def test_deny_overrides_allow(self):
        policy = ShellPolicy(allow=["rm", "ls"], deny=["rm"])
        with pytest.raises(ShellPolicyViolation, match="denied"):
            policy.check("rm file.txt")

    def test_command_not_in_allowlist(self):
        policy = ShellPolicy(allow=["ls", "cat"])
        with pytest.raises(ShellPolicyViolation, match="not in the allow list"):
            policy.check("curl http://evil.com")

    def test_no_policy_allows_everything(self):
        policy = ShellPolicy()
        assert policy.check("anything --works") == "anything --works"

    def test_pipeline_checks_first_command(self):
        policy = ShellPolicy(deny=["rm"])
        with pytest.raises(ShellPolicyViolation):
            policy.check("rm -rf / | grep something")

    def test_semicolon_checks_first_command(self):
        policy = ShellPolicy(deny=["rm"])
        with pytest.raises(ShellPolicyViolation):
            policy.check("rm -rf /; echo done")

    def test_extracts_basename(self):
        policy = ShellPolicy(allow=["python"])
        assert policy.check("/usr/bin/python script.py") == "/usr/bin/python script.py"

    def test_env_prefix_handled(self):
        policy = ShellPolicy(allow=["python"])
        assert policy.check("PYTHONPATH=/tmp python app.py") == "PYTHONPATH=/tmp python app.py"


class TestResourceLimits:
    def test_tool_count_limit(self):
        limits = ResourceLimits(max_tool_calls=5)
        limits.check_tool_count(5)  # OK
        with pytest.raises(SandboxViolation, match="Tool call limit"):
            limits.check_tool_count(6)

    def test_cost_limit(self):
        limits = ResourceLimits(max_cost_usd=1.0)
        limits.check_cost(0.50)  # OK
        with pytest.raises(SandboxViolation, match="Cost budget"):
            limits.check_cost(1.50)

    def test_output_truncation(self):
        limits = ResourceLimits(max_output_bytes=20)
        assert limits.truncate_output("short") == "short"
        long_output = "x" * 100
        truncated = limits.truncate_output(long_output)
        assert len(truncated) < 100
        assert "truncated" in truncated

    def test_zero_means_unlimited(self):
        limits = ResourceLimits()  # all zeros
        limits.check_tool_count(999999)
        limits.check_cost(999999.0)
        assert limits.truncate_output("x" * 10000) == "x" * 10000


class TestPIISafeLogging:
    def test_redacts_api_key(self):
        li = LoggingInterceptor()
        li.intercept("tool", {"api_key": "sk-secret-123", "query": "hello"})
        assert li.log[0]["input"]["api_key"] == "sk-secret-123"  # raw log preserved
        # But the logger output was sanitized (we can check _sanitize directly)
        sanitized = li._sanitize({"api_key": "sk-secret-123", "query": "hello"})
        assert "REDACTED" in sanitized
        assert "sk-secret-123" not in sanitized

    def test_redacts_password(self):
        li = LoggingInterceptor()
        sanitized = li._sanitize({"password": "hunter2", "user": "admin"})
        assert "REDACTED" in sanitized
        assert "hunter2" not in sanitized

    def test_redacts_nested(self):
        li = LoggingInterceptor()
        sanitized = li._sanitize({"config": {"token": "abc123"}, "name": "test"})
        assert "REDACTED" in sanitized
        assert "abc123" not in sanitized

    def test_truncates_long_input(self):
        li = LoggingInterceptor(max_input_length=50)
        sanitized = li._sanitize({"data": "x" * 200})
        assert len(sanitized) < 200
        assert "truncated" in sanitized

    def test_no_redaction_when_disabled(self):
        li = LoggingInterceptor(redact_fields=set())
        sanitized = li._sanitize({"api_key": "visible"})
        assert "visible" in sanitized


class TestSandboxedBuiltins:
    """Integration: builtins with sandbox enabled."""

    def test_read_file_blocked_by_sandbox(self, tmp_path):
        sandbox = PathSandbox(allowed_roots=[str(tmp_path)])
        registry = ToolRegistry()
        register_builtins(registry, path_sandbox=sandbox)

        handler = registry.get("read_file")
        result = handler({"path": "/etc/passwd"})
        assert "Error" in result
        assert "outside allowed roots" in result

    def test_read_file_allowed_in_sandbox(self, tmp_path):
        real = tmp_path / "safe.txt"
        real.write_text("safe content")

        sandbox = PathSandbox(allowed_roots=[str(tmp_path)])
        registry = ToolRegistry()
        register_builtins(registry, path_sandbox=sandbox)

        handler = registry.get("read_file")
        result = handler({"path": str(real)})
        assert result == "safe content"

    def test_write_file_blocked_by_sandbox(self, tmp_path):
        sandbox = PathSandbox(allowed_roots=[str(tmp_path)])
        registry = ToolRegistry()
        register_builtins(registry, path_sandbox=sandbox)

        handler = registry.get("write_file")
        result = handler({"path": "/etc/evil.txt", "content": "bad"})
        assert "Error" in result

    def test_shell_blocked_by_policy(self):
        policy = ShellPolicy(deny=["rm", "sudo"])
        registry = ToolRegistry()
        register_builtins(registry, shell_policy=policy)

        handler = registry.get("shell")
        result = handler({"command": "rm -rf /"})
        assert "Error" in result
        assert "denied" in result

    def test_shell_allowed_by_policy(self):
        policy = ShellPolicy(allow=["echo"])
        registry = ToolRegistry()
        register_builtins(registry, shell_policy=policy)

        handler = registry.get("shell")
        result = handler({"command": "echo hello"})
        assert "hello" in result
