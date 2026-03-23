"""Tool sandboxing — path restrictions, command policies, resource limits.

Provides security boundaries for built-in tools so agents can't read
sensitive files, run dangerous commands, or consume unbounded resources.

Usage::

    from agent_runner.sandbox import PathSandbox, ShellPolicy, ResourceLimits

    sandbox = PathSandbox(
        allowed_roots=["./src", "/tmp"],
        denied_patterns=["*.key", "*.pem", ".env*", "*secret*"],
    )
    sandbox.validate("/tmp/safe.txt")          # → OK
    sandbox.validate("/etc/passwd")            # → SandboxViolation
    sandbox.validate("./src/../../../etc/shadow")  # → SandboxViolation (symlink escape)

    policy = ShellPolicy(
        allow=["ls", "cat", "grep", "git", "npm", "python"],
        deny=["rm", "sudo", "curl", "wget", "chmod", "chown"],
        max_runtime=30,
        max_output_bytes=1_000_000,
    )
    policy.check("ls -la")                     # → OK
    policy.check("rm -rf /")                   # → ShellPolicyViolation

    limits = ResourceLimits(
        max_tool_calls=100,
        max_output_bytes=500_000,
        max_cost_usd=5.0,
        tool_timeout=30.0,
    )
"""

from __future__ import annotations

import fnmatch
import logging
import os
import shlex
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------

class SandboxViolation(Exception):
    """Raised when a tool call violates sandbox policy."""
    pass


class ShellPolicyViolation(SandboxViolation):
    """Raised when a shell command violates the command policy."""
    pass


# ------------------------------------------------------------------
# Path Sandbox
# ------------------------------------------------------------------

class PathSandbox:
    """Restrict file operations to allowed directories.

    Parameters
    ----------
    allowed_roots : list[str]
        Directories that file operations are restricted to.
        Paths are resolved to absolute. Default: [cwd].
    denied_patterns : list[str]
        Glob patterns for files that are always denied, even within
        allowed roots.  Example: ``["*.key", ".env*", "*secret*"]``
    """

    def __init__(
        self,
        allowed_roots: list[str] | None = None,
        denied_patterns: list[str] | None = None,
    ) -> None:
        if allowed_roots:
            self.allowed_roots = [os.path.realpath(os.path.abspath(r)) for r in allowed_roots]
        else:
            self.allowed_roots = [os.path.realpath(os.getcwd())]
        self.denied_patterns = denied_patterns or []

    def validate(self, path: str) -> str:
        """Validate and return the resolved absolute path.

        Raises ``SandboxViolation`` if the path is outside allowed roots
        or matches a denied pattern.
        """
        # Resolve to real path (follows symlinks, resolves ..)
        resolved = os.path.realpath(os.path.abspath(path))

        # Check against denied patterns (match on basename)
        basename = os.path.basename(resolved)
        for pattern in self.denied_patterns:
            if fnmatch.fnmatch(basename, pattern):
                raise SandboxViolation(
                    f"Path '{path}' matches denied pattern '{pattern}'"
                )

        # Check against allowed roots
        for root in self.allowed_roots:
            if resolved.startswith(root + os.sep) or resolved == root:
                return resolved

        raise SandboxViolation(
            f"Path '{path}' (resolved: {resolved}) is outside allowed roots: "
            f"{self.allowed_roots}"
        )


# ------------------------------------------------------------------
# Shell Policy
# ------------------------------------------------------------------

class ShellPolicy:
    """Control which shell commands the agent can execute.

    Parameters
    ----------
    allow : list[str] | None
        If set, only these base commands are allowed (allowlist mode).
    deny : list[str] | None
        Commands that are always denied (blocklist mode).
        If both allow and deny are set, deny takes precedence.
    max_runtime : float
        Maximum seconds a command can run.
    max_output_bytes : int
        Maximum bytes of stdout+stderr captured.
    """

    def __init__(
        self,
        allow: list[str] | None = None,
        deny: list[str] | None = None,
        max_runtime: float = 30.0,
        max_output_bytes: int = 1_000_000,
    ) -> None:
        self.allow = set(allow) if allow else None
        self.deny = set(deny) if deny else set()
        self.max_runtime = max_runtime
        self.max_output_bytes = max_output_bytes

    def check(self, command: str) -> str:
        """Validate a shell command.  Returns the command if OK.

        Raises ``ShellPolicyViolation`` if the command is not allowed.
        """
        base_cmd = self._extract_base_command(command)

        if base_cmd in self.deny:
            raise ShellPolicyViolation(
                f"Command '{base_cmd}' is denied by shell policy"
            )

        if self.allow is not None and base_cmd not in self.allow:
            raise ShellPolicyViolation(
                f"Command '{base_cmd}' is not in the allow list. "
                f"Allowed: {sorted(self.allow)}"
            )

        return command

    @staticmethod
    def _extract_base_command(command: str) -> str:
        """Extract the base executable from a shell command string.

        Handles pipes, semicolons, and common shell patterns.
        Returns the first command in a pipeline.
        """
        # Split on shell operators to get the first command
        for sep in (";", "&&", "||", "|"):
            command = command.split(sep)[0]

        command = command.strip()

        try:
            parts = shlex.split(command)
        except ValueError:
            # If shlex can't parse it, use simple split
            parts = command.split()

        if not parts:
            return ""

        # Handle env prefixes like "ENV=val cmd"
        for part in parts:
            if "=" not in part:
                return os.path.basename(part)

        return os.path.basename(parts[-1]) if parts else ""


# ------------------------------------------------------------------
# Resource Limits
# ------------------------------------------------------------------

@dataclass
class ResourceLimits:
    """Configurable resource limits for agent runs.

    Parameters
    ----------
    max_tool_calls : int
        Maximum total tool calls per run (0 = unlimited).
    max_output_bytes : int
        Maximum bytes returned by any single tool call.
        Output is truncated with a warning if exceeded.
    max_cost_usd : float
        Stop the run if estimated cost exceeds this (0 = unlimited).
    tool_timeout : float
        Per-tool-call timeout in seconds (0 = no timeout).
    """

    max_tool_calls: int = 0
    max_output_bytes: int = 0
    max_cost_usd: float = 0.0
    tool_timeout: float = 0.0

    def check_tool_count(self, count: int) -> None:
        """Raise if tool call count exceeds limit."""
        if self.max_tool_calls > 0 and count > self.max_tool_calls:
            raise SandboxViolation(
                f"Tool call limit exceeded: {count} > {self.max_tool_calls}"
            )

    def check_cost(self, cost: float) -> None:
        """Raise if estimated cost exceeds budget."""
        if self.max_cost_usd > 0 and cost > self.max_cost_usd:
            raise SandboxViolation(
                f"Cost budget exceeded: ${cost:.4f} > ${self.max_cost_usd:.2f}"
            )

    def truncate_output(self, output: str) -> str:
        """Truncate tool output if it exceeds the byte limit."""
        if self.max_output_bytes <= 0:
            return output
        encoded = output.encode("utf-8", errors="replace")
        if len(encoded) <= self.max_output_bytes:
            return output
        truncated = encoded[:self.max_output_bytes].decode("utf-8", errors="replace")
        return truncated + f"\n... (truncated at {self.max_output_bytes} bytes)"
