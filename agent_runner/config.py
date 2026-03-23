"""Load runner configuration from a JSON or YAML file.

Supports a single config file that replaces all CLI flags::

    {
      "model": "claude-sonnet-4-20250514",
      "max_turns": 25,
      "system_prompt": "You are a helpful assistant...",
      "shadow": true,
      "share": false,
      "approve": ["shell", "write_file"],
      "mcp": {
        "command": "npx -y @modelcontextprotocol/server-filesystem /tmp",
        "write_tools": ["write_file", "create_directory"]
      },
      "tools": {
        "shell": true,
        "write_file": true,
        "read_file": true,
        "calculator": true
      }
    }
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATHS = ["agent.json", "agent.yaml", "agent.yml"]


@dataclass
class MCPConfig:
    command: str = ""
    write_tools: list[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    """Merged configuration from file + CLI overrides."""

    model: str | None = None
    max_turns: int = 25
    max_tokens: int = 4096
    system_prompt: str = (
        "You are a helpful assistant with access to tools. "
        "Use the available tools to accomplish the user's request. "
        "Think step-by-step and use tools as needed."
    )
    shadow: bool = False
    share: bool = False
    stream: bool = False
    approve: list[str] = field(default_factory=list)
    mcp: MCPConfig | None = None
    tools: dict[str, bool] = field(default_factory=dict)
    # Security & sandbox
    sandbox_roots: list[str] = field(default_factory=list)
    sandbox_deny: list[str] = field(default_factory=lambda: ["*.key", "*.pem", ".env*"])
    shell_allow: list[str] = field(default_factory=list)
    shell_deny: list[str] = field(default_factory=lambda: ["sudo"])
    # Resource limits
    max_tool_calls: int = 0
    max_cost_usd: float = 0.0
    # Observability
    audit: bool = False
    audit_path: str = ".agent_audit.jsonl"
    # Approval policy
    approval_policy: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> AgentConfig:
        """Load config from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        text = path.read_text()
        try:
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    data = yaml.safe_load(text)
                except ImportError:
                    raise ImportError("PyYAML is required for YAML config files: pip install pyyaml")
            else:
                data = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Failed to parse config file {path}: {exc}") from exc

        # Validate against schema before parsing
        from agent_runner.config_schema import ConfigError, validate_config

        errors = validate_config(data)
        if errors:
            raise ConfigError(errors)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        mcp_data = data.get("mcp")
        mcp = None
        if mcp_data:
            mcp = MCPConfig(
                command=mcp_data.get("command", ""),
                write_tools=mcp_data.get("write_tools", []),
            )

        return cls(
            model=data.get("model"),
            max_turns=data.get("max_turns", 25),
            max_tokens=data.get("max_tokens", 4096),
            system_prompt=data.get("system_prompt", cls.system_prompt),
            shadow=data.get("shadow", False),
            share=data.get("share", False),
            stream=data.get("stream", False),
            approve=data.get("approve", []),
            mcp=mcp,
            tools=data.get("tools", {}),
            sandbox_roots=data.get("sandbox_roots", []),
            sandbox_deny=data.get("sandbox_deny", ["*.key", "*.pem", ".env*"]),
            shell_allow=data.get("shell_allow", []),
            shell_deny=data.get("shell_deny", ["sudo"]),
            max_tool_calls=data.get("max_tool_calls", 0),
            max_cost_usd=data.get("max_cost_usd", 0.0),
            audit=data.get("audit", False),
            audit_path=data.get("audit_path", ".agent_audit.jsonl"),
            approval_policy=data.get("approval_policy", []),
        )

    @classmethod
    def discover(cls) -> AgentConfig | None:
        """Look for a config file — checks ``AGENT_CONFIG`` env var first,
        then default file locations (``agent.json``, ``agent.yaml``)."""
        env_path = os.getenv("AGENT_CONFIG")
        if env_path:
            logger.info("Using config from AGENT_CONFIG=%s", env_path)
            return cls.from_file(env_path)
        for name in DEFAULT_CONFIG_PATHS:
            path = Path(name)
            if path.exists():
                logger.info("Using config file: %s", path)
                return cls.from_file(path)
        return None

    def merge_cli(self, args: Any) -> AgentConfig:
        """Override config values with non-default CLI arguments."""
        if getattr(args, "model", None):
            self.model = args.model
        if getattr(args, "max_turns", None) and args.max_turns != 25:
            self.max_turns = args.max_turns
        if getattr(args, "max_tokens", None) and args.max_tokens != 4096:
            self.max_tokens = args.max_tokens
        if getattr(args, "shadow", False):
            self.shadow = True
        if getattr(args, "share", False):
            self.share = True
        if getattr(args, "stream", False):
            self.stream = True
        if getattr(args, "approve", None):
            self.approve = [t.strip() for t in args.approve.split(",")]
        if getattr(args, "mcp", None):
            self.mcp = MCPConfig(command=args.mcp)
            if getattr(args, "mcp_writes", None):
                self.mcp.write_tools = [t.strip() for t in args.mcp_writes.split(",")]
        return self
