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

    @classmethod
    def from_file(cls, path: str | Path) -> AgentConfig:
        """Load config from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                data = yaml.safe_load(text)
            except ImportError:
                raise ImportError("PyYAML is required for YAML config files: pip install pyyaml")
        else:
            data = json.loads(text)

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
            system_prompt=data.get("system_prompt", cls.system_prompt),
            shadow=data.get("shadow", False),
            share=data.get("share", False),
            stream=data.get("stream", False),
            approve=data.get("approve", []),
            mcp=mcp,
            tools=data.get("tools", {}),
        )

    @classmethod
    def discover(cls) -> AgentConfig | None:
        """Look for a config file in the default locations."""
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
