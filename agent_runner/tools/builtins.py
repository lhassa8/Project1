"""Built-in tools that ship with the runner.

These are simple, safe tools useful for demos and testing.
Production deployments will register their own domain-specific tools.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from typing import Any

from agent_runner.tools.registry import ToolRegistry


def register_builtins(registry: ToolRegistry) -> None:
    """Add the default tool set to *registry*."""

    @registry.tool(
        name="calculator",
        description="Evaluate a mathematical expression. Supports basic arithmetic, exponents, and common math functions.",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. '2 + 3 * 4' or 'sqrt(144)'.",
                }
            },
            "required": ["expression"],
        },
        category="compute",
    )
    def calculator(params: dict[str, Any]) -> str:
        expr = params["expression"]
        # Allow only safe math builtins
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed["abs"] = abs
        allowed["round"] = round
        try:
            result = eval(expr, {"__builtins__": {}}, allowed)  # noqa: S307
            return str(result)
        except Exception as exc:
            return f"Error: {exc}"

    @registry.tool(
        name="shell",
        description="Execute a shell command and return its stdout and stderr. Use with caution.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run.",
                }
            },
            "required": ["command"],
        },
        category="system",
        is_write=True,
    )
    def shell(params: dict[str, Any]) -> str:
        cmd = params["command"]
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = ""
            if proc.stdout:
                output += proc.stdout
            if proc.stderr:
                output += f"\n[stderr]\n{proc.stderr}"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 30 seconds"
        except Exception as exc:
            return f"Error: {exc}"

    @registry.tool(
        name="read_file",
        description="Read the contents of a file and return it as text.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read.",
                }
            },
            "required": ["path"],
        },
        category="file",
    )
    def read_file(params: dict[str, Any]) -> str:
        try:
            with open(params["path"]) as f:
                return f.read()
        except Exception as exc:
            return f"Error: {exc}"

    @registry.tool(
        name="list_files",
        description="List files and directories at a given path. Returns names with trailing / for directories.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list. Defaults to current directory.",
                    "default": ".",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "If true, list files recursively (max 200 entries).",
                    "default": False,
                },
            },
        },
        category="file",
    )
    def list_files(params: dict[str, Any]) -> str:
        path = params.get("path", ".")
        recursive = params.get("recursive", False)
        try:
            if recursive:
                entries = []
                for root, dirs, files in os.walk(path):
                    for d in dirs:
                        entries.append(os.path.relpath(os.path.join(root, d), path) + "/")
                    for f in files:
                        entries.append(os.path.relpath(os.path.join(root, f), path))
                    if len(entries) >= 200:
                        entries = entries[:200]
                        entries.append("... (truncated at 200 entries)")
                        break
            else:
                raw = os.listdir(path)
                entries = []
                for name in sorted(raw):
                    full = os.path.join(path, name)
                    entries.append(name + "/" if os.path.isdir(full) else name)
            return "\n".join(entries) if entries else "(empty directory)"
        except Exception as exc:
            return f"Error: {exc}"

    @registry.tool(
        name="write_file",
        description="Write content to a file, creating it if it doesn't exist.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
        category="file",
        is_write=True,
    )
    def write_file(params: dict[str, Any]) -> str:
        try:
            with open(params["path"], "w") as f:
                f.write(params["content"])
            return f"Wrote {len(params['content'])} bytes to {params['path']}"
        except Exception as exc:
            return f"Error: {exc}"
