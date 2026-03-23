"""Human-readable diff generation for shadow mode changes.

Produces clear summaries of what a shadow run *would* do, suitable for
review by humans or automated approval systems.
"""

from __future__ import annotations

import difflib
import os
from typing import Any

from agent_runner.shadow.state import ShadowState


class ShadowDiff:
    """Generate diffs and summaries from a ShadowState."""

    def __init__(self, state: ShadowState) -> None:
        self.state = state

    def summary(self) -> str:
        """One-line-per-change summary suitable for terminal display.

        Example output::

            Shadow Run Summary (3 changes):
              CREATE  config/app.yaml       (142 bytes)
              MODIFY  src/main.py           (+12 / -3 lines)
              DELETE  old_config.json
        """
        changes = self.state.get_all_changes()
        if not changes:
            return "Shadow Run Summary: no changes captured."

        lines = [f"Shadow Run Summary ({len(changes)} changes):"]
        for change in changes:
            action = change["action"].upper()
            path = _short_path(change["path"])

            if action == "CREATE":
                lines.append(f"  {action:8s} {path:40s} ({change['size']} bytes)")
            elif action == "MODIFY":
                diff_stats = self._diff_stats(change["path"], change["content"])
                lines.append(f"  {action:8s} {path:40s} ({diff_stats})")
            elif action == "DELETE":
                lines.append(f"  {action:8s} {path}")
            elif action == "MKDIR":
                lines.append(f"  {action:8s} {path}/")

        return "\n".join(lines)

    def full_diff(self) -> str:
        """Unified diff of all file changes, like ``git diff``.

        Returns a string with standard unified diff format for each
        modified or created file.
        """
        changes = self.state.get_all_changes()
        parts: list[str] = []

        for change in changes:
            if change["action"] in ("mkdir",):
                continue

            path = change["path"]
            if change["action"] == "delete":
                original = _read_original(path)
                if original is not None:
                    diff = difflib.unified_diff(
                        original.splitlines(keepends=True),
                        [],
                        fromfile=f"a/{_short_path(path)}",
                        tofile="/dev/null",
                    )
                    parts.append("".join(diff))
            elif change["action"] == "create":
                diff = difflib.unified_diff(
                    [],
                    change["content"].splitlines(keepends=True),
                    fromfile="/dev/null",
                    tofile=f"b/{_short_path(path)}",
                )
                parts.append("".join(diff))
            elif change["action"] == "modify":
                original = _read_original(path)
                if original is not None:
                    diff = difflib.unified_diff(
                        original.splitlines(keepends=True),
                        change["content"].splitlines(keepends=True),
                        fromfile=f"a/{_short_path(path)}",
                        tofile=f"b/{_short_path(path)}",
                    )
                    parts.append("".join(diff))

        return "\n".join(parts) if parts else "(no file diffs)"

    def to_dict(self) -> list[dict[str, Any]]:
        """Structured representation for programmatic consumption."""
        return self.state.get_all_changes()

    def _diff_stats(self, path: str, new_content: str) -> str:
        """Return '+N / -M lines' stats for a modified file."""
        original = _read_original(path)
        if original is None:
            return f"{len(new_content)} bytes"

        old_lines = original.splitlines()
        new_lines = new_content.splitlines()
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

        added = 0
        removed = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                removed += i2 - i1
                added += j2 - j1
            elif tag == "delete":
                removed += i2 - i1
            elif tag == "insert":
                added += j2 - j1

        return f"+{added} / -{removed} lines"


def _short_path(path: str) -> str:
    """Shorten absolute paths relative to CWD for display."""
    try:
        return os.path.relpath(path)
    except ValueError:
        return path


def _read_original(path: str) -> str | None:
    """Read a file's current contents, or None if it doesn't exist."""
    try:
        with open(path) as f:
            return f.read()
    except (OSError, IOError):
        return None
