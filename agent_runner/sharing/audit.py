"""Immutable, append-only audit trail for agent runs.

Each entry is stored as a single JSON line in a ``.jsonl`` file.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class AuditEntry:
    """A single audit-log event."""

    timestamp: str  # ISO 8601
    event: str  # e.g. "run_started", "tool_called", "approval_requested", "approved", "denied", "replayed"
    run_id: str
    tool: str | None = None
    reviewer: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event": self.event,
            "run_id": self.run_id,
            "tool": self.tool,
            "reviewer": self.reviewer,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEntry:
        return cls(
            timestamp=data["timestamp"],
            event=data["event"],
            run_id=data["run_id"],
            tool=data.get("tool"),
            reviewer=data.get("reviewer"),
            details=data.get("details", {}),
        )


class AuditLog:
    """Append-only audit trail backed by a JSONL file.

    Parameters
    ----------
    path : str | Path
        Filesystem path for the log file.  Created on first write.
    """

    def __init__(self, path: str | Path = ".agent_audit.jsonl") -> None:
        self.path = Path(path)

    def record(self, entry: AuditEntry) -> None:
        """Append *entry* to the log file (one JSON line per entry)."""
        with self.path.open("a") as fh:
            fh.write(json.dumps(entry.to_dict(), default=str) + "\n")

    def query(
        self,
        run_id: str | None = None,
        event: str | None = None,
    ) -> list[AuditEntry]:
        """Read entries matching the supplied filters.

        If both *run_id* and *event* are ``None`` all entries are returned.
        """
        if not self.path.exists():
            return []

        results: list[AuditEntry] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if run_id is not None and data.get("run_id") != run_id:
                continue
            if event is not None and data.get("event") != event:
                continue
            results.append(AuditEntry.from_dict(data))
        return results

    def export_csv(self, path: str | Path) -> None:
        """Export the audit log as CSV for compliance reporting."""
        entries = self.query()
        fieldnames = ["timestamp", "event", "run_id", "tool", "reviewer", "details"]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for entry in entries:
                row = entry.to_dict()
                # Serialise the nested *details* dict so the CSV cell is a
                # flat string.
                row["details"] = json.dumps(row["details"], default=str)
                writer.writerow(row)
