"""Persistent store for agent runs awaiting approval.

Each run captures the full tool-call log, shadow captures, and approval
status.  A stakeholder can review and approve/reject via the web UI or API.
"""

from __future__ import annotations

import enum
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class RunStatus(enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REPLAYED = "replayed"


@dataclass
class RunRecord:
    """A single agent run awaiting review."""

    id: str
    created_at: float
    prompt: str
    final_text: str
    tool_call_log: list[dict[str, Any]]
    captured_writes: list[dict[str, Any]]
    status: RunStatus = RunStatus.PENDING
    reviewed_by: str | None = None
    reviewed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "prompt": self.prompt,
            "final_text": self.final_text,
            "tool_call_log": self.tool_call_log,
            "captured_writes": self.captured_writes,
            "status": self.status.value,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        return cls(
            id=data["id"],
            created_at=data["created_at"],
            prompt=data["prompt"],
            final_text=data["final_text"],
            tool_call_log=data["tool_call_log"],
            captured_writes=data["captured_writes"],
            status=RunStatus(data["status"]),
            reviewed_by=data.get("reviewed_by"),
            reviewed_at=data.get("reviewed_at"),
        )


class RunStore:
    """File-backed store for run records.

    Parameters
    ----------
    store_dir : str | Path
        Directory where run JSON files are persisted.
    """

    def __init__(self, store_dir: str | Path = ".agent_runs") -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        prompt: str,
        final_text: str,
        tool_call_log: list[dict[str, Any]],
        captured_writes: list[dict[str, Any]],
    ) -> RunRecord:
        """Create and persist a new run record.  Returns the record with its ID."""
        record = RunRecord(
            id=uuid.uuid4().hex[:12],
            created_at=time.time(),
            prompt=prompt,
            final_text=final_text,
            tool_call_log=tool_call_log,
            captured_writes=captured_writes,
        )
        self._write(record)
        return record

    def get(self, run_id: str) -> RunRecord | None:
        """Load a run record by ID."""
        path = self.store_dir / f"{run_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return RunRecord.from_dict(data)

    def list_pending(self) -> list[RunRecord]:
        """Return all runs with PENDING status."""
        records = []
        for path in sorted(self.store_dir.glob("*.json")):
            data = json.loads(path.read_text())
            record = RunRecord.from_dict(data)
            if record.status == RunStatus.PENDING:
                records.append(record)
        return records

    def approve(self, run_id: str, reviewer: str = "anonymous") -> RunRecord | None:
        """Mark a run as approved."""
        return self._update_status(run_id, RunStatus.APPROVED, reviewer)

    def reject(self, run_id: str, reviewer: str = "anonymous") -> RunRecord | None:
        """Mark a run as rejected."""
        return self._update_status(run_id, RunStatus.REJECTED, reviewer)

    def mark_replayed(self, run_id: str) -> RunRecord | None:
        """Mark a run as replayed (writes executed)."""
        return self._update_status(run_id, RunStatus.REPLAYED)

    def _update_status(
        self, run_id: str, status: RunStatus, reviewer: str | None = None
    ) -> RunRecord | None:
        record = self.get(run_id)
        if record is None:
            return None
        record.status = status
        record.reviewed_by = reviewer
        record.reviewed_at = time.time()
        self._write(record)
        return record

    def _write(self, record: RunRecord) -> None:
        path = self.store_dir / f"{record.id}.json"
        path.write_text(json.dumps(record.to_dict(), indent=2, default=str))
