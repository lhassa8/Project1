"""Transaction-safe replay of captured shadow writes.

Backs up originals before writing, rolls back on failure.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any

from agent_runner.shadow.state import ShadowState

logger = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    """Outcome of a replay attempt."""

    success: bool = True
    completed: list[dict[str, Any]] = field(default_factory=list)
    failed: dict[str, Any] | None = None
    rolled_back: bool = False


class TransactionReplay:
    """Execute shadow-captured writes with rollback on failure.

    Usage::

        replay = TransactionReplay(shadow_state)
        result = replay.execute()
        if result.success:
            print(f"Applied {len(result.completed)} changes")
        else:
            print(f"Failed at: {result.failed}")
            assert result.rolled_back  # originals restored
    """

    def __init__(self, state: ShadowState) -> None:
        self.state = state
        self._backups: list[tuple[str, str | None]] = []
        # (original_path, backup_path_or_None_if_didnt_exist)
        self._backup_dir: str | None = None

    def execute(self, dry_run: bool = False) -> ReplayResult:
        """Apply all captured changes to the real filesystem.

        Parameters
        ----------
        dry_run : bool
            If True, validate that all operations *could* succeed without
            actually writing anything.
        """
        changes = self.state.get_all_changes()
        if not changes:
            return ReplayResult(success=True)

        if dry_run:
            return self._dry_run(changes)

        # Create temp dir for backups
        self._backup_dir = tempfile.mkdtemp(prefix="agent_runner_backup_")
        result = ReplayResult()

        try:
            for change in changes:
                self._backup(change["path"])
                self._apply(change)
                result.completed.append(change)
        except Exception as exc:
            logger.error("Replay failed at %s: %s", change.get("path", "?"), exc)
            result.success = False
            result.failed = {**change, "error": str(exc)}
            self._rollback()
            result.rolled_back = True
        finally:
            self._cleanup_backups(keep=not result.success)

        return result

    def _dry_run(self, changes: list[dict[str, Any]]) -> ReplayResult:
        """Check that all operations are feasible without writing."""
        result = ReplayResult()
        for change in changes:
            path = change["path"]
            action = change["action"]

            if action in ("create", "modify"):
                parent = os.path.dirname(path)
                if not os.path.isdir(parent) and action != "mkdir":
                    result.success = False
                    result.failed = {**change, "error": f"Parent directory does not exist: {parent}"}
                    return result

            elif action == "delete":
                if not os.path.exists(path):
                    result.success = False
                    result.failed = {**change, "error": f"File does not exist: {path}"}
                    return result

            result.completed.append(change)
        return result

    def _backup(self, path: str) -> None:
        """Save the original file so we can restore on failure."""
        if os.path.exists(path):
            backup_path = os.path.join(self._backup_dir, os.path.basename(path) + f".{len(self._backups)}")
            shutil.copy2(path, backup_path)
            self._backups.append((path, backup_path))
        else:
            self._backups.append((path, None))

    def _apply(self, change: dict[str, Any]) -> None:
        """Apply a single change to the real filesystem."""
        action = change["action"]
        path = change["path"]

        if action == "mkdir":
            os.makedirs(path, exist_ok=True)
            logger.info("Created directory: %s", path)

        elif action == "create" or action == "modify":
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w") as f:
                f.write(change["content"])
            logger.info("Wrote %d bytes: %s", change["size"], path)

        elif action == "delete":
            os.remove(path)
            logger.info("Deleted: %s", path)

    def _rollback(self) -> None:
        """Restore all backed-up originals in reverse order."""
        logger.warning("Rolling back %d changes", len(self._backups))
        for original_path, backup_path in reversed(self._backups):
            try:
                if backup_path is not None:
                    # Restore original
                    shutil.copy2(backup_path, original_path)
                    logger.info("Restored: %s", original_path)
                else:
                    # File didn't exist before — remove it
                    if os.path.exists(original_path):
                        os.remove(original_path)
                        logger.info("Removed (didn't exist before): %s", original_path)
            except Exception as exc:
                logger.error("Rollback failed for %s: %s", original_path, exc)

    def _cleanup_backups(self, keep: bool = False) -> None:
        """Remove the temporary backup directory."""
        if self._backup_dir and os.path.isdir(self._backup_dir):
            if not keep:
                shutil.rmtree(self._backup_dir, ignore_errors=True)
            else:
                logger.info("Backup preserved at: %s", self._backup_dir)
        self._backups.clear()
        self._backup_dir = None
