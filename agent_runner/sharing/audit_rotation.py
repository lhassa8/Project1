"""Audit log rotation and retention policies.

Provides automatic rotation of JSONL audit log files by size or age,
with configurable retention and compression of archived logs.

Usage::

    from agent_runner.sharing.audit_rotation import RotatingAuditLog

    audit = RotatingAuditLog(
        path=".agent_audit.jsonl",
        max_bytes=10_000_000,      # rotate at 10MB
        backup_count=5,            # keep 5 archived logs
        compress=True,             # gzip old logs
        max_age_days=90,           # delete archives older than 90 days
    )
    audit.record(entry)
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_runner.sharing.audit import AuditEntry, AuditLog

logger = logging.getLogger(__name__)


@dataclass
class RetentionPolicy:
    """Defines how long audit logs are kept.

    Parameters
    ----------
    max_age_days : int
        Delete archived logs older than this (0 = keep forever).
    max_total_bytes : int
        Maximum total size of all archived logs (0 = unlimited).
        Oldest archives are deleted first when exceeded.
    max_files : int
        Maximum number of archived log files (0 = unlimited).
    """

    max_age_days: int = 90
    max_total_bytes: int = 0
    max_files: int = 10


class RotatingAuditLog(AuditLog):
    """Audit log with automatic rotation by size and retention policies.

    Extends ``AuditLog`` to add automatic rotation when the log file
    exceeds ``max_bytes``.  Archived logs are optionally compressed
    with gzip and subject to retention policies.

    Parameters
    ----------
    path : str | Path
        Base path for the active log file.
    max_bytes : int
        Rotate when the active file exceeds this size (default 10MB).
    backup_count : int
        Number of rotated backups to keep (default 5).
    compress : bool
        If True, compress rotated logs with gzip.
    retention : RetentionPolicy | None
        Retention policy for archived logs.
    """

    def __init__(
        self,
        path: str | Path = ".agent_audit.jsonl",
        max_bytes: int = 10_000_000,
        backup_count: int = 5,
        compress: bool = True,
        retention: RetentionPolicy | None = None,
    ) -> None:
        super().__init__(path=path)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.compress = compress
        self.retention = retention or RetentionPolicy()
        self._check_count = 0

    def record(self, entry: AuditEntry) -> None:
        """Append entry and rotate if needed."""
        super().record(entry)

        # Check rotation every 100 writes to avoid stat() on every call
        self._check_count += 1
        if self._check_count >= 100:
            self._check_count = 0
            self._maybe_rotate()

    def force_rotate(self) -> None:
        """Force an immediate log rotation regardless of size."""
        if self.path.exists() and self.path.stat().st_size > 0:
            self._rotate()

    def apply_retention(self) -> int:
        """Apply retention policy and return number of files deleted."""
        return self._enforce_retention()

    def list_archives(self) -> list[Path]:
        """Return all archived log files, newest first."""
        base = self.path
        archives = []
        for i in range(1, self.backup_count + 1):
            candidate = Path(f"{base}.{i}")
            compressed = Path(f"{base}.{i}.gz")
            if compressed.exists():
                archives.append(compressed)
            elif candidate.exists():
                archives.append(candidate)
        return archives

    def total_archive_size(self) -> int:
        """Return total size of all archived logs in bytes."""
        return sum(f.stat().st_size for f in self.list_archives() if f.exists())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _maybe_rotate(self) -> None:
        """Rotate if the active log exceeds max_bytes."""
        if not self.path.exists():
            return
        if self.path.stat().st_size >= self.max_bytes:
            self._rotate()

    def _rotate(self) -> None:
        """Perform the actual rotation.

        Shifts existing archives up by one (log.5 -> log.6 -> deleted,
        log.4 -> log.5, etc.), then renames the active log to log.1.
        """
        logger.info("Rotating audit log: %s (size: %d bytes)",
                     self.path, self.path.stat().st_size)

        # Delete the oldest backup if it would exceed backup_count
        for i in range(self.backup_count, 0, -1):
            src = Path(f"{self.path}.{i}")
            src_gz = Path(f"{self.path}.{i}.gz")
            dst = Path(f"{self.path}.{i + 1}")
            dst_gz = Path(f"{self.path}.{i + 1}.gz")

            if i >= self.backup_count:
                # Delete oldest
                src.unlink(missing_ok=True)
                src_gz.unlink(missing_ok=True)
            else:
                # Shift up
                if src_gz.exists():
                    dst_gz.unlink(missing_ok=True)
                    src_gz.rename(dst_gz)
                elif src.exists():
                    dst.unlink(missing_ok=True)
                    src.rename(dst)

        # Move active log to .1
        target = Path(f"{self.path}.1")
        shutil.move(str(self.path), str(target))

        # Compress if enabled
        if self.compress:
            self._compress_file(target)

        # Apply retention policy
        self._enforce_retention()

    def _compress_file(self, path: Path) -> None:
        """Gzip a file in place."""
        gz_path = Path(f"{path}.gz")
        try:
            with open(path, "rb") as f_in:
                with gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            path.unlink()
            logger.debug("Compressed %s -> %s", path, gz_path)
        except Exception as exc:
            logger.error("Failed to compress %s: %s", path, exc)

    def _enforce_retention(self) -> int:
        """Delete archives that violate the retention policy. Returns count deleted."""
        deleted = 0
        archives = self.list_archives()

        # Age-based retention
        if self.retention.max_age_days > 0:
            cutoff = time.time() - (self.retention.max_age_days * 86400)
            for archive in archives:
                if archive.exists() and archive.stat().st_mtime < cutoff:
                    archive.unlink()
                    deleted += 1
                    logger.info("Retention: deleted %s (too old)", archive)

        # Refresh list after age-based deletion
        archives = [a for a in self.list_archives() if a.exists()]

        # File count retention
        if self.retention.max_files > 0 and len(archives) > self.retention.max_files:
            for archive in archives[self.retention.max_files:]:
                if archive.exists():
                    archive.unlink()
                    deleted += 1
                    logger.info("Retention: deleted %s (too many files)", archive)

        # Size-based retention
        if self.retention.max_total_bytes > 0:
            archives = [a for a in self.list_archives() if a.exists()]
            total = sum(a.stat().st_size for a in archives)
            # Delete oldest first until under limit
            for archive in reversed(archives):
                if total <= self.retention.max_total_bytes:
                    break
                if archive.exists():
                    size = archive.stat().st_size
                    archive.unlink()
                    total -= size
                    deleted += 1
                    logger.info("Retention: deleted %s (total size exceeded)", archive)

        return deleted

    def query_archives(
        self,
        run_id: str | None = None,
        event: str | None = None,
    ) -> list[AuditEntry]:
        """Query across all archives (active log + compressed backups).

        Slower than querying just the active log, but useful for compliance.
        """
        results = self.query(run_id=run_id, event=event)

        for archive in self.list_archives():
            if not archive.exists():
                continue
            try:
                if str(archive).endswith(".gz"):
                    with gzip.open(archive, "rt") as f:
                        lines = f.read().splitlines()
                else:
                    lines = archive.read_text().splitlines()

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if run_id is not None and data.get("run_id") != run_id:
                        continue
                    if event is not None and data.get("event") != event:
                        continue
                    results.append(AuditEntry.from_dict(data))
            except Exception as exc:
                logger.warning("Failed to read archive %s: %s", archive, exc)

        return results
