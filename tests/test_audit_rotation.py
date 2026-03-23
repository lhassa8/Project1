"""Tests for audit log rotation and retention policies."""

import gzip
import json
import time
from pathlib import Path

import pytest

from agent_runner.sharing.audit import AuditEntry
from agent_runner.sharing.audit_rotation import (
    RetentionPolicy,
    RotatingAuditLog,
)


def _make_entry(run_id: str = "test123", event: str = "tool_called") -> AuditEntry:
    return AuditEntry(
        timestamp="2024-01-01T00:00:00Z",
        event=event,
        run_id=run_id,
        tool="calculator",
        details={"expression": "1+1"},
    )


class TestRotatingAuditLog:
    def test_basic_record_works(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        audit = RotatingAuditLog(path=log_path, max_bytes=10_000)

        audit.record(_make_entry())
        assert log_path.exists()
        lines = log_path.read_text().splitlines()
        assert len(lines) == 1

    def test_rotation_on_size(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        # Very small max_bytes to trigger rotation quickly
        audit = RotatingAuditLog(
            path=log_path, max_bytes=200, backup_count=3, compress=False,
        )
        # Force check on every write
        audit._check_count = 99

        # Write enough entries to trigger rotation
        for i in range(20):
            audit._check_count = 99  # force size check
            audit.record(_make_entry(run_id=f"run_{i}"))

        # Should have rotated at least once
        archives = audit.list_archives()
        assert len(archives) >= 1

    def test_force_rotate(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        audit = RotatingAuditLog(path=log_path, max_bytes=10_000_000, compress=False)

        audit.record(_make_entry())
        assert log_path.exists()

        audit.force_rotate()
        # Active log should be gone (moved to .1)
        archive = Path(f"{log_path}.1")
        assert archive.exists()

    def test_compression(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        audit = RotatingAuditLog(
            path=log_path, max_bytes=10_000_000, compress=True,
        )

        for i in range(10):
            audit.record(_make_entry(run_id=f"run_{i}"))

        audit.force_rotate()
        gz_path = Path(f"{log_path}.1.gz")
        assert gz_path.exists()
        plain_path = Path(f"{log_path}.1")
        assert not plain_path.exists()

        # Verify compressed content is valid
        with gzip.open(gz_path, "rt") as f:
            lines = f.read().splitlines()
        assert len(lines) == 10

    def test_backup_count_limit(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        audit = RotatingAuditLog(
            path=log_path, max_bytes=100, backup_count=2, compress=False,
        )

        # Do multiple rotations
        for i in range(5):
            for j in range(10):
                audit._check_count = 99
                audit.record(_make_entry(run_id=f"run_{i}_{j}"))

        # Should have at most backup_count archives
        archives = audit.list_archives()
        assert len(archives) <= 2

    def test_retention_by_age(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        audit = RotatingAuditLog(
            path=log_path,
            max_bytes=10_000_000,
            compress=False,
            retention=RetentionPolicy(max_age_days=0),  # 0 = keep forever
        )

        audit.record(_make_entry())
        audit.force_rotate()

        # With max_age_days=0, nothing should be deleted
        archives = audit.list_archives()
        assert len(archives) == 1

    def test_retention_by_file_count(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        audit = RotatingAuditLog(
            path=log_path,
            max_bytes=50,
            backup_count=10,
            compress=False,
            retention=RetentionPolicy(max_files=2),
        )

        for i in range(8):
            for j in range(5):
                audit._check_count = 99
                audit.record(_make_entry(run_id=f"run_{i}_{j}"))

        archives = audit.list_archives()
        assert len(archives) <= 2

    def test_retention_by_total_size(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        audit = RotatingAuditLog(
            path=log_path,
            max_bytes=100,
            backup_count=10,
            compress=False,
            retention=RetentionPolicy(max_total_bytes=500, max_files=0),
        )

        for i in range(10):
            for j in range(5):
                audit._check_count = 99
                audit.record(_make_entry(run_id=f"run_{i}_{j}"))

        total = audit.total_archive_size()
        assert total <= 500 or total == 0  # either under limit or all cleaned

    def test_list_archives_empty(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        audit = RotatingAuditLog(path=log_path)
        assert audit.list_archives() == []

    def test_query_active_log(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        audit = RotatingAuditLog(path=log_path)

        audit.record(_make_entry(run_id="abc", event="run_started"))
        audit.record(_make_entry(run_id="def", event="tool_called"))

        results = audit.query(run_id="abc")
        assert len(results) == 1
        assert results[0].run_id == "abc"

    def test_query_archives_across_rotations(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        audit = RotatingAuditLog(
            path=log_path, max_bytes=10_000_000, compress=True,
        )

        # Write entries to active log
        audit.record(_make_entry(run_id="old_run", event="run_started"))
        audit.force_rotate()

        # Write new entries to fresh active log
        audit.record(_make_entry(run_id="new_run", event="run_started"))

        # Query across all
        results = audit.query_archives(event="run_started")
        run_ids = {r.run_id for r in results}
        assert "old_run" in run_ids
        assert "new_run" in run_ids

    def test_total_archive_size(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        audit = RotatingAuditLog(
            path=log_path, max_bytes=10_000_000, compress=False,
        )

        for i in range(5):
            audit.record(_make_entry())
        audit.force_rotate()

        size = audit.total_archive_size()
        assert size > 0

    def test_apply_retention_returns_count(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        audit = RotatingAuditLog(
            path=log_path,
            max_bytes=10_000_000,
            compress=False,
            retention=RetentionPolicy(max_files=0),
        )

        # Create some archives
        audit.record(_make_entry())
        audit.force_rotate()
        audit.record(_make_entry())
        audit.force_rotate()

        # No retention limit set (max_files=0 means unlimited)
        deleted = audit.apply_retention()
        assert deleted == 0

    def test_no_rotation_on_empty_file(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        log_path.touch()
        audit = RotatingAuditLog(path=log_path)
        audit.force_rotate()  # should not crash
        # No archives created for empty file
        assert audit.list_archives() == []
