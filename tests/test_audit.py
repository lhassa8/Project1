"""Tests for the immutable audit trail."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_runner.sharing.audit import AuditEntry, AuditLog


def _make_entry(
    event: str = "tool_called",
    run_id: str = "run1",
    tool: str | None = "shell",
    reviewer: str | None = None,
    details: dict | None = None,
) -> AuditEntry:
    return AuditEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        event=event,
        run_id=run_id,
        tool=tool,
        reviewer=reviewer,
        details=details or {},
    )


class TestAuditRecord:
    def test_record_appends_to_file(self, tmp_path: Path):
        log = AuditLog(path=tmp_path / "audit.jsonl")
        log.record(_make_entry(event="run_started"))
        log.record(_make_entry(event="tool_called"))

        lines = (tmp_path / "audit.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["event"] == "run_started"

    def test_record_creates_file_if_missing(self, tmp_path: Path):
        log_path = tmp_path / "subdir" / "audit.jsonl"
        # parent must exist for open() to work — this validates that the
        # constructor does not pre-create the file.
        log_path.parent.mkdir(parents=True)
        log = AuditLog(path=log_path)
        assert not log_path.exists()
        log.record(_make_entry())
        assert log_path.exists()


class TestAuditQuery:
    def test_query_all(self, tmp_path: Path):
        log = AuditLog(path=tmp_path / "audit.jsonl")
        log.record(_make_entry(event="run_started", run_id="r1"))
        log.record(_make_entry(event="tool_called", run_id="r2"))
        assert len(log.query()) == 2

    def test_query_filter_by_run_id(self, tmp_path: Path):
        log = AuditLog(path=tmp_path / "audit.jsonl")
        log.record(_make_entry(event="run_started", run_id="r1"))
        log.record(_make_entry(event="tool_called", run_id="r2"))
        log.record(_make_entry(event="approved", run_id="r1"))

        results = log.query(run_id="r1")
        assert len(results) == 2
        assert all(e.run_id == "r1" for e in results)

    def test_query_filter_by_event(self, tmp_path: Path):
        log = AuditLog(path=tmp_path / "audit.jsonl")
        log.record(_make_entry(event="run_started", run_id="r1"))
        log.record(_make_entry(event="tool_called", run_id="r1"))
        log.record(_make_entry(event="tool_called", run_id="r2"))

        results = log.query(event="tool_called")
        assert len(results) == 2
        assert all(e.event == "tool_called" for e in results)

    def test_query_filter_by_both(self, tmp_path: Path):
        log = AuditLog(path=tmp_path / "audit.jsonl")
        log.record(_make_entry(event="run_started", run_id="r1"))
        log.record(_make_entry(event="tool_called", run_id="r1"))
        log.record(_make_entry(event="tool_called", run_id="r2"))

        results = log.query(run_id="r1", event="tool_called")
        assert len(results) == 1
        assert results[0].run_id == "r1"
        assert results[0].event == "tool_called"

    def test_query_returns_empty_when_file_missing(self, tmp_path: Path):
        log = AuditLog(path=tmp_path / "nonexistent.jsonl")
        assert log.query() == []

    def test_query_no_match(self, tmp_path: Path):
        log = AuditLog(path=tmp_path / "audit.jsonl")
        log.record(_make_entry(event="run_started", run_id="r1"))
        assert log.query(run_id="r999") == []


class TestAuditExportCSV:
    def test_export_csv_produces_valid_csv(self, tmp_path: Path):
        log = AuditLog(path=tmp_path / "audit.jsonl")
        log.record(_make_entry(event="run_started", run_id="r1", details={"info": "hello"}))
        log.record(_make_entry(event="approved", run_id="r1", reviewer="alice"))

        csv_path = tmp_path / "export.csv"
        log.export_csv(csv_path)

        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["event"] == "run_started"
        assert rows[0]["run_id"] == "r1"
        # details should be a JSON string
        details = json.loads(rows[0]["details"])
        assert details["info"] == "hello"
        assert rows[1]["reviewer"] == "alice"

    def test_export_csv_empty_log(self, tmp_path: Path):
        log = AuditLog(path=tmp_path / "audit.jsonl")
        csv_path = tmp_path / "empty.csv"
        log.export_csv(csv_path)

        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 0


class TestAuditEntryRoundtrip:
    def test_multiple_entries_roundtrip(self, tmp_path: Path):
        log = AuditLog(path=tmp_path / "audit.jsonl")
        entries = [
            _make_entry(event="run_started", run_id="r1"),
            _make_entry(event="tool_called", run_id="r1", tool="shell"),
            _make_entry(event="approval_requested", run_id="r1"),
            _make_entry(event="approved", run_id="r1", reviewer="bob"),
            _make_entry(event="replayed", run_id="r1"),
        ]
        for e in entries:
            log.record(e)

        restored = log.query()
        assert len(restored) == len(entries)
        for orig, rest in zip(entries, restored):
            assert orig.event == rest.event
            assert orig.run_id == rest.run_id
            assert orig.tool == rest.tool
            assert orig.reviewer == rest.reviewer
