"""Comprehensive tests for the SQLite-backed persistence layer."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone

import pytest

from agent_runner.sharing.audit import AuditEntry
from agent_runner.sharing.database import (
    CURRENT_SCHEMA_VERSION,
    Database,
    SQLiteRunStore,
)
from agent_runner.sharing.run_store import RunRecord, RunStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    """Yield a Database backed by a temporary file, closed after the test."""
    database = Database(db_path=tmp_path / "test.db")
    yield database
    database.close()


@pytest.fixture()
def store(db):
    """Yield an SQLiteRunStore wired to the temporary database."""
    return SQLiteRunStore(db)


def _make_run(store: SQLiteRunStore, prompt: str = "hello", **kwargs) -> RunRecord:
    """Helper to create a run with sensible defaults."""
    return store.save(
        prompt=prompt,
        final_text="world",
        tool_call_log=[{"tool": "echo", "args": {"msg": "hi"}}],
        captured_writes=[{"path": "/tmp/x", "content": "data"}],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Schema creation & migration
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_schema_version_is_set(self, db):
        assert db.schema_version == CURRENT_SCHEMA_VERSION

    def test_idempotent_migration(self, tmp_path):
        """Opening the same DB twice must not fail or re-apply migrations."""
        path = tmp_path / "idempotent.db"
        db1 = Database(db_path=path)
        db2 = Database(db_path=path)
        assert db1.schema_version == CURRENT_SCHEMA_VERSION
        assert db2.schema_version == CURRENT_SCHEMA_VERSION
        db1.close()
        db2.close()

    def test_tables_exist(self, db):
        tables = {
            row[0]
            for row in db.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "runs" in tables
        assert "audit_entries" in tables
        assert "run_tags" in tables
        assert "schema_version" in tables

    def test_wal_mode_enabled(self, db):
        mode = db.connection.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------


class TestCRUD:
    def test_save_and_get(self, store):
        rec = _make_run(store, prompt="test prompt")
        loaded = store.get(rec.id)
        assert loaded is not None
        assert loaded.id == rec.id
        assert loaded.prompt == "test prompt"
        assert loaded.final_text == "world"
        assert loaded.status == RunStatus.PENDING
        assert loaded.tool_call_log == [{"tool": "echo", "args": {"msg": "hi"}}]
        assert loaded.captured_writes == [{"path": "/tmp/x", "content": "data"}]

    def test_get_missing_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_approve(self, store):
        rec = _make_run(store)
        result = store.approve(rec.id, "alice")
        assert result is not None
        assert result.status == RunStatus.APPROVED
        assert result.reviewed_by == "alice"
        assert result.reviewed_at is not None

    def test_reject(self, store):
        rec = _make_run(store)
        result = store.reject(rec.id, "bob")
        assert result is not None
        assert result.status == RunStatus.REJECTED
        assert result.reviewed_by == "bob"

    def test_mark_replayed(self, store):
        rec = _make_run(store)
        store.approve(rec.id, "alice")
        result = store.mark_replayed(rec.id)
        assert result is not None
        assert result.status == RunStatus.REPLAYED

    def test_approve_nonexistent_returns_none(self, store):
        assert store.approve("nope") is None

    def test_reject_nonexistent_returns_none(self, store):
        assert store.reject("nope") is None

    def test_mark_replayed_nonexistent_returns_none(self, store):
        assert store.mark_replayed("nope") is None

    def test_list_pending(self, store):
        r1 = _make_run(store, prompt="a")
        r2 = _make_run(store, prompt="b")
        r3 = _make_run(store, prompt="c")
        store.approve(r2.id, "x")

        pending = store.list_pending()
        ids = [r.id for r in pending]
        assert r1.id in ids
        assert r3.id in ids
        assert r2.id not in ids


# ---------------------------------------------------------------------------
# Query methods
# ---------------------------------------------------------------------------


class TestQueries:
    def test_list_by_status(self, store):
        r1 = _make_run(store)
        r2 = _make_run(store)
        store.approve(r1.id, "x")

        approved = store.list_by_status(RunStatus.APPROVED)
        assert len(approved) == 1
        assert approved[0].id == r1.id

        pending = store.list_by_status(RunStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].id == r2.id

    def test_list_by_date_range(self, store):
        before = time.time()
        r1 = _make_run(store)
        after = time.time()
        # Slight future bump so we can exclude it.
        time.sleep(0.05)
        r2 = _make_run(store)

        results = store.list_by_date_range(before, after)
        ids = [r.id for r in results]
        assert r1.id in ids
        # r2 was created after `after`, so it should not appear.
        assert r2.id not in ids

    def test_search_by_prompt(self, store):
        _make_run(store, prompt="deploy the widgetizer")
        _make_run(store, prompt="run diagnostics")
        _make_run(store, prompt="widgetizer rollback")

        results = store.search_by_prompt("widgetizer")
        assert len(results) == 2

    def test_search_by_prompt_no_match(self, store):
        _make_run(store, prompt="something else")
        assert store.search_by_prompt("nonexistent") == []

    def test_count_by_status(self, store):
        r1 = _make_run(store)
        _make_run(store)
        _make_run(store)
        store.approve(r1.id, "x")

        counts = store.count_by_status()
        assert counts["pending"] == 2
        assert counts["approved"] == 1


# ---------------------------------------------------------------------------
# Tags / metadata
# ---------------------------------------------------------------------------


class TestTags:
    def test_save_with_tags(self, store):
        rec = _make_run(store, tags={"env": "staging", "team": "infra"})
        tags = store.get_tags(rec.id)
        assert tags == {"env": "staging", "team": "infra"}

    def test_set_tag_upsert(self, store):
        rec = _make_run(store)
        store.set_tag(rec.id, "env", "prod")
        assert store.get_tags(rec.id)["env"] == "prod"
        store.set_tag(rec.id, "env", "staging")
        assert store.get_tags(rec.id)["env"] == "staging"

    def test_list_by_tag(self, store):
        r1 = _make_run(store, tags={"env": "prod"})
        r2 = _make_run(store, tags={"env": "staging"})
        r3 = _make_run(store, tags={"env": "prod", "team": "core"})

        prod_runs = store.list_by_tag("env", "prod")
        ids = [r.id for r in prod_runs]
        assert r1.id in ids
        assert r3.id in ids
        assert r2.id not in ids

    def test_list_by_tag_key_only(self, store):
        r1 = _make_run(store, tags={"env": "prod"})
        r2 = _make_run(store)

        results = store.list_by_tag("env")
        assert len(results) == 1
        assert results[0].id == r1.id


# ---------------------------------------------------------------------------
# Purge (TTL-based cleanup)
# ---------------------------------------------------------------------------


class TestPurge:
    def test_purge_older_than(self, store, db):
        rec = _make_run(store)
        # Backdate the record to 100 days ago.
        db.connection.execute(
            "UPDATE runs SET created_at = ? WHERE id = ?",
            (time.time() - 100 * 86400, rec.id),
        )
        db.connection.commit()

        recent = _make_run(store)

        deleted = store.purge_older_than(days=30)
        assert deleted == 1
        assert store.get(rec.id) is None
        assert store.get(recent.id) is not None

    def test_purge_returns_zero_when_nothing_old(self, store):
        _make_run(store)
        assert store.purge_older_than(days=1) == 0

    def test_purge_cascades_tags(self, store, db):
        rec = _make_run(store, tags={"env": "prod"})
        db.connection.execute(
            "UPDATE runs SET created_at = ? WHERE id = ?",
            (time.time() - 100 * 86400, rec.id),
        )
        db.connection.commit()

        store.purge_older_than(days=30)
        assert store.get_tags(rec.id) == {}


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_writes(self, tmp_path):
        """Multiple threads writing simultaneously must not corrupt data."""
        db = Database(db_path=tmp_path / "threads.db")
        st = SQLiteRunStore(db)
        errors: list[Exception] = []
        results: list[RunRecord] = []
        lock = threading.Lock()

        def worker(thread_id: int) -> None:
            try:
                for i in range(10):
                    rec = st.save(
                        prompt=f"thread-{thread_id}-run-{i}",
                        final_text="ok",
                        tool_call_log=[],
                        captured_writes=[],
                    )
                    with lock:
                        results.append(rec)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors in threads: {errors}"
        assert len(results) == 50

        # Verify all are readable.
        for rec in results:
            assert st.get(rec.id) is not None

        db.close()

    def test_concurrent_read_write(self, tmp_path):
        """Readers must not block writers and vice-versa (WAL mode)."""
        db = Database(db_path=tmp_path / "rw.db")
        st = SQLiteRunStore(db)

        # Seed some data.
        for i in range(20):
            st.save(prompt=f"seed-{i}", final_text="ok",
                    tool_call_log=[], captured_writes=[])

        errors: list[Exception] = []

        def reader() -> None:
            try:
                for _ in range(30):
                    st.list_by_status(RunStatus.PENDING)
                    st.count_by_status()
            except Exception as exc:
                errors.append(exc)

        def writer() -> None:
            try:
                for i in range(10):
                    st.save(prompt=f"new-{i}", final_text="ok",
                            tool_call_log=[], captured_writes=[])
            except Exception as exc:
                errors.append(exc)

        threads = (
            [threading.Thread(target=reader) for _ in range(3)]
            + [threading.Thread(target=writer) for _ in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent r/w: {errors}"
        db.close()

    def test_thread_local_connections(self, tmp_path):
        """Each thread should receive its own connection object."""
        db = Database(db_path=tmp_path / "tl.db")
        connections: list[int] = []
        lock = threading.Lock()

        def capture() -> None:
            conn_id = id(db.connection)
            with lock:
                connections.append(conn_id)

        threads = [threading.Thread(target=capture) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All connection ids should be unique (different objects).
        assert len(set(connections)) == len(connections)
        db.close()


# ---------------------------------------------------------------------------
# Audit entries
# ---------------------------------------------------------------------------


class TestAudit:
    def test_record_and_query_audit(self, store):
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event="run_started",
            run_id="abc123",
            tool="echo",
            details={"key": "value"},
        )
        store.record_audit(entry)

        results = store.query_audit(run_id="abc123")
        assert len(results) == 1
        assert results[0].event == "run_started"
        assert results[0].tool == "echo"
        assert results[0].details == {"key": "value"}

    def test_query_audit_by_event(self, store):
        for event in ("run_started", "approved", "run_started"):
            store.record_audit(
                AuditEntry(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    event=event,
                    run_id="r1",
                )
            )

        started = store.query_audit(event="run_started")
        assert len(started) == 2

    def test_query_audit_no_filters(self, store):
        store.record_audit(
            AuditEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event="x",
                run_id="r1",
            )
        )
        store.record_audit(
            AuditEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event="y",
                run_id="r2",
            )
        )
        assert len(store.query_audit()) == 2

    def test_audit_entry_with_reviewer(self, store):
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event="approved",
            run_id="r1",
            reviewer="alice",
        )
        store.record_audit(entry)
        results = store.query_audit(run_id="r1")
        assert results[0].reviewer == "alice"


# ---------------------------------------------------------------------------
# Export to JSON
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_json_structure(self, store):
        rec = _make_run(store, prompt="backup test", tags={"env": "prod"})
        store.record_audit(
            AuditEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event="run_started",
                run_id=rec.id,
            )
        )

        raw = store.export_json()
        data = json.loads(raw)

        assert "exported_at" in data
        assert data["schema_version"] == CURRENT_SCHEMA_VERSION
        assert len(data["runs"]) == 1
        assert data["runs"][0]["prompt"] == "backup test"
        assert data["runs"][0]["tags"] == {"env": "prod"}
        assert len(data["audit_entries"]) == 1
        assert data["audit_entries"][0]["event"] == "run_started"

    def test_export_json_empty(self, store):
        data = json.loads(store.export_json())
        assert data["runs"] == []
        assert data["audit_entries"] == []

    def test_export_json_is_valid_json(self, store):
        _make_run(store)
        raw = store.export_json()
        # Must not raise.
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Transaction handling
# ---------------------------------------------------------------------------


class TestTransactions:
    def test_failed_transaction_rolls_back(self, db, store):
        """If an error occurs inside a transaction block, no data is committed."""
        rec = _make_run(store)

        with pytest.raises(ValueError):
            with db.transaction() as conn:
                conn.execute(
                    "UPDATE runs SET status = ? WHERE id = ?",
                    ("approved", rec.id),
                )
                raise ValueError("Simulated error")

        # The update must have been rolled back.
        loaded = store.get(rec.id)
        assert loaded is not None
        assert loaded.status == RunStatus.PENDING

    def test_successful_transaction_commits(self, db, store):
        rec = _make_run(store)
        with db.transaction() as conn:
            conn.execute(
                "UPDATE runs SET status = ? WHERE id = ?",
                ("approved", rec.id),
            )
        loaded = store.get(rec.id)
        assert loaded is not None
        assert loaded.status == RunStatus.APPROVED


# ---------------------------------------------------------------------------
# Database lifecycle
# ---------------------------------------------------------------------------


class TestDatabaseLifecycle:
    def test_close_and_reopen(self, tmp_path):
        path = tmp_path / "lifecycle.db"
        db = Database(db_path=path)
        st = SQLiteRunStore(db)
        rec = _make_run(st)
        db.close()

        db2 = Database(db_path=path)
        st2 = SQLiteRunStore(db2)
        loaded = st2.get(rec.id)
        assert loaded is not None
        assert loaded.prompt == "hello"
        db2.close()

    def test_in_memory_database(self):
        db = Database(db_path=":memory:")
        st = SQLiteRunStore(db)
        rec = _make_run(st)
        assert st.get(rec.id) is not None
        db.close()
