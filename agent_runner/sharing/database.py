"""SQLite-backed persistence layer for agent runs and audit entries.

Replaces the JSON file-backed ``RunStore`` and JSONL-backed ``AuditLog``
with a single SQLite database.  Designed for multi-threaded server use
with WAL mode, thread-local connections, and automatic schema migrations.

Only uses the stdlib ``sqlite3`` module -- no external dependencies.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from agent_runner.sharing.audit import AuditEntry
from agent_runner.sharing.run_store import RunRecord, RunStatus

# ---------------------------------------------------------------------------
# Schema versions -- each entry is (version, list_of_sql_statements)
# ---------------------------------------------------------------------------

_MIGRATIONS: list[tuple[int, list[str]]] = [
    (
        1,
        [
            # -- Core runs table --
            """
            CREATE TABLE IF NOT EXISTS runs (
                id              TEXT PRIMARY KEY,
                created_at      REAL    NOT NULL,
                prompt          TEXT    NOT NULL,
                final_text      TEXT    NOT NULL,
                tool_call_log   TEXT    NOT NULL,   -- JSON
                captured_writes TEXT    NOT NULL,   -- JSON
                status          TEXT    NOT NULL DEFAULT 'pending',
                reviewed_by     TEXT,
                reviewed_at     REAL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status)",
            "CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs (created_at)",
            # -- Audit entries table --
            """
            CREATE TABLE IF NOT EXISTS audit_entries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                event       TEXT    NOT NULL,
                run_id      TEXT    NOT NULL,
                tool        TEXT,
                reviewer    TEXT,
                details     TEXT    NOT NULL DEFAULT '{}'  -- JSON
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_audit_run_id ON audit_entries (run_id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_entries (event)",
            # -- Run tags / metadata --
            """
            CREATE TABLE IF NOT EXISTS run_tags (
                run_id  TEXT    NOT NULL,
                key     TEXT    NOT NULL,
                value   TEXT    NOT NULL,
                PRIMARY KEY (run_id, key),
                FOREIGN KEY (run_id) REFERENCES runs (id) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_run_tags_key_value ON run_tags (key, value)",
            # -- Schema version tracking --
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version     INTEGER PRIMARY KEY,
                applied_at  TEXT    NOT NULL
            )
            """,
        ],
    ),
]

CURRENT_SCHEMA_VERSION = _MIGRATIONS[-1][0]


# ---------------------------------------------------------------------------
# Database — manages SQLite connections with thread-local pooling
# ---------------------------------------------------------------------------


class Database:
    """Thread-safe SQLite database manager.

    Each thread receives its own ``sqlite3.Connection`` via a
    ``threading.local()`` store.  The database is opened in WAL mode so
    that readers never block writers.

    Parameters
    ----------
    db_path:
        Filesystem path for the SQLite database file.  Use ``":memory:"``
        for an in-memory database (mostly useful in tests).
    """

    def __init__(self, db_path: str | Path = ".agent_runs.db") -> None:
        self._db_path = str(db_path)
        self._local = threading.local()
        self._lock = threading.Lock()

        # Run migrations on the *calling* thread's connection.
        self._migrate()

    # -- connection pool (thread-local) ------------------------------------

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the ``sqlite3.Connection`` for the current thread.

        A new connection is lazily created when a thread accesses it for
        the first time.
        """
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._make_connection()
            self._local.conn = conn
        return conn

    def _make_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    # -- transaction context manager ---------------------------------------

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a transactional scope around a series of operations.

        On success the transaction is committed; on failure it is rolled
        back.  Uses ``BEGIN IMMEDIATE`` so that write transactions are
        serialized at the SQLite level.
        """
        conn = self.connection
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    # -- schema migrations -------------------------------------------------

    def _current_version(self) -> int:
        """Return the latest applied schema version, or 0 if none."""
        conn = self.connection
        try:
            row = conn.execute(
                "SELECT MAX(version) AS v FROM schema_version"
            ).fetchone()
            return row["v"] if row and row["v"] is not None else 0
        except sqlite3.OperationalError:
            # schema_version table does not exist yet
            return 0

    def _migrate(self) -> None:
        """Apply all outstanding migrations in order."""
        current = self._current_version()
        for version, statements in _MIGRATIONS:
            if version <= current:
                continue
            with self.transaction() as conn:
                for sql in statements:
                    conn.execute(sql)
                conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (version, datetime.now(timezone.utc).isoformat()),
                )

    @property
    def schema_version(self) -> int:
        """The current schema version of the database."""
        return self._current_version()

    # -- cleanup -----------------------------------------------------------

    def close(self) -> None:
        """Close the current thread's connection (if any)."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def close_all(self) -> None:
        """Best-effort close of the calling thread's connection.

        In practice each thread should call ``close()`` when it shuts
        down.  This method is an alias kept for semantic clarity.
        """
        self.close()


# ---------------------------------------------------------------------------
# SQLiteRunStore — drop-in replacement for RunStore
# ---------------------------------------------------------------------------


class SQLiteRunStore:
    """SQLite-backed implementation of the ``RunStore`` interface.

    Parameters
    ----------
    db:
        A ``Database`` instance that manages the underlying connection.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # -- interface methods (matching RunStore) ------------------------------

    def save(
        self,
        prompt: str,
        final_text: str,
        tool_call_log: list[dict[str, Any]],
        captured_writes: list[dict[str, Any]],
        *,
        tags: dict[str, str] | None = None,
    ) -> RunRecord:
        """Create and persist a new run record.

        Parameters
        ----------
        tags:
            Optional key/value metadata attached to the run for later
            filtering.
        """
        record = RunRecord(
            id=uuid.uuid4().hex[:12],
            created_at=time.time(),
            prompt=prompt,
            final_text=final_text,
            tool_call_log=tool_call_log,
            captured_writes=captured_writes,
        )
        with self._db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO runs
                    (id, created_at, prompt, final_text,
                     tool_call_log, captured_writes, status,
                     reviewed_by, reviewed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.created_at,
                    record.prompt,
                    record.final_text,
                    json.dumps(record.tool_call_log, default=str),
                    json.dumps(record.captured_writes, default=str),
                    record.status.value,
                    record.reviewed_by,
                    record.reviewed_at,
                ),
            )
            if tags:
                for key, value in tags.items():
                    conn.execute(
                        "INSERT INTO run_tags (run_id, key, value) VALUES (?, ?, ?)",
                        (record.id, key, value),
                    )
        return record

    def get(self, run_id: str) -> RunRecord | None:
        """Load a run record by ID."""
        row = self._db.connection.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_pending(self) -> list[RunRecord]:
        """Return all runs with PENDING status, ordered by creation time."""
        return self.list_by_status(RunStatus.PENDING)

    def approve(self, run_id: str, reviewer: str = "anonymous") -> RunRecord | None:
        """Mark a run as approved."""
        return self._update_status(run_id, RunStatus.APPROVED, reviewer)

    def reject(self, run_id: str, reviewer: str = "anonymous") -> RunRecord | None:
        """Mark a run as rejected."""
        return self._update_status(run_id, RunStatus.REJECTED, reviewer)

    def mark_replayed(self, run_id: str) -> RunRecord | None:
        """Mark a run as replayed (writes executed)."""
        return self._update_status(run_id, RunStatus.REPLAYED)

    # -- extended query methods --------------------------------------------

    def list_by_status(self, status: RunStatus) -> list[RunRecord]:
        """Return all runs matching *status*, ordered by creation time."""
        rows = self._db.connection.execute(
            "SELECT * FROM runs WHERE status = ? ORDER BY created_at",
            (status.value,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_by_date_range(
        self,
        start: float,
        end: float,
    ) -> list[RunRecord]:
        """Return runs created within *[start, end]* (epoch seconds)."""
        rows = self._db.connection.execute(
            "SELECT * FROM runs WHERE created_at >= ? AND created_at <= ? ORDER BY created_at",
            (start, end),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def search_by_prompt(self, query: str) -> list[RunRecord]:
        """Full-text search on the *prompt* column (SQL ``LIKE``)."""
        rows = self._db.connection.execute(
            "SELECT * FROM runs WHERE prompt LIKE ? ORDER BY created_at",
            (f"%{query}%",),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count_by_status(self) -> dict[str, int]:
        """Return ``{status_value: count}`` for every status present."""
        rows = self._db.connection.execute(
            "SELECT status, COUNT(*) AS cnt FROM runs GROUP BY status"
        ).fetchall()
        return {row["status"]: row["cnt"] for row in rows}

    def list_by_tag(self, key: str, value: str | None = None) -> list[RunRecord]:
        """Return runs that have the given tag *key* (and optionally *value*)."""
        if value is not None:
            rows = self._db.connection.execute(
                """
                SELECT r.* FROM runs r
                JOIN run_tags t ON r.id = t.run_id
                WHERE t.key = ? AND t.value = ?
                ORDER BY r.created_at
                """,
                (key, value),
            ).fetchall()
        else:
            rows = self._db.connection.execute(
                """
                SELECT r.* FROM runs r
                JOIN run_tags t ON r.id = t.run_id
                WHERE t.key = ?
                ORDER BY r.created_at
                """,
                (key,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_tags(self, run_id: str) -> dict[str, str]:
        """Return all tags for a given run."""
        rows = self._db.connection.execute(
            "SELECT key, value FROM run_tags WHERE run_id = ?", (run_id,)
        ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        """Set or overwrite a tag on a run."""
        with self._db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO run_tags (run_id, key, value)
                VALUES (?, ?, ?)
                ON CONFLICT (run_id, key) DO UPDATE SET value = excluded.value
                """,
                (run_id, key, value),
            )

    # -- TTL cleanup -------------------------------------------------------

    def purge_older_than(self, days: int) -> int:
        """Delete runs (and their tags) older than *days* days.

        Returns the number of rows deleted.
        """
        cutoff = time.time() - days * 86400
        with self._db.transaction() as conn:
            # Tags are removed by ON DELETE CASCADE.
            cursor = conn.execute(
                "DELETE FROM runs WHERE created_at < ?", (cutoff,)
            )
            deleted = cursor.rowcount
        return deleted

    # -- audit entries -----------------------------------------------------

    def record_audit(self, entry: AuditEntry) -> None:
        """Persist an audit entry into the ``audit_entries`` table."""
        with self._db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO audit_entries
                    (timestamp, event, run_id, tool, reviewer, details)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.timestamp,
                    entry.event,
                    entry.run_id,
                    entry.tool,
                    entry.reviewer,
                    json.dumps(entry.details, default=str),
                ),
            )

    def query_audit(
        self,
        run_id: str | None = None,
        event: str | None = None,
    ) -> list[AuditEntry]:
        """Query audit entries with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if event is not None:
            clauses.append("event = ?")
            params.append(event)

        sql = "SELECT * FROM audit_entries"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id"

        rows = self._db.connection.execute(sql, params).fetchall()
        return [
            AuditEntry(
                timestamp=row["timestamp"],
                event=row["event"],
                run_id=row["run_id"],
                tool=row["tool"],
                reviewer=row["reviewer"],
                details=json.loads(row["details"]),
            )
            for row in rows
        ]

    # -- export ------------------------------------------------------------

    def export_json(self) -> str:
        """Export all runs, tags, and audit entries as a JSON string."""
        conn = self._db.connection

        run_rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at"
        ).fetchall()
        runs = []
        for row in run_rows:
            record_dict = self._row_to_record(row).to_dict()
            tag_rows = conn.execute(
                "SELECT key, value FROM run_tags WHERE run_id = ?",
                (row["id"],),
            ).fetchall()
            record_dict["tags"] = {r["key"]: r["value"] for r in tag_rows}
            runs.append(record_dict)

        audit_rows = conn.execute(
            "SELECT * FROM audit_entries ORDER BY id"
        ).fetchall()
        audits = [
            {
                "timestamp": r["timestamp"],
                "event": r["event"],
                "run_id": r["run_id"],
                "tool": r["tool"],
                "reviewer": r["reviewer"],
                "details": json.loads(r["details"]),
            }
            for r in audit_rows
        ]

        payload = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": self._db.schema_version,
            "runs": runs,
            "audit_entries": audits,
        }
        return json.dumps(payload, indent=2, default=str)

    # -- internals ---------------------------------------------------------

    def _update_status(
        self,
        run_id: str,
        status: RunStatus,
        reviewer: str | None = None,
    ) -> RunRecord | None:
        reviewed_at = time.time()
        with self._db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE runs
                   SET status = ?, reviewed_by = ?, reviewed_at = ?
                 WHERE id = ?
                """,
                (status.value, reviewer, reviewed_at, run_id),
            )
            if cursor.rowcount == 0:
                return None
        # Re-read after commit so the record reflects the new state.
        return self.get(run_id)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            id=row["id"],
            created_at=row["created_at"],
            prompt=row["prompt"],
            final_text=row["final_text"],
            tool_call_log=json.loads(row["tool_call_log"]),
            captured_writes=json.loads(row["captured_writes"]),
            status=RunStatus(row["status"]),
            reviewed_by=row["reviewed_by"],
            reviewed_at=row["reviewed_at"],
        )
