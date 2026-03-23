"""Tests for the run store and approval server."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from http.client import HTTPConnection
from threading import Thread

from agent_runner.sharing.run_store import RunStore, RunRecord, RunStatus
from agent_runner.sharing.server import create_approval_app


class TestRunStore:
    def test_save_and_get(self, tmp_path):
        store = RunStore(store_dir=tmp_path / "runs")
        record = store.save(
            prompt="test prompt",
            final_text="test response",
            tool_call_log=[{"tool": "echo", "action": "allow"}],
            captured_writes=[{"tool": "write_file", "input": {"path": "x.txt"}}],
        )
        assert record.status == RunStatus.PENDING
        assert len(record.id) == 12

        loaded = store.get(record.id)
        assert loaded is not None
        assert loaded.prompt == "test prompt"
        assert loaded.captured_writes[0]["tool"] == "write_file"

    def test_get_missing_returns_none(self, tmp_path):
        store = RunStore(store_dir=tmp_path / "runs")
        assert store.get("nonexistent") is None

    def test_list_pending(self, tmp_path):
        store = RunStore(store_dir=tmp_path / "runs")
        store.save("p1", "r1", [], [])
        store.save("p2", "r2", [], [])
        assert len(store.list_pending()) == 2

    def test_approve(self, tmp_path):
        store = RunStore(store_dir=tmp_path / "runs")
        record = store.save("p", "r", [], [])

        updated = store.approve(record.id, reviewer="alice")
        assert updated.status == RunStatus.APPROVED
        assert updated.reviewed_by == "alice"
        assert updated.reviewed_at is not None

        reloaded = store.get(record.id)
        assert reloaded.status == RunStatus.APPROVED

    def test_reject(self, tmp_path):
        store = RunStore(store_dir=tmp_path / "runs")
        record = store.save("p", "r", [], [])

        updated = store.reject(record.id, reviewer="bob")
        assert updated.status == RunStatus.REJECTED

    def test_mark_replayed(self, tmp_path):
        store = RunStore(store_dir=tmp_path / "runs")
        record = store.save("p", "r", [], [])
        store.approve(record.id)

        updated = store.mark_replayed(record.id)
        assert updated.status == RunStatus.REPLAYED

    def test_roundtrip_serialization(self, tmp_path):
        store = RunStore(store_dir=tmp_path / "runs")
        record = store.save(
            prompt="deploy",
            final_text="done",
            tool_call_log=[{"tool": "shell", "action": "mock", "input": {"command": "ls"}}],
            captured_writes=[{"tool": "shell", "input": {"command": "rm -rf /tmp/test"}}],
        )
        d = record.to_dict()
        restored = RunRecord.from_dict(d)
        assert restored.id == record.id
        assert restored.prompt == record.prompt
        assert restored.captured_writes == record.captured_writes


class TestApprovalServer:
    def test_list_and_approve_via_api(self, tmp_path):
        store = RunStore(store_dir=tmp_path / "runs")
        record = store.save("test", "response", [], [{"tool": "w", "input": {}}])

        server = create_approval_app(store, host="127.0.0.1", port=0)
        port = server.server_address[1]
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            conn = HTTPConnection("127.0.0.1", port)

            # List pending
            conn.request("GET", "/runs")
            resp = conn.getresponse()
            assert resp.status == 200
            data = json.loads(resp.read())
            assert len(data) == 1
            assert data[0]["id"] == record.id

            # Get single run
            conn.request("GET", f"/runs/{record.id}")
            resp = conn.getresponse()
            assert resp.status == 200
            data = json.loads(resp.read())
            assert data["prompt"] == "test"

            # Approve
            body = json.dumps({"reviewer": "tester"}).encode()
            conn.request("POST", f"/runs/{record.id}/approve", body=body,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            assert resp.status == 200
            data = json.loads(resp.read())
            assert data["status"] == "approved"
            assert data["reviewed_by"] == "tester"

            # Review page (HTML)
            conn.request("GET", f"/review/{record.id}")
            resp = conn.getresponse()
            assert resp.status == 200
            html = resp.read().decode()
            assert "Approve" in html

        finally:
            server.shutdown()

    def test_reject_via_api(self, tmp_path):
        store = RunStore(store_dir=tmp_path / "runs")
        record = store.save("test", "resp", [], [])

        server = create_approval_app(store, host="127.0.0.1", port=0)
        port = server.server_address[1]
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            conn = HTTPConnection("127.0.0.1", port)
            body = json.dumps({"reviewer": "reviewer2"}).encode()
            conn.request("POST", f"/runs/{record.id}/reject", body=body,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            assert resp.status == 200
            data = json.loads(resp.read())
            assert data["status"] == "rejected"
        finally:
            server.shutdown()


class TestApprovalServerAuth:
    """Token-based authentication for the approval server."""

    def _start_server(self, store, auth_token=None):
        server = create_approval_app(store, host="127.0.0.1", port=0, auth_token=auth_token)
        port = server.server_address[1]
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, port

    def test_no_auth_required_when_token_none(self, tmp_path):
        store = RunStore(store_dir=tmp_path / "runs")
        store.save("p", "r", [], [])
        server, port = self._start_server(store, auth_token=None)
        try:
            conn = HTTPConnection("127.0.0.1", port)
            conn.request("GET", "/runs")
            resp = conn.getresponse()
            assert resp.status == 200
        finally:
            server.shutdown()

    def test_request_without_token_returns_401(self, tmp_path):
        store = RunStore(store_dir=tmp_path / "runs")
        store.save("p", "r", [], [])
        server, port = self._start_server(store, auth_token="s3cret")
        try:
            conn = HTTPConnection("127.0.0.1", port)
            conn.request("GET", "/runs")
            resp = conn.getresponse()
            assert resp.status == 401
            data = json.loads(resp.read())
            assert "Unauthorized" in data["error"]
        finally:
            server.shutdown()

    def test_request_with_wrong_token_returns_401(self, tmp_path):
        store = RunStore(store_dir=tmp_path / "runs")
        store.save("p", "r", [], [])
        server, port = self._start_server(store, auth_token="s3cret")
        try:
            conn = HTTPConnection("127.0.0.1", port)
            conn.request("GET", "/runs", headers={"Authorization": "Bearer wrong"})
            resp = conn.getresponse()
            assert resp.status == 401
        finally:
            server.shutdown()

    def test_request_with_correct_token_succeeds(self, tmp_path):
        store = RunStore(store_dir=tmp_path / "runs")
        record = store.save("p", "r", [], [])
        server, port = self._start_server(store, auth_token="s3cret")
        try:
            conn = HTTPConnection("127.0.0.1", port)
            conn.request("GET", "/runs", headers={"Authorization": "Bearer s3cret"})
            resp = conn.getresponse()
            assert resp.status == 200
            data = json.loads(resp.read())
            assert len(data) == 1
        finally:
            server.shutdown()

    def test_post_without_token_returns_401(self, tmp_path):
        store = RunStore(store_dir=tmp_path / "runs")
        record = store.save("p", "r", [], [])
        server, port = self._start_server(store, auth_token="s3cret")
        try:
            conn = HTTPConnection("127.0.0.1", port)
            body = json.dumps({"reviewer": "tester"}).encode()
            conn.request("POST", f"/runs/{record.id}/approve", body=body,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            assert resp.status == 401
        finally:
            server.shutdown()

    def test_post_with_correct_token_succeeds(self, tmp_path):
        store = RunStore(store_dir=tmp_path / "runs")
        record = store.save("p", "r", [], [])
        server, port = self._start_server(store, auth_token="s3cret")
        try:
            conn = HTTPConnection("127.0.0.1", port)
            body = json.dumps({"reviewer": "tester"}).encode()
            conn.request("POST", f"/runs/{record.id}/approve", body=body,
                         headers={
                             "Content-Type": "application/json",
                             "Authorization": "Bearer s3cret",
                         })
            resp = conn.getresponse()
            assert resp.status == 200
            data = json.loads(resp.read())
            assert data["status"] == "approved"
        finally:
            server.shutdown()

    def test_review_page_accessible_without_token(self, tmp_path):
        """The HTML review page should be accessible without the auth header."""
        store = RunStore(store_dir=tmp_path / "runs")
        record = store.save("p", "r", [], [{"tool": "w", "input": {}}])
        server, port = self._start_server(store, auth_token="s3cret")
        try:
            conn = HTTPConnection("127.0.0.1", port)
            conn.request("GET", f"/review/{record.id}")
            resp = conn.getresponse()
            assert resp.status == 200
            html = resp.read().decode()
            assert "auth_token" in html
            assert "s3cret" in html  # token embedded in hidden field
        finally:
            server.shutdown()
