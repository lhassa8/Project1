"""Tests for agent_runner.sharing.webhooks."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import unittest
from unittest.mock import MagicMock, patch, call

from agent_runner.sharing.run_store import RunRecord, RunStatus, RunStore
from agent_runner.sharing.webhooks import (
    ApprovalCallbackHandler,
    WebhookConfig,
    WebhookNotifier,
    _sign_payload,
    verify_signature,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(**overrides) -> RunRecord:
    defaults = dict(
        id="run123",
        created_at=1700000000.0,
        prompt="Write hello world",
        final_text="print('hello world')",
        tool_call_log=[{"tool": "shell", "action": "allow", "input": {"cmd": "echo hi"}}],
        captured_writes=[{"tool": "write_file", "input": {"path": "/tmp/a.py", "content": "x"}}],
        status=RunStatus.PENDING,
        reviewed_by=None,
        reviewed_at=None,
    )
    defaults.update(overrides)
    return RunRecord(**defaults)


SECRET = "test-secret-key"


# ---------------------------------------------------------------------------
# HMAC signing / verification
# ---------------------------------------------------------------------------

class TestHMAC(unittest.TestCase):
    def test_sign_and_verify_roundtrip(self):
        payload = b'{"run_id": "abc"}'
        sig = _sign_payload(payload, SECRET)
        self.assertTrue(verify_signature(payload, SECRET, sig))

    def test_verify_rejects_bad_signature(self):
        payload = b'{"run_id": "abc"}'
        self.assertFalse(verify_signature(payload, SECRET, "bad-sig"))

    def test_verify_rejects_wrong_secret(self):
        payload = b'{"run_id": "abc"}'
        sig = _sign_payload(payload, SECRET)
        self.assertFalse(verify_signature(payload, "wrong-secret", sig))

    def test_verify_rejects_tampered_payload(self):
        payload = b'{"run_id": "abc"}'
        sig = _sign_payload(payload, SECRET)
        tampered = b'{"run_id": "xyz"}'
        self.assertFalse(verify_signature(tampered, SECRET, sig))

    def test_signature_matches_stdlib_hmac(self):
        payload = b"hello"
        expected = hmac.new(
            SECRET.encode("utf-8"), payload, hashlib.sha256
        ).hexdigest()
        self.assertEqual(_sign_payload(payload, SECRET), expected)


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------

class TestPayloadConstruction(unittest.TestCase):
    def test_approval_payload_fields(self):
        record = _make_record()
        payload = WebhookNotifier._build_approval_payload(record)

        self.assertEqual(payload["event"], "approval_needed")
        self.assertEqual(payload["run_id"], "run123")
        self.assertEqual(payload["prompt"], "Write hello world")
        self.assertEqual(len(payload["tool_call_log"]), 1)
        self.assertEqual(len(payload["captured_writes_summary"]), 1)
        self.assertIn("timestamp", payload)

    def test_captured_writes_summary_structure(self):
        record = _make_record()
        payload = WebhookNotifier._build_approval_payload(record)
        summary = payload["captured_writes_summary"][0]

        self.assertEqual(summary["tool"], "write_file")
        self.assertIn("path", summary["keys"])
        self.assertIn("content", summary["keys"])

    def test_empty_captured_writes(self):
        record = _make_record(captured_writes=[])
        payload = WebhookNotifier._build_approval_payload(record)
        self.assertEqual(payload["captured_writes_summary"], [])


# ---------------------------------------------------------------------------
# Retry logic (mock urllib)
# ---------------------------------------------------------------------------

class TestRetryLogic(unittest.TestCase):
    def _make_notifier(self, **cfg_overrides):
        cfg = WebhookConfig(
            url="https://example.com/hook",
            secret=SECRET,
            retry_count=3,
            timeout=5,
            **cfg_overrides,
        )
        return WebhookNotifier([cfg])

    @patch("agent_runner.sharing.webhooks.urllib.request.urlopen")
    def test_success_on_first_try(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        notifier = self._make_notifier()
        record = _make_record()
        results = notifier.notify_approval_needed(record)

        self.assertTrue(results["https://example.com/hook"])
        self.assertEqual(mock_urlopen.call_count, 1)

    @patch("agent_runner.sharing.webhooks.time.sleep")
    @patch("agent_runner.sharing.webhooks.urllib.request.urlopen")
    def test_retries_on_failure_then_succeeds(self, mock_urlopen, mock_sleep):
        # First two calls fail, third succeeds
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            ConnectionError("refused"),
            ConnectionError("refused"),
            mock_resp,
        ]

        notifier = self._make_notifier()
        record = _make_record()
        results = notifier.notify_approval_needed(record)

        self.assertTrue(results["https://example.com/hook"])
        self.assertEqual(mock_urlopen.call_count, 3)
        # Exponential backoff: sleep(1), sleep(2)
        self.assertEqual(mock_sleep.call_count, 2)
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)

    @patch("agent_runner.sharing.webhooks.time.sleep")
    @patch("agent_runner.sharing.webhooks.urllib.request.urlopen")
    def test_all_retries_exhausted(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = ConnectionError("refused")

        notifier = self._make_notifier()
        record = _make_record()
        results = notifier.notify_approval_needed(record)

        self.assertFalse(results["https://example.com/hook"])
        self.assertEqual(mock_urlopen.call_count, 3)

    @patch("agent_runner.sharing.webhooks.urllib.request.urlopen")
    def test_request_has_correct_signature_header(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        notifier = self._make_notifier()
        record = _make_record()
        notifier.notify_approval_needed(record)

        req = mock_urlopen.call_args[0][0]
        sig_header = req.get_header("X-agent-signature")
        self.assertIsNotNone(sig_header)

        # Verify the signature matches the payload
        self.assertTrue(verify_signature(req.data, SECRET, sig_header))

    @patch("agent_runner.sharing.webhooks.urllib.request.urlopen")
    def test_multiple_endpoints(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        configs = [
            WebhookConfig(url="https://a.example.com/hook", secret="s1"),
            WebhookConfig(url="https://b.example.com/hook", secret="s2"),
        ]
        notifier = WebhookNotifier(configs)
        record = _make_record()
        results = notifier.notify_approval_needed(record)

        self.assertEqual(len(results), 2)
        self.assertTrue(results["https://a.example.com/hook"])
        self.assertTrue(results["https://b.example.com/hook"])

    @patch("agent_runner.sharing.webhooks.urllib.request.urlopen")
    def test_status_changed_notification(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        notifier = self._make_notifier()
        record = _make_record(status=RunStatus.APPROVED)
        results = notifier.notify_status_changed(
            record, RunStatus.PENDING, RunStatus.APPROVED
        )

        self.assertTrue(results["https://example.com/hook"])
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        self.assertEqual(body["event"], "status_changed")
        self.assertEqual(body["old_status"], "pending")
        self.assertEqual(body["new_status"], "approved")


# ---------------------------------------------------------------------------
# ApprovalCallbackHandler
# ---------------------------------------------------------------------------

class TestApprovalCallbackHandler(unittest.TestCase):
    def setUp(self):
        self.store = MagicMock(spec=RunStore)
        self.handler = ApprovalCallbackHandler(self.store, SECRET)

    def _make_request(self, payload: dict, secret: str | None = None):
        body = json.dumps(payload).encode("utf-8")
        sig = _sign_payload(body, secret or SECRET)
        headers = {"X-Agent-Signature": sig}
        return body, headers

    def test_approve_run(self):
        record = _make_record(status=RunStatus.APPROVED, reviewed_by="alice")
        self.store.get.return_value = _make_record()
        self.store.approve.return_value = record

        body, headers = self._make_request({
            "run_id": "run123",
            "action": "approve",
            "reviewer": "alice",
        })
        result = self.handler.handle(body, headers)

        self.assertEqual(result["status_code"], 200)
        self.assertEqual(result["body"]["status"], "approved")
        self.store.approve.assert_called_once_with("run123", "alice")

    def test_reject_run(self):
        record = _make_record(status=RunStatus.REJECTED, reviewed_by="bob")
        self.store.get.return_value = _make_record()
        self.store.reject.return_value = record

        body, headers = self._make_request({
            "run_id": "run123",
            "action": "reject",
            "reviewer": "bob",
        })
        result = self.handler.handle(body, headers)

        self.assertEqual(result["status_code"], 200)
        self.assertEqual(result["body"]["status"], "rejected")
        self.store.reject.assert_called_once_with("run123", "bob")

    def test_invalid_signature_rejected(self):
        body, headers = self._make_request(
            {"run_id": "run123", "action": "approve"},
            secret="wrong-secret",
        )
        # Override the header with the wrong-secret signature but handler
        # expects the real SECRET
        result = self.handler.handle(body, headers)
        self.assertEqual(result["status_code"], 401)

    def test_missing_signature_rejected(self):
        body = json.dumps({"run_id": "run123", "action": "approve"}).encode()
        result = self.handler.handle(body, {})
        self.assertEqual(result["status_code"], 401)

    def test_malformed_json(self):
        body = b"not json at all"
        sig = _sign_payload(body, SECRET)
        result = self.handler.handle(body, {"X-Agent-Signature": sig})
        self.assertEqual(result["status_code"], 400)

    def test_missing_action_field(self):
        body, headers = self._make_request({"run_id": "run123"})
        result = self.handler.handle(body, headers)
        self.assertEqual(result["status_code"], 400)

    def test_invalid_action_field(self):
        body, headers = self._make_request({
            "run_id": "run123",
            "action": "delete",
        })
        result = self.handler.handle(body, headers)
        self.assertEqual(result["status_code"], 400)

    def test_run_not_found(self):
        self.store.get.return_value = None

        body, headers = self._make_request({
            "run_id": "nonexistent",
            "action": "approve",
        })
        result = self.handler.handle(body, headers)
        self.assertEqual(result["status_code"], 404)

    def test_default_reviewer_is_webhook(self):
        record = _make_record(status=RunStatus.APPROVED, reviewed_by="webhook")
        self.store.get.return_value = _make_record()
        self.store.approve.return_value = record

        body, headers = self._make_request({
            "run_id": "run123",
            "action": "approve",
        })
        result = self.handler.handle(body, headers)

        self.assertEqual(result["status_code"], 200)
        self.store.approve.assert_called_once_with("run123", "webhook")


if __name__ == "__main__":
    unittest.main()
