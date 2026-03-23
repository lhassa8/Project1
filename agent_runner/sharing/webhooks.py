"""Async webhook-based approval workflow system.

Sends webhook notifications when runs need approval and receives
approval/rejection callbacks.  Uses only stdlib (no third-party deps).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from agent_runner.sharing.run_store import RunRecord, RunStatus, RunStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WebhookConfig:
    """Configuration for a single webhook endpoint."""

    url: str
    secret: str
    timeout: int = 30
    retry_count: int = 3
    headers: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HMAC helpers
# ---------------------------------------------------------------------------

def _sign_payload(payload_bytes: bytes, secret: str) -> str:
    """Return hex-encoded HMAC-SHA256 signature for *payload_bytes*."""
    return hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()


def verify_signature(payload_bytes: bytes, secret: str, signature: str) -> bool:
    """Verify an HMAC-SHA256 signature (constant-time comparison)."""
    expected = _sign_payload(payload_bytes, secret)
    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# WebhookNotifier
# ---------------------------------------------------------------------------

class WebhookNotifier:
    """Sends webhook POST requests when run events occur.

    Supports multiple endpoints, HMAC-SHA256 signing, and retries with
    exponential backoff.

    Parameters
    ----------
    configs : list[WebhookConfig]
        One or more webhook endpoint configurations.
    """

    def __init__(self, configs: list[WebhookConfig]) -> None:
        self.configs = list(configs)

    # -- public API --------------------------------------------------------

    def notify_approval_needed(self, run_record: RunRecord) -> dict[str, bool]:
        """Notify all endpoints that a run needs approval.

        Returns a mapping of ``{url: success_bool}`` for each endpoint.
        """
        payload = self._build_approval_payload(run_record)
        return self._broadcast("approval_needed", payload)

    def notify_status_changed(
        self,
        run_record: RunRecord,
        old_status: RunStatus,
        new_status: RunStatus,
    ) -> dict[str, bool]:
        """Notify all endpoints that a run's status has changed.

        Returns a mapping of ``{url: success_bool}`` for each endpoint.
        """
        payload = {
            "event": "status_changed",
            "run_id": run_record.id,
            "old_status": old_status.value,
            "new_status": new_status.value,
            "reviewed_by": run_record.reviewed_by,
            "reviewed_at": run_record.reviewed_at,
            "timestamp": time.time(),
        }
        return self._broadcast("status_changed", payload)

    # -- internal helpers --------------------------------------------------

    @staticmethod
    def _build_approval_payload(record: RunRecord) -> dict[str, Any]:
        writes_summary = [
            {"tool": w.get("tool", "unknown"), "keys": list(w.get("input", {}).keys())}
            for w in record.captured_writes
        ]
        return {
            "event": "approval_needed",
            "run_id": record.id,
            "prompt": record.prompt,
            "tool_call_log": record.tool_call_log,
            "captured_writes_summary": writes_summary,
            "created_at": record.created_at,
            "timestamp": time.time(),
        }

    def _broadcast(self, event: str, payload: dict[str, Any]) -> dict[str, bool]:
        """Send *payload* to every configured endpoint, return results."""
        results: dict[str, bool] = {}
        threads: list[threading.Thread] = []

        def _send(cfg: WebhookConfig) -> None:
            results[cfg.url] = self._send_with_retry(cfg, payload)

        for cfg in self.configs:
            t = threading.Thread(target=_send, args=(cfg,), daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return results

    def _send_with_retry(
        self, cfg: WebhookConfig, payload: dict[str, Any]
    ) -> bool:
        """Try to POST *payload* to *cfg.url* with exponential backoff."""
        payload_bytes = json.dumps(payload, default=str).encode("utf-8")
        signature = _sign_payload(payload_bytes, cfg.secret)

        headers = {
            "Content-Type": "application/json",
            "X-Agent-Signature": signature,
            **cfg.headers,
        }

        last_exc: Exception | None = None
        for attempt in range(cfg.retry_count):
            try:
                req = urllib.request.Request(
                    cfg.url,
                    data=payload_bytes,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
                    if resp.status < 300:
                        return True
                    last_exc = RuntimeError(f"HTTP {resp.status}")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc

            if attempt < cfg.retry_count - 1:
                backoff = 2 ** attempt  # 1s, 2s, 4s, ...
                logger.warning(
                    "Webhook %s attempt %d failed (%s), retrying in %ds",
                    cfg.url,
                    attempt + 1,
                    last_exc,
                    backoff,
                )
                time.sleep(backoff)

        logger.error(
            "Webhook %s failed after %d attempts: %s",
            cfg.url,
            cfg.retry_count,
            last_exc,
        )
        return False


# ---------------------------------------------------------------------------
# ApprovalCallbackHandler
# ---------------------------------------------------------------------------

class ApprovalCallbackHandler:
    """Receives webhook callbacks and updates run status.

    Designed to be mounted at ``POST /api/v1/webhooks/approval-callback``
    inside the existing stdlib HTTP server.

    Parameters
    ----------
    store : RunStore
        The backing run store.
    webhook_secret : str
        Shared secret used to verify inbound HMAC signatures.
    """

    def __init__(self, store: RunStore, webhook_secret: str) -> None:
        self.store = store
        self.webhook_secret = webhook_secret

    def handle(self, body_bytes: bytes, headers: dict[str, str]) -> dict[str, Any]:
        """Process an inbound approval callback.

        Parameters
        ----------
        body_bytes : bytes
            Raw request body.
        headers : dict[str, str]
            Request headers (keys are case-insensitive-friendly; the caller
            should normalise or pass the raw mapping).

        Returns
        -------
        dict
            A response dict with ``status_code`` and ``body`` keys.
        """
        # -- Validate signature -------------------------------------------
        signature = headers.get("X-Agent-Signature", "")
        if not signature or not verify_signature(
            body_bytes, self.webhook_secret, signature
        ):
            return {
                "status_code": 401,
                "body": {"error": "Invalid or missing signature"},
            }

        # -- Parse payload ------------------------------------------------
        try:
            payload = json.loads(body_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {
                "status_code": 400,
                "body": {"error": "Malformed JSON payload"},
            }

        run_id = payload.get("run_id")
        action = payload.get("action")  # "approve" or "reject"
        reviewer = payload.get("reviewer", "webhook")

        if not run_id or action not in ("approve", "reject"):
            return {
                "status_code": 400,
                "body": {
                    "error": "Missing or invalid fields: run_id, action (approve|reject)"
                },
            }

        # -- Update store -------------------------------------------------
        record = self.store.get(run_id)
        if record is None:
            return {
                "status_code": 404,
                "body": {"error": f"Run '{run_id}' not found"},
            }

        if action == "approve":
            updated = self.store.approve(run_id, reviewer)
        else:
            updated = self.store.reject(run_id, reviewer)

        if updated is None:
            return {
                "status_code": 500,
                "body": {"error": "Failed to update run status"},
            }

        return {
            "status_code": 200,
            "body": {
                "run_id": updated.id,
                "status": updated.status.value,
                "reviewed_by": updated.reviewed_by,
                "reviewed_at": updated.reviewed_at,
            },
        }
