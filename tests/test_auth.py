"""Tests for agent_runner.auth."""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from agent_runner.auth import (
    APIKey,
    AuthManager,
    KEY_PREFIX,
    Permission,
    Role,
    ROLE_PERMISSIONS,
    require_auth,
    require_permission,
    _hash_key,
)


# ---------------------------------------------------------------------------
# Role / Permission mapping
# ---------------------------------------------------------------------------

class TestRolePermissions(unittest.TestCase):
    def test_viewer_has_view_runs(self):
        self.assertIn(Permission.VIEW_RUNS, ROLE_PERMISSIONS[Role.VIEWER])

    def test_viewer_has_view_audit(self):
        self.assertIn(Permission.VIEW_AUDIT, ROLE_PERMISSIONS[Role.VIEWER])

    def test_viewer_cannot_approve(self):
        self.assertNotIn(Permission.APPROVE_RUNS, ROLE_PERMISSIONS[Role.VIEWER])

    def test_viewer_cannot_manage_keys(self):
        self.assertNotIn(Permission.MANAGE_KEYS, ROLE_PERMISSIONS[Role.VIEWER])

    def test_operator_can_approve_and_reject(self):
        perms = ROLE_PERMISSIONS[Role.OPERATOR]
        self.assertIn(Permission.APPROVE_RUNS, perms)
        self.assertIn(Permission.REJECT_RUNS, perms)
        self.assertIn(Permission.REPLAY_RUNS, perms)

    def test_operator_cannot_manage_keys(self):
        self.assertNotIn(Permission.MANAGE_KEYS, ROLE_PERMISSIONS[Role.OPERATOR])

    def test_admin_has_all_permissions(self):
        self.assertEqual(ROLE_PERMISSIONS[Role.ADMIN], set(Permission))


# ---------------------------------------------------------------------------
# Key generation and validation
# ---------------------------------------------------------------------------

class TestAuthManager(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.keys_path = Path(self.tmpdir) / "keys.json"
        self.mgr = AuthManager(keys_path=self.keys_path)

    def tearDown(self):
        if self.keys_path.exists():
            self.keys_path.unlink()
        os.rmdir(self.tmpdir)

    def test_create_key_returns_id_and_raw(self):
        key_id, raw_key = self.mgr.create_api_key(Role.VIEWER, "test key")
        self.assertIsInstance(key_id, str)
        self.assertTrue(raw_key.startswith(KEY_PREFIX))

    def test_create_key_persists_to_file(self):
        self.mgr.create_api_key(Role.VIEWER, "test key")
        self.assertTrue(self.keys_path.exists())
        data = json.loads(self.keys_path.read_text())
        self.assertEqual(len(data), 1)

    def test_authenticate_valid_key(self):
        key_id, raw_key = self.mgr.create_api_key(Role.OPERATOR, "ops key")
        api_key = self.mgr.authenticate(raw_key)
        self.assertIsNotNone(api_key)
        self.assertEqual(api_key.key_id, key_id)
        self.assertEqual(api_key.role, Role.OPERATOR)

    def test_authenticate_invalid_key(self):
        self.mgr.create_api_key(Role.VIEWER, "test key")
        result = self.mgr.authenticate("agr_bogus_key_that_doesnt_exist")
        self.assertIsNone(result)

    def test_authenticate_after_reload(self):
        """Keys survive a fresh AuthManager instance."""
        key_id, raw_key = self.mgr.create_api_key(Role.ADMIN, "admin key")
        mgr2 = AuthManager(keys_path=self.keys_path)
        api_key = mgr2.authenticate(raw_key)
        self.assertIsNotNone(api_key)
        self.assertEqual(api_key.key_id, key_id)

    def test_key_hash_is_sha256(self):
        key_id, raw_key = self.mgr.create_api_key(Role.VIEWER, "test")
        expected_hash = _hash_key(raw_key)
        # Check the stored key has matching hash
        stored = self.mgr._keys[key_id]
        self.assertEqual(stored.key_hash, expected_hash)

    def test_authorize_viewer_view_runs(self):
        _, raw_key = self.mgr.create_api_key(Role.VIEWER, "v")
        api_key = self.mgr.authenticate(raw_key)
        self.assertTrue(self.mgr.authorize(api_key, Permission.VIEW_RUNS))

    def test_authorize_viewer_cannot_approve(self):
        _, raw_key = self.mgr.create_api_key(Role.VIEWER, "v")
        api_key = self.mgr.authenticate(raw_key)
        self.assertFalse(self.mgr.authorize(api_key, Permission.APPROVE_RUNS))

    def test_authorize_admin_can_manage_keys(self):
        _, raw_key = self.mgr.create_api_key(Role.ADMIN, "a")
        api_key = self.mgr.authenticate(raw_key)
        self.assertTrue(self.mgr.authorize(api_key, Permission.MANAGE_KEYS))


# ---------------------------------------------------------------------------
# Key expiry
# ---------------------------------------------------------------------------

class TestKeyExpiry(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.keys_path = Path(self.tmpdir) / "keys.json"
        self.mgr = AuthManager(keys_path=self.keys_path)

    def tearDown(self):
        if self.keys_path.exists():
            self.keys_path.unlink()
        os.rmdir(self.tmpdir)

    def test_key_with_expiry_works_before_expiry(self):
        _, raw_key = self.mgr.create_api_key(
            Role.VIEWER, "temp", expires_in_days=30
        )
        api_key = self.mgr.authenticate(raw_key)
        self.assertIsNotNone(api_key)

    def test_expired_key_returns_none(self):
        _, raw_key = self.mgr.create_api_key(
            Role.VIEWER, "temp", expires_in_days=1
        )
        # Force the key to be expired by backdating expires_at
        for k in self.mgr._keys.values():
            k.expires_at = time.time() - 1
        api_key = self.mgr.authenticate(raw_key)
        self.assertIsNone(api_key)

    def test_key_without_expiry_never_expires(self):
        _, raw_key = self.mgr.create_api_key(Role.VIEWER, "permanent")
        api_key = self.mgr.authenticate(raw_key)
        self.assertIsNotNone(api_key)
        self.assertFalse(api_key.is_expired())

    def test_is_expired_method(self):
        key = APIKey(
            key_id="x",
            key_hash="h",
            role=Role.VIEWER,
            created_at=time.time(),
            expires_at=time.time() - 100,
            description="expired",
            is_active=True,
        )
        self.assertTrue(key.is_expired())

    def test_is_not_expired_method(self):
        key = APIKey(
            key_id="x",
            key_hash="h",
            role=Role.VIEWER,
            created_at=time.time(),
            expires_at=time.time() + 100000,
            description="future",
            is_active=True,
        )
        self.assertFalse(key.is_expired())


# ---------------------------------------------------------------------------
# Key revocation
# ---------------------------------------------------------------------------

class TestKeyRevocation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.keys_path = Path(self.tmpdir) / "keys.json"
        self.mgr = AuthManager(keys_path=self.keys_path)

    def tearDown(self):
        if self.keys_path.exists():
            self.keys_path.unlink()
        os.rmdir(self.tmpdir)

    def test_revoke_existing_key(self):
        key_id, raw_key = self.mgr.create_api_key(Role.OPERATOR, "ops")
        result = self.mgr.revoke_key(key_id)
        self.assertTrue(result)

    def test_revoked_key_cannot_authenticate(self):
        key_id, raw_key = self.mgr.create_api_key(Role.OPERATOR, "ops")
        self.mgr.revoke_key(key_id)
        api_key = self.mgr.authenticate(raw_key)
        self.assertIsNone(api_key)

    def test_revoke_nonexistent_key(self):
        result = self.mgr.revoke_key("nonexistent-id")
        self.assertFalse(result)

    def test_revocation_persists(self):
        key_id, raw_key = self.mgr.create_api_key(Role.ADMIN, "admin")
        self.mgr.revoke_key(key_id)
        mgr2 = AuthManager(keys_path=self.keys_path)
        api_key = mgr2.authenticate(raw_key)
        self.assertIsNone(api_key)

    def test_list_keys_shows_revoked(self):
        key_id, _ = self.mgr.create_api_key(Role.VIEWER, "v")
        self.mgr.revoke_key(key_id)
        keys = self.mgr.list_keys()
        self.assertEqual(len(keys), 1)
        self.assertFalse(keys[0].is_active)


# ---------------------------------------------------------------------------
# list_keys
# ---------------------------------------------------------------------------

class TestListKeys(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.keys_path = Path(self.tmpdir) / "keys.json"
        self.mgr = AuthManager(keys_path=self.keys_path)

    def tearDown(self):
        if self.keys_path.exists():
            self.keys_path.unlink()
        os.rmdir(self.tmpdir)

    def test_list_empty(self):
        self.assertEqual(self.mgr.list_keys(), [])

    def test_list_returns_all_keys(self):
        self.mgr.create_api_key(Role.VIEWER, "v")
        self.mgr.create_api_key(Role.OPERATOR, "o")
        self.mgr.create_api_key(Role.ADMIN, "a")
        self.assertEqual(len(self.mgr.list_keys()), 3)

    def test_list_keys_hides_hash(self):
        self.mgr.create_api_key(Role.VIEWER, "v")
        keys = self.mgr.list_keys()
        self.assertEqual(keys[0].key_hash, "")


# ---------------------------------------------------------------------------
# require_auth / require_permission decorators
# ---------------------------------------------------------------------------

def _make_mock_handler(auth_header: str | None = None):
    """Create a mock HTTP handler with controllable headers and response."""
    handler = MagicMock()
    headers = {}
    if auth_header:
        headers["Authorization"] = auth_header

    handler.headers = headers
    handler.wfile = io.BytesIO()

    # Make send_response, send_header, end_headers work
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()

    return handler


class TestRequireAuth(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.keys_path = Path(self.tmpdir) / "keys.json"
        self.mgr = AuthManager(keys_path=self.keys_path)
        self.key_id, self.raw_key = self.mgr.create_api_key(
            Role.OPERATOR, "test-key"
        )

    def tearDown(self):
        if self.keys_path.exists():
            self.keys_path.unlink()
        os.rmdir(self.tmpdir)

    def test_valid_auth_passes_api_key(self):
        called_with = {}

        @require_auth(self.mgr)
        def handler_method(self_handler, *, api_key):
            called_with["api_key"] = api_key

        mock_handler = _make_mock_handler(f"Bearer {self.raw_key}")
        handler_method(mock_handler)
        self.assertIn("api_key", called_with)
        self.assertEqual(called_with["api_key"].key_id, self.key_id)

    def test_missing_auth_header_returns_401(self):
        @require_auth(self.mgr)
        def handler_method(self_handler, *, api_key):
            pass  # should not be reached

        mock_handler = _make_mock_handler()
        handler_method(mock_handler)
        mock_handler.send_response.assert_called_once_with(401)

    def test_invalid_key_returns_401(self):
        @require_auth(self.mgr)
        def handler_method(self_handler, *, api_key):
            pass

        mock_handler = _make_mock_handler("Bearer agr_invalid_key_here")
        handler_method(mock_handler)
        mock_handler.send_response.assert_called_once_with(401)


class TestRequirePermission(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.keys_path = Path(self.tmpdir) / "keys.json"
        self.mgr = AuthManager(keys_path=self.keys_path)

    def tearDown(self):
        if self.keys_path.exists():
            self.keys_path.unlink()
        os.rmdir(self.tmpdir)

    def test_authorized_request_succeeds(self):
        _, raw_key = self.mgr.create_api_key(Role.OPERATOR, "ops")
        called = {"reached": False}

        @require_permission(self.mgr, Permission.APPROVE_RUNS)
        def handler_method(self_handler, *, api_key):
            called["reached"] = True

        mock_handler = _make_mock_handler(f"Bearer {raw_key}")
        handler_method(mock_handler)
        self.assertTrue(called["reached"])

    def test_insufficient_permission_returns_403(self):
        _, raw_key = self.mgr.create_api_key(Role.VIEWER, "viewer")

        @require_permission(self.mgr, Permission.APPROVE_RUNS)
        def handler_method(self_handler, *, api_key):
            pass  # should not be reached

        mock_handler = _make_mock_handler(f"Bearer {raw_key}")
        handler_method(mock_handler)
        mock_handler.send_response.assert_called_once_with(403)

    def test_missing_auth_returns_401(self):
        @require_permission(self.mgr, Permission.VIEW_RUNS)
        def handler_method(self_handler, *, api_key):
            pass

        mock_handler = _make_mock_handler()
        handler_method(mock_handler)
        mock_handler.send_response.assert_called_once_with(401)

    def test_admin_has_all_permissions(self):
        _, raw_key = self.mgr.create_api_key(Role.ADMIN, "admin")
        results = {}

        for perm in Permission:
            @require_permission(self.mgr, perm)
            def handler_method(self_handler, *, api_key, _perm=perm):
                results[_perm] = True

            mock_handler = _make_mock_handler(f"Bearer {raw_key}")
            handler_method(mock_handler)

        self.assertEqual(len(results), len(Permission))

    def test_expired_key_returns_401_on_permission_check(self):
        key_id, raw_key = self.mgr.create_api_key(
            Role.ADMIN, "admin", expires_in_days=1
        )
        # Force expiry
        for k in self.mgr._keys.values():
            k.expires_at = time.time() - 1

        @require_permission(self.mgr, Permission.VIEW_RUNS)
        def handler_method(self_handler, *, api_key):
            pass

        mock_handler = _make_mock_handler(f"Bearer {raw_key}")
        handler_method(mock_handler)
        mock_handler.send_response.assert_called_once_with(401)

    def test_revoked_key_returns_401_on_permission_check(self):
        key_id, raw_key = self.mgr.create_api_key(Role.ADMIN, "admin")
        self.mgr.revoke_key(key_id)

        @require_permission(self.mgr, Permission.VIEW_RUNS)
        def handler_method(self_handler, *, api_key):
            pass

        mock_handler = _make_mock_handler(f"Bearer {raw_key}")
        handler_method(mock_handler)
        mock_handler.send_response.assert_called_once_with(401)


# ---------------------------------------------------------------------------
# APIKey dataclass helpers
# ---------------------------------------------------------------------------

class TestAPIKeyDataclass(unittest.TestCase):
    def test_to_dict_roundtrip(self):
        key = APIKey(
            key_id="abc",
            key_hash="deadbeef",
            role=Role.OPERATOR,
            created_at=1700000000.0,
            expires_at=None,
            description="test",
            is_active=True,
        )
        d = key.to_dict()
        restored = APIKey.from_dict(d)
        self.assertEqual(restored.key_id, key.key_id)
        self.assertEqual(restored.role, key.role)

    def test_safe_dict_excludes_hash(self):
        key = APIKey(
            key_id="abc",
            key_hash="deadbeef",
            role=Role.VIEWER,
            created_at=1700000000.0,
            expires_at=None,
            description="test",
        )
        safe = key.safe_dict()
        self.assertNotIn("key_hash", safe)
        self.assertIn("key_id", safe)


if __name__ == "__main__":
    unittest.main()
