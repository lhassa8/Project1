"""RBAC and API key authentication for the agent runner.

Provides role-based access control, API key lifecycle management,
and decorator helpers for protecting HTTP handler methods.
Uses only stdlib.
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import secrets
import time
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

KEY_PREFIX = "agr_"


# ---------------------------------------------------------------------------
# Roles & Permissions
# ---------------------------------------------------------------------------

class Role(enum.Enum):
    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"


class Permission(enum.Enum):
    VIEW_RUNS = "view_runs"
    APPROVE_RUNS = "approve_runs"
    REJECT_RUNS = "reject_runs"
    REPLAY_RUNS = "replay_runs"
    MANAGE_KEYS = "manage_keys"
    VIEW_AUDIT = "view_audit"


ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.VIEWER: {
        Permission.VIEW_RUNS,
        Permission.VIEW_AUDIT,
    },
    Role.OPERATOR: {
        Permission.VIEW_RUNS,
        Permission.APPROVE_RUNS,
        Permission.REJECT_RUNS,
        Permission.REPLAY_RUNS,
        Permission.VIEW_AUDIT,
    },
    Role.ADMIN: set(Permission),  # all permissions
}


# ---------------------------------------------------------------------------
# APIKey dataclass
# ---------------------------------------------------------------------------

@dataclass
class APIKey:
    """Represents a stored API key (hash only -- never the raw key)."""

    key_id: str
    key_hash: str
    role: Role
    created_at: float
    expires_at: float | None
    description: str
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "key_hash": self.key_hash,
            "role": self.role.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "description": self.description,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> APIKey:
        return cls(
            key_id=data["key_id"],
            key_hash=data["key_hash"],
            role=Role(data["role"]),
            created_at=data["created_at"],
            expires_at=data.get("expires_at"),
            description=data.get("description", ""),
            is_active=data.get("is_active", True),
        )

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def safe_dict(self) -> dict[str, Any]:
        """Return a dict *without* the key_hash (safe for listing)."""
        d = self.to_dict()
        d.pop("key_hash", None)
        return d


# ---------------------------------------------------------------------------
# AuthManager
# ---------------------------------------------------------------------------

def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class AuthManager:
    """Manages API key lifecycle and RBAC authorisation.

    Keys are persisted in a JSON file so they survive restarts.

    Parameters
    ----------
    keys_path : str | Path
        Path to the JSON file for key storage.
    """

    def __init__(self, keys_path: str | Path = "keys.json") -> None:
        self.keys_path = Path(keys_path)
        self._keys: dict[str, APIKey] = {}
        self._load()

    # -- Key lifecycle -----------------------------------------------------

    def create_api_key(
        self,
        role: Role,
        description: str = "",
        expires_in_days: int | None = None,
    ) -> tuple[str, str]:
        """Generate a new API key.

        Returns
        -------
        (key_id, raw_key)
            The *raw_key* is shown once; only the SHA-256 hash is stored.
        """
        raw_token = secrets.token_urlsafe(32)
        raw_key = f"{KEY_PREFIX}{raw_token}"
        key_id = secrets.token_urlsafe(8)

        now = time.time()
        expires_at = (
            now + expires_in_days * 86400 if expires_in_days is not None else None
        )

        api_key = APIKey(
            key_id=key_id,
            key_hash=_hash_key(raw_key),
            role=role,
            created_at=now,
            expires_at=expires_at,
            description=description,
            is_active=True,
        )
        self._keys[key_id] = api_key
        self._save()
        return key_id, raw_key

    def authenticate(self, raw_key: str) -> APIKey | None:
        """Validate *raw_key* and return the matching :class:`APIKey`.

        Returns ``None`` if the key is unknown, inactive, or expired.
        """
        hashed = _hash_key(raw_key)
        for api_key in self._keys.values():
            if api_key.key_hash == hashed:
                if not api_key.is_active:
                    return None
                if api_key.is_expired():
                    return None
                return api_key
        return None

    def authorize(self, api_key: APIKey, permission: Permission) -> bool:
        """Return ``True`` if *api_key*'s role grants *permission*."""
        allowed = ROLE_PERMISSIONS.get(api_key.role, set())
        return permission in allowed

    def revoke_key(self, key_id: str) -> bool:
        """Deactivate a key.  Returns ``True`` if the key was found."""
        api_key = self._keys.get(key_id)
        if api_key is None:
            return False
        api_key.is_active = False
        self._save()
        return True

    def list_keys(self) -> list[APIKey]:
        """Return all keys **without** their hashes."""
        result: list[APIKey] = []
        for k in self._keys.values():
            # Return a shallow copy with hash redacted
            copy = APIKey(
                key_id=k.key_id,
                key_hash="",
                role=k.role,
                created_at=k.created_at,
                expires_at=k.expires_at,
                description=k.description,
                is_active=k.is_active,
            )
            result.append(copy)
        return result

    # -- Persistence -------------------------------------------------------

    def _load(self) -> None:
        if not self.keys_path.exists():
            return
        try:
            data = json.loads(self.keys_path.read_text())
            for entry in data:
                api_key = APIKey.from_dict(entry)
                self._keys[api_key.key_id] = api_key
        except Exception:
            logger.exception("Failed to load keys from %s", self.keys_path)

    def _save(self) -> None:
        data = [k.to_dict() for k in self._keys.values()]
        self.keys_path.write_text(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# Decorator helpers
# ---------------------------------------------------------------------------

def _extract_bearer_token(handler: Any) -> str | None:
    """Extract the Bearer token from an ``http.server`` handler's headers."""
    auth_header = handler.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    return None


def _respond_json(handler: Any, data: Any, status: int = 200) -> None:
    """Write a JSON response on an ``http.server`` BaseHTTPRequestHandler."""
    body = json.dumps(data, indent=2, default=str).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def require_auth(auth_manager: AuthManager) -> Callable:
    """Decorator that authenticates the request via Bearer token.

    On success the wrapped function receives an extra keyword argument
    ``api_key`` containing the validated :class:`APIKey`.

    Usage::

        @require_auth(auth_manager)
        def do_GET(self, *, api_key: APIKey):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(handler_self: Any, *args: Any, **kwargs: Any) -> Any:
            token = _extract_bearer_token(handler_self)
            if token is None:
                _respond_json(
                    handler_self,
                    {"error": "Missing Authorization header"},
                    401,
                )
                return None
            api_key = auth_manager.authenticate(token)
            if api_key is None:
                _respond_json(
                    handler_self,
                    {"error": "Invalid or expired API key"},
                    401,
                )
                return None
            kwargs["api_key"] = api_key
            return func(handler_self, *args, **kwargs)

        return wrapper

    return decorator


def require_permission(
    auth_manager: AuthManager, permission: Permission
) -> Callable:
    """Decorator that chains authentication **and** permission check.

    Usage::

        @require_permission(auth_manager, Permission.APPROVE_RUNS)
        def do_POST(self, *, api_key: APIKey):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(handler_self: Any, *args: Any, **kwargs: Any) -> Any:
            token = _extract_bearer_token(handler_self)
            if token is None:
                _respond_json(
                    handler_self,
                    {"error": "Missing Authorization header"},
                    401,
                )
                return None
            api_key = auth_manager.authenticate(token)
            if api_key is None:
                _respond_json(
                    handler_self,
                    {"error": "Invalid or expired API key"},
                    401,
                )
                return None
            if not auth_manager.authorize(api_key, permission):
                _respond_json(
                    handler_self,
                    {
                        "error": f"Forbidden: requires {permission.value} permission",
                    },
                    403,
                )
                return None
            kwargs["api_key"] = api_key
            return func(handler_self, *args, **kwargs)

        return wrapper

    return decorator
