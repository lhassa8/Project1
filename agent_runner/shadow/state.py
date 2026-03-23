"""Virtual filesystem overlay for shadow mode.

Maintains an in-memory layer of proposed file changes so that reads-after-writes
work correctly during a shadow run.  The real filesystem is never modified.

Usage::

    state = ShadowState()
    state.write_file("/tmp/hello.txt", "world")
    state.read_file("/tmp/hello.txt")     # → "world" (from virtual layer)
    state.read_file("/etc/hostname")      # → real file contents (passthrough)
    state.list_files("/tmp")              # → merges virtual + real entries
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FileEntry:
    """A virtual file in the shadow layer."""

    path: str
    content: str
    deleted: bool = False


class ShadowState:
    """In-memory overlay that captures writes and serves consistent reads.

    All paths are resolved to absolute paths for consistent lookup.
    """

    def __init__(self) -> None:
        self._files: dict[str, FileEntry] = {}
        self._dirs_created: set[str] = set()

    # ------------------------------------------------------------------
    # Write operations (captured, never hit disk)
    # ------------------------------------------------------------------

    def write_file(self, path: str, content: str) -> str:
        """Record a virtual file write.  Returns a realistic success message."""
        abspath = os.path.abspath(path)
        self._files[abspath] = FileEntry(path=abspath, content=content)
        # Track parent dirs as implicitly created
        self._dirs_created.add(os.path.dirname(abspath))
        return f"Wrote {len(content)} bytes to {path}"

    def delete_file(self, path: str) -> str:
        """Record a virtual file deletion."""
        abspath = os.path.abspath(path)
        self._files[abspath] = FileEntry(path=abspath, content="", deleted=True)
        return f"Deleted {path}"

    def mkdir(self, path: str) -> str:
        """Record a virtual directory creation."""
        abspath = os.path.abspath(path)
        self._dirs_created.add(abspath)
        return f"Created directory {path}"

    # ------------------------------------------------------------------
    # Read operations (check virtual layer first, then real FS)
    # ------------------------------------------------------------------

    def read_file(self, path: str) -> str:
        """Read from virtual layer if file was written, otherwise from real FS."""
        abspath = os.path.abspath(path)

        # Check virtual layer first
        entry = self._files.get(abspath)
        if entry is not None:
            if entry.deleted:
                return f"Error: [Errno 2] No such file or directory: '{path}'"
            return entry.content

        # Fall through to real filesystem
        try:
            with open(abspath) as f:
                return f.read()
        except Exception as exc:
            return f"Error: {exc}"

    def list_files(self, path: str, recursive: bool = False) -> str:
        """List directory contents, merging virtual and real entries."""
        abspath = os.path.abspath(path)

        # Gather real entries
        real_entries: set[str] = set()
        try:
            if recursive:
                for root, dirs, files in os.walk(abspath):
                    for d in dirs:
                        real_entries.add(os.path.relpath(os.path.join(root, d), abspath) + "/")
                    for f in files:
                        real_entries.add(os.path.relpath(os.path.join(root, f), abspath))
                    if len(real_entries) >= 200:
                        break
            else:
                for name in os.listdir(abspath):
                    full = os.path.join(abspath, name)
                    real_entries.add(name + "/" if os.path.isdir(full) else name)
        except FileNotFoundError:
            # Directory might only exist in virtual layer
            pass
        except Exception as exc:
            if not self._has_virtual_entries_under(abspath):
                return f"Error: {exc}"

        # Merge virtual entries
        merged = set(real_entries)
        for fpath, entry in self._files.items():
            if entry.deleted:
                # Remove from listing if it was in real FS
                rel = os.path.relpath(fpath, abspath)
                merged.discard(rel)
                continue
            # If this file is directly under the listed path
            if os.path.dirname(fpath) == abspath:
                rel = os.path.basename(fpath)
                merged.add(rel)
            elif recursive and fpath.startswith(abspath + os.sep):
                rel = os.path.relpath(fpath, abspath)
                merged.add(rel)

        # Add virtual directories
        for dpath in self._dirs_created:
            if os.path.dirname(dpath) == abspath:
                rel = os.path.basename(dpath) + "/"
                merged.add(rel)
            elif recursive and dpath.startswith(abspath + os.sep):
                rel = os.path.relpath(dpath, abspath) + "/"
                merged.add(rel)

        entries = sorted(merged)
        if len(entries) > 200:
            entries = entries[:200]
            entries.append("... (truncated at 200 entries)")

        return "\n".join(entries) if entries else "(empty directory)"

    def file_exists(self, path: str) -> bool:
        """Check if a file exists (virtual or real)."""
        abspath = os.path.abspath(path)
        entry = self._files.get(abspath)
        if entry is not None:
            return not entry.deleted
        return os.path.exists(abspath)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_all_changes(self) -> list[dict[str, Any]]:
        """Return all virtual changes for review."""
        changes = []
        for abspath, entry in self._files.items():
            if entry.deleted:
                changes.append({"action": "delete", "path": abspath})
            else:
                changes.append({
                    "action": "create" if not os.path.exists(abspath) else "modify",
                    "path": abspath,
                    "content": entry.content,
                    "size": len(entry.content),
                })
        for dpath in self._dirs_created:
            if not os.path.isdir(dpath):
                changes.append({"action": "mkdir", "path": dpath})
        return changes

    def has_changes(self) -> bool:
        """True if any writes were captured."""
        return bool(self._files) or bool(self._dirs_created)

    def _has_virtual_entries_under(self, dirpath: str) -> bool:
        """Check if any virtual files exist under a directory."""
        prefix = dirpath + os.sep
        return any(p.startswith(prefix) for p in self._files) or any(
            d.startswith(prefix) for d in self._dirs_created
        )
