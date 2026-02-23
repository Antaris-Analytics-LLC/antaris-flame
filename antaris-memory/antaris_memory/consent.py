"""
ConsentStore — manages user opt-in/revoke for anonymous data collection.

Stores a single local JSON flag file in the memory workspace.
Default state: opt-in is False (off by default, user must explicitly enable).

Usage:
    from antaris_memory.consent import ConsentStore

    store = ConsentStore("/path/to/workspace")
    store.load()

    if store.is_opted_in():
        # collect data
        pass

    store.opt_in()      # user grants consent
    store.revoke()      # user revokes consent
    store.save()
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

CONSENT_VERSION = "1.1"
_CONSENT_FILENAME = ".consent.json"


class ConsentStore:
    """Local consent flag store for anonymous semantic data collection.

    Attributes:
        workspace: Path to the memory workspace directory.
        semantic_opt_in: Whether user has opted in to semantic data sharing.
        consented_at: ISO timestamp when consent was granted (or None).
        revoked_at: ISO timestamp when consent was revoked (or None).
        consent_version: Version of consent schema at time of consent.
    """

    def __init__(self, workspace: str, filename: str = _CONSENT_FILENAME):
        self.workspace = Path(workspace)
        self._path = self.workspace / filename
        self._lock = threading.Lock()

        # Consent state
        self.semantic_opt_in: bool = False
        self.consented_at: Optional[str] = None
        self.revoked_at: Optional[str] = None
        self.consent_version: str = CONSENT_VERSION

    # ── Persistence ───────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load consent state from disk. Returns True if file existed."""
        if not self._path.exists():
            return False
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                self.semantic_opt_in = bool(data.get("semantic_opt_in", False))
                self.consented_at = data.get("consented_at")
                self.revoked_at = data.get("revoked_at")
                self.consent_version = data.get("consent_version", CONSENT_VERSION)
            return True
        except (json.JSONDecodeError, OSError):
            return False

    def save(self) -> str:
        """Persist consent state to disk. Returns path written."""
        self.workspace.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "semantic_opt_in": self.semantic_opt_in,
                "consented_at": self.consented_at,
                "revoked_at": self.revoked_at,
                "consent_version": self.consent_version,
            }
        tmp_path = str(self._path) + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, str(self._path))
        return str(self._path)

    # ── Consent Actions ───────────────────────────────────────────────────────

    def opt_in(self) -> None:
        """Grant consent for anonymous semantic data collection."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.semantic_opt_in = True
            self.consented_at = now
            self.revoked_at = None
            self.consent_version = CONSENT_VERSION

    def revoke(self) -> None:
        """Revoke consent. Collection stops immediately — no future data transmitted.

        Previously transmitted data cannot be identified or deleted (it is
        permanently anonymous with no user linkage). Future transmissions
        stop immediately upon calling this method.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.semantic_opt_in = False
            self.revoked_at = now

    # ── State Queries ─────────────────────────────────────────────────────────

    def is_opted_in(self) -> bool:
        """Return True if user has active opt-in consent."""
        with self._lock:
            return bool(self.semantic_opt_in)

    def status(self) -> dict:
        """Return current consent state as a dict."""
        with self._lock:
            return {
                "semantic_opt_in": self.semantic_opt_in,
                "consented_at": self.consented_at,
                "revoked_at": self.revoked_at,
                "consent_version": self.consent_version,
            }

    def __repr__(self) -> str:
        return (
            f"ConsentStore(opted_in={self.semantic_opt_in}, "
            f"consented_at={self.consented_at!r})"
        )
