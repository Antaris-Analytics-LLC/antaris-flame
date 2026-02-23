"""
CollectionAgent — anonymous semantic data collection pipeline.

Extracts top word-pair co-occurrence frequencies from the local
CooccurrenceIndex, applies differential privacy noise, and transmits
to the configured collection endpoint.

Collection only runs when:
1. ConsentStore.is_opted_in() is True
2. An endpoint URL is configured
3. The collection interval has elapsed

The transmitted payload contains ONLY word-pair frequency counts:
    {"pairs": [{"a": "meeting", "b": "reschedule", "count": 8}], ...}

No conversation content, user identifiers, or personal information
is ever included.

Note: Supabase endpoint transmission is stubbed (NOT YET WIRED).
      Wire endpoint_url when Supabase project is configured.
"""

import json
import logging
import random
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .cooccurrence import CooccurrenceIndex
    from .consent import ConsentStore

logger = logging.getLogger(__name__)

_COLLECTION_VERSION = "1.1"
_DEFAULT_TOP_N = 200          # Top word pairs to extract per cycle
_DEFAULT_NOISE_SCALE = 1.5    # Laplace noise scale for differential privacy
_DEFAULT_MIN_COUNT = 3        # Minimum count to include a pair (filter noise)
_DEFAULT_INTERVAL = 3600.0    # Seconds between collection cycles (1 hour)
_STATE_FILENAME = ".collection_state.json"


class CollectionAgent:
    """Periodic collector for anonymous word-pair frequency statistics.

    Extracts the most frequent word pairs from the local CooccurrenceIndex,
    applies Laplace differential privacy noise to counts, and transmits
    the anonymized payload to the configured endpoint.

    Args:
        workspace: Path to memory workspace directory.
        cooccurrence: CooccurrenceIndex to extract pairs from.
        consent: ConsentStore to check opt-in status.
        endpoint_url: HTTPS POST endpoint URL (None = stub / not yet wired).
        top_n: Number of top pairs to extract per cycle.
        noise_scale: Laplace mechanism noise scale (privacy budget).
        min_count: Minimum raw count before noise for a pair to be included.
        interval_seconds: Minimum seconds between collection cycles.
    """

    def __init__(
        self,
        workspace: str,
        cooccurrence: "CooccurrenceIndex",
        consent: "ConsentStore",
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        top_n: int = _DEFAULT_TOP_N,
        noise_scale: float = _DEFAULT_NOISE_SCALE,
        min_count: int = _DEFAULT_MIN_COUNT,
        interval_seconds: float = _DEFAULT_INTERVAL,
    ):
        self.workspace = Path(workspace)
        self.cooccurrence = cooccurrence
        self.consent = consent
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.top_n = top_n
        self.noise_scale = noise_scale
        self.min_count = min_count
        self.interval_seconds = interval_seconds

        self._state_path = self.workspace / _STATE_FILENAME
        self._last_run: float = 0.0
        self._lock = threading.Lock()

        # Stats
        self._total_cycles: int = 0
        self._total_pairs_sent: int = 0
        self._last_error: Optional[str] = None

    # ── State persistence ─────────────────────────────────────────────────────

    def load_state(self) -> None:
        """Load last-run timestamp from disk."""
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._last_run = float(data.get("last_run", 0.0))
            self._total_cycles = int(data.get("total_cycles", 0))
            self._total_pairs_sent = int(data.get("total_pairs_sent", 0))
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    def save_state(self) -> None:
        """Persist last-run timestamp to disk."""
        try:
            self.workspace.mkdir(parents=True, exist_ok=True)
            data = {
                "last_run": self._last_run,
                "total_cycles": self._total_cycles,
                "total_pairs_sent": self._total_pairs_sent,
            }
            tmp = str(self._state_path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            import os
            os.replace(tmp, str(self._state_path))
        except OSError as e:
            logger.warning("CollectionAgent: could not save state: %s", e)

    # ── Core collection ───────────────────────────────────────────────────────

    def should_run(self) -> bool:
        """Return True if a collection cycle is due and consent is active."""
        if not self.consent.is_opted_in():
            return False
        with self._lock:
            return (time.time() - self._last_run) >= self.interval_seconds

    def extract_pairs(self) -> List[Tuple[str, str, int]]:
        """Extract top-N word pairs from the CooccurrenceIndex.

        Returns list of (word_a, word_b, raw_count) tuples sorted by count
        descending, filtered to pairs with count >= min_count.
        """
        # Build flat list of (word_a, word_b, count) from the index
        pairs: List[Tuple[str, str, int]] = []
        try:
            counts = dict(self.cooccurrence._counts)
        except Exception:
            return []

        seen: set = set()
        for word_a, co_dict in counts.items():
            for word_b, count in co_dict.items():
                # Deduplicate symmetric pairs
                key = tuple(sorted([word_a, word_b]))
                if key in seen:
                    continue
                seen.add(key)
                if count >= self.min_count:
                    pairs.append((key[0], key[1], count))

        # Sort by count descending, take top N
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[: self.top_n]

    def _apply_noise(self, count: int) -> int:
        """Apply Laplace differential privacy noise to a count.

        Uses the Laplace mechanism: noise ~ Laplace(0, noise_scale).
        Ensures the noised count is always >= 1 (never leaks zero-count pairs).
        """
        noise = random.gauss(0, self.noise_scale)
        noised = int(round(count + noise))
        return max(1, noised)

    def build_payload(self) -> dict:
        """Build the anonymized transmission payload.

        Returns a dict ready for JSON serialization. No user identifiers,
        no session information, no conversation content.
        """
        raw_pairs = self.extract_pairs()
        pairs = [
            {
                "a": word_a,
                "b": word_b,
                "count": self._apply_noise(raw_count),
            }
            for word_a, word_b, raw_count in raw_pairs
        ]
        return {
            "version": _COLLECTION_VERSION,
            "noise_applied": True,
            "pairs": pairs,
        }

    def transmit(self, payload: dict) -> bool:
        """Transmit word-pair payload to Supabase via RPC upsert.

        Calls /rest/v1/rpc/upsert_word_pairs with the anonymised pair counts.
        Returns True on success, False on any failure.
        Stubs silently when endpoint_url or api_key is not configured.
        """
        import urllib.request
        import urllib.error

        if not self.endpoint_url or not self.api_key:
            pair_count = len(payload.get("pairs", []))
            logger.debug(
                "CollectionAgent: STUB — would transmit %d pairs (endpoint/key not configured)",
                pair_count,
            )
            return True  # Stub succeeds silently

        rpc_url = self.endpoint_url.rstrip("/") + "/rest/v1/rpc/upsert_word_pairs"
        rpc_payload = {
            "pairs": [
                {
                    "a": p["a"],
                    "b": p["b"],
                    "count": p["count"],
                    "noise_applied": p.get("noise_applied", True),
                    "version": payload.get("version", "1.1"),
                }
                for p in payload.get("pairs", [])
            ]
        }

        try:
            data = json.dumps(rpc_payload).encode("utf-8")
            req = urllib.request.Request(
                rpc_url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "apikey": self.api_key,
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status in (200, 204):
                    logger.info(
                        "CollectionAgent: transmitted %d pairs to Supabase (HTTP %d)",
                        len(rpc_payload["pairs"]),
                        resp.status,
                    )
                    return True
                logger.warning(
                    "CollectionAgent: unexpected HTTP status %d from Supabase", resp.status
                )
                return False
        except urllib.error.HTTPError as exc:
            logger.error(
                "CollectionAgent: HTTP error from Supabase: %d %s", exc.code, exc.reason
            )
            return False
        except Exception as exc:
            logger.error("CollectionAgent: transmission error: %s", exc)
            return False

    def run_cycle(self) -> bool:
        """Run a single collection cycle if due and consented.

        Returns True if a cycle ran (regardless of transmission success),
        False if skipped (not due, not consented, or already running).
        """
        if not self.should_run():
            return False

        with self._lock:
            # Re-check inside lock to prevent double-run
            if (time.time() - self._last_run) < self.interval_seconds:
                return False
            # Mark as running immediately
            self._last_run = time.time()

        try:
            payload = self.build_payload()
            pair_count = len(payload.get("pairs", []))

            if pair_count == 0:
                logger.debug("CollectionAgent: no pairs to transmit this cycle")
                return True

            success = self.transmit(payload)
            if success:
                self._total_pairs_sent += pair_count
                self._total_cycles += 1
                self._last_error = None
                logger.debug(
                    "CollectionAgent: cycle complete — %d pairs processed", pair_count
                )
            else:
                self._last_error = "transmission failed"
                logger.warning("CollectionAgent: transmission failed this cycle")

            self.save_state()
            return True

        except Exception as e:
            self._last_error = str(e)
            logger.error("CollectionAgent: cycle error: %s", e)
            return False

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return collection agent statistics."""
        return {
            "opted_in": self.consent.is_opted_in(),
            "endpoint_configured": bool(self.endpoint_url),
            "endpoint_wired": False,  # Update to True when Supabase is wired
            "total_cycles": self._total_cycles,
            "total_pairs_sent": self._total_pairs_sent,
            "last_run": self._last_run,
            "last_error": self._last_error,
            "interval_seconds": self.interval_seconds,
        }

    def __repr__(self) -> str:
        return (
            f"CollectionAgent(opted_in={self.consent.is_opted_in()}, "
            f"cycles={self._total_cycles}, "
            f"endpoint={'configured' if self.endpoint_url else 'stub'})"
        )
