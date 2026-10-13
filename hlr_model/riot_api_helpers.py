"""Opt-in, sequential Riot client and read-through cache. Importing does no I/O."""
from __future__ import annotations

import json
import time
from collections import Counter, deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests


class CollectionDisabled(RuntimeError):
    pass


class AuthenticationError(RuntimeError):
    pass


class RiotRequestError(RuntimeError):
    """Safe error text: never includes headers, token, or response body."""


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value):
    """Only call for V2-owned checkpoints; legacy files are always read-only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    temporary.replace(path)


def _limits(value):
    return [(int(n), float(seconds)) for part in value.split(",")
            for n, seconds in [part.split(":")]] if value else []


class RiotClient:
    """One client per key/run; conservative defaults plus advertised limits.

    App and method limits are tracked by host and endpoint class. Response
    counts also gate requests, accounting for usage outside this process.
    Service limits are handled by 429. Do not run other collectors with this
    key concurrently. Sessions/credentials never enter serialized state.
    """

    def __init__(self, api_key, *, enabled=False, retries=6,
                 app_limits=((20, 1), (100, 120))):
        self.enabled = enabled
        self.retries = retries
        self.app_limits = list(app_limits)
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"X-Riot-Token": api_key})
        self.has_key = bool(api_key)
        self.events = {}
        self.advertised = {}
        self.blocked_until = {}
        self.metrics = Counter()

    def require_enabled(self):
        if not self.enabled:
            raise CollectionDisabled("Collection requires RUN_COLLECTION = True.")
        if not self.has_key:
            raise AuthenticationError("Set RIOT_API_KEY in the repository-root .env.")

    def _pace(self, host, method):
        keys = [(host, "app"), (host, method)]
        while True:
            now = time.monotonic()
            wait = 0.0
            for key in keys:
                limits = self.advertised.get(key, self.app_limits if key[1] == "app" else [])
                events = self.events.setdefault(key, deque())
                longest = max([seconds for _, seconds in limits] + [120])
                while events and events[0] <= now - longest - 1:
                    events.popleft()
                wait = max(wait, self.blocked_until.get(key, 0) - now)
                for cap, window in limits:
                    recent = [t for t in events if t > now - window - 0.1]
                    if len(recent) >= cap:
                        wait = max(wait, recent[-cap] + window + 0.1 - now)
            if wait <= 0:
                for key in keys:
                    self.events[key].append(now)
                return
            time.sleep(wait)

    def _observe(self, headers, host, method):
        now = time.monotonic()
        for prefix, key in [("App", (host, "app")), ("Method", (host, method))]:
            limits = _limits(headers.get(f"X-{prefix}-Rate-Limit", ""))
            if limits:
                self.advertised[key] = limits
            for count, window in _limits(headers.get(f"X-{prefix}-Rate-Limit-Count", "")):
                matching = [cap for cap, seconds in self.advertised.get(key, []) if seconds == window]
                if matching and count >= min(matching):
                    self.blocked_until[key] = max(self.blocked_until.get(key, 0), now + window + 0.1)

    def get(self, host, path, *, method, params=None):
        self.require_enabled()
        for attempt in range(self.retries + 1):
            self._pace(host, method)
            self.metrics["http_attempts"] += 1
            try:
                response = self.session.get(
                    f"https://{host}.api.riotgames.com{path}", params=params, timeout=(10, 45))
            except (requests.Timeout, requests.ConnectionError):
                if attempt == self.retries:
                    raise RiotRequestError("Network request failed after bounded retries.") from None
                time.sleep(min(60, 2 ** attempt))
                self.metrics["transient_retries"] += 1
                continue
            self._observe(response.headers, host, method)
            status = response.status_code
            if status in (401, 403):
                raise AuthenticationError("Riot rejected the key (401/403); renew it or check API permissions.")
            if status == 404:
                raise RiotRequestError("Riot returned 404; record unavailable, retry on a later resume.")
            if status == 429:
                raw = response.headers.get("Retry-After")
                try:
                    delay = float(raw)
                except (TypeError, ValueError):
                    try:
                        delay = (parsedate_to_datetime(raw) - datetime.now(timezone.utc)).total_seconds()
                    except (TypeError, ValueError):
                        delay = min(120, 2 ** attempt + 10)
                self.metrics["rate_limit_retries"] += 1
                self.blocked_until[(host, "app")] = time.monotonic() + max(0, delay) + 1
            elif status in (408, 500, 502, 503, 504):
                self.metrics["transient_retries"] += 1
                time.sleep(min(60, 2 ** attempt))
            elif 200 <= status < 300:
                try:
                    return response.json()
                except ValueError:
                    raise RiotRequestError("Riot returned malformed JSON.") from None
            else:
                raise RiotRequestError(f"Riot returned HTTP {status}.")
            if attempt == self.retries:
                raise RiotRequestError(f"HTTP {status} persisted after bounded retries; resume later.")

    def match_ids(self, puuid, **params):
        return self.get("americas", f"/lol/match/v5/matches/by-puuid/{puuid}/ids",
                        method="match_ids", params=params)

    def match(self, match_id):
        return self.get("americas", f"/lol/match/v5/matches/{match_id}", method="match_detail")

    def masters(self):
        # Optional inspection endpoint only: today's membership is not study-period evidence.
        return self.get("na1", "/lol/league/v4/masterleagues/by-queue/RANKED_SOLO_5x5",
                        method="master_league")


class MatchStore:
    """Global match-ID cache, legacy roots searched without copying or writing."""

    def __init__(self, own_root, legacy_roots=(), client=None):
        self.own_root = Path(own_root)
        self.roots = [self.own_root, *map(Path, legacy_roots)]
        self.client = client
        self.metrics = Counter()
        self.seen = set()

    @staticmethod
    def valid(payload, match_id):
        return (isinstance(payload, dict)
                and isinstance(payload.get("metadata"), dict)
                and isinstance(payload.get("info"), dict)
                and payload.get("metadata", {}).get("matchId") == match_id
                and isinstance(payload.get("info", {}).get("participants"), list)
                and bool(payload["info"].get("gameVersion")))

    def get(self, match_id, *, download=False):
        if not isinstance(match_id, str) or not match_id.startswith("NA1_") or not match_id[4:].isdigit():
            raise ValueError("Expected a North America Match-V5 ID.")
        for root in self.roots:
            path = root / f"{match_id}.json"
            if path.exists():
                try:
                    payload = read_json(path)
                    if self.valid(payload, match_id):
                        if match_id not in self.seen:
                            self.metrics["unique_cached_matches"] += 1
                            self.seen.add(match_id)
                        return payload
                except (OSError, ValueError, TypeError):
                    pass
                self.metrics["invalid_cache_reads"] += 1
        if not download:
            return None
        if self.client is None:
            raise CollectionDisabled("No enabled Riot client supplied.")
        payload = self.client.match(match_id)
        if not self.valid(payload, match_id):
            raise RiotRequestError("Incomplete/mismatched Riot match payload.")
        path = self.own_root / f"{match_id}.json"
        # Preserve even an invalid pre-existing file for inspection.
        if path.exists():
            path.rename(path.with_name(path.name + f".invalid-{time.time_ns()}"))
        atomic_json(path, payload)
        self.metrics["new_match_downloads"] += 1
        self.seen.add(match_id)
        return payload
