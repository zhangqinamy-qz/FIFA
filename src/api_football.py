"""api-football.com (api-sports.io) client with on-disk JSON cache.

Direct v3 endpoint, not RapidAPI. Reads API_FOOTBALL_KEY from .env (project root)
or from the process environment.

Caching: every request is hashed on (endpoint, params) and stored as JSON under
data/raw/api_football/<endpoint>/<hash>.json. Re-running a notebook does not
spend requests.

Rate-limit handling: the response headers x-ratelimit-requests-remaining and
x-ratelimit-requests-limit are surfaced via `last_rate_limit()`. On HTTP 429 we
sleep until the minute window resets (60s default) and retry once.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = PROJECT_ROOT / "data" / "raw" / "api_football"
DEFAULT_HOST = "v3.football.api-sports.io"


def _load_env() -> None:
    """Populate os.environ from FIFA/.env if not already set. Tiny parser, no dep."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_env()

_LAST_HEADERS: dict[str, str] = {}


def _api_key() -> str:
    key = os.environ.get("API_FOOTBALL_KEY")
    if not key:
        raise RuntimeError("API_FOOTBALL_KEY not set in environment or .env")
    return key


def _api_host() -> str:
    return os.environ.get("API_FOOTBALL_HOST", DEFAULT_HOST)


def _cache_path(endpoint: str, params: dict[str, Any]) -> Path:
    canon = json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.sha1(canon.encode()).hexdigest()[:16]
    folder = CACHE_ROOT / endpoint.strip("/").replace("/", "_")
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{digest}.json.gz"


def _cache_read(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _cache_write(path: Path, body: dict[str, Any]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(body, f)


def last_rate_limit() -> dict[str, str]:
    """Return rate-limit headers from the most recent live HTTP call."""
    return dict(_LAST_HEADERS)


def request(
    endpoint: str,
    params: dict[str, Any] | None = None,
    *,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """GET https://<host>/<endpoint>?<params>, cached on disk by default.

    Returns the parsed JSON body (dict with at least `response`, `errors`,
    `results`, `paging` keys per api-football conventions).
    """
    global _LAST_HEADERS
    params = params or {}
    cache = _cache_path(endpoint, params)

    if use_cache and not force_refresh and cache.exists():
        return _cache_read(cache)

    url = f"https://{_api_host()}/{endpoint.lstrip('/')}"
    headers = {"x-apisports-key": _api_key()}

    for attempt in range(2):
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        _LAST_HEADERS = {k: v for k, v in resp.headers.items() if k.lower().startswith("x-")}
        if resp.status_code == 429:
            time.sleep(61)
            continue
        resp.raise_for_status()
        body = resp.json()
        break
    else:
        raise RuntimeError(f"api-football: 429 twice for {endpoint} {params}")

    if body.get("errors"):
        # api-football returns 200 even on validation errors; surface them.
        errors = body["errors"]
        if isinstance(errors, dict) and errors:
            raise RuntimeError(f"api-football errors for {endpoint} {params}: {errors}")

    if use_cache:
        _cache_write(cache, body)
    return body


def paged_request(
    endpoint: str,
    params: dict[str, Any] | None = None,
    *,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Iterate api-football's `paging` pages and concatenate the `response` arrays."""
    base = dict(params or {})
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        call = dict(base) if page == 1 else {**base, "page": page}
        body = request(endpoint, call, use_cache=use_cache)
        out.extend(body.get("response", []))
        paging = body.get("paging", {}) or {}
        total = paging.get("total", 1)
        current = paging.get("current", page)
        if current >= total:
            break
        page = current + 1
    return out


# --- Convenience wrappers --------------------------------------------------

def get_leagues(**filters: Any) -> list[dict[str, Any]]:
    """Pass e.g. id=1 for World Cup, type='Cup', country='World'."""
    return paged_request("leagues", filters)


def get_fixtures(league: int, season: int) -> list[dict[str, Any]]:
    """All fixtures for one league+season."""
    return paged_request("fixtures", {"league": league, "season": season})


def get_lineups(fixture: int) -> list[dict[str, Any]]:
    """Lineups for one fixture. Returns two dicts (one per team) when available."""
    body = request("fixtures/lineups", {"fixture": fixture})
    return body.get("response", []) or []


def get_fixture_players(fixture: int) -> list[dict[str, Any]]:
    """Per-player statistics for one fixture (includes appearances/minutes)."""
    body = request("fixtures/players", {"fixture": fixture})
    return body.get("response", []) or []


def status() -> dict[str, Any]:
    """Account status: subscription tier, daily/min usage, request quota."""
    return request("status", use_cache=False)
