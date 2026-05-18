"""Pull /fixtures/statistics for every in-scope fixture.

Resumable via the gzip cache. Stats coverage starts ~2018; pre-2018 fixtures
return empty responses (still cached so we don't ask twice).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import api_football as af  # noqa: E402
from scripts.pull_api_football_lineups import LEAGUES, collect_fixture_ids  # noqa: E402


def main() -> None:
    print("Collecting fixture IDs (from cached /fixtures)...", flush=True)
    fixture_ids = collect_fixture_ids()
    print(f"Total fixtures in scope: {len(fixture_ids):,}", flush=True)

    already_cached = 0
    for _, _, fid in fixture_ids:
        if af._cache_path("fixtures/statistics", {"fixture": fid}).exists():
            already_cached += 1
    to_fetch = len(fixture_ids) - already_cached
    print(f"Already cached: {already_cached:,} | To fetch: {to_fetch:,}", flush=True)
    if to_fetch == 0:
        print("Nothing to do.")
        return

    start = time.time()
    fetched = 0
    empty = 0
    for i, (lid, year, fid) in enumerate(fixture_ids, 1):
        cache = af._cache_path("fixtures/statistics", {"fixture": fid})
        is_cached = cache.exists()
        try:
            body = af.request("fixtures/statistics", {"fixture": fid})
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR fixture={fid}: {exc}", flush=True)
            continue
        if not is_cached:
            fetched += 1
        if not body.get("response"):
            empty += 1
        if fetched and fetched % 200 == 0 and not is_cached:
            rl = af.last_rate_limit()
            elapsed = time.time() - start
            rate = fetched / elapsed if elapsed else 0
            eta = (to_fetch - fetched) / rate / 60 if rate else 0
            print(f"  [{i:,}/{len(fixture_ids):,}] fetched={fetched:,} empty={empty:,} "
                  f"| {rate:.1f} req/s | ETA {eta:.1f} min | "
                  f"daily remaining={rl.get('x-ratelimit-requests-remaining','?')}",
                  flush=True)

    elapsed = time.time() - start
    cache_dir = af.CACHE_ROOT / "fixtures_statistics"
    cache_size = sum(p.stat().st_size for p in cache_dir.glob("*.json.gz")) / 1e6
    print()
    print(f"Done in {elapsed/60:.1f} min")
    print(f"  fetched this run: {fetched:,}")
    print(f"  empty responses : {empty:,}")
    print(f"Cache size: {cache_size:.1f} MB")


if __name__ == "__main__":
    main()
