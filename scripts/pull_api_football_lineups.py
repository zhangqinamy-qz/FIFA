"""Pull lineups for every fixture in the in-scope international competitions.

Resumable: every /fixtures/lineups call is cached as gzipped JSON, so
re-running this script after an interruption only sends requests for
the fixtures that haven't been cached yet.

Skips Friendlies (league 10) — competitive-only per project decision.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import api_football as af  # noqa: E402

LEAGUES = {
    1:  "World Cup",
    4:  "Euro",
    5:  "UEFA Nations League",
    6:  "AFCON",
    7:  "Asian Cup",
    9:  "Copa America",
    22: "Gold Cup",
    29: "WC Qual Africa",
    30: "WC Qual Asia",
    31: "WC Qual CONCACAF",
    32: "WC Qual Europe",
    33: "WC Qual Oceania",
    34: "WC Qual S America",
    37: "WC Qual Intercont",
}


def collect_fixture_ids() -> list[tuple[int, int, int]]:
    """Return (league_id, season, fixture_id) for every in-scope fixture.

    Re-reads from cache when possible (fixtures were fetched during scoping).
    """
    ids: list[tuple[int, int, int]] = []
    for lid in LEAGUES:
        body = af.request("leagues", {"id": lid})
        seasons = [s["year"] for s in body["response"][0]["seasons"]]
        for year in seasons:
            fx = af.get_fixtures(league=lid, season=year)
            for f in fx:
                ids.append((lid, year, f["fixture"]["id"]))
    return ids


def main() -> None:
    print("Collecting fixture IDs...", flush=True)
    fixture_ids = collect_fixture_ids()
    print(f"Total fixtures in scope: {len(fixture_ids):,}", flush=True)

    cache_dir = af.CACHE_ROOT / "fixtures_lineups"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Pre-count what's already cached so we can show resume progress.
    already_cached = 0
    for _, _, fid in fixture_ids:
        if af._cache_path("fixtures/lineups", {"fixture": fid}).exists():
            already_cached += 1
    to_fetch = len(fixture_ids) - already_cached
    print(f"Already cached: {already_cached:,} | To fetch: {to_fetch:,}", flush=True)

    if to_fetch == 0:
        print("Nothing to do.")
        return

    start = time.time()
    fetched = 0
    empty = 0
    errors = 0
    for i, (lid, year, fid) in enumerate(fixture_ids, 1):
        cache = af._cache_path("fixtures/lineups", {"fixture": fid})
        is_cached = cache.exists()
        try:
            lineups = af.get_lineups(fid)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"  ERROR fixture={fid} league={lid} season={year}: {exc}", flush=True)
            continue
        if not is_cached:
            fetched += 1
        if not lineups:
            empty += 1
        if fetched and fetched % 100 == 0 and not is_cached:
            rl = af.last_rate_limit()
            elapsed = time.time() - start
            rate = fetched / elapsed if elapsed else 0
            eta_s = (to_fetch - fetched) / rate if rate else 0
            print(
                f"  [{i:,}/{len(fixture_ids):,}] fetched={fetched:,} "
                f"empty={empty:,} errors={errors:,} | "
                f"{rate:.1f} req/s | ETA {eta_s/60:.1f} min | "
                f"daily remaining={rl.get('x-ratelimit-requests-remaining','?')}",
                flush=True,
            )

    elapsed = time.time() - start
    print()
    print(f"Done in {elapsed/60:.1f} min", flush=True)
    print(f"  fetched this run: {fetched:,}", flush=True)
    print(f"  empty responses : {empty:,}  (no lineup published)", flush=True)
    print(f"  errors          : {errors:,}", flush=True)
    print(f"Cache size: {sum(p.stat().st_size for p in cache_dir.glob('*.json.gz'))/1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
