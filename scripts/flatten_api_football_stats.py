"""Flatten cached /fixtures/statistics responses into a wide per-team-match CSV.

Output: data/processed/api_lineups_stats.csv
  one row per (fixture_id, team_id), columns:
    fixture_id, team_id,
    shots_on_goal, shots_off_goal, total_shots, blocked_shots,
    shots_insidebox, shots_outsidebox,
    fouls, corner_kicks, offsides,
    ball_possession (float, 0-1),
    yellow_cards, red_cards,
    goalkeeper_saves, total_passes, passes_accurate, passes_pct (float, 0-1),
    expected_goals (float, when present)

Rows for fixtures with no stat coverage are skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import api_football as af  # noqa: E402
from scripts.pull_api_football_lineups import collect_fixture_ids  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"

# api-football "type" -> our column name
FIELD_MAP = {
    "Shots on Goal":     "shots_on_goal",
    "Shots off Goal":    "shots_off_goal",
    "Total Shots":       "total_shots",
    "Blocked Shots":     "blocked_shots",
    "Shots insidebox":   "shots_insidebox",
    "Shots outsidebox":  "shots_outsidebox",
    "Fouls":             "fouls",
    "Corner Kicks":      "corner_kicks",
    "Offsides":          "offsides",
    "Ball Possession":   "ball_possession",
    "Yellow Cards":      "yellow_cards",
    "Red Cards":         "red_cards",
    "Goalkeeper Saves":  "goalkeeper_saves",
    "Total passes":      "total_passes",
    "Passes accurate":   "passes_accurate",
    "Passes %":          "passes_pct",
    "expected_goals":    "expected_goals",
}


def _coerce(field: str, raw):
    """Coerce raw api value to numeric (handle '54%' strings, None, etc.)"""
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        s = raw.strip()
        if s.endswith("%"):
            try:
                return float(s.rstrip("%")) / 100.0
            except ValueError:
                return None
        try:
            return float(s)
        except ValueError:
            return None
    return raw


def main() -> None:
    fixture_ids = collect_fixture_ids()
    print(f"In-scope fixtures: {len(fixture_ids):,}")

    rows: list[dict] = []
    n_with_stats = 0
    for _, _, fid in fixture_ids:
        cache = af._cache_path("fixtures/statistics", {"fixture": fid})
        if not cache.exists():
            continue
        body = af._cache_read(cache)
        resp = body.get("response") or []
        if not resp:
            continue
        n_with_stats += 1
        for team_stats in resp:
            row = {"fixture_id": fid, "team_id": team_stats["team"]["id"]}
            for s in team_stats.get("statistics") or []:
                col = FIELD_MAP.get(s.get("type"))
                if col:
                    row[col] = _coerce(col, s.get("value"))
            rows.append(row)

    df = pd.DataFrame(rows)
    out = DATA / "processed" / "api_lineups_stats.csv"
    df.to_csv(out, index=False)
    print(f"Fixtures with stats: {n_with_stats:,}")
    print(f"Rows written       : {len(df):,} -> {out.name} ({out.stat().st_size/1e6:.2f} MB)")
    print()
    print("Coverage per column:")
    print(df.notna().mean().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
