"""Build the api-football -> Transfermarkt player_id mapping and save.

Output: data/processed/api_to_tm_players.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import transfermarkt as tm  # noqa: E402
from src.api_football_join import build_join  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    print("Loading TM profiles, perfs, team-country map...")
    profiles = tm.load_profiles()
    perfs = tm.load_national_performances()
    team_country = tm.build_team_country_map(perfs, profiles, senior_only=True)
    print(f"  TM profiles : {len(profiles):,}")
    print(f"  TM perfs    : {len(perfs):,}")
    print(f"  Senior teams: {len(team_country):,}")
    print()

    print("Loading api-football lineups CSVs...")
    matches = pd.read_csv(DATA / "processed" / "api_lineups_matches.csv")
    players = pd.read_csv(DATA / "processed" / "api_lineups_players.csv")
    print(f"  api matches: {len(matches):,}")
    print(f"  api players: {len(players):,}")
    print(f"  unique api player_ids: {players.player_id.nunique():,}")
    print(f"  unique api teams      : {matches.team_id.nunique():,}")
    print()

    print("Building join...")
    join = build_join(matches, players, profiles, perfs, team_country)
    out = DATA / "processed" / "api_to_tm_players.csv"
    join.to_csv(out, index=False)

    print()
    print(f"=== Result ({len(join):,} rows, saved to {out.name}) ===")
    quality = join.match_quality.value_counts()
    print(quality.to_string())
    print()
    matched = join[join.tm_player_id.notna()]
    print(f"Matched: {len(matched):,}/{len(join):,} ({len(matched)/len(join):.1%})")
    print()
    print("=== Match rate per country (top 15 by player count) ===")
    per_country = join.groupby("country").agg(
        n_players=("api_player_id", "count"),
        matched=("tm_player_id", lambda s: s.notna().sum()),
    )
    per_country["match_rate"] = per_country["matched"] / per_country["n_players"]
    print(per_country.sort_values("n_players", ascending=False).head(15).to_string())
    print()
    print("=== Worst match rates (>= 50 players, bottom 10) ===")
    worst = per_country[per_country["n_players"] >= 50].sort_values("match_rate").head(10)
    print(worst.to_string())
    print()
    print("=== Unmatched countries (no TM team found) ===")
    unmatched_country = (join[join.tm_player_id.isna()]
                         .groupby("api_team_name").size().sort_values(ascending=False).head(20))
    print(unmatched_country.to_string())


if __name__ == "__main__":
    main()
