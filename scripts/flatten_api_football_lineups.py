"""Flatten cached /fixtures and /fixtures/lineups responses into two CSVs.

Outputs (in data/processed/):
  api_lineups_matches.csv   one row per (fixture, team) -> 2 rows per fixture
                            cols: fixture_id, league_id, season, date_utc, venue,
                                  home_team_id, home_team_name, away_team_id,
                                  away_team_name, team_id, team_name, is_home,
                                  formation, coach_id, coach_name,
                                  home_goals, away_goals
  api_lineups_players.csv   one row per (fixture, team, player)
                            cols: fixture_id, team_id, player_id, player_name,
                                  number, pos, grid, is_starter

Re-reads from the gzipped JSON cache, no API requests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

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

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def main() -> None:
    matches_rows: list[dict] = []
    players_rows: list[dict] = []
    n_fix_total = 0
    n_fix_with_lineup = 0

    for lid in LEAGUES:
        body = af.request("leagues", {"id": lid})
        seasons = [s["year"] for s in body["response"][0]["seasons"]]
        for year in seasons:
            fixtures = af.get_fixtures(league=lid, season=year)
            for f in fixtures:
                n_fix_total += 1
                fid = f["fixture"]["id"]
                date_utc = f["fixture"]["date"]
                venue = (f["fixture"].get("venue") or {}).get("name")
                home = f["teams"]["home"]
                away = f["teams"]["away"]
                goals = f.get("goals") or {}
                home_goals = goals.get("home")
                away_goals = goals.get("away")

                lineups = af.get_lineups(fid)
                if not lineups:
                    continue
                n_fix_with_lineup += 1

                for lu in lineups:
                    team_id = lu["team"]["id"]
                    team_name = lu["team"]["name"]
                    is_home = team_id == home["id"]
                    coach = lu.get("coach") or {}
                    matches_rows.append({
                        "fixture_id":      fid,
                        "league_id":       lid,
                        "season":          year,
                        "date_utc":        date_utc,
                        "venue":           venue,
                        "home_team_id":    home["id"],
                        "home_team_name":  home["name"],
                        "away_team_id":    away["id"],
                        "away_team_name":  away["name"],
                        "team_id":         team_id,
                        "team_name":       team_name,
                        "is_home":         is_home,
                        "formation":       lu.get("formation"),
                        "coach_id":        coach.get("id"),
                        "coach_name":      coach.get("name"),
                        "home_goals":      home_goals,
                        "away_goals":      away_goals,
                    })

                    for slot, starter in [("startXI", True), ("substitutes", False)]:
                        for entry in lu.get(slot) or []:
                            p = entry.get("player") or {}
                            players_rows.append({
                                "fixture_id":   fid,
                                "team_id":      team_id,
                                "player_id":    p.get("id"),
                                "player_name":  p.get("name"),
                                "number":       p.get("number"),
                                "pos":          p.get("pos"),
                                "grid":         p.get("grid"),
                                "is_starter":   starter,
                            })

    matches = pd.DataFrame(matches_rows)
    players = pd.DataFrame(players_rows)

    matches_path = OUT_DIR / "api_lineups_matches.csv"
    players_path = OUT_DIR / "api_lineups_players.csv"
    matches.to_csv(matches_path, index=False)
    players.to_csv(players_path, index=False)

    print(f"Fixtures total: {n_fix_total:,}")
    print(f"Fixtures with lineup: {n_fix_with_lineup:,} "
          f"({n_fix_with_lineup / n_fix_total:.1%})")
    print()
    print(f"matches rows: {len(matches):,}  -> {matches_path} "
          f"({matches_path.stat().st_size / 1e6:.2f} MB)")
    print(f"players rows: {len(players):,}  -> {players_path} "
          f"({players_path.stat().st_size / 1e6:.2f} MB)")
    print()
    print("Players per league:")
    print(matches.groupby("league_id").size().rename("team_lineups").to_string())
    print()
    print("Coverage by season:")
    coverage = matches.assign(year=matches.date_utc.str.slice(0, 4)).groupby("year").size()
    print(coverage.to_string())
    print()
    print("Unique counts:")
    print(f"  unique players : {players.player_id.nunique():,}")
    print(f"  unique coaches : {matches.coach_id.nunique():,}")
    print(f"  unique teams   : {matches.team_id.nunique():,}")


if __name__ == "__main__":
    main()
