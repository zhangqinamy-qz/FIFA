"""Manager features from api-football coach-per-fixture data.

Uses `data/processed/api_lineups_matches.csv` which has coach_id + team_id +
date for every covered international fixture. Coverage 2010, 2016+. Gives 16x
more rows than the prior Fjelstul WC-only approach.

Three pre-match features per side, computed with .shift(1) (no leakage):
  mgr_career_matches  — int. fixtures this coach has managed across all teams
  mgr_career_wr       — career win rate (excluding draws? no: wins / total)
  mgr_tenure_days     — days since this coach's first match with this team
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_MATCHES_PATH = Path("data/processed/api_lineups_matches.csv")
DEFAULT_JOIN_PATH = Path("data/processed/training_to_api_fixture.csv")


def _per_team_match_results(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team-fixture with (team_id, coach_id, date, win, draw)."""
    m = matches.copy()
    m["date"] = pd.to_datetime(m["date_utc"], utc=True).dt.tz_localize(None)
    m = m.dropna(subset=["coach_id"])
    m["coach_id"] = m["coach_id"].astype(int)
    # Match outcome from this team's perspective
    m["gf"] = np.where(m["is_home"], m["home_goals"], m["away_goals"])
    m["ga"] = np.where(m["is_home"], m["away_goals"], m["home_goals"])
    m = m.dropna(subset=["gf", "ga"])
    m["win"] = (m["gf"] > m["ga"]).astype(int)
    m["draw"] = (m["gf"] == m["ga"]).astype(int)
    return m[["fixture_id", "team_id", "team_name", "coach_id", "date", "win", "draw"]].copy()


def per_team_manager_features(
    matches_path: Path = DEFAULT_MATCHES_PATH,
) -> pd.DataFrame:
    """Pre-match manager features per (fixture_id, team_id).

    Columns: fixture_id, team_id, coach_id, date,
             mgr_career_matches, mgr_career_wr, mgr_tenure_days
    """
    matches = pd.read_csv(matches_path)
    long = _per_team_match_results(matches)
    long = long.sort_values(["coach_id", "date", "fixture_id"]).reset_index(drop=True)

    # Career cumulative across ALL teams the coach has managed: shift(1) to exclude current
    g_coach = long.groupby("coach_id", sort=False, group_keys=False)
    long["mgr_career_matches"] = g_coach.cumcount()  # already excludes current row
    long["cum_wins"] = g_coach["win"].cumsum() - long["win"]
    long["mgr_career_wr"] = np.where(
        long["mgr_career_matches"] > 0,
        long["cum_wins"] / long["mgr_career_matches"].clip(lower=1),
        0.0,
    )

    # Tenure with current team: days since first appearance with this (coach_id, team_id)
    long = long.sort_values(["coach_id", "team_id", "date", "fixture_id"]).reset_index(drop=True)
    g_ct = long.groupby(["coach_id", "team_id"], sort=False, group_keys=False)
    long["first_match_with_team"] = g_ct["date"].transform("min")
    long["mgr_tenure_days"] = (long["date"] - long["first_match_with_team"]).dt.days

    return long[["fixture_id", "team_id", "coach_id", "date",
                 "mgr_career_matches", "mgr_career_wr", "mgr_tenure_days"]].copy()


def attach_manager_features_to_training(
    training: pd.DataFrame,
    matches_path: Path = DEFAULT_MATCHES_PATH,
    join_path: Path = DEFAULT_JOIN_PATH,
) -> pd.DataFrame:
    """Attach home_/away_ manager features to a training-matches frame.

    Requires columns: date, home_team, away_team.

    Adds:
      home_mgr_career_matches, away_mgr_career_matches
      home_mgr_career_wr,      away_mgr_career_wr
      home_mgr_tenure_days,    away_mgr_tenure_days
    Plus diff features (home - away) for the same.

    Rows with no fixture-id mapping, or no coach data, get NaN.
    """
    mgr = per_team_manager_features(matches_path)

    matches = pd.read_csv(matches_path)
    fix_home = matches[matches["is_home"]][["fixture_id", "team_id"]].rename(columns={"team_id": "home_team_id"})
    fix_away = matches[~matches["is_home"]][["fixture_id", "team_id"]].rename(columns={"team_id": "away_team_id"})
    fixtures = fix_home.merge(fix_away, on="fixture_id", how="inner")

    join_map = pd.read_csv(join_path)
    join_map["date"] = pd.to_datetime(join_map["date"])
    join_map = join_map.dropna(subset=["fixture_id"]).copy()
    join_map["fixture_id"] = join_map["fixture_id"].astype(int)
    join_map = join_map[["date", "home_team", "away_team", "fixture_id"]].merge(fixtures, on="fixture_id", how="left")

    feat_cols = ["mgr_career_matches", "mgr_career_wr", "mgr_tenure_days"]
    home_feats = mgr.rename(columns={"team_id": "home_team_id"})[["fixture_id", "home_team_id"] + feat_cols]
    home_feats = home_feats.rename(columns={c: f"home_{c}" for c in feat_cols})
    away_feats = mgr.rename(columns={"team_id": "away_team_id"})[["fixture_id", "away_team_id"] + feat_cols]
    away_feats = away_feats.rename(columns={c: f"away_{c}" for c in feat_cols})

    enriched = join_map.merge(home_feats, on=["fixture_id", "home_team_id"], how="left")
    enriched = enriched.merge(away_feats, on=["fixture_id", "away_team_id"], how="left")
    for c in feat_cols:
        enriched[f"diff_{c}"] = enriched[f"home_{c}"] - enriched[f"away_{c}"]

    out = training.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.merge(
        enriched.drop(columns=["fixture_id", "home_team_id", "away_team_id"]),
        on=["date", "home_team", "away_team"],
        how="left",
    )
    return out


if __name__ == "__main__":
    mgr = per_team_manager_features()
    print(f"rows: {len(mgr):,}")
    print(mgr.describe().round(2).to_string())
    print()
    # Show a familiar coach
    france = mgr.merge(
        pd.read_csv(DEFAULT_MATCHES_PATH)[["fixture_id", "team_name", "coach_id", "coach_name"]].drop_duplicates(),
        on=["fixture_id", "coach_id"], how="left",
    )
    france = france[france["team_name"] == "France"].sort_values("date").tail(5)
    print("France latest 5 fixtures (manager features):")
    cols = ["date", "coach_name", "mgr_career_matches", "mgr_career_wr", "mgr_tenure_days"]
    print(france[cols].to_string(index=False))
