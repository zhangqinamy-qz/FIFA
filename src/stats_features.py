"""Per-team rolling in-match stats from api-football.

Builds a per-team-per-fixture long table of stats (shots-on-target for/against,
possession, pass accuracy, etc.), then computes prior-match rolling averages
per team. Stats coverage in api-football starts 2018, so features are NaN
before that.

Joined to training matches via `data/processed/training_to_api_fixture.csv`,
which maps (date, home_team, away_team) -> api fixture_id.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_STATS_PATH = Path("data/processed/api_lineups_stats.csv")
DEFAULT_MATCHES_PATH = Path("data/processed/api_lineups_matches.csv")
DEFAULT_JOIN_PATH = Path("data/processed/training_to_api_fixture.csv")


def _build_team_stats_long(
    stats_path: Path = DEFAULT_STATS_PATH,
    matches_path: Path = DEFAULT_MATCHES_PATH,
) -> pd.DataFrame:
    """Return a long table (one row per team-fixture) with per-match for/against stats.

    Columns: fixture_id, team_id, team_name, date (Timestamp),
             sot_for, total_shots_for, possession_for, pass_pct_for,
             sot_against, total_shots_against
    """
    stats = pd.read_csv(stats_path)
    matches = pd.read_csv(matches_path)
    matches["date"] = pd.to_datetime(matches["date_utc"], utc=True).dt.tz_localize(None)

    base = matches[["fixture_id", "team_id", "team_name", "is_home", "date"]].copy()
    s = stats[[
        "fixture_id", "team_id", "shots_on_goal", "total_shots",
        "ball_possession", "passes_pct",
    ]].rename(columns={
        "shots_on_goal": "sot_for",
        "total_shots": "total_shots_for",
        "ball_possession": "possession_for",
        "passes_pct": "pass_pct_for",
    })

    long = base.merge(s, on=["fixture_id", "team_id"], how="left")

    # Opponent stats via self-join on fixture_id (the other team's "for" = our "against")
    opp = stats[["fixture_id", "team_id", "shots_on_goal", "total_shots"]].rename(
        columns={
            "team_id": "opp_team_id",
            "shots_on_goal": "sot_against",
            "total_shots": "total_shots_against",
        }
    )
    long = long.merge(opp, on="fixture_id", how="left")
    # Drop the self row (where opp == self) — keep only the actual opponent
    long = long[long["team_id"] != long["opp_team_id"]].drop(columns="opp_team_id")

    # If a fixture had stats only for one side, the merge may produce duplicates; dedup
    long = long.sort_values(["team_id", "date", "fixture_id"]).drop_duplicates(
        subset=["fixture_id", "team_id"], keep="first"
    ).reset_index(drop=True)
    return long


def per_team_rolling_stats(
    rolling_window: int = 10,
    min_periods: int = 3,
    stats_path: Path = DEFAULT_STATS_PATH,
    matches_path: Path = DEFAULT_MATCHES_PATH,
) -> pd.DataFrame:
    """Return per-fixture, per-team pre-match rolling averages of in-match stats.

    Output one row per (fixture_id, team_id) with columns:
      team_id, fixture_id, date,
      sot_for_rN, sot_against_rN, total_shots_for_rN, total_shots_against_rN,
      possession_for_rN, pass_pct_for_rN, n_rN
    where N = rolling_window.

    Rolling uses .shift(1) so the current match is excluded from its own history.
    """
    long = _build_team_stats_long(stats_path, matches_path)
    g = long.groupby("team_id", sort=False, group_keys=False)

    def _roll_mean(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(window=rolling_window, min_periods=min_periods).mean()

    def _roll_count(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(window=rolling_window, min_periods=1).count()

    cols_for = ["sot_for", "total_shots_for", "possession_for", "pass_pct_for"]
    cols_against = ["sot_against", "total_shots_against"]
    suffix = f"_r{rolling_window}"

    for c in cols_for + cols_against:
        long[c + suffix] = g[c].apply(_roll_mean).reset_index(level=0, drop=True)
    long["n" + suffix] = g["sot_for"].apply(_roll_count).reset_index(level=0, drop=True)

    keep = ["team_id", "team_name", "fixture_id", "date"] + [c + suffix for c in cols_for + cols_against] + ["n" + suffix]
    return long[keep].copy()


def attach_stats_features_to_training(
    training: pd.DataFrame,
    rolling_window: int = 10,
    min_periods: int = 3,
    stats_path: Path = DEFAULT_STATS_PATH,
    matches_path: Path = DEFAULT_MATCHES_PATH,
    join_path: Path = DEFAULT_JOIN_PATH,
) -> pd.DataFrame:
    """Attach home_/away_ rolling-stat features to a training-matches frame.

    `training` needs columns: date, home_team, away_team.
    Joins via training_to_api_fixture.csv -> api fixture_id -> api lineups
    (which has home_team_id/away_team_id) -> rolling per-team stats.

    Adds for each side N:
      home_sot_for_rN, home_sot_against_rN, home_total_shots_for_rN, ...
      away_sot_for_rN, away_sot_against_rN, away_total_shots_for_rN, ...
    Plus diff features:
      sot_for_diff_rN, sot_against_diff_rN, sot_net_diff_rN

    Rows with no fixture_id, or where rolling history is too thin, get NaN.
    """
    suffix = f"_r{rolling_window}"
    rolling = per_team_rolling_stats(
        rolling_window=rolling_window,
        min_periods=min_periods,
        stats_path=stats_path,
        matches_path=matches_path,
    )

    matches = pd.read_csv(matches_path)
    matches["date"] = pd.to_datetime(matches["date_utc"], utc=True).dt.tz_localize(None)
    # One row per fixture with home/away team ids
    fix_home = (
        matches[matches["is_home"]][["fixture_id", "team_id"]]
        .rename(columns={"team_id": "home_team_id"})
    )
    fix_away = (
        matches[~matches["is_home"]][["fixture_id", "team_id"]]
        .rename(columns={"team_id": "away_team_id"})
    )
    fixtures = fix_home.merge(fix_away, on="fixture_id", how="inner")

    join_map = pd.read_csv(join_path)
    join_map = join_map[["date", "home_team", "away_team", "fixture_id"]].copy()
    join_map["date"] = pd.to_datetime(join_map["date"])
    join_map = join_map.dropna(subset=["fixture_id"])
    join_map["fixture_id"] = join_map["fixture_id"].astype(int)
    join_map = join_map.merge(fixtures, on="fixture_id", how="left")

    # Rolling stats per (fixture_id, team_id)
    feat_cols = [c for c in rolling.columns if c.endswith(suffix)]
    home_feats = rolling.rename(columns={"team_id": "home_team_id"})[
        ["fixture_id", "home_team_id"] + feat_cols
    ]
    home_feats = home_feats.rename(columns={c: f"home_{c}" for c in feat_cols})
    away_feats = rolling.rename(columns={"team_id": "away_team_id"})[
        ["fixture_id", "away_team_id"] + feat_cols
    ]
    away_feats = away_feats.rename(columns={c: f"away_{c}" for c in feat_cols})

    enriched = join_map.merge(home_feats, on=["fixture_id", "home_team_id"], how="left")
    enriched = enriched.merge(away_feats, on=["fixture_id", "away_team_id"], how="left")

    # Diff features (home - away) on key signals
    for c in feat_cols:
        enriched[f"diff_{c}"] = enriched[f"home_{c}"] - enriched[f"away_{c}"]
    # Net SoT diff = (sot_for - sot_against) home minus same on away
    sot_for = f"sot_for{suffix}"
    sot_against = f"sot_against{suffix}"
    enriched[f"sot_net_diff{suffix}"] = (
        (enriched[f"home_{sot_for}"] - enriched[f"home_{sot_against}"])
        - (enriched[f"away_{sot_for}"] - enriched[f"away_{sot_against}"])
    )

    out = training.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.merge(
        enriched.drop(columns=["fixture_id", "home_team_id", "away_team_id"]),
        on=["date", "home_team", "away_team"],
        how="left",
    )
    return out


if __name__ == "__main__":
    rolling = per_team_rolling_stats(rolling_window=10)
    print(f"rolling rows: {len(rolling):,}")
    print(rolling.head(8).to_string())
