"""Per-team form features: days since last match + rolling W/D/L + goal diff.

For each match, looks BACKWARD through that team's history. No future leakage.
Works on the entire match history so features exist for all training rows.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def add_form_features(
    matches: pd.DataFrame,
    rolling_window: int = 10,
    days_cap: int = 365,
) -> pd.DataFrame:
    """Add days-of-rest and rolling form features per team to a matches frame.

    `matches` needs columns: date, home_team, away_team, home_score, away_score.

    Adds (per side):
      home_days_rest, away_days_rest          — days since previous match (capped)
      home_form_wins_l10, away_form_wins_l10  — wins in last `rolling_window` matches
      home_form_draws_l10, away_form_draws_l10
      home_form_gd_l10, away_form_gd_l10      — sum of goal differences (signed)
      home_form_n_l10, away_form_n_l10        — actual sample size (capped by available history)
    """
    df = matches.copy().reset_index().rename(columns={"index": "_orig_index"})

    # Build long format: one row per team-match
    home_rows = df[["_orig_index", "date", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "gf", "away_score": "ga"}
    ).assign(side="home")
    away_rows = df[["_orig_index", "date", "away_team", "home_score", "away_score"]].rename(
        columns={"away_team": "team", "away_score": "gf", "home_score": "ga"}
    ).assign(side="away")
    long = pd.concat([home_rows, away_rows], ignore_index=True)
    long = long.sort_values(["team", "date", "_orig_index"]).reset_index(drop=True)
    long["win"]  = (long["gf"] > long["ga"]).astype(int)
    long["draw"] = (long["gf"] == long["ga"]).astype(int)
    long["gd"]   = long["gf"] - long["ga"]

    g = long.groupby("team", sort=False, group_keys=False)
    long["prev_date"] = g["date"].shift(1)
    long["days_rest"] = (long["date"] - long["prev_date"]).dt.days.fillna(days_cap).clip(upper=days_cap)

    # Rolling over PRIOR matches: shift(1) excludes the current match.
    def _roll(s, fn):
        return s.shift(1).rolling(window=rolling_window, min_periods=1).apply(fn, raw=True)

    long["form_wins_l10"]  = g["win"].apply(lambda s:  s.shift(1).rolling(window=rolling_window, min_periods=1).sum()).reset_index(level=0, drop=True)
    long["form_draws_l10"] = g["draw"].apply(lambda s: s.shift(1).rolling(window=rolling_window, min_periods=1).sum()).reset_index(level=0, drop=True)
    long["form_gd_l10"]    = g["gd"].apply(lambda s:   s.shift(1).rolling(window=rolling_window, min_periods=1).sum()).reset_index(level=0, drop=True)
    long["form_n_l10"]     = g["gd"].apply(lambda s:   s.shift(1).rolling(window=rolling_window, min_periods=1).count()).reset_index(level=0, drop=True)
    long[["form_wins_l10","form_draws_l10","form_gd_l10","form_n_l10"]] = long[[
        "form_wins_l10","form_draws_l10","form_gd_l10","form_n_l10"
    ]].fillna(0)

    # Pivot back: one row per original match with home_/away_ feature pairs
    keep_cols = ["_orig_index", "side", "days_rest",
                 "form_wins_l10", "form_draws_l10", "form_gd_l10", "form_n_l10"]
    feat_long = long[keep_cols]
    home_feat = feat_long[feat_long["side"] == "home"].drop(columns="side").set_index("_orig_index")
    away_feat = feat_long[feat_long["side"] == "away"].drop(columns="side").set_index("_orig_index")
    home_feat = home_feat.add_prefix("home_")
    away_feat = away_feat.add_prefix("away_")

    out = df.set_index("_orig_index").join([home_feat, away_feat]).reset_index(drop=True)
    return out
