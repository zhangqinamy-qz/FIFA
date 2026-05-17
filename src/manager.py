"""Fjelstul manager features for WC matches.

Computes prior-WC experience and win rate for each team's primary manager at
a given WC. Joins to our match data via team_name + tournament_id.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from src.fjelstul import DEFAULT_DATA_DIR, TEAM_NAME_FIXES, BASE_URL

MANAGER_FILE_URL = f"{BASE_URL}/manager_appearances.csv"


def _ensure_local(filename: str, data_dir: Path | str = DEFAULT_DATA_DIR) -> Path:
    p = Path(data_dir) / filename
    if not p.exists():
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MANAGER_FILE_URL, p)
    return p


def load_manager_appearances(data_dir: Path | str = DEFAULT_DATA_DIR,
                              mens_only: bool = True) -> pd.DataFrame:
    p = _ensure_local("manager_appearances.csv", data_dir)
    df = pd.read_csv(p, parse_dates=["match_date"])
    if mens_only:
        df = df[df["tournament_name"].str.contains("Men", na=False)]
    return df


def load_matches(data_dir: Path | str = DEFAULT_DATA_DIR,
                  mens_only: bool = True) -> pd.DataFrame:
    p = Path(data_dir) / "matches.csv"
    df = pd.read_csv(p, parse_dates=["match_date"])
    if mens_only:
        df = df[df["tournament_name"].str.contains("Men", na=False)]
    return df


def primary_manager_per_team(ma: pd.DataFrame) -> pd.DataFrame:
    """For each (team_name, tournament_id), pick the manager who coached the
    most matches (ties broken by the earliest match_date)."""
    grouped = (
        ma.groupby(["team_name", "tournament_id", "manager_id", "family_name", "given_name"])
          .agg(n_matches=("match_id", "nunique"),
               first_date=("match_date", "min"))
          .reset_index()
    )
    grouped = grouped.sort_values(
        ["team_name", "tournament_id", "n_matches", "first_date"],
        ascending=[True, True, False, True],
    )
    return grouped.drop_duplicates(["team_name", "tournament_id"], keep="first")


def manager_career_stats(
    ma: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """Build a frame of (manager_id, tournament_id) -> prior-WC stats.

    For each manager + tournament, computes their prior-WC totals (matches
    coached and win rate) using only WC matches that ended *before* this
    tournament's start date — no look-ahead.
    """
    # Match result per (team, match_id)
    res = matches[["match_id", "tournament_id", "match_date",
                   "home_team_name", "away_team_name",
                   "home_team_win", "away_team_win", "draw"]].copy()
    home = res.rename(columns={"home_team_name": "team_name"})[
        ["match_id", "tournament_id", "match_date", "team_name", "home_team_win", "draw"]
    ].assign(win=lambda d: d["home_team_win"]).drop(columns=["home_team_win"])
    away = res.rename(columns={"away_team_name": "team_name"})[
        ["match_id", "tournament_id", "match_date", "team_name", "away_team_win", "draw"]
    ].assign(win=lambda d: d["away_team_win"]).drop(columns=["away_team_win"])
    long = pd.concat([home, away], ignore_index=True)

    # Join: manager who managed team at match
    mgr_at_match = ma[["match_id", "team_name", "manager_id"]].drop_duplicates()
    mgr_results = long.merge(mgr_at_match, on=["match_id", "team_name"], how="inner")

    # For each manager × tournament, compute *prior* stats (cumulative over
    # tournaments earlier than this one).
    tourn_dates = matches.groupby("tournament_id")["match_date"].min().rename("tourn_start")
    mgr_results = mgr_results.merge(tourn_dates, left_on="tournament_id", right_index=True)
    mgr_results = mgr_results.sort_values(["manager_id", "match_date"])

    rows = []
    # All tournaments each manager has touched, sorted
    for manager_id, sub in mgr_results.groupby("manager_id"):
        seen = sub.drop_duplicates(["tournament_id"])[["tournament_id", "tourn_start"]].sort_values("tourn_start")
        for _, t in seen.iterrows():
            prior = sub[sub["match_date"] < t["tourn_start"]]
            rows.append({
                "manager_id": manager_id,
                "tournament_id": t["tournament_id"],
                "prior_wc_matches": int(len(prior)),
                "prior_wc_wins": int(prior["win"].sum()),
                "prior_wc_draws": int(prior["draw"].sum()),
            })
    out = pd.DataFrame(rows)
    out["prior_wc_win_rate"] = out.apply(
        lambda r: (r["prior_wc_wins"] / r["prior_wc_matches"]) if r["prior_wc_matches"] else 0.0,
        axis=1,
    )
    return out


def manager_features_for_matches(
    matches_in: pd.DataFrame,
    *,
    ma: pd.DataFrame,
    wc_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Return home_/away_ manager features aligned to `matches_in` index.

    `matches_in` is our match frame (must have date, home_team, away_team,
    tournament). Only WC rows are populated; others get NaN.

    Features per side:
      mgr_prior_wc_matches  — prior-WC matches coached (capped at 30)
      mgr_prior_wc_win_rate — win rate over prior WC matches (0 if none)
    """
    primary = primary_manager_per_team(ma)
    stats = manager_career_stats(ma, wc_matches)
    primary = primary.merge(stats, on=["manager_id", "tournament_id"], how="left")

    # Reverse name map: our match team_name → Fjelstul team_name
    reverse_fixes = {v: k for k, v in TEAM_NAME_FIXES.items() if v != k}
    reverse_fixes.setdefault("South Korea", "Korea Republic")

    out = pd.DataFrame(index=matches_in.index, dtype=float, columns=[
        "home_mgr_prior_wc_matches", "home_mgr_prior_wc_win_rate",
        "away_mgr_prior_wc_matches", "away_mgr_prior_wc_win_rate",
    ])

    by_key = primary.set_index(["team_name", "tournament_id"])
    for idx, row in matches_in.iterrows():
        if row.get("tournament") != "FIFA World Cup":
            continue
        year = pd.Timestamp(row["date"]).year
        tid = f"WC-{year}"
        for side in ("home", "away"):
            team_raw = row[f"{side}_team"]
            team_fj = reverse_fixes.get(team_raw, team_raw)
            try:
                info = by_key.loc[(team_fj, tid)]
            except KeyError:
                continue
            out.at[idx, f"{side}_mgr_prior_wc_matches"] = min(float(info["prior_wc_matches"]), 30.0)
            out.at[idx, f"{side}_mgr_prior_wc_win_rate"] = float(info["prior_wc_win_rate"])
    return out
