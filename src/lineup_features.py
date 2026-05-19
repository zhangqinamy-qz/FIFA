"""Real-lineup age + top-value-share features from api-football starting XIs.

For each api fixture, take the starting XI, join to Transfermarkt via the
api_to_tm_players mapping (has DOB), look up each player's TM market value at
fixture date, then aggregate per (fixture_id, team_id):
  - outfield_age_lineup: mean age of non-GK starters
  - top1_share_lineup:   max value / total value among matched starters

This refactors the eligibility-pool approximation (top-20 outfielders from a
senior team's career pool) into a real-roster feature using only players who
actually played that day. The Fjelstul refactor (WC-only) gave a marginal
improvement; this dense-data version tests whether 16x more rows extracts more
signal.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_LINEUPS_PATH = Path("data/processed/api_lineups_players.csv")
DEFAULT_MATCHES_PATH = Path("data/processed/api_lineups_matches.csv")
DEFAULT_JOIN_TM_PATH = Path("data/processed/api_to_tm_players.csv")
DEFAULT_TM_MV_PATH = Path("data/raw/transfermarkt/player_market_value.csv")
DEFAULT_TM_PROFILES_PATH = Path("data/raw/transfermarkt/player_profiles.csv")
DEFAULT_TRAINING_JOIN_PATH = Path("data/processed/training_to_api_fixture.csv")


def per_team_lineup_features(
    lineups_path: Path = DEFAULT_LINEUPS_PATH,
    matches_path: Path = DEFAULT_MATCHES_PATH,
    join_path: Path = DEFAULT_JOIN_TM_PATH,
    mv_path: Path = DEFAULT_TM_MV_PATH,
    profiles_path: Path = DEFAULT_TM_PROFILES_PATH,
) -> pd.DataFrame:
    """Per (fixture_id, team_id) aggregates from real starting XI.

    Returns columns: fixture_id, team_id, n_matched,
                     outfield_age_lineup, gk_age_lineup, top1_share_lineup

    GK age is kept separate from outfielder age — GKs peak later (~30+) and
    blending them just adds noise to the outfielder signal.
    """
    # Step 1: starters with api player_id
    lp = pd.read_csv(lineups_path)
    lp = lp[lp["is_starter"] == True].dropna(subset=["player_id"]).copy()
    lp["api_player_id"] = lp["player_id"].astype(int)
    lp["api_pos"] = lp["pos"]

    # Step 2: join to TM via api_to_tm_players (drops unmatched)
    j = pd.read_csv(join_path)
    j = j[j["match_quality"] != "unmatched"].dropna(subset=["tm_player_id"]).copy()
    j["api_player_id"] = j["api_player_id"].astype(int)
    j["tm_player_id"] = j["tm_player_id"].astype(int)
    j["dob"] = pd.to_datetime(j["tm_date_of_birth"], errors="coerce")
    lp = lp.merge(j[["api_player_id", "tm_player_id", "dob"]], on="api_player_id", how="inner")

    # Step 3: fixture date
    m = pd.read_csv(matches_path, usecols=["fixture_id", "date_utc"]).drop_duplicates()
    m["date"] = pd.to_datetime(m["date_utc"], utc=True).dt.tz_localize(None)
    lp = lp.merge(m[["fixture_id", "date"]], on="fixture_id", how="left").dropna(subset=["date"])

    # Step 4: TM market value at/before fixture date (merge_asof)
    mv = pd.read_csv(mv_path, usecols=["player_id", "date_unix", "value"]).rename(columns={"date_unix": "date"})
    mv["date"] = pd.to_datetime(mv["date"], errors="coerce")
    mv = mv.dropna(subset=["date", "value"]).rename(columns={"player_id": "tm_player_id"})
    mv["tm_player_id"] = mv["tm_player_id"].astype(int)
    mv = mv.sort_values("date").reset_index(drop=True)
    lp_sorted = lp.sort_values("date").reset_index(drop=True)
    lp_sorted = pd.merge_asof(
        lp_sorted, mv, on="date", by="tm_player_id",
        direction="backward", allow_exact_matches=True,
    )
    # If a player has no prior mv, leave value=NaN (won't count toward top1_share)

    # Step 5: GK flag — use api `pos` first, fall back to TM profile position
    prof = pd.read_csv(profiles_path, usecols=["player_id", "position"], low_memory=False)
    prof = prof.rename(columns={"player_id": "tm_player_id"})
    prof["tm_player_id"] = prof["tm_player_id"].astype(int)
    prof["is_gk_tm"] = prof["position"].str.contains("Goal", case=False, na=False)
    lp_sorted = lp_sorted.merge(prof[["tm_player_id", "is_gk_tm"]], on="tm_player_id", how="left")
    lp_sorted["is_gk_tm"] = lp_sorted["is_gk_tm"].fillna(False)

    def is_gk(row):
        if pd.notna(row["api_pos"]) and row["api_pos"] in ("G",):
            return True
        return bool(row["is_gk_tm"])
    lp_sorted["is_gk"] = lp_sorted.apply(is_gk, axis=1)

    # Step 6: per-player age at fixture date
    lp_sorted["age"] = (lp_sorted["date"] - lp_sorted["dob"]).dt.days / 365.25

    # Step 7: aggregate per (fixture_id, team_id) — keep GK and outfield ages separate
    rows = []
    for (fid, tid), g in lp_sorted.groupby(["fixture_id", "team_id"], sort=False):
        outfield = g[~g["is_gk"]]
        gks = g[g["is_gk"]]
        n_outfield = len(outfield)
        n_gk = len(gks)
        mean_outfield_age = float(outfield["age"].mean()) if n_outfield > 0 else np.nan
        mean_gk_age = float(gks["age"].mean()) if n_gk > 0 else np.nan
        total_val = float(g["value"].fillna(0).sum())
        top1 = float(g["value"].fillna(0).max())
        top1_share = (top1 / total_val) if total_val > 0 else np.nan
        rows.append({
            "fixture_id": fid, "team_id": tid,
            "n_matched": len(g), "n_outfield": n_outfield, "n_gk": n_gk,
            "outfield_age_lineup": mean_outfield_age,
            "gk_age_lineup": mean_gk_age,
            "top1_share_lineup": top1_share,
        })
    return pd.DataFrame(rows)


def attach_lineup_features_to_training(
    training: pd.DataFrame,
    lineups_path: Path = DEFAULT_LINEUPS_PATH,
    matches_path: Path = DEFAULT_MATCHES_PATH,
    join_path: Path = DEFAULT_JOIN_TM_PATH,
    mv_path: Path = DEFAULT_TM_MV_PATH,
    profiles_path: Path = DEFAULT_TM_PROFILES_PATH,
    training_join_path: Path = DEFAULT_TRAINING_JOIN_PATH,
    min_matched: int = 6,
) -> pd.DataFrame:
    """Adds home_/away_ lineup-based age + top1_share + diff features."""
    feats = per_team_lineup_features(lineups_path, matches_path, join_path, mv_path, profiles_path)
    feats = feats[feats["n_matched"] >= min_matched]

    matches = pd.read_csv(matches_path)
    fix_home = matches[matches["is_home"]][["fixture_id", "team_id"]].rename(columns={"team_id": "home_team_id"})
    fix_away = matches[~matches["is_home"]][["fixture_id", "team_id"]].rename(columns={"team_id": "away_team_id"})
    fixtures = fix_home.merge(fix_away, on="fixture_id", how="inner")

    join_map = pd.read_csv(training_join_path)
    join_map["date"] = pd.to_datetime(join_map["date"])
    join_map = join_map.dropna(subset=["fixture_id"]).copy()
    join_map["fixture_id"] = join_map["fixture_id"].astype(int)
    join_map = join_map[["date", "home_team", "away_team", "fixture_id"]].merge(fixtures, on="fixture_id", how="left")

    cols = ["outfield_age_lineup", "gk_age_lineup", "top1_share_lineup", "n_matched"]
    home = feats.rename(columns={"team_id": "home_team_id"})[
        ["fixture_id", "home_team_id"] + cols
    ].rename(columns={c: f"home_{c}" for c in cols})
    away = feats.rename(columns={"team_id": "away_team_id"})[
        ["fixture_id", "away_team_id"] + cols
    ].rename(columns={c: f"away_{c}" for c in cols})

    enriched = join_map.merge(home, on=["fixture_id", "home_team_id"], how="left")
    enriched = enriched.merge(away, on=["fixture_id", "away_team_id"], how="left")
    enriched["outfield_age_lineup_diff"] = enriched["home_outfield_age_lineup"] - enriched["away_outfield_age_lineup"]
    enriched["gk_age_lineup_diff"]       = enriched["home_gk_age_lineup"]       - enriched["away_gk_age_lineup"]
    enriched["top1_share_lineup_diff"]   = enriched["home_top1_share_lineup"]   - enriched["away_top1_share_lineup"]

    out = training.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.merge(
        enriched.drop(columns=["fixture_id", "home_team_id", "away_team_id"]),
        on=["date", "home_team", "away_team"], how="left",
    )
    return out


if __name__ == "__main__":
    feats = per_team_lineup_features()
    print(f"per-team rows: {len(feats):,}")
    print(feats.describe().round(2).to_string())
