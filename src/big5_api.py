"""Big-5 league exposure on dense api-football lineups.

For each api fixture, take the lineup (starters), join to TM player_id via
`api_to_tm_players.csv`, look up each player's club at fixture date in
`transfer_history.csv`, mark Big-5 clubs, aggregate per (fixture_id, team_id).

Returns per-team big5 share — unweighted (count-based) plus value-weighted
(when TM market value at fixture date is available).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.big5 import BIG5_COMPETITION_IDS, load_team_details, load_transfer_history
from src.transfermarkt import load_market_value, DEFAULT_DATA_DIR


DEFAULT_LINEUPS_PATH = Path("data/processed/api_lineups_players.csv")
DEFAULT_MATCHES_PATH = Path("data/processed/api_lineups_matches.csv")
DEFAULT_JOIN_TM_PATH = Path("data/processed/api_to_tm_players.csv")
DEFAULT_TRAINING_JOIN_PATH = Path("data/processed/training_to_api_fixture.csv")


def build_club_to_big5_map() -> dict[int, str]:
    td = load_team_details(DEFAULT_DATA_DIR)
    td["is_big5"] = td["competition_id"].isin(BIG5_COMPETITION_IDS)
    td = td.sort_values(["club_id", "is_big5"], ascending=[True, False]).drop_duplicates("club_id", keep="first")
    return dict(zip(td.loc[td["is_big5"], "club_id"].astype(int), td.loc[td["is_big5"], "competition_id"]))


def lineup_player_clubs(
    lineups_path: Path = DEFAULT_LINEUPS_PATH,
    matches_path: Path = DEFAULT_MATCHES_PATH,
    join_path: Path = DEFAULT_JOIN_TM_PATH,
    starters_only: bool = True,
) -> pd.DataFrame:
    """One row per (fixture_id, team_id, tm_player_id, date) with club_id + is_big5."""
    lineups = pd.read_csv(lineups_path)
    if starters_only:
        lineups = lineups[lineups["is_starter"] == True]
    lineups = lineups.dropna(subset=["player_id"]).copy()
    lineups["api_player_id"] = lineups["player_id"].astype(int)

    j = pd.read_csv(join_path)
    j = j[j["match_quality"] != "unmatched"].dropna(subset=["tm_player_id"]).copy()
    j["api_player_id"] = j["api_player_id"].astype(int)
    j["tm_player_id"]  = j["tm_player_id"].astype(int)

    lineups = lineups.merge(
        j[["api_player_id", "tm_player_id"]], on="api_player_id", how="inner"
    )

    matches = pd.read_csv(matches_path, usecols=["fixture_id", "date_utc"]).drop_duplicates()
    matches["date"] = pd.to_datetime(matches["date_utc"], utc=True).dt.tz_localize(None)
    lineups = lineups.merge(matches[["fixture_id", "date"]], on="fixture_id", how="left")
    lineups = lineups.dropna(subset=["date"])

    # Club at date via merge_asof on transfer_history
    transfers = load_transfer_history(DEFAULT_DATA_DIR)
    transfers = transfers.dropna(subset=["player_id", "to_team_id"]).copy()
    transfers["player_id"]   = transfers["player_id"].astype(int)
    transfers["to_team_id"]  = transfers["to_team_id"].astype(int)
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"])

    # merge_asof requires both sides sorted by the `on` column globally and non-null
    transfers = transfers.dropna(subset=["transfer_date"]).sort_values("transfer_date").reset_index(drop=True)
    lineups   = lineups.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    out = pd.merge_asof(
        lineups,
        transfers.rename(columns={"player_id": "tm_player_id"})[
            ["tm_player_id", "transfer_date", "to_team_id"]
        ],
        left_on="date", right_on="transfer_date",
        by="tm_player_id", direction="backward", allow_exact_matches=True,
    )
    out = out.rename(columns={"to_team_id": "club_id"})

    # Fallback for players with no prior transfer: use their earliest-known from_team_id
    needs_fb = out[out["club_id"].isna()][["tm_player_id"]].drop_duplicates()
    if len(needs_fb):
        first = (
            transfers.sort_values(["player_id", "transfer_date"])
            .groupby("player_id", as_index=False).first()
            [["player_id", "from_team_id"]]
            .rename(columns={"player_id": "tm_player_id"})
        )
        fb = needs_fb.merge(first, on="tm_player_id", how="left")
        fb_map = dict(zip(fb["tm_player_id"], fb["from_team_id"]))
        out["club_id"] = out["club_id"].fillna(out["tm_player_id"].map(fb_map))

    big5 = build_club_to_big5_map()
    out["is_big5"] = out["club_id"].map(lambda c: bool(c) and int(c) in big5 if pd.notna(c) else False).astype(int)
    return out[["fixture_id", "team_id", "tm_player_id", "date", "club_id", "is_big5"]].copy()


def per_team_big5_features(
    lineups_path: Path = DEFAULT_LINEUPS_PATH,
    matches_path: Path = DEFAULT_MATCHES_PATH,
    join_path: Path = DEFAULT_JOIN_TM_PATH,
) -> pd.DataFrame:
    """Per (fixture_id, team_id): unweighted Big-5 share + matched count.

    Columns: fixture_id, team_id, big5_count, n_matched, big5_share
    """
    pc = lineup_player_clubs(lineups_path, matches_path, join_path)
    agg = pc.groupby(["fixture_id", "team_id"]).agg(
        n_matched=("tm_player_id", "size"),
        big5_count=("is_big5", "sum"),
    ).reset_index()
    agg["big5_share"] = np.where(agg["n_matched"] > 0, agg["big5_count"] / agg["n_matched"], 0.0)
    return agg


def attach_big5_features_to_training(
    training: pd.DataFrame,
    lineups_path: Path = DEFAULT_LINEUPS_PATH,
    matches_path: Path = DEFAULT_MATCHES_PATH,
    join_path: Path = DEFAULT_JOIN_TM_PATH,
    training_join_path: Path = DEFAULT_TRAINING_JOIN_PATH,
    min_matched: int = 5,
) -> pd.DataFrame:
    """Adds home_/away_big5_share + diff_big5_share to a training frame.

    `min_matched=5` drops sides where fewer than 5 starters were matched to TM
    (data-quality floor — otherwise share is noisy).
    """
    big5 = per_team_big5_features(lineups_path, matches_path, join_path)
    big5 = big5[big5["n_matched"] >= min_matched]

    matches = pd.read_csv(matches_path)
    fix_home = matches[matches["is_home"]][["fixture_id", "team_id"]].rename(columns={"team_id": "home_team_id"})
    fix_away = matches[~matches["is_home"]][["fixture_id", "team_id"]].rename(columns={"team_id": "away_team_id"})
    fixtures = fix_home.merge(fix_away, on="fixture_id", how="inner")

    join_map = pd.read_csv(training_join_path)
    join_map["date"] = pd.to_datetime(join_map["date"])
    join_map = join_map.dropna(subset=["fixture_id"]).copy()
    join_map["fixture_id"] = join_map["fixture_id"].astype(int)
    join_map = join_map[["date", "home_team", "away_team", "fixture_id"]].merge(
        fixtures, on="fixture_id", how="left"
    )

    feat_cols = ["big5_share", "n_matched", "big5_count"]
    home_feats = big5.rename(columns={"team_id": "home_team_id"})[
        ["fixture_id", "home_team_id"] + feat_cols
    ].rename(columns={c: f"home_{c}" for c in feat_cols})
    away_feats = big5.rename(columns={"team_id": "away_team_id"})[
        ["fixture_id", "away_team_id"] + feat_cols
    ].rename(columns={c: f"away_{c}" for c in feat_cols})

    enriched = join_map.merge(home_feats, on=["fixture_id", "home_team_id"], how="left")
    enriched = enriched.merge(away_feats, on=["fixture_id", "away_team_id"], how="left")
    enriched["diff_big5_share"] = enriched["home_big5_share"] - enriched["away_big5_share"]

    out = training.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.merge(
        enriched.drop(columns=["fixture_id", "home_team_id", "away_team_id"]),
        on=["date", "home_team", "away_team"], how="left",
    )
    return out


if __name__ == "__main__":
    per_team = per_team_big5_features()
    print(f"per-team rows: {len(per_team):,}")
    print(per_team.describe().round(2).to_string())
    print()
    matches = pd.read_csv(DEFAULT_MATCHES_PATH)
    nm = matches[["fixture_id", "team_id", "team_name", "season"]].drop_duplicates()
    samp = per_team.merge(nm, on=["fixture_id", "team_id"], how="left")
    print("WC 2022 big5 shares (sample):")
    print(samp[samp["season"] == 2022].sort_values("big5_share", ascending=False).head(20)[
        ["team_name", "big5_share", "big5_count", "n_matched"]
    ].to_string(index=False))
