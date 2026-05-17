"""Big-5 league exposure: count of squad currently playing in top European leagues.

For each WC squad we look up each player's club at June 1 of the tournament
year (using TM transfer history). Then check that club's competition against
the Big-5 league set.

Catch: salimt's `team_details.csv` reflects ~2023 league assignments. A club
that was Bundesliga in 2010 but got relegated by 2020 will be classified by
its 2023 league. For most Big-5 stalwarts (Bayern, Real, Liverpool, etc.) this
is fine, but it's slightly biased for promoted/relegated clubs.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.transfermarkt import DEFAULT_DATA_DIR, _ensure_local

BIG5_COMPETITION_IDS = {"GB1", "ES1", "L1", "IT1", "FR1"}
BIG5_NAMES = {"GB1": "Premier League", "ES1": "LaLiga", "L1": "Bundesliga",
              "IT1": "Serie A", "FR1": "Ligue 1"}

# transfer_history is LFS — needs the media URL, not raw.
TRANSFER_HISTORY_MEDIA_URL = (
    "https://media.githubusercontent.com/media/salimt/football-datasets/main/"
    "datalake/transfermarkt/transfer_history/transfer_history.csv"
)
TEAM_DETAILS_URL = (
    "https://raw.githubusercontent.com/salimt/football-datasets/main/"
    "datalake/transfermarkt/team_details/team_details.csv"
)


def _download_if_missing(filename: str, url: str, data_dir: Path | str = DEFAULT_DATA_DIR) -> Path:
    p = Path(data_dir) / filename
    if not p.exists():
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, p)
    return p


def load_transfer_history(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    p = _download_if_missing("transfer_history.csv", TRANSFER_HISTORY_MEDIA_URL, data_dir)
    df = pd.read_csv(p, parse_dates=["transfer_date"], usecols=[
        "player_id", "transfer_date", "from_team_id", "to_team_id",
    ])
    return df


def load_team_details(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    p = _download_if_missing("team_details.csv", TEAM_DETAILS_URL, data_dir)
    return pd.read_csv(p, usecols=["club_id", "competition_id", "competition_name", "country_name"])


def build_club_to_big5_map(team_details: pd.DataFrame) -> dict[int, str]:
    """club_id -> Big5 competition_id (or absent if not Big5).
    For clubs appearing in multiple competitions across seasons, prefer Big5 over others."""
    # Drop duplicates and keep one row per club; prefer rows where competition_id is in Big5.
    td = team_details.copy()
    td["is_big5"] = td["competition_id"].isin(BIG5_COMPETITION_IDS)
    td = td.sort_values(["club_id", "is_big5"], ascending=[True, False]).drop_duplicates("club_id", keep="first")
    big5 = td[td["is_big5"]]
    return dict(zip(big5["club_id"].astype(int), big5["competition_id"]))


def player_clubs_at_date(
    player_ids: list[int] | np.ndarray,
    as_of_date: pd.Timestamp,
    transfers: pd.DataFrame,
) -> dict[int, int]:
    """For each player_id, return their club_id as of `as_of_date`.

    Logic: the player's club at time t is the `to_team_id` of their most
    recent transfer with `transfer_date <= t`. If no such transfer exists,
    the player was at the `from_team_id` of their earliest transfer (i.e.
    their starting club) — return that.
    """
    as_of = pd.Timestamp(as_of_date)
    sub = transfers[transfers["player_id"].isin(player_ids)].sort_values(["player_id", "transfer_date"])

    out: dict[int, int] = {}
    for pid, grp in sub.groupby("player_id"):
        past = grp[grp["transfer_date"] <= as_of]
        if not past.empty:
            club = past.iloc[-1]["to_team_id"]
        else:
            # No transfers before this date — assume player was at their first known from-club.
            club = grp.iloc[0]["from_team_id"]
        if pd.notna(club):
            out[int(pid)] = int(club)
    return out


def wc_big5_for_matches(
    matches: pd.DataFrame,
    *,
    squads: pd.DataFrame,
    players: pd.DataFrame,
    fj_to_tm: pd.DataFrame,
    market_value: pd.DataFrame,
    transfers: pd.DataFrame,
    club_to_big5: dict[int, str],
    top_n: int = 23,
    min_matched: int = 10,
) -> pd.DataFrame:
    """Per-WC big5 features overlaid onto a matches frame (WC rows only)."""
    from src.fjelstul import TEAM_NAME_FIXES
    reverse_fixes = {v: k for k, v in TEAM_NAME_FIXES.items() if v != k}
    reverse_fixes.setdefault("South Korea", "Korea Republic")

    cols = ["big5_share_top_n", "big5_value_share", "big5_count_top_n"]
    out = pd.DataFrame(index=matches.index, dtype=float,
                       columns=[f"{side}_{c}" for side in ("home","away") for c in cols])

    cache: dict[tuple, dict] = {}
    for idx, row in matches.iterrows():
        date = pd.Timestamp(row["date"])
        year = date.year
        tid = f"WC-{year}"
        for side in ("home", "away"):
            team_raw = row[f"{side}_team"]
            team_fj = reverse_fixes.get(team_raw, team_raw)
            key = (team_fj, tid, date.normalize())
            if key not in cache:
                cache[key] = wc_big5_exposure(
                    team_fj, tid, squads=squads, players=players, fj_to_tm=fj_to_tm,
                    market_value=market_value, transfers=transfers,
                    club_to_big5=club_to_big5, as_of_date=date, top_n=top_n,
                )
            info = cache[key]
            if info.get("missing") or info.get("n_matched", 0) < min_matched:
                continue
            for c in cols:
                out.at[idx, f"{side}_{c}"] = info.get(c)
    return out


def wc_big5_exposure(
    team_name: str,
    tournament_id: str,
    *,
    squads: pd.DataFrame,
    players: pd.DataFrame,
    fj_to_tm: pd.DataFrame,
    market_value: pd.DataFrame,
    transfers: pd.DataFrame,
    club_to_big5: dict[int, str],
    as_of_date: str | pd.Timestamp | None = None,
    top_n: int = 23,
) -> dict:
    """Big-5 exposure for a WC squad.

    Returns:
        big5_count_top_n: number of top-N players (by value) at a Big-5 club on as_of_date
        big5_share_top_n: fraction (0..1)
        big5_value_share: value-weighted share of top-N value at Big-5 clubs
    """
    roster = squads[(squads["team_name"] == team_name) & (squads["tournament_id"] == tournament_id)]
    if roster.empty:
        return {"missing": True, "big5_count_top_n": 0, "big5_share_top_n": 0.0, "big5_value_share": 0.0,
                "n_matched": 0, "n_with_club": 0}

    if as_of_date is None:
        year = int(tournament_id.split("-")[-1])
        as_of_date = pd.Timestamp(year, 6, 1)
    as_of = pd.Timestamp(as_of_date)

    # roster -> TM ids
    joined = roster.merge(
        fj_to_tm[["fjelstul_player_id", "tm_player_id"]],
        left_on="player_id", right_on="fjelstul_player_id", how="left",
    ).dropna(subset=["tm_player_id"])
    joined["tm_player_id"] = joined["tm_player_id"].astype(int)
    matched_ids = joined["tm_player_id"].unique()
    if len(matched_ids) == 0:
        return {"missing": True, "big5_count_top_n": 0, "big5_share_top_n": 0.0, "big5_value_share": 0.0,
                "n_matched": 0, "n_with_club": 0}

    # TM market values as of date
    mv = market_value[market_value["player_id"].isin(matched_ids) & (market_value["date"] <= as_of)]
    latest_val = (mv.sort_values("date").groupby("player_id", as_index=False).tail(1)[["player_id", "value"]])
    if latest_val.empty:
        return {"missing": True, "big5_count_top_n": 0, "big5_share_top_n": 0.0, "big5_value_share": 0.0,
                "n_matched": int(len(matched_ids)), "n_with_club": 0}

    # Top-N by value
    top = latest_val.sort_values("value", ascending=False).head(top_n).copy()

    # Player clubs at as_of_date
    clubs = player_clubs_at_date(top["player_id"].tolist(), as_of, transfers)
    top["club_id"] = top["player_id"].map(clubs)
    top["is_big5"] = top["club_id"].map(lambda c: c in club_to_big5 if pd.notna(c) else False)

    total_val = float(top["value"].sum())
    big5_val  = float(top.loc[top["is_big5"], "value"].sum())
    return {
        "missing": False,
        "n_matched": int(len(matched_ids)),
        "n_with_club": int(top["club_id"].notna().sum()),
        "big5_count_top_n": int(top["is_big5"].sum()),
        "big5_share_top_n": float(top["is_big5"].mean()),
        "big5_value_share": (big5_val / total_val) if total_val > 0 else 0.0,
    }
