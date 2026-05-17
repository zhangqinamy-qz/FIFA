"""Fjelstul World Cup roster loader and matcher to Transfermarkt.

Data source: https://github.com/jfjelstul/worldcup

Fjelstul has per-tournament squads (real 23-player rosters) for every men's WC
1930-2022, but uses its own player_id space and links only to Wikipedia. To get
market values for those players we join to Transfermarkt profiles on
(normalised name, date of birth).
"""
from __future__ import annotations

import re
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv"

FILES = {
    "squads":   f"{BASE_URL}/squads.csv",
    "players":  f"{BASE_URL}/players.csv",
    "teams":    f"{BASE_URL}/teams.csv",
    "matches":  f"{BASE_URL}/matches.csv",
}

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "fjelstul"

# Fjelstul team_name -> our match-data home_team / away_team names.
# Most match exactly; this only lists the divergences.
TEAM_NAME_FIXES = {
    "United States": "United States",
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Czech Republic": "Czech Republic",
    "Czechoslovakia": "Czechoslovakia",
    "West Germany": "Germany",
    "East Germany": "Germany DR",
    "Yugoslavia": "Yugoslavia",
    "Soviet Union": "Soviet Union",
    "FR Yugoslavia": "Serbia and Montenegro",
    "Serbia and Montenegro": "Serbia and Montenegro",
    "China PR": "China",
}


def download_data(data_dir: Path | str = DEFAULT_DATA_DIR, overwrite: bool = False) -> dict[str, Path]:
    """Download the four CSVs we need into `data_dir`."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, url in FILES.items():
        dest = data_dir / Path(url).name
        if dest.exists() and not overwrite:
            paths[key] = dest
            continue
        print(f"Downloading {key} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        paths[key] = dest
    return paths


def _ensure_local(data_dir: Path | str, filename: str) -> Path:
    """Return path to filename, re-downloading from GitHub if missing."""
    p = Path(data_dir) / filename
    if not p.exists():
        download_data(data_dir)
    return p


def load_squads(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(_ensure_local(data_dir, "squads.csv"))


def load_players(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(_ensure_local(data_dir, "players.csv"), parse_dates=["birth_date"])
    return df


def normalise_name(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace, drop punctuation."""
    if not isinstance(name, str):
        return ""
    # Decompose accented chars into base letter + combining mark, then drop marks.
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase and remove anything that isn't a letter, digit, or space.
    s = re.sub(r"[^a-z0-9\s]", " ", ascii_only.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _fjelstul_full_name(row) -> str:
    given = row.get("given_name", "") or ""
    family = row.get("family_name", "") or ""
    return f"{given} {family}".strip()


def _tm_clean_name(name: str) -> str:
    """Transfermarkt names look like 'Miroslav Klose (10)' — strip the (id) suffix."""
    if not isinstance(name, str):
        return ""
    return re.sub(r"\s*\(\d+\)\s*$", "", name)


def match_to_transfermarkt(
    fj_players: pd.DataFrame,
    tm_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Build a mapping (fjelstul_player_id, tm_player_id) by (name, DOB) join.

    Strategy: normalised full name AND exact birth date must match. Players
    with no DOB on either side are dropped.

    Returns DataFrame with columns: fjelstul_player_id, tm_player_id,
                                    name_norm, birth_date, fj_name, tm_name.
    Duplicate matches (same key) are resolved by keeping the first TM record.
    """
    fj = fj_players[["player_id", "family_name", "given_name", "birth_date"]].copy()
    fj["name_norm"] = fj.apply(lambda r: normalise_name(_fjelstul_full_name(r)), axis=1)
    fj["birth_date"] = pd.to_datetime(fj["birth_date"], errors="coerce").dt.date
    fj = fj.dropna(subset=["birth_date"])
    fj = fj[fj["name_norm"] != ""]
    fj = fj.rename(columns={"player_id": "fjelstul_player_id"})

    tm = tm_profiles[["player_id", "player_name", "date_of_birth"]].copy()
    tm["name_norm"] = tm["player_name"].apply(_tm_clean_name).apply(normalise_name)
    tm["birth_date"] = pd.to_datetime(tm["date_of_birth"], errors="coerce").dt.date
    tm = tm.drop(columns=["date_of_birth"])
    tm = tm.dropna(subset=["birth_date"])
    tm = tm[tm["name_norm"] != ""]
    tm = tm.rename(columns={
        "player_id": "tm_player_id",
        "player_name": "tm_name",
    })
    # Drop duplicate (name, dob) on the TM side — keeps the first occurrence.
    tm = tm.drop_duplicates(subset=["name_norm", "birth_date"], keep="first")

    exact = fj.merge(tm, on=["name_norm", "birth_date"], how="inner")

    # Second pass: for FJ players still unmatched, try DOB-anchored partial-name
    # match (one side's name is a substring of the other). Handles mononyms like
    # "Neymar" in FJ vs "Neymar da Silva Santos Junior" in TM.
    unmatched = fj[~fj["fjelstul_player_id"].isin(exact["fjelstul_player_id"])]
    by_dob = tm.groupby("birth_date")
    partial_rows = []
    for _, fj_row in unmatched.iterrows():
        if fj_row["birth_date"] not in by_dob.groups:
            continue
        candidates = by_dob.get_group(fj_row["birth_date"])
        fj_n = fj_row["name_norm"]
        # Substring either way, lowercased and accent-free already.
        hits = candidates[candidates["name_norm"].apply(
            lambda tm_n: fj_n in tm_n or tm_n in fj_n
        )]
        if len(hits) == 1:
            tm_row = hits.iloc[0]
            partial_rows.append({
                "fjelstul_player_id": fj_row["fjelstul_player_id"],
                "tm_player_id": tm_row["tm_player_id"],
                "name_norm": fj_n,
                "birth_date": fj_row["birth_date"],
                "tm_name": tm_row["tm_name"],
                "match_type": "partial",
            })
        # If multiple candidates share the DOB and overlapping name, skip — ambiguous.

    partial = pd.DataFrame(partial_rows)
    exact["match_type"] = "exact"
    merged = pd.concat([exact, partial], ignore_index=True)
    merged["fj_name"] = merged.apply(_fjelstul_full_name, axis=1) if {"family_name", "given_name"}.issubset(merged.columns) else merged["name_norm"]
    return merged[[
        "fjelstul_player_id", "tm_player_id", "name_norm",
        "birth_date", "fj_name", "tm_name", "match_type",
    ]].reset_index(drop=True)


def wc_squad_value(
    team_name: str,
    tournament_id: str,
    *,
    squads: pd.DataFrame,
    players: pd.DataFrame,
    fj_to_tm: pd.DataFrame,
    market_value: pd.DataFrame,
    injuries: pd.DataFrame | None = None,
    as_of_date: str | pd.Timestamp | None = None,
    top_n: int = 23,
    n_gk: int = 3,
) -> dict:
    """Full squad-feature dict computed on the real Fjelstul roster.

    Returns value, age (outfield + GK), top1_share, and injury features
    (aggregate + rank-aware) for the team's 23-man WC squad. Same shape as
    `transfermarkt.historical_squad_value` so they're drop-in compatible.

    Args:
        team_name: Fjelstul team_name (e.g. "Brazil", "Korea Republic").
        tournament_id: Fjelstul tournament_id (e.g. "WC-2014").
        squads: Fjelstul squads DataFrame (player_id + position_code per WC).
        players: Fjelstul players DataFrame (player_id + birth_date).
        fj_to_tm: mapping from `match_to_transfermarkt`.
        market_value: Transfermarkt market value DataFrame.
        injuries: Transfermarkt player_injuries DataFrame.
        as_of_date: defaults to June 1 of tournament year.
    """
    from src.transfermarkt import squad_features_from_players, _empty_feature_dict

    roster = squads[(squads["team_name"] == team_name) & (squads["tournament_id"] == tournament_id)]
    if roster.empty:
        return {"team_name": team_name, "tournament_id": tournament_id,
                "n_roster": 0, "n_matched": 0, **_empty_feature_dict()}

    if as_of_date is None:
        year = int(tournament_id.split("-")[-1])
        as_of_date = pd.Timestamp(year, 6, 1)
    as_of = pd.Timestamp(as_of_date)

    # Join roster -> TM player_id + DOB (for age) + GK flag (from Fjelstul position_code)
    joined = roster.merge(
        fj_to_tm[["fjelstul_player_id", "tm_player_id"]],
        left_on="player_id", right_on="fjelstul_player_id", how="left",
    )
    joined = joined.merge(
        players[["player_id", "birth_date"]].rename(columns={"player_id": "fjelstul_player_id"}),
        on="fjelstul_player_id", how="left",
    )
    joined = joined.dropna(subset=["tm_player_id", "birth_date"]).copy()
    joined["tm_player_id"] = joined["tm_player_id"].astype(int)
    joined["age"] = (as_of - pd.to_datetime(joined["birth_date"])).dt.days / 365.25
    joined["is_gk"] = joined["position_code"].eq("GK")

    # Pull market values
    mv = market_value[
        market_value["player_id"].isin(joined["tm_player_id"]) & (market_value["date"] <= as_of)
    ]
    latest = (mv.sort_values("date").groupby("player_id", as_index=False).tail(1)
                [["player_id", "value"]])
    latest = latest.rename(columns={"player_id": "tm_player_id"})
    player_table = joined.merge(latest, on="tm_player_id", how="inner")[
        ["tm_player_id", "value", "age", "is_gk"]
    ].rename(columns={"tm_player_id": "player_id"})

    feats = squad_features_from_players(player_table, as_of, injuries=injuries, top_n=top_n, n_gk=n_gk)
    return {
        "team_name": team_name,
        "tournament_id": tournament_id,
        "n_roster": int(len(roster)),
        "n_matched": int(joined["tm_player_id"].nunique()),
        **feats,
    }


def wc_features_for_matches(
    matches: pd.DataFrame,
    *,
    squads: pd.DataFrame,
    players: pd.DataFrame,
    fj_to_tm: pd.DataFrame,
    market_value: pd.DataFrame,
    injuries: pd.DataFrame | None = None,
    top_n: int = 23,
    min_matched: int = 10,
) -> pd.DataFrame:
    """Add real-roster squad features to WC matches.

    Returns a frame indexed by the matches' original index with columns:
      home_*, away_* for top_n_value_eur, outfield_mean_age, gk_mean_age,
      top1_share, injured_value_share, injured_top1, injured_top3_share, n_matched.

    `as_of_date` per match = match_date (captures in-tournament injuries like
    Neymar 2014/2022). Teams with `n_matched < min_matched` get NaN in all
    feature columns — caller should fall back to pool-based values for those.
    """
    feat_cols = [
        "top_n_value_eur", "outfield_mean_age", "gk_mean_age",
        "top1_share", "injured_value_share", "injured_top1",
        "injured_top3_share", "n_matched",
    ]
    out = pd.DataFrame(index=matches.index, columns=[f"{side}_{c}" for side in ("home","away") for c in feat_cols], dtype=float)

    # Reverse: our match team-names → Fjelstul team_names.
    reverse_fixes = {v: k for k, v in TEAM_NAME_FIXES.items() if v != k}
    # Special cases where multiple FJ names map to same match name (e.g. Germany);
    # `reverse_fixes` gets whichever wins the dict overwrite, but those FJ names
    # are split by era so a year-keyed lookup works for what we need:
    reverse_fixes.setdefault("South Korea", "Korea Republic")

    cache: dict[tuple, dict] = {}
    for idx, row in matches.iterrows():
        date = pd.Timestamp(row["date"])
        year = int(row["date"].year) if hasattr(row["date"], "year") else date.year
        tid = f"WC-{year}"
        for side in ("home", "away"):
            team_raw = row[f"{side}_team"]
            team_fj = reverse_fixes.get(team_raw, team_raw)
            key = (team_fj, tid, date.normalize())
            if key not in cache:
                cache[key] = wc_squad_value(
                    team_fj, tid, squads=squads, players=players,
                    fj_to_tm=fj_to_tm, market_value=market_value,
                    injuries=injuries, as_of_date=date, top_n=top_n,
                )
            info = cache[key]
            if info.get("missing") or info.get("n_matched", 0) < min_matched:
                continue
            for c in feat_cols:
                out.at[idx, f"{side}_{c}"] = info.get(c)
    return out
