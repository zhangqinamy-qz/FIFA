"""Transfermarkt national-team squad-value loader.

Data source: https://github.com/salimt/football-datasets

We download three CSVs once into data/raw/transfermarkt/ and use them to compute
national-team squad market values at a given date.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/salimt/football-datasets/main/datalake/transfermarkt"

FILES = {
    "national_performances": f"{BASE_URL}/player_national_performances/player_national_performances.csv",
    "market_value":          f"{BASE_URL}/player_market_value/player_market_value.csv",
    "profiles":              f"{BASE_URL}/player_profiles/player_profiles.csv",
}

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "transfermarkt"

# Map our match-data team names -> the names that appear in Transfermarkt profiles
# (so callers can use the same country labels as in src/data.py).
COUNTRY_ALIASES = {
    "South Korea": "Korea, South",
    "North Korea": "Korea, North",
    "USA": "United States",
    "Republic of Ireland": "Ireland",
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde",
    "DR Congo": "Congo DR",
    "Czechia": "Czech Republic",
}


def download_data(data_dir: Path | str = DEFAULT_DATA_DIR, overwrite: bool = False) -> dict[str, Path]:
    """Download the three CSVs from GitHub. Returns a dict of {key: local_path}."""
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


def load_national_performances(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    p = Path(data_dir) / "player_national_performances.csv"
    return pd.read_csv(p)


def load_market_value(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    p = Path(data_dir) / "player_market_value.csv"
    df = pd.read_csv(p, parse_dates=["date_unix"])
    df = df.rename(columns={"date_unix": "date"})
    return df


def load_profiles(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    p = Path(data_dir) / "player_profiles.csv"
    cols = ["player_id", "player_name", "citizenship", "date_of_birth",
            "main_position", "position", "current_club_id", "current_club_name"]
    df = pd.read_csv(p, usecols=cols)
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    return df


def build_team_country_map(
    perfs: pd.DataFrame,
    profiles: pd.DataFrame,
    min_players: int = 3,
    senior_only: bool = True,
) -> pd.DataFrame:
    """Map team_id -> country name by majority citizenship of associated players.

    Each country has multiple Transfermarkt team_ids (senior + U21 + U19 + youth).
    When `senior_only=True`, we keep just the senior team for each country, identified
    as the team_id with the highest max(matches) among that country's teams — senior
    legends accrue 100+ caps, while youth players cap out around 20–25.

    Returns DataFrame columns: team_id, country, n_players, max_matches, confidence.
    `confidence` is the share of players whose citizenship matches the assigned country.
    Teams with fewer than `min_players` linked are dropped.
    """
    merged = perfs.merge(profiles[["player_id", "citizenship"]], on="player_id", how="left")
    merged = merged.dropna(subset=["citizenship"])
    # Transfermarkt encodes multi-citizenship as e.g. "Germany  Poland" (double-space).
    # Each player may also represent a country they aren't a citizen of (rare), so we
    # vote by *all* citizenships a player holds — exploding multi-citizenship rows.
    merged = merged.assign(
        citizenship=merged["citizenship"].str.split(r"\s{2,}")
    ).explode("citizenship")
    merged["citizenship"] = merged["citizenship"].str.strip()
    merged = merged[merged["citizenship"] != ""]

    rows = []
    for team_id, group in merged.groupby("team_id"):
        # Count unique players per citizenship (since a multi-citizen player is now in
        # multiple rows). This way "Argentina  Italy" players contribute 1 to each.
        per_citizenship = group.drop_duplicates(["player_id", "citizenship"])
        top = per_citizenship["citizenship"].value_counts()
        n_unique = group["player_id"].nunique()
        if n_unique < min_players:
            continue
        country = top.index[0]
        rows.append({
            "team_id": int(team_id),
            "country": country,
            "n_players": n_unique,
            "max_matches": int(group["matches"].max()),
            "confidence": top.iloc[0] / n_unique,
        })
    df = pd.DataFrame(rows)

    if senior_only:
        # For each country, keep the team_id with the highest max_matches.
        df = (df.sort_values(["country", "max_matches"], ascending=[True, False])
                .groupby("country", as_index=False)
                .head(1))

    return df.sort_values("max_matches", ascending=False).reset_index(drop=True)


def _latest_value_on_or_before(player_id: int, as_of: pd.Timestamp,
                               market_value: pd.DataFrame) -> Optional[float]:
    sub = market_value[(market_value["player_id"] == player_id) & (market_value["date"] <= as_of)]
    if sub.empty:
        return None
    return sub.sort_values("date").iloc[-1]["value"]


def squad_value(
    country: str,
    as_of_date: str | pd.Timestamp = None,
    *,
    team_map: pd.DataFrame,
    perfs: pd.DataFrame,
    market_value: pd.DataFrame,
    career_states: tuple[str, ...] = ("CURRENT_NATIONAL_PLAYER",),
    top_n: int = 23,
) -> dict:
    """Compute squad market-value aggregates for a national team.

    Args:
        country: country name (matches the 'country' column in `team_map`).
        as_of_date: snapshot date; defaults to today.
        team_map: output of `build_team_country_map`.
        perfs: national performances DataFrame.
        market_value: market value DataFrame (with `date` column parsed).
        career_states: which career_state values to include.
        top_n: also return sum of top-N most valuable players.

    Returns dict with: country, team_id, as_of, n_players, n_with_value,
                       total_value_eur, mean_value_eur, top_n_value_eur.
    """
    as_of = pd.Timestamp(as_of_date) if as_of_date is not None else pd.Timestamp.today().normalize()
    resolved = COUNTRY_ALIASES.get(country, country)
    row = team_map[team_map["country"] == resolved]
    if row.empty:
        raise ValueError(f"No team_id found for country={country!r} (resolved to {resolved!r}). "
                         f"Available examples: {team_map['country'].head(10).tolist()}")
    team_id = int(row.sort_values("max_matches", ascending=False).iloc[0]["team_id"])

    players = perfs[(perfs["team_id"] == team_id) & (perfs["career_state"].isin(career_states))]
    player_ids = players["player_id"].unique()

    # Filter market_value once, then group by player and pick the latest on or before as_of
    mv = market_value[market_value["player_id"].isin(player_ids) & (market_value["date"] <= as_of)]
    latest = (
        mv.sort_values("date")
          .groupby("player_id", as_index=False)
          .tail(1)
          [["player_id", "value"]]
    )
    values = latest["value"].to_numpy()
    values_sorted = np.sort(values)[::-1]

    return {
        "country": country,
        "team_id": team_id,
        "as_of": as_of.date().isoformat(),
        "n_players": len(player_ids),
        "n_with_value": len(values),
        "total_value_eur": float(values.sum()) if len(values) else 0.0,
        "mean_value_eur": float(values.mean()) if len(values) else 0.0,
        "top_n_value_eur": float(values_sorted[:top_n].sum()) if len(values) else 0.0,
    }


def historical_squad_value(
    country: str,
    as_of_date: str | pd.Timestamp,
    *,
    team_map: pd.DataFrame,
    perfs: pd.DataFrame,
    profiles: pd.DataFrame,
    market_value: pd.DataFrame,
    min_age: int = 17,
    max_age: int = 37,
    top_n: int = 23,
) -> dict:
    """Approximate squad market value at a historical date.

    Strategy: take ALL players who ever played for the country's senior team_id
    (any career_state), keep those aged between `min_age` and `max_age` on
    `as_of_date`, and look up their latest market value on or before that date.

    Note: this is an approximation. Without per-tournament rosters we can't know
    exactly who would have been called up for, e.g., the 2010 WC. We assume the
    set of "eligible active players" is the right proxy.
    """
    as_of = pd.Timestamp(as_of_date)
    resolved = COUNTRY_ALIASES.get(country, country)
    row = team_map[team_map["country"] == resolved]
    if row.empty:
        return {"country": country, "as_of": as_of.date().isoformat(),
                "n_with_value": 0, "total_value_eur": 0.0,
                "mean_value_eur": 0.0, "top_n_value_eur": 0.0, "missing": True}
    team_id = int(row.sort_values("max_matches", ascending=False).iloc[0]["team_id"])

    # All players who ever played for this national team
    candidate_ids = perfs.loc[perfs["team_id"] == team_id, "player_id"].unique()
    cand = profiles[profiles["player_id"].isin(candidate_ids) & profiles["date_of_birth"].notna()].copy()
    age = (as_of - cand["date_of_birth"]).dt.days / 365.25
    cand = cand[(age >= min_age) & (age <= max_age)]
    eligible_ids = cand["player_id"].to_numpy()

    mv = market_value[market_value["player_id"].isin(eligible_ids) & (market_value["date"] <= as_of)]
    latest = (
        mv.sort_values("date")
          .groupby("player_id", as_index=False)
          .tail(1)
          [["player_id", "value"]]
    )
    values = latest["value"].to_numpy()
    values_sorted = np.sort(values)[::-1]

    return {
        "country": country,
        "team_id": team_id,
        "as_of": as_of.date().isoformat(),
        "n_eligible": len(eligible_ids),
        "n_with_value": len(values),
        "total_value_eur": float(values.sum()) if len(values) else 0.0,
        "mean_value_eur": float(values.mean()) if len(values) else 0.0,
        "top_n_value_eur": float(values_sorted[:top_n].sum()) if len(values) else 0.0,
        "missing": len(values) == 0,
    }


def squad_value_for_matches(
    matches: pd.DataFrame,
    *,
    team_map: pd.DataFrame,
    perfs: pd.DataFrame,
    profiles: pd.DataFrame,
    market_value: pd.DataFrame,
    top_n: int = 23,
    snapshot: str = "month",
) -> pd.DataFrame:
    """Attach home_top_n_value_eur and away_top_n_value_eur to a matches frame.

    For efficiency, we cache squad-value lookups per (country, snapshot_key).
    `snapshot` controls the resolution: "month" rounds the match date to its
    month start, "year" rounds to its year. Coarser snapshots = faster.
    """
    if snapshot == "month":
        key_fn = lambda d: pd.Timestamp(d.year, d.month, 1)
    elif snapshot == "year":
        key_fn = lambda d: pd.Timestamp(d.year, 7, 1)  # mid-year snapshot
    else:
        raise ValueError(f"snapshot must be 'month' or 'year', got {snapshot!r}")

    out = matches.copy()
    out["snapshot"] = out["date"].apply(key_fn)
    pairs = pd.concat([
        out[["home_team", "snapshot"]].rename(columns={"home_team": "country"}),
        out[["away_team", "snapshot"]].rename(columns={"away_team": "country"}),
    ]).drop_duplicates().reset_index(drop=True)

    cache: dict[tuple, float] = {}
    for _, p in pairs.iterrows():
        key = (p["country"], p["snapshot"])
        info = historical_squad_value(
            p["country"], p["snapshot"],
            team_map=team_map, perfs=perfs, profiles=profiles, market_value=market_value,
            top_n=top_n,
        )
        cache[key] = info["top_n_value_eur"]

    out["home_top_n_value_eur"] = out.apply(lambda r: cache.get((r["home_team"], r["snapshot"])), axis=1)
    out["away_top_n_value_eur"] = out.apply(lambda r: cache.get((r["away_team"], r["snapshot"])), axis=1)
    return out.drop(columns=["snapshot"])


def squad_value_table(
    countries: list[str],
    as_of_date: str | pd.Timestamp = None,
    *,
    team_map: pd.DataFrame,
    perfs: pd.DataFrame,
    market_value: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    rows = [squad_value(c, as_of_date, team_map=team_map, perfs=perfs,
                        market_value=market_value, **kwargs) for c in countries]
    return pd.DataFrame(rows).sort_values("top_n_value_eur", ascending=False).reset_index(drop=True)
