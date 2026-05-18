"""Join api-football player_ids to Transfermarkt player_ids.

api-football lineups use abbreviated names ("E. Martínez", "L. Messi"),
no DOB. We match by:
  1. Restricting the TM candidate pool to players who have appeared for
     the country's senior national team (via `tm.build_team_country_map`).
  2. Within that pool, matching on (surname + first-name initial). When the
     api name is multi-token full, we also try full-name match.

Outputs a mapping (api_player_id, tm_player_id, country, match_quality)
suitable for joining api_lineups_players.csv to TM market values, ages,
injuries.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import transfermarkt as tm
from src.fjelstul import normalise_name

# Country-name fixes: api-football team_name -> name as it appears in
# tm.build_team_country_map() (which uses TM `citizenship` strings).
# Lookup is case-insensitive (see _country_key below), so case doesn't matter.
API_TO_TM_COUNTRY = {
    "USA": "United States",
    "South Korea": "Korea, South",
    "North Korea": "Korea, North",
    "Korea Republic": "Korea, South",
    "Republic of Ireland": "Ireland",
    "Rep. Of Ireland": "Ireland",
    "Ivory Coast": "Cote d'Ivoire",
    "Cape Verde Islands": "Cape Verde",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Turkey": "Türkiye",
    "Bosnia & Herzegovina": "Bosnia-Herzegovina",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Hong Kong": "Hongkong",
    "FYR Macedonia": "North Macedonia",
    "Macedonia": "North Macedonia",
    "St. Kitts and Nevis": "St. Kitts & Nevis",
    "St. Vincent / Grenadines": "St. Vincent & Grenadinen",
    "New Caledonia": "Neukaledonien",
    "Curaçao": "Curacao",
    "Gambia": "The Gambia",
}


def _country_key(name: str) -> str:
    """Case-fold + light normalisation so alias matching is forgiving."""
    if not isinstance(name, str):
        return ""
    return name.strip().casefold()


def _split_tokens(name: str) -> list[str]:
    return [t for t in normalise_name(name).split(" ") if t]


def _surname_initial(name: str) -> tuple[str | None, str | None]:
    """Return (initial, surname). For 'E. Martínez' -> ('e', 'martinez').
    For full names, take the last token as surname and first letter of
    the first token as initial."""
    tokens = _split_tokens(name)
    if not tokens:
        return None, None
    surname = tokens[-1]
    first = tokens[0]
    initial = first[0] if first else None
    return initial, surname


def build_join(
    api_matches: pd.DataFrame,
    api_players: pd.DataFrame,
    tm_profiles: pd.DataFrame,
    tm_perfs: pd.DataFrame,
    team_country: pd.DataFrame,
) -> pd.DataFrame:
    """Return a DataFrame with columns:
        api_player_id, api_player_name, api_team_name, country,
        tm_player_id, tm_player_name, tm_date_of_birth, match_quality, n_candidates
    where match_quality is one of:
        'exact_name'    - normalised full name match (within country pool)
        'surname_initial' - surname + first-name initial match (within country pool)
        'surname_only'  - unique surname match (within country pool)
        'unmatched'     - no candidate
    """
    # 1. Resolve api team_name -> TM country -> TM team_id -> candidate player_ids
    api_team_names = api_matches[["team_id", "team_name"]].drop_duplicates().copy()
    # Build case-insensitive alias and TM-country lookups
    alias_map = {_country_key(k): v for k, v in API_TO_TM_COUNTRY.items()}
    tm_by_key = {_country_key(c): c for c in team_country["country"]}

    def _resolve(name: str) -> str | None:
        k = _country_key(name)
        if not k:
            return None
        # explicit alias first
        if k in alias_map:
            target = alias_map[k]
            return target if _country_key(target) in tm_by_key else None
        # direct case-insensitive hit
        if k in tm_by_key:
            return tm_by_key[k]
        return None

    api_team_names["country"] = api_team_names["team_name"].apply(_resolve)
    team_country_lookup = team_country.set_index("country")["team_id"].to_dict()
    api_team_names["tm_team_id"] = api_team_names["country"].map(team_country_lookup)
    missing_country = api_team_names[api_team_names["tm_team_id"].isna()]
    if not missing_country.empty:
        print(f"  WARN: {len(missing_country)} api teams have no TM equivalent "
              f"(likely TM under-coverage)")
        print(f"    e.g. {missing_country['team_name'].head(15).tolist()}")

    # 2. Build TM candidate pool per country
    perfs_by_team = tm_perfs.groupby("team_id")["player_id"].apply(set).to_dict()
    profiles = tm_profiles.copy()
    profiles["name_norm"] = profiles["player_name"].apply(
        lambda s: normalise_name(str(s).split("(")[0])  # strip "(player_id)" suffix
    )
    profiles["surname"] = profiles["name_norm"].apply(
        lambda n: n.split(" ")[-1] if n else ""
    )
    profiles["first_initial"] = profiles["name_norm"].apply(
        lambda n: n.split(" ")[0][:1] if n and n.split(" ")[0] else ""
    )

    # Unique (api_player_id, api_player_name, api_team_id) — same player across many
    # fixtures should resolve to the same TM id.
    api_unique = (api_players[["player_id", "player_name", "team_id"]]
                  .dropna(subset=["player_id"])
                  .drop_duplicates())
    api_unique = api_unique.merge(
        api_team_names[["team_id", "team_name", "country", "tm_team_id"]],
        on="team_id", how="left",
    )

    rows = []
    for _, r in api_unique.iterrows():
        out = {
            "api_player_id":   int(r["player_id"]),
            "api_player_name": r["player_name"],
            "api_team_name":   r["team_name"],
            "country":         r["country"],
            "tm_player_id":    None,
            "tm_player_name":  None,
            "tm_date_of_birth": None,
            "match_quality":   "unmatched",
            "n_candidates":    0,
        }
        tm_team_id = r["tm_team_id"]
        if pd.isna(tm_team_id):
            rows.append(out)
            continue
        candidate_ids = perfs_by_team.get(int(tm_team_id), set())
        if not candidate_ids:
            rows.append(out)
            continue
        pool = profiles[profiles["player_id"].isin(candidate_ids)]
        if pool.empty:
            rows.append(out)
            continue
        out["n_candidates"] = len(pool)

        api_name = str(r["player_name"])
        api_norm = normalise_name(api_name.split("(")[0])
        initial, surname = _surname_initial(api_name)

        # Strategy 1: full-name match if api name has multiple tokens and looks full
        api_tokens = _split_tokens(api_name)
        is_initial_form = len(api_tokens) >= 2 and len(api_tokens[0]) == 1
        if api_norm and not is_initial_form:
            full = pool[pool["name_norm"] == api_norm]
            if len(full) == 1:
                m = full.iloc[0]
                out.update(tm_player_id=int(m["player_id"]),
                           tm_player_name=m["player_name"],
                           tm_date_of_birth=m["date_of_birth"],
                           match_quality="exact_name")
                rows.append(out)
                continue

        # Strategy 2: surname + initial
        if surname and initial:
            sub = pool[(pool["surname"] == surname) & (pool["first_initial"] == initial)]
            if len(sub) == 1:
                m = sub.iloc[0]
                out.update(tm_player_id=int(m["player_id"]),
                           tm_player_name=m["player_name"],
                           tm_date_of_birth=m["date_of_birth"],
                           match_quality="surname_initial")
                rows.append(out)
                continue
            elif len(sub) > 1:
                # Tie-break: prefer record with non-null DOB and youngest non-retired
                # heuristic (most recent dob — but actually api lineups are recent, so
                # picking the player most recently active is best). Use first.
                m = sub.iloc[0]
                out.update(tm_player_id=int(m["player_id"]),
                           tm_player_name=m["player_name"],
                           tm_date_of_birth=m["date_of_birth"],
                           match_quality="surname_initial_ambiguous")
                rows.append(out)
                continue

        # Strategy 3: unique surname only
        if surname:
            sub = pool[pool["surname"] == surname]
            if len(sub) == 1:
                m = sub.iloc[0]
                out.update(tm_player_id=int(m["player_id"]),
                           tm_player_name=m["player_name"],
                           tm_date_of_birth=m["date_of_birth"],
                           match_quality="surname_only")
                rows.append(out)
                continue

        rows.append(out)

    return pd.DataFrame(rows)
