"""Map api-football fixture_id to rows in matches_competitive.csv.

Outputs data/processed/training_to_api_fixture.csv with columns:
  date, home_team, away_team, fixture_id (nullable), shift_days (0 or -1)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def _key(date: pd.Timestamp, a: str, b: str) -> str:
    pair = "|".join(sorted([str(a), str(b)]))
    return f"{date.date()}|{pair}"


def main() -> None:
    mc = pd.read_csv(DATA / "processed" / "matches_competitive.csv", parse_dates=["date"])
    am = pd.read_csv(DATA / "processed" / "api_lineups_matches.csv", parse_dates=["date_utc"])

    api_fix = (am.assign(date=am["date_utc"].dt.tz_convert(None).dt.normalize())
                 [["fixture_id", "date", "home_team_name", "away_team_name"]]
                 .drop_duplicates(subset="fixture_id"))

    mc = mc[(mc["date"] >= api_fix["date"].min()) &
            (mc["date"] <= api_fix["date"].max())].copy()
    mc["key"] = mc.apply(lambda r: _key(r["date"], r["home_team"], r["away_team"]), axis=1)

    api_fix = api_fix.copy()
    api_fix["key_same"] = api_fix.apply(
        lambda r: _key(r["date"], r["home_team_name"], r["away_team_name"]), axis=1)
    api_fix["key_minus"] = api_fix.apply(
        lambda r: _key(r["date"] - pd.Timedelta(days=1),
                       r["home_team_name"], r["away_team_name"]), axis=1)

    by_same = api_fix.set_index("key_same")["fixture_id"]
    by_minus = api_fix.set_index("key_minus")["fixture_id"]

    out_rows = []
    for _, r in mc.iterrows():
        fid = by_same.get(r["key"])
        shift = 0
        if pd.isna(fid):
            fid = by_minus.get(r["key"])
            shift = -1 if not pd.isna(fid) else None
        out_rows.append({
            "date":       r["date"],
            "home_team":  r["home_team"],
            "away_team":  r["away_team"],
            "tournament": r["tournament"],
            "fixture_id": int(fid) if not pd.isna(fid) else None,
            "shift_days": shift,
        })

    out = pd.DataFrame(out_rows)
    out_path = DATA / "processed" / "training_to_api_fixture.csv"
    out.to_csv(out_path, index=False)

    matched = out["fixture_id"].notna()
    print(f"{matched.sum():,} / {len(out):,} training rows matched ({matched.mean():.1%}) "
          f"-> {out_path.name}")
    print()
    print("By tournament:")
    print(out.groupby("tournament").agg(
        rows=("fixture_id", "count"),
        matched=("fixture_id", lambda s: s.notna().sum()),
    ).assign(rate=lambda d: d["matched"]/d["rows"])
       .sort_values("rows", ascending=False).head(20).to_string())


if __name__ == "__main__":
    main()
