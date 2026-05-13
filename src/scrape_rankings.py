"""
Scrapes FIFA rankings from api.fifa.com for dates after the Kaggle dataset cutoff.
Saves results to data/raw/fifa_ranking_scraped.csv in the same format as the Kaggle files.

Usage:
    python src/scrape_rankings.py
"""

import requests
import pandas as pd
import time
from pathlib import Path

BASE_URL = "https://api.fifa.com/api/v3/fifarankings/rankings/rankingsbyschedule"
OUT_PATH = Path("data/raw/fifa_ranking_scraped.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36 Edg/148.0.0.0",
    "Origin":     "https://inside.fifa.com",
    "Referer":    "https://inside.fifa.com/",
}

# Anchor: inside.fifa.com shows dateId=id14415 for June 20, 2024.
# IDs increment by 1 per calendar day.
_ANCHOR_DATE = pd.Timestamp("2024-06-20")
_ANCHOR_ID   = 14415

def _date_to_schedule_id(date_str: str) -> str:
    """Return the rankingScheduleId for a YYYYMMDD string."""
    dt = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")
    date_id = _ANCHOR_ID + (dt - _ANCHOR_DATE).days
    return f"id{date_id}"

# Known FIFA men's ranking release dates after the June 2024 Kaggle cutoff.
# Sourced from inside.fifa.com date picker.
KNOWN_DATES = [
    "20240718",  # July 2024
    "20240919",  # September 2024
    "20241024",  # October 2024
    "20241128",  # November 2024
    "20241219",  # December 2024
    "20250403",  # April 2025
    "20250710",  # July 2025
    "20250918",  # September 2025
    "20251015",  # October 2025
    "20251119",  # November 2025
    "20251219",  # December 2025
    "20260119",  # January 2026
    "20260401",  # April 2026
]


_FORMAT_CUTOVER = pd.Timestamp("2025-10-15")  # FRS_ format starts here


def fetch_ranking(date_str: str) -> pd.DataFrame | None:
    dt = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")
    if dt >= _FORMAT_CUTOVER:
        schedule_id = f"FRS_Male_Football_{date_str}"
    else:
        schedule_id = _date_to_schedule_id(date_str)

    url = f"{BASE_URL}?rankingScheduleId={schedule_id}&language=en"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            print(f"  FAIL {date_str}: HTTP {resp.status_code}")
            return None
        results = resp.json().get("Results", [])
    except Exception as e:
        print(f"  FAIL {date_str}: {e}")
        return None

    if not results:
        print(f"  FAIL {date_str}: no results")
        return None

    rank_date = dt
    rows = [{
        "rank":            r["Rank"],
        "country_full":    r["TeamName"][0]["Description"],
        "country_abrv":    r["IdCountry"],
        "total_points":    r["TotalPoints"],
        "previous_points": r["PrevPoints"],
        "rank_change":     r["RankingMovement"],
        "confederation":   r["ConfederationName"],
        "rank_date":       rank_date,
    } for r in results]
    print(f"  OK {date_str}: {len(rows)} teams")
    return pd.DataFrame(rows)


def main():
    kaggle_file = Path("data/raw/fifa_ranking-2024-06-20.csv")
    cutoff = pd.Timestamp("2024-06-20")
    if kaggle_file.exists():
        existing = pd.read_csv(kaggle_file, parse_dates=["rank_date"])
        cutoff = existing["rank_date"].max()
    print(f"Kaggle data cutoff: {cutoff.date()}")

    frames = []
    for date_str in KNOWN_DATES:
        rank_date = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")
        if rank_date <= cutoff:
            print(f"  skip {date_str} (before cutoff)")
            continue
        df = fetch_ranking(date_str)
        if df is not None:
            frames.append(df)
        time.sleep(0.5)

    if not frames:
        print("No new rankings found.")
        return

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(out):,} rows across {len(frames)} ranking dates -> {OUT_PATH}")


if __name__ == "__main__":
    main()
