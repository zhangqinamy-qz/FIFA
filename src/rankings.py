import pandas as pd

# Maps FIFA ranking country_full names → results dataset team names where they differ
NAME_MAP = {
    'IR Iran': 'Iran',
    'Korea Republic': 'South Korea',
    'Korea DPR': 'North Korea',
    'USA': 'United States',
    'Türkiye': 'Turkey',
    'Czechia': 'Czech Republic',
    'China PR': 'China',
    'Chinese Taipei': 'Taiwan',
    'North Macedonia': 'North Macedonia',
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Cabo Verde': 'Cape Verde',
    'São Tomé and Príncipe': 'São Tomé and Príncipe',
    'St. Kitts and Nevis': 'Saint Kitts and Nevis',
    'St. Lucia': 'Saint Lucia',
    'St. Vincent and the Grenadines': 'Saint Vincent and the Grenadines',
    'Trinidad and Tobago': 'Trinidad and Tobago',
    'Antigua and Barbuda': 'Antigua and Barbuda',
    'Guinea Bissau': 'Guinea-Bissau',
    'Côte d\'Ivoire': 'Ivory Coast',
    'Congo DR': 'DR Congo',
    'Congo': 'Congo',
}

def load_rankings(path: str, scraped_path: str = None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['rank_date'])
    if scraped_path:
        import os
        if os.path.exists(scraped_path):
            scraped = pd.read_csv(scraped_path, parse_dates=['rank_date'])
            df = pd.concat([df, scraped], ignore_index=True).drop_duplicates(
                subset=['rank_date', 'country_full']
            )
    df['team'] = df['country_full'].map(lambda x: NAME_MAP.get(x, x))
    df = df[['rank_date', 'team', 'rank', 'total_points', 'confederation']].sort_values('rank_date')
    return df


class RankingLookup:
    def __init__(self, rankings_df: pd.DataFrame):
        # Build a dict: team → sorted list of (rank_date, rank, points)
        self._index: dict[str, pd.DataFrame] = {
            team: grp.reset_index(drop=True)
            for team, grp in rankings_df.groupby('team')
        }

    def get(self, team: str, date: pd.Timestamp) -> dict:
        """Return the most recent FIFA rank and points for a team before a given date."""
        if team not in self._index:
            return {'rank': None, 'total_points': None}
        grp = self._index[team]
        past = grp[grp['rank_date'] <= date]
        if past.empty:
            # Use earliest available if no ranking exists before match date
            row = grp.iloc[0]
        else:
            row = past.iloc[-1]
        return {'rank': row['rank'], 'total_points': row['total_points']}
