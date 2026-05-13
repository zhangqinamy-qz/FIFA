import pandas as pd

FRIENDLY_LABELS = {
    # Standard friendlies
    'Friendly', 'Friendly tournament', 'FIFA Series',
    # Named invitationals / sponsor tournaments
    'Al Ain International Cup', 'King Hassan II Tournament', 'King\'s Cup',
    'Kirin Cup', 'Kirin Challenge Cup', 'Korea Cup', 'Dragon Cup', 'Dynasty Cup',
    'Lunar New Year Cup', 'Great Wall Cup', 'Merdeka Tournament', 'Merlion Cup',
    'Malta International Tournament', 'Cyprus International Tournament',
    'Jordan International Tournament', 'Navruz Cup', 'Nehru Cup',
    'Mahinda Rajapaksa Cup', 'VFF Cup', 'Beijing International Friendship Tournament',
    'Guangzhou International Friendship Tournament', 'United Arab Emirates Friendship Tournament',
    'Benedikt Fontana Cup', 'Hungary Heritage Cup', 'ABCS Tournament', 'Dunhill Cup',
    'Rous Cup', 'Tournoi de France', 'Mundialito', 'Scania 100 Tournament', 'OSN Cup',
    'Outrigger Challenge Cup', 'Marlboro Cup', 'Matthews Cup', 'Joe Robbie Cup',
    'Miami Cup', 'USA Cup', 'Atlantic Heritage Cup', 'Balkan Cup', 'Baltic Cup',
    'Nordic Championship', 'SKN Football Festival', 'Dakar Tournament', 'Mapinduzi Cup',
    'Simba Tournament', 'Nile Basin Tournament', 'Amílcar Cabral Cup', 'Mukuru 4 Nations',
    'Mauritius Four Nations Cup', 'Morocco, Capital of African Football',
    'Tournament Burkina Faso', 'Indonesia Tournament', 'TIFOCO Tournament',
    'Coupe de l\'Outre-Mer', 'CONCACAF Series', 'Copa Artigas', 'Copa Félix Bogado',
    'Copa Juan Pinto Durán', 'Copa Lipton', 'Copa Paz del Chaco', 'Copa del Pacífico',
    'Copa Confraternidad', 'Windward Islands Tournament', 'Marianas Cup',
    'MSG Prime Minister\'s Cup', 'Prime Minister\'s Cup', 'Soccer Ashes', 'Trans-Tasman Cup',
    'The Other Final', 'World Unity Cup', 'Unity Cup', 'Millennium Cup',
    'Cup of Ancient Civilizations', 'Niamh Challenge Cup', 'Intercontinental Cup',
    'Nations Cup', 'NAFU Championship', 'Four Nations Tournament', 'Four Nations\' Cup',
    'Three Nations Cup', 'Tri Nation Tournament', 'Tri-Nations Series',
    # Non-FIFA entities
    'CONIFA Africa Football Cup', 'CONIFA Asia Cup', 'CONIFA European Football Cup',
    'CONIFA South America Football Cup', 'CONIFA World Football Cup',
    'CONIFA World Football Cup qualification', 'ConIFA Challenger Cup',
    'FIFI Wild Cup', 'ELF Cup', 'Viva World Cup', 'Muratti Vase',
    'Island Games', 'Tynwald Hill Tournament', 'Corsica Cup',
    # Multi-sport games (U23/amateur squads)
    'Asian Games', 'Southeast Asian Games', 'East Asian Games', 'South Asian Games',
    'Pacific Games', 'Pacific Mini Games', 'South Pacific Games', 'South Pacific Mini Games',
    'Indian Ocean Island Games', 'All-African Games', 'Inter Games', 'Afro-Asian Games',
}

def _build_name_map(former_names_path: str) -> dict[str, tuple]:
    """Returns {former_name: (current_name, start_date, end_date)}."""
    df = pd.read_csv(former_names_path, parse_dates=['start_date', 'end_date'])
    mapping = {}
    for _, row in df.iterrows():
        mapping[row['former']] = (row['current'], row['start_date'], row['end_date'])
    return mapping

def normalize_team_names(df: pd.DataFrame, former_names_path: str) -> pd.DataFrame:
    name_map = _build_name_map(former_names_path)
    def resolve(team, date):
        if team in name_map:
            current, start, end = name_map[team]
            if start <= date <= end:
                return current
        return team
    df = df.copy()
    df['home_team'] = df.apply(lambda r: resolve(r['home_team'], r['date']), axis=1)
    df['away_team'] = df.apply(lambda r: resolve(r['away_team'], r['date']), axis=1)
    return df

def load_results(path: str, since_year: int = 1998, former_names_path: str = None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])
    df = df[df['date'].dt.year >= since_year].copy()
    df = df.dropna(subset=['home_score', 'away_score'])
    df = df.sort_values('date').reset_index(drop=True)
    if former_names_path:
        df = normalize_team_names(df, former_names_path)
    return df

def filter_competitive(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df['tournament'].isin(FRIENDLY_LABELS)].reset_index(drop=True)
