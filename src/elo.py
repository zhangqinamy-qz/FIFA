import pandas as pd
import numpy as np
from src.confederations import confederation_k_multiplier

# Matches containing these strings are treated as major tournaments
MAJOR_TOURNAMENTS = [
    'UEFA Euro', 'Copa América', 'Africa Cup of Nations',
    'AFC Asian Cup', 'CONCACAF Gold Cup', 'OFC Nations Cup',
]

def _tournament_weight(tournament: str) -> float:
    t = str(tournament)
    if 'FIFA World Cup' in t and 'qualification' not in t.lower():
        return 2.0
    if any(major in t for major in MAJOR_TOURNAMENTS):
        return 1.5
    if 'qualification' in t.lower() or 'qualifier' in t.lower():
        return 1.0
    return 0.75

def _goal_diff_multiplier(gd: int) -> float:
    """Standard World Football Elo goal difference multiplier."""
    if gd <= 1:
        return 1.0
    elif gd == 2:
        return 1.5
    elif gd == 3:
        return 1.75
    else:
        return 1.75 + (gd - 3) / 8


class EloModel:
    def __init__(
        self,
        k: float = 32,
        home_advantage: float = 100,
        initial_rating: float = 1500,
        regression_factor: float = 0.1,
    ):
        self.k = k
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.regression_factor = regression_factor  # annual pull toward 1500
        self.ratings: dict[str, float] = {}
        self._history: list[dict] = []
        self._current_year: int = None

    def _get(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)

    def _expected(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def _regress_to_mean(self):
        """Pull all ratings toward 1500 at the start of each new year."""
        self.ratings = {
            team: self.initial_rating + (1 - self.regression_factor) * (r - self.initial_rating)
            for team, r in self.ratings.items()
        }

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, match in df.iterrows():
            year = match['date'].year
            if self._current_year is None:
                self._current_year = year
            elif year > self._current_year:
                self._regress_to_mean()
                self._current_year = year

            home, away = match['home_team'], match['away_team']
            neutral = match.get('neutral', False)
            r_home = self._get(home) + (0 if neutral else self.home_advantage)
            r_away = self._get(away)

            exp_home = self._expected(r_home, r_away)
            hs, as_ = match['home_score'], match['away_score']
            actual = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)

            gd = abs(hs - as_)
            conf_mult = (confederation_k_multiplier(home) + confederation_k_multiplier(away)) / 2
            k_adjusted = self.k * min(_tournament_weight(match['tournament']) * _goal_diff_multiplier(gd), 2.0) * conf_mult

            pre_home = self._get(home)
            pre_away = self._get(away)

            self.ratings[home] = pre_home + k_adjusted * (actual - exp_home)
            self.ratings[away] = pre_away + k_adjusted * ((1 - actual) - (1 - exp_home))

            self._history.append({'date': match['date'], 'team': home, 'elo': pre_home})
            self._history.append({'date': match['date'], 'team': away, 'elo': pre_away})

            rows.append({
                **match.to_dict(),
                'home_elo_pre': pre_home,
                'away_elo_pre': pre_away,
                'home_win_prob': exp_home,
                'k_adjusted': k_adjusted,
            })

        return pd.DataFrame(rows)

    def current_ratings(self) -> pd.DataFrame:
        return (
            pd.DataFrame(list(self.ratings.items()), columns=['team', 'elo'])
            .sort_values('elo', ascending=False)
            .reset_index(drop=True)
        )

    def rating_history(self, teams: list[str]) -> pd.DataFrame:
        df = pd.DataFrame(self._history)
        return df[df['team'].isin(teams)].reset_index(drop=True)
