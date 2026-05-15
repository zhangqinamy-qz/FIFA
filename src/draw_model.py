import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def match_outcome(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return 2  # home win
    if home_score == away_score:
        return 1  # draw
    return 0      # away win


def effective_elo_diff(
    home_elo: float | pd.Series,
    away_elo: float | pd.Series,
    neutral: bool | pd.Series,
    home_advantage: float = 100.0,
) -> float | pd.Series:
    ha = np.where(neutral, 0.0, home_advantage) if isinstance(neutral, pd.Series) else (0.0 if neutral else home_advantage)
    return (home_elo + ha) - away_elo


class DrawModel:
    """Multinomial logistic regression: effective Elo diff -> (P_away, P_draw, P_home).

    Class labels follow `match_outcome`: 0=away_win, 1=draw, 2=home_win.
    """

    def __init__(self, home_advantage: float = 100.0):
        self.home_advantage = home_advantage
        self.clf = LogisticRegression(max_iter=1000)
        self._fitted = False

    def _features(self, elo_diff):
        x = np.asarray(elo_diff, dtype=float).reshape(-1, 1) / 400.0
        return x

    def fit(self, matches_with_elo: pd.DataFrame) -> "DrawModel":
        """`matches_with_elo` must have columns: home_elo_pre, away_elo_pre, neutral,
        home_score, away_score.
        """
        diff = effective_elo_diff(
            matches_with_elo['home_elo_pre'],
            matches_with_elo['away_elo_pre'],
            matches_with_elo['neutral'].fillna(False) if 'neutral' in matches_with_elo else False,
            self.home_advantage,
        )
        y = matches_with_elo.apply(
            lambda r: match_outcome(r['home_score'], r['away_score']), axis=1
        ).to_numpy()
        self.clf.fit(self._features(diff), y)
        self._fitted = True
        return self

    def predict_proba(self, home_elo, away_elo, neutral=False) -> np.ndarray:
        """Returns array shape (n, 3): columns are [P_away, P_draw, P_home]."""
        if not self._fitted:
            raise RuntimeError("DrawModel not fitted")
        diff = effective_elo_diff(home_elo, away_elo, neutral, self.home_advantage)
        proba = self.clf.predict_proba(self._features(diff))
        # Ensure column order matches class labels 0,1,2
        classes = list(self.clf.classes_)
        order = [classes.index(c) for c in (0, 1, 2)]
        return proba[:, order]

    def predict_proba_single(self, home_elo: float, away_elo: float, neutral: bool = False) -> dict:
        p = self.predict_proba(np.array([home_elo]), np.array([away_elo]), neutral)[0]
        return {'away_win': p[0], 'draw': p[1], 'home_win': p[2]}
