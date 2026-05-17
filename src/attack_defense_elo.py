"""Attack/Defense Elo: two ratings per team.

Standard Elo collapses a team into one number. That loses asymmetries —
Germany 2018 had a top-tier attack and a leaky defense, but single Elo
averages them into one mediocre rating.

Here each team has two ratings:
    attack  — how many goals they tend to score
    defense — how many goals they tend to concede

Per-match expected goals (Poisson rate):
    lam_h = exp(c + (A_home - D_away) / SCALE + ha)
    lam_a = exp(c + (A_away - D_home) / SCALE)

where `c = log(avg_goals_per_team_per_match)` ≈ log(1.4) sets the baseline.

Update rule (goals residual, not binary win/loss):
    A_home += K * (actual_home_goals - lam_h)
    D_away -= K * (actual_home_goals - lam_h)     # they conceded more than expected
    A_away += K * (actual_away_goals - lam_a)
    D_home -= K * (actual_away_goals - lam_a)

Other knobs from src.elo (regression-to-mean, tournament weighting,
confederation multiplier) are preserved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.confederations import confederation_k_multiplier
from src.elo import MAJOR_TOURNAMENTS, _tournament_weight


class AttackDefenseEloModel:
    """Two-rating Elo. Ratings updated on goal residuals each match.

    Args:
        k: Elo K-factor applied per goal of residual. Goals residuals are
            typically ±0.5..±3 (not ±1 like binary outcomes), so K should be
            smaller than the standard 32 — defaults to 6.
        scale: rating spread per "1.0 in log(lam)" unit. 200 means a
            +200 attack rating doubles your expected goal rate against an
            average opponent (since exp(200/200)≈2.72, well, with c subtracted).
        log_avg_goals: baseline rate `c`. Default log(1.4) ≈ 0.336 — historical
            average goals per team per match in international football.
        home_advantage: added to `lam_h` linear term. 0.25 ≈ exp(0.25)=1.28x
            more goals at home, in line with empirical home advantage.
        initial_rating: starting attack/defense for unseen teams.
        regression_factor: annual pull toward initial_rating.
    """

    def __init__(
        self,
        k: float = 6.0,
        scale: float = 200.0,
        log_avg_goals: float = 0.336,
        home_advantage: float = 0.25,
        initial_rating: float = 1500.0,
        regression_factor: float = 0.1,
    ):
        self.k = k
        self.scale = scale
        self.log_avg_goals = log_avg_goals
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.regression_factor = regression_factor
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self._history: list[dict] = []
        self._current_year: int | None = None

    def _get_a(self, team: str) -> float:
        return self.attack.get(team, self.initial_rating)

    def _get_d(self, team: str) -> float:
        return self.defense.get(team, self.initial_rating)

    def _regress_to_mean(self):
        for team in list(self.attack):
            self.attack[team] = self.initial_rating + (1 - self.regression_factor) * (self.attack[team] - self.initial_rating)
        for team in list(self.defense):
            self.defense[team] = self.initial_rating + (1 - self.regression_factor) * (self.defense[team] - self.initial_rating)

    def _expected_goals(self, attack: float, defense: float, ha_term: float) -> float:
        x = self.log_avg_goals + (attack - defense) / self.scale + ha_term
        # Clip to avoid numeric blowup on extreme ratings early in training
        return float(np.exp(np.clip(x, -3, 3)))

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, match in df.iterrows():
            year = match["date"].year
            if self._current_year is None:
                self._current_year = year
            elif year > self._current_year:
                self._regress_to_mean()
                self._current_year = year

            home, away = match["home_team"], match["away_team"]
            neutral = bool(match.get("neutral", False))
            ha = 0.0 if neutral else self.home_advantage

            A_h = self._get_a(home); D_h = self._get_d(home)
            A_a = self._get_a(away); D_a = self._get_d(away)

            lam_h = self._expected_goals(A_h, D_a, ha)
            lam_a = self._expected_goals(A_a, D_h, 0.0)

            hs, as_ = int(match["home_score"]), int(match["away_score"])

            # Tournament + confederation weight (same logic as single Elo)
            conf_mult = (confederation_k_multiplier(home) + confederation_k_multiplier(away)) / 2
            k_adj = self.k * min(_tournament_weight(match["tournament"]), 2.0) * conf_mult

            # Goal residuals, compressed to dampen tiny-sample blowups.
            # sign * sqrt(|r|) keeps direction but shrinks ±5-goal residuals to ±2.2.
            def _compress(r):
                return float(np.sign(r) * np.sqrt(abs(r)))
            r_h = _compress(hs - lam_h)
            r_a = _compress(as_ - lam_a)

            # Store pre-match ratings for the predict view
            rows.append({
                **match.to_dict(),
                "home_attack_pre": A_h, "home_defense_pre": D_h,
                "away_attack_pre": A_a, "away_defense_pre": D_a,
                "exp_home_goals": lam_h, "exp_away_goals": lam_a,
            })

            self.attack[home]  = A_h + k_adj * r_h
            self.defense[away] = D_a - k_adj * r_h
            self.attack[away]  = A_a + k_adj * r_a
            self.defense[home] = D_h - k_adj * r_a

            self._history.append({"date": match["date"], "team": home, "attack": A_h, "defense": D_h})
            self._history.append({"date": match["date"], "team": away, "attack": A_a, "defense": D_a})

        return pd.DataFrame(rows)

    def predict_match(self, home: str, away: str, neutral: bool = False) -> dict:
        ha = 0.0 if neutral else self.home_advantage
        lam_h = self._expected_goals(self._get_a(home), self._get_d(away), ha)
        lam_a = self._expected_goals(self._get_a(away), self._get_d(home), 0.0)
        return {"lam_home": lam_h, "lam_away": lam_a}

    def current_ratings(self) -> pd.DataFrame:
        """High attack = scores a lot; high defense = concedes few. Sum is overall skill."""
        teams = sorted(set(self.attack) | set(self.defense))
        return (
            pd.DataFrame({
                "team": teams,
                "attack":  [self._get_a(t) for t in teams],
                "defense": [self._get_d(t) for t in teams],
            })
            .assign(total=lambda d: d["attack"] + d["defense"])
            .sort_values("total", ascending=False)
            .reset_index(drop=True)
        )
