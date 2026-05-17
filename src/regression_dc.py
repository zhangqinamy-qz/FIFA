"""Regression Dixon-Coles: alpha/beta as linear functions of team features.

Standard Dixon-Coles fits a free per-team attack/defense parameter. That works
when every team has plenty of historical matches, but fails for cold-start
teams (new lineups, new countries, 2026 squad rebuilds).

Regression-DC instead parameterises:

    alpha_team_at(t) = w_a . features_team(t)
    beta_team_at(t)  = w_b . features_team(t)

So the strength of "France" depends on France's features (Elo, squad value,
FIFA points) *at match time t*, not on a static team_id. New teams or new
lineups get sensible alpha/beta from their features alone.

Match likelihood (unchanged from DC):
    lam_h = exp(w_a . x_home - w_b . x_away + gamma)
    lam_a = exp(w_a . x_away - w_b . x_home)

Plus the same Dixon-Coles tau correction for low scores.

Identifiability: we use mean-centered features and NO intercept in the
alpha/beta regressions. gamma absorbs the global home advantage; an average
team gets alpha = beta = 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


class RegressionDixonColesModel:
    """Feature-driven Dixon-Coles.

    Args:
        feature_cols: list of per-team feature names. The matches frame must
            contain `home_{col}` and `away_{col}` for each.
        xi: time-decay rate (days^-1). Default 0.0018 ≈ half-weight per year.
        max_goals: score grid truncation for outcome probabilities.
    """

    def __init__(self, feature_cols: list[str], xi: float = 0.0018, max_goals: int = 10):
        self.feature_cols = list(feature_cols)
        self.K = len(feature_cols)
        self.xi = xi
        self.max_goals = max_goals
        # Fitted parameters
        self.w_a: np.ndarray | None = None  # K-vector, attack coefficients
        self.w_b: np.ndarray | None = None  # K-vector, defense coefficients
        self.home_adv: float | None = None
        self.rho: float | None = None
        # Feature mean for centering at predict time
        self.feature_mean_: np.ndarray | None = None
        self._opt_result = None

    def _unpack(self, params: np.ndarray):
        w_a = params[: self.K]
        w_b = params[self.K : 2 * self.K]
        gamma = params[2 * self.K]
        rho = params[2 * self.K + 1]
        return w_a, w_b, gamma, rho

    def _neg_log_lik_and_grad(self, params, X_home, X_away, hg, ag, weights):
        """Analytic gradient version. X_home/X_away: (M, K) feature matrices,
        already mean-centered."""
        w_a, w_b, gamma, rho = self._unpack(params)

        alpha_home = X_home @ w_a
        alpha_away = X_away @ w_a
        beta_home  = X_home @ w_b
        beta_away  = X_away @ w_b

        lam_h = np.exp(alpha_home - beta_away + gamma)
        lam_a = np.exp(alpha_away - beta_home)

        # Base Poisson log-likelihood
        ll = hg * np.log(lam_h) - lam_h + ag * np.log(lam_a) - lam_a

        # Dixon-Coles tau correction on 4 low-score cells
        is_00 = (hg == 0) & (ag == 0)
        is_01 = (hg == 0) & (ag == 1)
        is_10 = (hg == 1) & (ag == 0)
        is_11 = (hg == 1) & (ag == 1)

        tau = np.ones_like(lam_h)
        tau[is_00] = 1 - lam_h[is_00] * lam_a[is_00] * rho
        tau[is_01] = 1 + lam_h[is_01] * rho
        tau[is_10] = 1 + lam_a[is_10] * rho
        tau[is_11] = 1 - rho

        if (tau <= 0).any():
            return 1e10, np.zeros_like(params)

        ll = ll + np.log(tau)
        nll = -(weights * ll).sum()

        # --- Analytic gradient ---
        dlogtau_dlamh = np.zeros_like(lam_h)
        dlogtau_dlama = np.zeros_like(lam_a)
        dlogtau_drho = np.zeros_like(lam_h)

        dlogtau_dlamh[is_00] = -lam_a[is_00] * rho / tau[is_00]
        dlogtau_dlama[is_00] = -lam_h[is_00] * rho / tau[is_00]
        dlogtau_drho[is_00]  = -lam_h[is_00] * lam_a[is_00] / tau[is_00]

        dlogtau_dlamh[is_01] = rho / tau[is_01]
        dlogtau_drho[is_01]  = lam_h[is_01] / tau[is_01]

        dlogtau_dlama[is_10] = rho / tau[is_10]
        dlogtau_drho[is_10]  = lam_a[is_10] / tau[is_10]

        dlogtau_drho[is_11] = -1.0 / tau[is_11]

        # d log L / d log(lam_h) = (hg - lam_h) + dlogtau_dlamh * lam_h
        coef_lh = weights * ((hg - lam_h) + dlogtau_dlamh * lam_h)
        coef_la = weights * ((ag - lam_a) + dlogtau_dlama * lam_a)

        # log(lam_h) = (X_home - X_away_for_beta) · ... is not quite linear, careful:
        # log(lam_h) = w_a . X_home - w_b . X_away + gamma
        # log(lam_a) = w_a . X_away - w_b . X_home
        # d log(lam_h) / d w_a[k] = X_home[:, k]
        # d log(lam_h) / d w_b[k] = -X_away[:, k]
        # d log(lam_a) / d w_a[k] = X_away[:, k]
        # d log(lam_a) / d w_b[k] = -X_home[:, k]

        # d log L / d w_a = X_home.T @ coef_lh + X_away.T @ coef_la
        # d log L / d w_b = -X_away.T @ coef_lh - X_home.T @ coef_la
        grad_w_a = -(X_home.T @ coef_lh + X_away.T @ coef_la)  # NLL = -log L
        grad_w_b = -(-X_away.T @ coef_lh - X_home.T @ coef_la)
        grad_gamma = -np.sum(coef_lh)  # gamma only in lam_h
        grad_rho = -np.sum(weights * dlogtau_drho)

        grad = np.concatenate([grad_w_a, grad_w_b, [grad_gamma, grad_rho]])
        return nll, grad

    def fit(self, df: pd.DataFrame, ref_date: pd.Timestamp | None = None) -> "RegressionDixonColesModel":
        """Fit regression coefficients on a DataFrame of past matches.

        Required columns: date, home_score, away_score, plus home_{feat} and
        away_{feat} for each feature in self.feature_cols.
        """
        df = df.dropna(subset=["home_score", "away_score"] +
                                [f"home_{c}" for c in self.feature_cols] +
                                [f"away_{c}" for c in self.feature_cols]).copy()
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)

        if ref_date is None:
            ref_date = df["date"].max()
        days_back = (ref_date - df["date"]).dt.days.to_numpy().astype(float)
        weights = np.exp(-self.xi * days_back) if self.xi > 0 else np.ones(len(df))

        X_home_raw = df[[f"home_{c}" for c in self.feature_cols]].to_numpy(dtype=float)
        X_away_raw = df[[f"away_{c}" for c in self.feature_cols]].to_numpy(dtype=float)
        # Center features. Use pooled mean over both home and away appearances.
        all_features = np.vstack([X_home_raw, X_away_raw])
        self.feature_mean_ = all_features.mean(axis=0)
        X_home = X_home_raw - self.feature_mean_
        X_away = X_away_raw - self.feature_mean_

        hg = df["home_score"].to_numpy()
        ag = df["away_score"].to_numpy()

        x0 = np.zeros(2 * self.K + 2)
        x0[2 * self.K] = 0.25       # gamma initial
        x0[2 * self.K + 1] = -0.10  # rho initial

        result = minimize(
            self._neg_log_lik_and_grad, x0,
            args=(X_home, X_away, hg, ag, weights),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 500, "disp": False},
        )
        w_a, w_b, gamma, rho = self._unpack(result.x)
        self.w_a = w_a
        self.w_b = w_b
        self.home_adv = float(gamma)
        self.rho = float(rho)
        self._opt_result = result
        return self

    def _rates_from_features(self, x_home: np.ndarray, x_away: np.ndarray,
                              neutral: bool = False) -> tuple[float, float]:
        x_h = x_home - self.feature_mean_
        x_a = x_away - self.feature_mean_
        alpha_h = float(x_h @ self.w_a)
        alpha_a = float(x_a @ self.w_a)
        beta_h  = float(x_h @ self.w_b)
        beta_a  = float(x_a @ self.w_b)
        gamma = 0.0 if neutral else self.home_adv
        lam_h = float(np.exp(alpha_h - beta_a + gamma))
        lam_a = float(np.exp(alpha_a - beta_h))
        return lam_h, lam_a

    def score_matrix_from_features(self, x_home: np.ndarray, x_away: np.ndarray,
                                    neutral: bool = False) -> np.ndarray:
        lam_h, lam_a = self._rates_from_features(x_home, x_away, neutral)
        x = np.arange(self.max_goals + 1)
        px = poisson.pmf(x, lam_h)
        py = poisson.pmf(x, lam_a)
        m = np.outer(px, py).copy()
        m[0, 0] *= 1 - lam_h * lam_a * self.rho
        m[0, 1] *= 1 + lam_h * self.rho
        m[1, 0] *= 1 + lam_a * self.rho
        m[1, 1] *= 1 - self.rho
        return m

    def predict_match_from_features(self, x_home: np.ndarray, x_away: np.ndarray,
                                     neutral: bool = False) -> dict:
        lam_h, lam_a = self._rates_from_features(x_home, x_away, neutral)
        m = self.score_matrix_from_features(x_home, x_away, neutral)
        total = float(m.sum())
        p_home = float(np.tril(m, -1).sum()) / total
        p_draw = float(np.diag(m).sum()) / total
        p_away = float(np.triu(m, 1).sum()) / total
        return {
            "lam_home": lam_h, "lam_away": lam_a,
            "p_away_win": p_away, "p_draw": p_draw, "p_home_win": p_home,
        }

    def predict_many(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Vectorised prediction for a matches frame with home_/away_ feature cols."""
        X_h = matches[[f"home_{c}" for c in self.feature_cols]].to_numpy(dtype=float) - self.feature_mean_
        X_a = matches[[f"away_{c}" for c in self.feature_cols]].to_numpy(dtype=float) - self.feature_mean_
        neutral = matches.get("neutral", pd.Series([False] * len(matches), index=matches.index)).fillna(False).to_numpy()

        alpha_h = X_h @ self.w_a
        alpha_a = X_a @ self.w_a
        beta_h  = X_h @ self.w_b
        beta_a  = X_a @ self.w_b
        gamma_vec = np.where(neutral, 0.0, self.home_adv)
        lam_h = np.exp(alpha_h - beta_a + gamma_vec)
        lam_a = np.exp(alpha_a - beta_h)

        # Score grids per row
        x = np.arange(self.max_goals + 1)
        p_home_win = np.empty(len(matches))
        p_draw     = np.empty(len(matches))
        p_away_win = np.empty(len(matches))
        for i in range(len(matches)):
            px = poisson.pmf(x, lam_h[i])
            py = poisson.pmf(x, lam_a[i])
            m = np.outer(px, py).copy()
            m[0, 0] *= 1 - lam_h[i] * lam_a[i] * self.rho
            m[0, 1] *= 1 + lam_h[i] * self.rho
            m[1, 0] *= 1 + lam_a[i] * self.rho
            m[1, 1] *= 1 - self.rho
            total = m.sum()
            p_home_win[i] = np.tril(m, -1).sum() / total
            p_draw[i]     = np.diag(m).sum() / total
            p_away_win[i] = np.triu(m, 1).sum() / total

        out = matches.copy()
        out["lam_home"] = lam_h
        out["lam_away"] = lam_a
        out["p_home_win"] = p_home_win
        out["p_draw"] = p_draw
        out["p_away_win"] = p_away_win
        return out

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": self.feature_cols,
            "w_attack": self.w_a,
            "w_defense": self.w_b,
        })
