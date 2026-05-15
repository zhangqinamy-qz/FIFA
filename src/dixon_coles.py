"""Dixon-Coles bivariate Poisson model for football match outcomes.

Reference: Dixon & Coles (1997), "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market".

For each match between home team i and away team j:
    lambda_home = exp(alpha_i - beta_j + gamma)
    lambda_away = exp(alpha_j - beta_i)

Goals scored are modelled as Poisson(lambda) for each side, with a low-score
correction tau(x, y) that boosts the probability mass of 0-0, 1-0, 0-1, and 1-1
results to match empirical football scorelines.

Parameters fit by maximum likelihood with optional exponential time decay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def _dc_correction_matrix(score_matrix: np.ndarray, lam_h: float, lam_a: float, rho: float) -> np.ndarray:
    """Apply the Dixon-Coles low-score adjustment in place and return the matrix."""
    score_matrix = score_matrix.copy()
    score_matrix[0, 0] *= 1 - lam_h * lam_a * rho
    score_matrix[0, 1] *= 1 + lam_h * rho
    score_matrix[1, 0] *= 1 + lam_a * rho
    score_matrix[1, 1] *= 1 - rho
    return score_matrix


class DixonColesModel:
    """Maximum-likelihood Dixon-Coles fit on historical match scores.

    Each team gets an attack strength `alpha` and defense strength `beta`. A
    shared `gamma` is the log home-advantage and a shared `rho` is the low-score
    coupling. Both alpha and beta vectors are re-centered to mean 0 to remove
    the additive identifiability degeneracy.

    Args:
        xi: time-decay rate per day. weight = exp(-xi * days_back). The default
            of 0.0018 gives ~half weight after 1 year, ~quarter weight after 2.
            Set to 0 to disable decay.
        max_goals: truncation of the score grid when computing outcome
            probabilities. 10 covers >99.9% of mass for realistic lambdas.
    """

    def __init__(self, xi: float = 0.0018, max_goals: int = 10):
        self.xi = xi
        self.max_goals = max_goals
        self.teams: list[str] | None = None
        self.attack: dict[str, float] | None = None
        self.defense: dict[str, float] | None = None
        self.home_adv: float | None = None
        self.rho: float | None = None
        self._opt_result = None

    # --- internal: parameter packing/unpacking ---
    def _unpack(self, params: np.ndarray, n: int):
        alpha = params[:n]
        beta = params[n:2 * n]
        gamma = params[2 * n]
        rho = params[2 * n + 1]
        # Re-center for identifiability (additive shift to alpha/beta is unobservable)
        alpha = alpha - alpha.mean()
        beta = beta - beta.mean()
        return alpha, beta, gamma, rho

    # --- internal: vectorised negative log-likelihood + analytic gradient ---
    def _neg_log_lik_and_grad(self, params, home_idx, away_idx, hg, ag, weights, n):
        alpha, beta, gamma, rho = self._unpack(params, n)
        lam_h = np.exp(alpha[home_idx] - beta[away_idx] + gamma)
        lam_a = np.exp(alpha[away_idx] - beta[home_idx])

        # Poisson log-likelihood (drop constant log factorial terms)
        ll = hg * np.log(lam_h) - lam_h + ag * np.log(lam_a) - lam_a

        # Dixon-Coles tau adjustment on the four low-score cells
        is_00 = (hg == 0) & (ag == 0)
        is_01 = (hg == 0) & (ag == 1)
        is_10 = (hg == 1) & (ag == 0)
        is_11 = (hg == 1) & (ag == 1)

        tau = np.ones_like(lam_h)
        tau[is_00] = 1 - lam_h[is_00] * lam_a[is_00] * rho
        tau[is_01] = 1 + lam_h[is_01] * rho
        tau[is_10] = 1 + lam_a[is_10] * rho
        tau[is_11] = 1 - rho

        # If any tau goes non-positive the model is invalid; penalise heavily so
        # the optimiser walks away from that region.
        if (tau <= 0).any():
            return 1e10, np.zeros_like(params)

        ll = ll + np.log(tau)
        nll = -(weights * ll).sum()

        # --- Analytic gradient ---
        # Poisson part: d log L / d lam_h = hg/lam_h - 1, then chain via lam_h.
        # tau part: d log tau / d lam_{h,a} below; multiplied by lam_{h,a} (chain).
        dlogtau_dlamh = np.zeros_like(lam_h)
        dlogtau_dlama = np.zeros_like(lam_a)
        dlogtau_drho = np.zeros_like(lam_h)

        # tau cells (only the four low-score patterns contribute)
        dlogtau_dlamh[is_00] = -lam_a[is_00] * rho / tau[is_00]
        dlogtau_dlama[is_00] = -lam_h[is_00] * rho / tau[is_00]
        dlogtau_drho[is_00]  = -lam_h[is_00] * lam_a[is_00] / tau[is_00]

        dlogtau_dlamh[is_01] = rho / tau[is_01]
        dlogtau_drho[is_01]  = lam_h[is_01] / tau[is_01]

        dlogtau_dlama[is_10] = rho / tau[is_10]
        dlogtau_drho[is_10]  = lam_a[is_10] / tau[is_10]

        # (1,1): tau = 1 - rho, no lam dependence
        dlogtau_drho[is_11] = -1.0 / tau[is_11]

        # Per-match d log L / d (something) coefficients, after chain rule onto
        # log-rates (multiplied by lam since lam = exp(linear term)):
        coef_lh = (hg - lam_h) + dlogtau_dlamh * lam_h
        coef_la = (ag - lam_a) + dlogtau_dlama * lam_a

        # Scatter into per-team gradients of log L (centered alpha/beta).
        grad_alpha_c = np.zeros(n)
        grad_beta_c = np.zeros(n)
        np.add.at(grad_alpha_c, home_idx, weights * coef_lh)
        np.add.at(grad_alpha_c, away_idx, weights * coef_la)
        np.add.at(grad_beta_c, away_idx, -weights * coef_lh)
        np.add.at(grad_beta_c, home_idx, -weights * coef_la)

        # d NLL / d centered = -d log L / d centered
        grad_alpha_c = -grad_alpha_c
        grad_beta_c = -grad_beta_c

        # Centering chain rule: alpha_centered = alpha_raw - mean(alpha_raw)
        # => d NLL / d alpha_raw[i] = grad_alpha_c[i] - mean(grad_alpha_c)
        grad_alpha = grad_alpha_c - grad_alpha_c.mean()
        grad_beta = grad_beta_c - grad_beta_c.mean()

        grad_gamma = -np.sum(weights * coef_lh)  # only home rate depends on gamma
        grad_rho = -np.sum(weights * dlogtau_drho)

        grad = np.concatenate([grad_alpha, grad_beta, [grad_gamma, grad_rho]])
        return nll, grad

    def _neg_log_lik(self, params, home_idx, away_idx, hg, ag, weights, n):
        # Kept as a thin wrapper for finite-difference gradient checks.
        return self._neg_log_lik_and_grad(params, home_idx, away_idx, hg, ag, weights, n)[0]

    def fit(self, df: pd.DataFrame, ref_date: pd.Timestamp | None = None) -> "DixonColesModel":
        """Fit alpha, beta, gamma, rho on a DataFrame of past matches.

        Required columns: date, home_team, away_team, home_score, away_score.
        """
        df = df.dropna(subset=["home_score", "away_score"]).copy()
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)

        if ref_date is None:
            ref_date = df["date"].max()
        days_back = (ref_date - df["date"]).dt.days.to_numpy().astype(float)
        weights = np.exp(-self.xi * days_back) if self.xi > 0 else np.ones(len(df))

        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        team_idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        home_idx = df["home_team"].map(team_idx).to_numpy()
        away_idx = df["away_team"].map(team_idx).to_numpy()
        hg = df["home_score"].to_numpy()
        ag = df["away_score"].to_numpy()

        # Initial guess: zeros for alpha/beta, sensible defaults for gamma/rho.
        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.25       # gamma — moderate log home advantage
        x0[2 * n + 1] = -0.10  # rho — slight negative coupling

        result = minimize(
            self._neg_log_lik_and_grad, x0,
            args=(home_idx, away_idx, hg, ag, weights, n),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 500, "disp": False},
        )
        alpha, beta, gamma, rho = self._unpack(result.x, n)

        self.attack = dict(zip(self.teams, alpha))
        self.defense = dict(zip(self.teams, beta))
        self.home_adv = float(gamma)
        self.rho = float(rho)
        self._opt_result = result
        return self

    # --- inference ---
    def _rates(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        a_h = self.attack.get(home, 0.0)
        d_h = self.defense.get(home, 0.0)
        a_a = self.attack.get(away, 0.0)
        d_a = self.defense.get(away, 0.0)
        gamma = 0.0 if neutral else self.home_adv
        lam_h = float(np.exp(a_h - d_a + gamma))
        lam_a = float(np.exp(a_a - d_h))
        return lam_h, lam_a

    def score_matrix(self, home: str, away: str, neutral: bool = False) -> np.ndarray:
        """Return P(home_goals=x, away_goals=y) matrix up to `max_goals`."""
        lam_h, lam_a = self._rates(home, away, neutral)
        x = np.arange(self.max_goals + 1)
        px = poisson.pmf(x, lam_h)
        py = poisson.pmf(x, lam_a)
        m = np.outer(px, py)
        return _dc_correction_matrix(m, lam_h, lam_a, self.rho)

    def predict_match(self, home: str, away: str, neutral: bool = False) -> dict:
        """Return outcome probs, expected goals, and most likely scoreline.

        Outcome probs are renormalised to sum to 1 — without this, the
        max_goals truncation drops a tiny tail of mass.
        """
        lam_h, lam_a = self._rates(home, away, neutral)
        m = self.score_matrix(home, away, neutral)
        total = float(m.sum())
        p_home = float(np.tril(m, -1).sum()) / total
        p_draw = float(np.diag(m).sum()) / total
        p_away = float(np.triu(m, 1).sum()) / total
        most_likely = np.unravel_index(np.argmax(m), m.shape)
        return {
            "lam_home": lam_h, "lam_away": lam_a,
            "p_away_win": p_away, "p_draw": p_draw, "p_home_win": p_home,
            "most_likely_score": f"{most_likely[0]}-{most_likely[1]}",
            "most_likely_p": float(m[most_likely]) / total,
        }

    def predict_many(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Vectorised-ish prediction for a DataFrame of matches.

        Required columns: home_team, away_team, neutral.
        Adds columns: p_away_win, p_draw, p_home_win, lam_home, lam_away.
        """
        out = matches.copy()
        rows = [self.predict_match(r["home_team"], r["away_team"], bool(r.get("neutral", False)))
                for _, r in matches.iterrows()]
        out[["p_away_win", "p_draw", "p_home_win", "lam_home", "lam_away"]] = pd.DataFrame(
            [[r["p_away_win"], r["p_draw"], r["p_home_win"], r["lam_home"], r["lam_away"]] for r in rows],
            index=out.index,
        )
        return out

    def ratings_table(self) -> pd.DataFrame:
        """Compact view of fitted parameters per team."""
        return (
            pd.DataFrame({
                "team": self.teams,
                "attack": [self.attack[t] for t in self.teams],
                "defense": [self.defense[t] for t in self.teams],
            })
            .assign(total=lambda d: d["attack"] + d["defense"])
            .sort_values("total", ascending=False)
            .reset_index(drop=True)
        )
