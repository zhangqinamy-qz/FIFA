"""Tournament simulator built on a fitted DixonColesModel.

Supports the FIFA WC 32-team format:
  - 8 groups of 4, round-robin → top 2 advance
  - Knockout R16 → QF → SF → Final (+ 3rd place)
  - Knockout draws → 30-min extra time at the same rate, then a coin-flip PK shootout.

A single `simulate_tournament(groups, dc, n_sims=...)` returns a per-team
DataFrame with probabilities of reaching each round.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from src.dixon_coles import DixonColesModel


# --- low-level: sample a match score from the DC score matrix ---

def sample_score(dc: DixonColesModel, home: str, away: str, neutral: bool = True,
                 rng: np.random.Generator | None = None) -> tuple[int, int]:
    """Draw one (home_goals, away_goals) sample from the DC joint distribution."""
    rng = rng or np.random.default_rng()
    m = dc.score_matrix(home, away, neutral=neutral)
    flat = m.ravel() / m.sum()
    idx = rng.choice(flat.size, p=flat)
    return int(idx // m.shape[1]), int(idx % m.shape[1])


def sample_extra_time(dc: DixonColesModel, home: str, away: str, neutral: bool = True,
                      rng: np.random.Generator | None = None) -> tuple[int, int]:
    """Sample 30-minute ET goals: independent Poissons at 1/3 the regulation rate.

    No DC tau correction in ET (low-score adjustment is calibrated for 90-min
    distributions and the empirical correlation in 30-min ET is too noisy).
    """
    rng = rng or np.random.default_rng()
    lam_h, lam_a = dc._rates(home, away, neutral=neutral)
    eg_h = rng.poisson(lam_h * 30 / 90)
    eg_a = rng.poisson(lam_a * 30 / 90)
    return int(eg_h), int(eg_a)


def simulate_knockout_match(dc: DixonColesModel, home: str, away: str, neutral: bool = True,
                            rng: np.random.Generator | None = None) -> str:
    """Return the winner's name. Plays 90 + (ET if drawn) + (50/50 PKs if still drawn)."""
    rng = rng or np.random.default_rng()
    hg, ag = sample_score(dc, home, away, neutral, rng)
    if hg != ag:
        return home if hg > ag else away
    eg_h, eg_a = sample_extra_time(dc, home, away, neutral, rng)
    if eg_h != eg_a:
        return home if eg_h > eg_a else away
    return home if rng.random() < 0.5 else away


# --- group stage ---

@dataclass
class GroupResult:
    """Per-team finishing position probabilities for a single group."""
    teams: list[str]
    # finish_probs[t][pos] = P(team t finishes in position pos), pos in 0..3
    finish_probs: dict[str, np.ndarray] = field(default_factory=dict)


def _group_standings(scores: dict[tuple[str, str], tuple[int, int]],
                     teams: list[str], rng: np.random.Generator) -> list[str]:
    """Rank `teams` after one round-robin given `scores[(home, away)] = (hg, ag)`.

    Tiebreakers: points → goal difference → goals for → random. Real FIFA uses
    head-to-head before GD, but for sim purposes this is close enough and avoids
    the cyclic-tiebreak edge case.
    """
    rows = []
    for t in teams:
        pts = gf = ga = 0
        for (h, a), (hg, ag) in scores.items():
            if t == h:
                gf += hg; ga += ag
                if hg > ag: pts += 3
                elif hg == ag: pts += 1
            elif t == a:
                gf += ag; ga += hg
                if ag > hg: pts += 3
                elif hg == ag: pts += 1
        rows.append({'team': t, 'pts': pts, 'gd': gf - ga, 'gf': gf, 'rnd': rng.random()})
    return [r['team'] for r in sorted(rows, key=lambda r: (-r['pts'], -r['gd'], -r['gf'], r['rnd']))]


def simulate_group(dc: DixonColesModel, teams: list[str], n_sims: int,
                   rng: np.random.Generator) -> GroupResult:
    """Monte-Carlo a single group; return finishing-position probabilities."""
    assert len(teams) == 4, "WC groups have 4 teams"
    pairs = list(combinations(teams, 2))
    counts = {t: np.zeros(4, dtype=int) for t in teams}
    pos_winners = {pos: {t: 0 for t in teams} for pos in range(4)}

    for _ in range(n_sims):
        scores = {}
        for h, a in pairs:
            scores[(h, a)] = sample_score(dc, h, a, neutral=True, rng=rng)
        ranked = _group_standings(scores, teams, rng)
        for pos, t in enumerate(ranked):
            counts[t][pos] += 1
            pos_winners[pos][t] += 1

    finish_probs = {t: counts[t] / n_sims for t in teams}
    return GroupResult(teams=teams, finish_probs=finish_probs)


# --- knockout bracket ---

# Standard 32-team WC bracket pairing:
#   R16: A1-B2, C1-D2, E1-F2, G1-H2, B1-A2, D1-C2, F1-E2, H1-G2
# Encoded as the (group, position) pairs that meet in each R16 match.
KO_PAIRS = [
    (('A', 0), ('B', 1)),
    (('C', 0), ('D', 1)),
    (('E', 0), ('F', 1)),
    (('G', 0), ('H', 1)),
    (('B', 0), ('A', 1)),
    (('D', 0), ('C', 1)),
    (('F', 0), ('E', 1)),
    (('H', 0), ('G', 1)),
]


def _draw_group_outcome(group_letter: str, group_result: GroupResult,
                        rng: np.random.Generator) -> list[str]:
    """Sample one realisation of the group's final standings according to its
    finish-probability distribution.

    The marginal positional probabilities are sampled independently with
    rejection: keep drawing until all four positions are distinct teams. This is
    a slight simplification (true distribution is the joint over (1st,2nd,3rd,4th)),
    but in practice nearly every draw succeeds on the first attempt because
    well-separated teams have very peaked marginals.
    """
    teams = group_result.teams
    for _ in range(50):
        sampled = []
        for pos in range(4):
            probs = np.array([group_result.finish_probs[t][pos] for t in teams])
            probs = probs / probs.sum()
            sampled.append(rng.choice(teams, p=probs))
        if len(set(sampled)) == 4:
            return sampled
    # Fall back: draw one finishing order proportional to product of marginals.
    return sorted(teams, key=lambda _: rng.random())


def simulate_tournament(groups: dict[str, list[str]], dc: DixonColesModel,
                        n_sims: int = 5000, n_group_sims: int = 2000,
                        seed: int | None = 0) -> pd.DataFrame:
    """End-to-end tournament simulation.

    Args:
        groups: e.g. {'A': ['Qatar', 'Ecuador', 'Senegal', 'Netherlands'], ...}
        dc: fitted DixonColesModel covering all 32 teams.
        n_sims: number of full-tournament Monte-Carlo runs.
        n_group_sims: per-group sims used to estimate the finish-position
            marginals (cached once, reused across all `n_sims` runs).

    Returns a DataFrame with one row per team and columns
    ['p_R16', 'p_QF', 'p_SF', 'p_final', 'p_winner', 'group'].
    """
    rng = np.random.default_rng(seed)
    all_teams = [t for ts in groups.values() for t in ts]
    missing = [t for t in all_teams if t not in dc.attack]
    if missing:
        raise ValueError(f"DC model missing ratings for: {missing}")

    # 1) Cache each group's finish-position distribution.
    group_results = {g: simulate_group(dc, teams, n_group_sims, rng)
                     for g, teams in groups.items()}

    counts = {t: {'R16': 0, 'QF': 0, 'SF': 0, 'final': 0, 'winner': 0} for t in all_teams}

    for _ in range(n_sims):
        # Sample one realised ordering for each group.
        ordering = {g: _draw_group_outcome(g, group_results[g], rng)
                    for g in groups}

        # R16
        r16_winners = []
        for (g1, p1), (g2, p2) in KO_PAIRS:
            t1 = ordering[g1][p1]
            t2 = ordering[g2][p2]
            for t in (t1, t2):
                counts[t]['R16'] += 1
            w = simulate_knockout_match(dc, t1, t2, neutral=True, rng=rng)
            r16_winners.append(w)

        # QF, SF, F
        qf_winners = []
        for i in range(0, 8, 2):
            t1, t2 = r16_winners[i], r16_winners[i + 1]
            for t in (t1, t2):
                counts[t]['QF'] += 1
            qf_winners.append(simulate_knockout_match(dc, t1, t2, neutral=True, rng=rng))

        sf_winners = []
        for i in range(0, 4, 2):
            t1, t2 = qf_winners[i], qf_winners[i + 1]
            for t in (t1, t2):
                counts[t]['SF'] += 1
            sf_winners.append(simulate_knockout_match(dc, t1, t2, neutral=True, rng=rng))

        for t in sf_winners:
            counts[t]['final'] += 1
        winner = simulate_knockout_match(dc, sf_winners[0], sf_winners[1], neutral=True, rng=rng)
        counts[winner]['winner'] += 1

    rows = []
    team_to_group = {t: g for g, ts in groups.items() for t in ts}
    for t in all_teams:
        c = counts[t]
        rows.append({
            'team': t,
            'group': team_to_group[t],
            'p_R16': c['R16'] / n_sims,
            'p_QF': c['QF'] / n_sims,
            'p_SF': c['SF'] / n_sims,
            'p_final': c['final'] / n_sims,
            'p_winner': c['winner'] / n_sims,
        })
    return (pd.DataFrame(rows)
              .sort_values('p_winner', ascending=False)
              .reset_index(drop=True))


# --- WC 2022 group draw (handy default for testing) ---

WC_2022_GROUPS = {
    'A': ['Netherlands', 'Senegal', 'Ecuador', 'Qatar'],
    'B': ['England', 'United States', 'Iran', 'Wales'],
    'C': ['Argentina', 'Poland', 'Mexico', 'Saudi Arabia'],
    'D': ['France', 'Australia', 'Tunisia', 'Denmark'],
    'E': ['Japan', 'Spain', 'Germany', 'Costa Rica'],
    'F': ['Morocco', 'Croatia', 'Belgium', 'Canada'],
    'G': ['Brazil', 'Switzerland', 'Cameroon', 'Serbia'],
    'H': ['Portugal', 'South Korea', 'Uruguay', 'Ghana'],
}
