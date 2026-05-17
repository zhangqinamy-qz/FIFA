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
                            rng: np.random.Generator | None = None,
                            outcome_model=None) -> str:
    """Return the winner's name. Plays 90 + (ET if drawn) + (50/50 PKs if still drawn).

    If `outcome_model` is provided (callable: (home, away) -> (p_away, p_draw, p_home)),
    use it for the 90-minute outcome instead of sampling from DC's score matrix. ET
    and PKs still use DC for goal-rate sampling. This lets you plug an ensemble
    classifier into the bracket while keeping DC for the goal-level dynamics.
    """
    rng = rng or np.random.default_rng()
    if outcome_model is not None:
        p_away, p_draw, p_home = outcome_model(home, away)
        idx = rng.choice(3, p=[p_away, p_draw, p_home])
        if idx == 2:
            return home
        if idx == 0:
            return away
        # idx == 1 → draw, fall through to ET
    else:
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


# --- 48-team format (WC 2026, official KO template per FIFA December 2025 draw) ---
# 12 groups × 4 teams (A-L). Top 2 per group + 8 best 3rd-placed → R32.
# R32 match numbers 73-88. Slot encoding:
#   ('A', 0) = Group A winner (1st), ('A', 1) = Group A runner-up (2nd)
#   ('T', None) = a best-third slot (assigned post-group stage per FIFA's
#                 third-place combination table; we approximate by ranking the
#                 8 best thirds and assigning them in T-slot order).
KO_PAIRS_48 = [
    (('A', 1), ('B', 1)),       # M73: 2A vs 2B
    (('E', 0), ('T', None)),    # M74: 1E vs best third (from A/B/C/D/F)
    (('F', 0), ('C', 1)),       # M75: 1F vs 2C
    (('C', 0), ('F', 1)),       # M76: 1C vs 2F
    (('I', 0), ('T', None)),    # M77: 1I vs best third (from C/D/F/G/H)
    (('E', 1), ('I', 1)),       # M78: 2E vs 2I
    (('A', 0), ('T', None)),    # M79: 1A vs best third (from C/E/F/H/I)
    (('L', 0), ('T', None)),    # M80: 1L vs best third (from E/H/I/J/K)
    (('D', 0), ('T', None)),    # M81: 1D vs best third (from B/E/F/I/J)
    (('G', 0), ('T', None)),    # M82: 1G vs best third (from A/E/H/I/J)
    (('K', 1), ('L', 1)),       # M83: 2K vs 2L
    (('H', 0), ('J', 1)),       # M84: 1H vs 2J
    (('B', 0), ('T', None)),    # M85: 1B vs best third (from E/F/G/I/J)
    (('J', 0), ('H', 1)),       # M86: 1J vs 2H
    (('K', 0), ('T', None)),    # M87: 1K vs best third (from D/E/I/J/L)
    (('D', 1), ('G', 1)),       # M88: 2D vs 2G
]

# R16 pairings (M89-M96): (idx_of_R32_match_a, idx_of_R32_match_b) — indices into KO_PAIRS_48 above.
R16_PAIRS_48 = [
    (1, 4),    # M89: W74 vs W77
    (0, 2),    # M90: W73 vs W75
    (3, 5),    # M91: W76 vs W78
    (6, 7),    # M92: W79 vs W80
    (10, 11),  # M93: W83 vs W84
    (8, 9),    # M94: W81 vs W82
    (13, 15),  # M95: W86 vs W88
    (12, 14),  # M96: W85 vs W87
]

# QF pairings (M97-M100): indices into R16_PAIRS_48
QF_PAIRS_48 = [
    (0, 1),    # M97: W89 vs W90
    (4, 5),    # M98: W93 vs W94
    (2, 3),    # M99: W91 vs W92
    (6, 7),    # M100: W95 vs W96
]

# SF pairings (M101-M102): indices into QF_PAIRS_48
SF_PAIRS_48 = [
    (0, 1),    # M101: W97 vs W98
    (2, 3),    # M102: W99 vs W100
]

# Match-number labels for the bracket viz
R32_MATCH_NUMBERS = list(range(73, 89))   # 73..88
R16_MATCH_NUMBERS = list(range(89, 97))   # 89..96
QF_MATCH_NUMBERS  = list(range(97, 101))  # 97..100
SF_MATCH_NUMBERS  = [101, 102]
THIRD_PLACE_MATCH = 103
FINAL_MATCH       = 104


def _select_best_thirds(third_place_records: list[dict], k: int = 8) -> list[str]:
    """Pick the k best 3rd-placed teams across all groups.

    `third_place_records` is a list of dicts with keys: team, pts, gd, gf, rnd.
    Same tiebreaker chain as group sorting: pts → gd → gf → random.
    """
    ranked = sorted(third_place_records,
                    key=lambda r: (-r['pts'], -r['gd'], -r['gf'], r['rnd']))
    return [r['team'] for r in ranked[:k]]


def simulate_group_with_stats(dc: DixonColesModel, teams: list[str], n_sims: int,
                               rng: np.random.Generator) -> tuple[GroupResult, dict[str, dict]]:
    """Like simulate_group but also tracks avg pts/gd/gf per (team, finishing position).

    Returns (group_result, per_position_stats) where per_position_stats[(team, pos)]
    has keys 'pts', 'gd', 'gf' (averaged across sims where the team finished `pos`).
    These stats feed the best-thirds tiebreaker in the 48-team sim.
    """
    assert len(teams) == 4
    pairs = list(combinations(teams, 2))
    counts = {t: np.zeros(4, dtype=int) for t in teams}
    pos_stats = {(t, pos): [] for t in teams for pos in range(4)}

    for _ in range(n_sims):
        scores = {}
        for h, a in pairs:
            scores[(h, a)] = sample_score(dc, h, a, neutral=True, rng=rng)
        ranked = _group_standings(scores, teams, rng)
        # Compute pts/gd/gf for each team
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
            pos = ranked.index(t)
            counts[t][pos] += 1
            pos_stats[(t, pos)].append({'pts': pts, 'gd': gf - ga, 'gf': gf})

    finish_probs = {t: counts[t] / n_sims for t in teams}
    # Aggregate pos_stats to mean per (team, pos)
    avg_stats = {}
    for (t, pos), records in pos_stats.items():
        if records:
            avg_stats[(t, pos)] = {
                'pts': np.mean([r['pts'] for r in records]),
                'gd':  np.mean([r['gd']  for r in records]),
                'gf':  np.mean([r['gf']  for r in records]),
            }
    return GroupResult(teams=teams, finish_probs=finish_probs), avg_stats


def simulate_tournament_48(groups: dict[str, list[str]], dc: DixonColesModel,
                            n_sims: int = 5000, n_group_sims: int = 2000,
                            seed: int | None = 0,
                            outcome_model=None,
                            return_bracket: bool = False):
    """48-team WC 2026 simulator. 12 groups × 4, top 2 + 8 best thirds → R32.

    Args:
        groups: 12 groups labeled 'A'..'L', each with 4 teams.
        outcome_model: optional callable (home, away) -> (p_away, p_draw, p_home).
            If provided, used for knockout 90-minute outcomes. Group stage still
            uses DC for score sampling (needed for GD/GF tiebreakers).
        return_bracket: if True, returns (summary_df, bracket_data). The
            bracket_data dict has per-slot team frequencies across all sims so
            the caller can compute the "most likely path" visualization.
    """
    from collections import defaultdict
    assert len(groups) == 12, f"48-team format needs 12 groups, got {len(groups)}"
    rng = np.random.default_rng(seed)
    all_teams = [t for ts in groups.values() for t in ts]
    missing = [t for t in all_teams if t not in dc.attack]
    if missing:
        raise ValueError(f"DC model missing ratings for: {missing}")

    # Per-group: finish-position distribution + per-position avg pts/gd/gf
    group_results: dict[str, GroupResult] = {}
    group_stats: dict[str, dict] = {}
    for g, teams in groups.items():
        gr, st = simulate_group_with_stats(dc, teams, n_group_sims, rng)
        group_results[g] = gr
        group_stats[g] = st

    counts = {t: {'R32': 0, 'R16': 0, 'QF': 0, 'SF': 0, 'final': 0, 'winner': 0} for t in all_teams}
    # Per-match-slot occupancy. Keys: match number (73..104), values: dict with
    # 'team_a', 'team_b', 'winner' → defaultdict[team] = count.
    match_occ: dict[int, dict[str, dict]] = {
        m: {'team_a': defaultdict(int), 'team_b': defaultdict(int), 'winner': defaultdict(int)}
        for m in list(range(73, 89)) + list(range(89, 97)) + list(range(97, 101)) + [101, 102, 104]
    }

    for _ in range(n_sims):
        # One realised standings per group
        ordering = {g: _draw_group_outcome(g, group_results[g], rng)
                    for g in groups}

        # Collect 3rd-placed teams + their stats
        third_records = []
        for g in groups:
            t3 = ordering[g][2]
            stats = group_stats[g].get((t3, 2))
            if stats is None:
                stats = {'pts': 0, 'gd': 0, 'gf': 0}
            third_records.append({'team': t3, **stats, 'rnd': rng.random()})
        best_thirds = _select_best_thirds(third_records, k=8)

        # Build slot map for group-positioned slots
        slot_map = {}
        for g, std in ordering.items():
            for pos in range(4):
                slot_map[(g, pos)] = std[pos]

        # R32 — assign best thirds to ('T', None) slots in order of appearance
        r32_winners = []
        third_idx = 0
        for k, (sa, sb) in enumerate(KO_PAIRS_48):
            t1 = best_thirds[third_idx] if sa == ('T', None) else slot_map[sa]
            if sa == ('T', None):
                third_idx += 1
            t2 = best_thirds[third_idx] if sb == ('T', None) else slot_map[sb]
            if sb == ('T', None):
                third_idx += 1
            for t in (t1, t2):
                counts[t]['R32'] += 1
            w = simulate_knockout_match(dc, t1, t2, neutral=True, rng=rng, outcome_model=outcome_model)
            r32_winners.append(w)
            m_no = R32_MATCH_NUMBERS[k]
            match_occ[m_no]['team_a'][t1] += 1
            match_occ[m_no]['team_b'][t2] += 1
            match_occ[m_no]['winner'][w]  += 1

        # R16 — official pairing per Wikipedia/FIFA template
        r16_winners = []
        for k, (a_idx, b_idx) in enumerate(R16_PAIRS_48):
            t1, t2 = r32_winners[a_idx], r32_winners[b_idx]
            for t in (t1, t2):
                counts[t]['R16'] += 1
            w = simulate_knockout_match(dc, t1, t2, neutral=True, rng=rng, outcome_model=outcome_model)
            r16_winners.append(w)
            m_no = R16_MATCH_NUMBERS[k]
            match_occ[m_no]['team_a'][t1] += 1
            match_occ[m_no]['team_b'][t2] += 1
            match_occ[m_no]['winner'][w]  += 1

        # QF — official pairing
        qf_winners = []
        for k, (a_idx, b_idx) in enumerate(QF_PAIRS_48):
            t1, t2 = r16_winners[a_idx], r16_winners[b_idx]
            for t in (t1, t2):
                counts[t]['QF'] += 1
            w = simulate_knockout_match(dc, t1, t2, neutral=True, rng=rng, outcome_model=outcome_model)
            qf_winners.append(w)
            m_no = QF_MATCH_NUMBERS[k]
            match_occ[m_no]['team_a'][t1] += 1
            match_occ[m_no]['team_b'][t2] += 1
            match_occ[m_no]['winner'][w]  += 1

        # SF — official pairing
        sf_winners = []
        for k, (a_idx, b_idx) in enumerate(SF_PAIRS_48):
            t1, t2 = qf_winners[a_idx], qf_winners[b_idx]
            for t in (t1, t2):
                counts[t]['SF'] += 1
            w = simulate_knockout_match(dc, t1, t2, neutral=True, rng=rng, outcome_model=outcome_model)
            sf_winners.append(w)
            m_no = SF_MATCH_NUMBERS[k]
            match_occ[m_no]['team_a'][t1] += 1
            match_occ[m_no]['team_b'][t2] += 1
            match_occ[m_no]['winner'][w]  += 1

        # Final
        for t in sf_winners:
            counts[t]['final'] += 1
        winner = simulate_knockout_match(dc, sf_winners[0], sf_winners[1], neutral=True, rng=rng, outcome_model=outcome_model)
        counts[winner]['winner'] += 1
        match_occ[FINAL_MATCH]['team_a'][sf_winners[0]] += 1
        match_occ[FINAL_MATCH]['team_b'][sf_winners[1]] += 1
        match_occ[FINAL_MATCH]['winner'][winner]        += 1

    rows = []
    team_to_group = {t: g for g, ts in groups.items() for t in ts}
    for t in all_teams:
        c = counts[t]
        rows.append({
            'team': t,
            'group': team_to_group[t],
            'p_R32': c['R32'] / n_sims,
            'p_R16': c['R16'] / n_sims,
            'p_QF':  c['QF']  / n_sims,
            'p_SF':  c['SF']  / n_sims,
            'p_final': c['final'] / n_sims,
            'p_winner': c['winner'] / n_sims,
        })
    summary_df = (pd.DataFrame(rows)
                    .sort_values('p_winner', ascending=False)
                    .reset_index(drop=True))

    if not return_bracket:
        return summary_df

    # Build bracket_data: per match, the most-likely team on each side + the
    # full probability distribution over occupants. Match wirings (R16 inputs
    # come from R32 winners etc.) included for the front-end to render the tree.
    bracket = {'n_sims': n_sims, 'matches': [], 'wiring': {
        'r32': [(R32_MATCH_NUMBERS[i], list(p)) for i, p in enumerate(
            [(f'{sa[0]}{sa[1]+1}' if sa != ('T', None) else '3rd',
              f'{sb[0]}{sb[1]+1}' if sb != ('T', None) else '3rd') for sa, sb in KO_PAIRS_48])],
        'r16': [(R16_MATCH_NUMBERS[i], R32_MATCH_NUMBERS[a], R32_MATCH_NUMBERS[b])
                for i, (a, b) in enumerate(R16_PAIRS_48)],
        'qf':  [(QF_MATCH_NUMBERS[i],  R16_MATCH_NUMBERS[a], R16_MATCH_NUMBERS[b])
                for i, (a, b) in enumerate(QF_PAIRS_48)],
        'sf':  [(SF_MATCH_NUMBERS[i],  QF_MATCH_NUMBERS[a],  QF_MATCH_NUMBERS[b])
                for i, (a, b) in enumerate(SF_PAIRS_48)],
        'final': (FINAL_MATCH, SF_MATCH_NUMBERS[0], SF_MATCH_NUMBERS[1]),
    }}
    for m_no, slots in match_occ.items():
        def topk(d, k=4):
            return sorted(({'team': t, 'p': c / n_sims} for t, c in d.items()),
                          key=lambda x: -x['p'])[:k]
        bracket['matches'].append({
            'match_no': m_no,
            'team_a_top': topk(slots['team_a']),
            'team_b_top': topk(slots['team_b']),
            'winner_top': topk(slots['winner']),
            # Modal (most-likely) entries for the bracket diagram
            'mode_a':  max(slots['team_a'].items(), key=lambda x: x[1])[0] if slots['team_a'] else None,
            'mode_b':  max(slots['team_b'].items(), key=lambda x: x[1])[0] if slots['team_b'] else None,
            'mode_w':  max(slots['winner'].items(), key=lambda x: x[1])[0] if slots['winner'] else None,
            'p_mode_w': max(slots['winner'].values()) / n_sims if slots['winner'] else 0.0,
        })
    bracket['matches'].sort(key=lambda m: m['match_no'])
    return summary_df, bracket


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


# --- WC 2026 group draw (official, December 2025) ---

WC_2026_GROUPS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}
