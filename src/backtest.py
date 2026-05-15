import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from src.elo import EloModel
from src.draw_model import DrawModel, match_outcome
from src.dixon_coles import DixonColesModel

def walk_forward_wc(df: pd.DataFrame, wc_years: list[int], **elo_kwargs) -> pd.DataFrame:
    records = []
    for year in wc_years:
        train = df[df['date'].dt.year < year]
        test = df[(df['date'].dt.year == year) & (df['tournament'] == 'FIFA World Cup')]

        if test.empty:
            continue

        model = EloModel(**elo_kwargs)
        model.fit(train)

        preds = []
        for _, match in test.iterrows():
            home, away = match['home_team'], match['away_team']
            r_home = model._get(home) + (0 if match.get('neutral', True) else model.home_advantage)
            r_away = model._get(away)
            p = model._expected(r_home, r_away)
            actual = 1 if match['home_score'] > match['away_score'] else 0
            preds.append({'home_win_prob': p, 'home_win': actual})

        preds_df = pd.DataFrame(preds)
        y_true = preds_df['home_win']
        y_prob = preds_df['home_win_prob'].clip(1e-6, 1 - 1e-6)

        records.append({
            'year': year,
            'n_matches': len(preds_df),
            'log_loss': log_loss(y_true, y_prob),
            'accuracy': (preds_df['home_win_prob'].round() == y_true).mean(),
            'predictions': preds_df,
        })

    summary = pd.DataFrame([{k: v for k, v in r.items() if k != 'predictions'} for r in records])
    summary['raw'] = [r['predictions'] for r in records]
    return summary


def walk_forward_wc_3way(
    df: pd.DataFrame,
    wc_years: list[int],
    *,
    draw_train_min_year: int = 1990,
    **elo_kwargs,
) -> pd.DataFrame:
    """3-outcome (away/draw/home) walk-forward backtest.

    For each WC year:
      1. Train Elo on all competitive matches before `year`.
      2. Fit a multinomial-logit DrawModel on the same matches (filtered to
         >= `draw_train_min_year` so the warmup Elo noise is excluded).
      3. Predict each WC match as (P_away, P_draw, P_home).
      4. Score 3-class log loss and argmax accuracy.

    Returns a summary DataFrame with one row per year and a `raw` column holding
    the per-match prediction DataFrames.
    """
    records = []
    home_advantage = elo_kwargs.get('home_advantage', 100)

    for year in wc_years:
        train = df[df['date'].dt.year < year]
        test = df[(df['date'].dt.year == year) & (df['tournament'] == 'FIFA World Cup')]
        if test.empty:
            continue

        model = EloModel(**elo_kwargs)
        train_enriched = model.fit(train)

        # Fit the draw model on Elo-enriched training matches (skip cold-start years)
        train_for_draw = train_enriched[
            train_enriched['date'].dt.year >= draw_train_min_year
        ].copy()
        draw_model = DrawModel(home_advantage=home_advantage)
        draw_model.fit(train_for_draw)

        preds = []
        for _, match in test.iterrows():
            home, away = match['home_team'], match['away_team']
            neutral = bool(match.get('neutral', True))
            r_home = model._get(home)
            r_away = model._get(away)
            proba = draw_model.predict_proba(
                np.array([r_home]), np.array([r_away]), np.array([neutral])
            )[0]
            preds.append({
                'date': match['date'],
                'home_team': home,
                'away_team': away,
                'neutral': neutral,
                'home_elo': r_home,
                'away_elo': r_away,
                'p_away': proba[0],
                'p_draw': proba[1],
                'p_home': proba[2],
                'outcome': match_outcome(match['home_score'], match['away_score']),
            })

        preds_df = pd.DataFrame(preds)
        y_true = preds_df['outcome'].to_numpy()
        y_proba = preds_df[['p_away', 'p_draw', 'p_home']].to_numpy()
        y_proba = np.clip(y_proba, 1e-6, 1 - 1e-6)
        y_pred = y_proba.argmax(axis=1)

        records.append({
            'year': year,
            'n_matches': len(preds_df),
            'log_loss': log_loss(y_true, y_proba, labels=[0, 1, 2]),
            'accuracy': float((y_pred == y_true).mean()),
            'predictions': preds_df,
        })

    summary = pd.DataFrame([{k: v for k, v in r.items() if k != 'predictions'} for r in records])
    summary['raw'] = [r['predictions'] for r in records]
    return summary


def walk_forward_wc_dc(
    df: pd.DataFrame,
    wc_years: list[int],
    *,
    xi: float = 0.0018,
    train_min_year: int = 1990,
    max_goals: int = 10,
) -> pd.DataFrame:
    """3-outcome walk-forward backtest using the Dixon-Coles bivariate Poisson.

    Mirrors `walk_forward_wc_3way` but fits a `DixonColesModel` on raw scores
    instead of an Elo + multinomial-logit pipeline. Each WC year:
      1. Train DC on competitive matches in [train_min_year, year).
      2. For each WC match, compute (P_away, P_draw, P_home) from the score
         matrix. WC matches are treated as neutral by default (matches the data
         convention: `neutral` defaults to True in `walk_forward_wc_3way`).
      3. Score 3-class log loss and argmax accuracy.

    Args:
        xi: time-decay rate (per day) for the DC fit. Default 0.0018 ≈ 1y half-life.
        train_min_year: skip very-old matches; combined with xi this controls
            both the training cost and the effective sample.
        max_goals: score-grid truncation for outcome probability integration.
    """
    records = []
    for year in wc_years:
        train = df[(df['date'].dt.year >= train_min_year) & (df['date'].dt.year < year)]
        test = df[(df['date'].dt.year == year) & (df['tournament'] == 'FIFA World Cup')]
        if test.empty:
            continue

        model = DixonColesModel(xi=xi, max_goals=max_goals)
        ref_date = pd.Timestamp(f'{year}-01-01')
        model.fit(train, ref_date=ref_date)

        preds = []
        for _, match in test.iterrows():
            home, away = match['home_team'], match['away_team']
            neutral = bool(match.get('neutral', True))
            p = model.predict_match(home, away, neutral=neutral)
            preds.append({
                'date': match['date'],
                'home_team': home,
                'away_team': away,
                'neutral': neutral,
                'lam_home': p['lam_home'],
                'lam_away': p['lam_away'],
                'p_away': p['p_away_win'],
                'p_draw': p['p_draw'],
                'p_home': p['p_home_win'],
                'most_likely_score': p['most_likely_score'],
                'outcome': match_outcome(match['home_score'], match['away_score']),
            })

        preds_df = pd.DataFrame(preds)
        y_true = preds_df['outcome'].to_numpy()
        y_proba = preds_df[['p_away', 'p_draw', 'p_home']].to_numpy()
        y_proba = np.clip(y_proba, 1e-6, 1 - 1e-6)
        y_pred = y_proba.argmax(axis=1)

        records.append({
            'year': year,
            'n_matches': len(preds_df),
            'log_loss': log_loss(y_true, y_proba, labels=[0, 1, 2]),
            'accuracy': float((y_pred == y_true).mean()),
            'predictions': preds_df,
        })

    summary = pd.DataFrame([{k: v for k, v in r.items() if k != 'predictions'} for r in records])
    summary['raw'] = [r['predictions'] for r in records]
    return summary
