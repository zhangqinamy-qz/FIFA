import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from src.elo import EloModel

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
