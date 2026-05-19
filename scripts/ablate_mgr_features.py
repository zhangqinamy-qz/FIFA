"""Ablate individual Mgr+Big5 sub-features to find which one hurts WC 2018.

Hypothesis: career_wr is elite-team-correlated (already in Elo) and adds noise.
Career_matches + tenure are independent signals worth keeping.
"""
import sys, warnings
sys.path.append('.')
warnings.filterwarnings('ignore')

import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from catboost import CatBoostClassifier

from src.elo import EloModel
from src.draw_model import match_outcome
from src.regression_dc import RegressionDixonColesModel
from src.manager_api import attach_manager_features_to_training
from src.big5_api import attach_big5_features_to_training


HA, FLOOR = 100, 1e5

df_all = pd.read_csv('data/processed/matches_competitive.csv', parse_dates=['date']).dropna(subset=['home_score','away_score'])
df_sv  = pd.read_csv('data/processed/matches_with_squad_value.csv', parse_dates=['date'])
elo = EloModel()
elo_enriched = elo.fit(df_all)
df = df_sv.merge(elo_enriched[['date','home_team','away_team','home_elo_pre','away_elo_pre']],
                 on=['date','home_team','away_team'], how='left').dropna(subset=['home_elo_pre','away_elo_pre'])
df['neutral'] = df['neutral'].fillna(False)
df['elo_diff_norm'] = ((df['home_elo_pre'] + (~df['neutral']).astype(int)*HA) - df['away_elo_pre'])/400.0
df['log_value_diff']    = np.log10(df['home_top_n_value_eur'].clip(lower=FLOOR)/df['away_top_n_value_eur'].clip(lower=FLOOR))
df['outfield_age_diff'] = df['home_outfield_mean_age'] - df['away_outfield_mean_age']
df['top1_share_diff']   = df['home_top1_share'] - df['away_top1_share']
df['fifa_points_diff']  = (df['home_fifa_points'].fillna(0) - df['away_fifa_points'].fillna(0))/1000.0
df['home_elo']  = df['home_elo_pre']/100.0
df['away_elo']  = df['away_elo_pre']/100.0
df['home_logv'] = np.log10(df['home_top_n_value_eur'].clip(lower=FLOOR))
df['away_logv'] = np.log10(df['away_top_n_value_eur'].clip(lower=FLOOR))
df['home_fpts'] = df['home_fifa_points'].fillna(0)/1000.0
df['away_fpts'] = df['away_fifa_points'].fillna(0)/1000.0
df['outcome']   = df.apply(lambda r: match_outcome(r['home_score'], r['away_score']), axis=1)

df = attach_manager_features_to_training(df)
df = attach_big5_features_to_training(df)
df['diff_mgr_career_matches_n'] = df['diff_mgr_career_matches'] / 30.0
df['diff_mgr_career_wr_n']      = df['diff_mgr_career_wr']
df['diff_mgr_tenure_days_n']    = df['diff_mgr_tenure_days'] / 365.0

FEAT = ['elo_diff_norm','log_value_diff','outfield_age_diff','top1_share_diff','fifa_points_diff']
RDC_FEAT = ['elo','logv','fpts']
EXTRAS_ALL = ['diff_big5_share','diff_mgr_career_matches_n','diff_mgr_career_wr_n','diff_mgr_tenure_days_n']

needed = FEAT + [f'{s}_{c}' for s in ('home','away') for c in RDC_FEAT] + ['outcome']
valid = df.dropna(subset=needed)
valid = valid[(valid['home_top_n_value_eur']>FLOOR)&(valid['away_top_n_value_eur']>FLOOR)].copy()
for c in EXTRAS_ALL:
    valid[c] = valid[c].fillna(0)


def score(y, p):
    p = np.clip(p, 1e-6, 1-1e-6)
    oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1
    return {'log_loss': log_loss(y, p, labels=[0,1,2]),
            'accuracy': float((p.argmax(axis=1)==y).mean()),
            'brier':    float(np.mean(np.sum((p-oh)**2, axis=1)))}

def fit_xgb(X, y):
    return xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=300, max_depth=4,
                             learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             eval_metric='mlogloss', n_jobs=4, verbosity=0).fit(X, y)
def fit_mlp(Xs, y):
    return MLPClassifier(hidden_layer_sizes=(32,16), activation='relu', max_iter=500,
                         learning_rate_init=0.005, alpha=1e-3, early_stopping=True, random_state=0).fit(Xs, y)
def fit_cb(X, y):
    return CatBoostClassifier(iterations=400, depth=4, learning_rate=0.05, loss_function='MultiClass',
                              verbose=False, random_seed=0, thread_count=4).fit(X, y)
def po(c, X):
    p = c.predict_proba(X); cl = np.asarray(c.classes_).flatten(); o = [list(cl).index(x) for x in (0,1,2)]
    return p[:, o]


def backtest(feat_cols, label):
    preds = {}
    for year in [2010, 2014, 2018, 2022]:
        train = valid[valid['date'].dt.year < year]
        test  = valid[(valid['date'].dt.year == year) & (valid['tournament']=='FIFA World Cup')]
        if test.empty: continue
        y_train = train['outcome'].to_numpy(); y_test = test['outcome'].to_numpy()
        Xtr = train[feat_cols].to_numpy(); Xte = test[feat_cols].to_numpy()
        sc = StandardScaler().fit(Xtr)
        p_mlp = po(fit_mlp(sc.transform(Xtr), y_train), sc.transform(Xte))
        p_xgb = po(fit_xgb(Xtr, y_train), Xte)
        p_cb  = po(fit_cb(Xtr, y_train),  Xte)
        rdc = RegressionDixonColesModel(RDC_FEAT, xi=0.00038).fit(train)
        p_rdc = rdc.predict_many(test)[['p_away_win','p_draw','p_home_win']].to_numpy()
        preds[year] = {'y': y_test, 'mlp': p_mlp, 'xgb': p_xgb, 'cb': p_cb, 'rdc': p_rdc}
    yall = np.concatenate([d['y'] for d in preds.values()])
    pall, rows = [], []
    for yr, d in preds.items():
        avg = (d['mlp']+d['xgb']+d['cb']+d['rdc'])/4.0
        avg /= avg.sum(axis=1, keepdims=True)
        pall.append(avg)
        rows.append({'wc': yr, **score(d['y'], avg)})
    s_all = score(yall, np.vstack(pall))
    print(f'\n=== {label} ({len(feat_cols)} feats) ===')
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f'  Agg: LL={s_all["log_loss"]:.4f}  acc={s_all["accuracy"]:.4f}  brier={s_all["brier"]:.4f}')


# Per-feature ablation
backtest(FEAT, 'BASELINE')
backtest(FEAT + ['diff_big5_share'], '+Big5 only')
backtest(FEAT + ['diff_mgr_career_matches_n'], '+Mgr-career-matches only')
backtest(FEAT + ['diff_mgr_career_wr_n'], '+Mgr-career-WR only')
backtest(FEAT + ['diff_mgr_tenure_days_n'], '+Mgr-tenure only')
# Pairs and curated sets
backtest(FEAT + ['diff_mgr_career_matches_n', 'diff_mgr_tenure_days_n'], '+Mgr (no WR)')
backtest(FEAT + ['diff_mgr_career_matches_n', 'diff_mgr_tenure_days_n', 'diff_big5_share'], '+Mgr(no WR)+Big5')
