import pandas as pd
import numpy as np

def implied_prob(decimal_odds: pd.Series) -> pd.Series:
    return 1 / decimal_odds

def kelly_fraction(prob: float, decimal_odds: float) -> float:
    b = decimal_odds - 1
    return max(0.0, (b * prob - (1 - prob)) / b)

def simulate_bets(df: pd.DataFrame, strategy: str = 'flat', threshold: float = 0.0, fraction: float = 1.0) -> pd.DataFrame:
    df = df.copy()
    bankroll = 1000.0
    bankrolls = []
    profits = []

    for _, row in df.iterrows():
        edge = row['home_win_prob'] - row['home_implied']
        if edge <= threshold:
            bankrolls.append(bankroll)
            profits.append(0)
            continue

        if strategy == 'flat':
            stake = 10.0
        else:
            f = kelly_fraction(row['home_win_prob'], row['home_odds']) * fraction
            stake = bankroll * f

        won = row['home_win'] == 1
        pnl = stake * (row['home_odds'] - 1) if won else -stake
        bankroll += pnl
        bankrolls.append(bankroll)
        profits.append(pnl)

    df['bankroll'] = bankrolls
    df['pnl'] = profits
    df['cumulative_profit'] = df['pnl'].cumsum()
    return df
