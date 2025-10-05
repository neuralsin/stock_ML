# utils/helpers.py

import pandas as pd
import numpy as np
import logging

def load_csv(path):
    """Load CSV safely"""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

def save_csv(df, path):
    """Save DataFrame to CSV safely"""
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        logging.error(f"Error saving {path}: {e}")

def calculate_drawdown(pnl_list):
    """Return max drawdown from PnL list"""
    if not pnl_list:
        return 0
    cum = np.cumsum(pnl_list)
    drawdown = np.maximum.accumulate(cum) - cum
    return max(drawdown)

def profit_factor(pnl_list):
    """Calculate profit factor"""
    wins = sum([p for p in pnl_list if p > 0])
    losses = abs(sum([p for p in pnl_list if p < 0]))
    return wins / losses if losses > 0 else wins

# finished
