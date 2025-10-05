# utils/helpers.py

import math
import pandas as pd

def calculate_quantity(loss_per_trade, signal_std, max_qty):
    if signal_std == 0:
        return max_qty
    qty = math.floor(loss_per_trade / signal_std)
    return min(qty, max_qty)

def calculate_pnl(trade, exit_price):
    if trade['trade_type'] == "Long":
        pnl = (exit_price - trade['entry_price']) * trade['qty']
    else:
        pnl = (trade['entry_price'] - exit_price) * trade['qty']
    return pnl

def load_csv_safe(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return pd.DataFrame()

#finished
