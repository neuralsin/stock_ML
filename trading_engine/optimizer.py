# trading_engine/optimizer.py

import itertools
import logging
import numpy as np
from copy import deepcopy
from trading_engine.engine import TradingEngine

# Optimization ranges
MEAN_STD_WINDOW_RANGE = [15, 25, 35]
MA_WINDOWS = [(20, 50, 100), (10, 30, 60)]
RSI_WINDOWS = [10, 14, 20]
RSI_OVERSOLD_RANGE = [25, 30, 35]
RSI_OVERBOUGHT_RANGE = [65, 70, 75]
REVERSE_STRATEGY_OPTIONS = [True, False]

def evaluate_trades(trades):
    if not trades:
        return {'total_pnl': 0, 'max_dd': 0, 'profit_factor': 0}
    df = deepcopy(trades)
    pnl_list = [t['pnl'] for t in df]
    total_pnl = sum(pnl_list)
    cum = np.cumsum(pnl_list)
    drawdown = np.maximum.accumulate(cum) - cum
    max_dd = max(drawdown) if len(drawdown) > 0 else 0
    wins = sum([p for p in pnl_list if p > 0])
    losses = abs(sum([p for p in pnl_list if p < 0]))
    profit_factor = wins / losses if losses > 0 else wins
    return {'total_pnl': total_pnl, 'max_dd': max_dd, 'profit_factor': profit_factor}

def optimize_parameters(symbol, df):
    best_metrics = {'total_pnl': -np.inf, 'params': None}

    for mean_std, ma_window, rsi_w, rsi_os, rsi_ob, rev in itertools.product(
        MEAN_STD_WINDOW_RANGE, MA_WINDOWS, RSI_WINDOWS, RSI_OVERSOLD_RANGE, RSI_OVERBOUGHT_RANGE, REVERSE_STRATEGY_OPTIONS
    ):
        engine = TradingEngine(symbol, deepcopy(df))
        engine.MEAN_STD_WINDOW = mean_std
        engine.MA30_WINDOW, engine.MA50_WINDOW, engine.MA100_WINDOW = ma_window
        engine.RSI_WINDOW = rsi_w
        engine.RSI_OVERSOLD = rsi_os
        engine.RSI_OVERBOUGHT = rsi_ob
        engine.REVERSE_STRATEGY = rev

        trades = engine.virtual_trade()
        metrics = evaluate_trades(trades)
        score = metrics['total_pnl'] - 0.5 * metrics['max_dd']
        best_score = best_metrics['total_pnl'] - 0.5 * best_metrics.get('max_dd', 0)

        if score > best_score:
            best_metrics.update(metrics)
            best_metrics['params'] = {
                'MEAN_STD_WINDOW': mean_std,
                'MA30_WINDOW': engine.MA30_WINDOW,
                'MA50_WINDOW': engine.MA50_WINDOW,
                'MA100_WINDOW': engine.MA100_WINDOW,
                'RSI_WINDOW': rsi_w,
                'RSI_OVERSOLD': rsi_os,
                'RSI_OVERBOUGHT': rsi_ob,
                'REVERSE_STRATEGY': rev
            }

    logging.info(f"[{symbol}] Best Params: {best_metrics['params']}, Metrics: {best_metrics}")
    return best_metrics

# finished