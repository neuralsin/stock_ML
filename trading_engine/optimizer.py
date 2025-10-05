# trading_engine/optimizer.py

import itertools
import logging
from copy import deepcopy
from trading_engine.engine import TradingEngine
from trading_engine.indicators import calculate_indicators
import numpy as np

# Parameter ranges for optimization
MEAN_STD_WINDOW_RANGE = [15, 25, 35]
MA_WINDOWS = [(20, 50, 100), (10, 30, 60)]
RSI_WINDOWS = [10, 14, 20]
RSI_OVERSOLD_RANGE = [25, 30, 35]
RSI_OVERBOUGHT_RANGE = [65, 70, 75]
REVERSE_STRATEGY_OPTIONS = [True, False]

def evaluate_trades(trades):
    if not trades:
        return {'total_pnl': 0, 'max_dd': 0, 'profit_factor': 0}
    df = trades.copy() if isinstance(trades, list) else pd.DataFrame(trades)
    total_pnl = df['pnl'].sum()
    cum = (df['pnl'] + 1).cumprod()
    drawdown = cum.cummax() - cum
    max_dd = drawdown.max()
    wins = df.loc[df['pnl'] > 0, 'pnl'].sum()
    losses = abs(df.loc[df['pnl'] < 0, 'pnl'].sum())
    profit_factor = wins / losses if losses > 0 else wins
    return {'total_pnl': total_pnl, 'max_dd': max_dd, 'profit_factor': profit_factor}

def optimize_parameters(symbol, df):
    best_metrics = {'total_pnl': -np.inf, 'params': None}

    for mean_std, ma_window, rsi_w, rsi_os, rsi_ob, rev in itertools.product(
        MEAN_STD_WINDOW_RANGE, MA_WINDOWS, RSI_WINDOWS, RSI_OVERSOLD_RANGE, RSI_OVERBOUGHT_RANGE, REVERSE_STRATEGY_OPTIONS
    ):
        MEAN_STD_WINDOW = mean_std
        MA30_WINDOW, MA50_WINDOW, MA100_WINDOW = ma_window
        RSI_WINDOW = rsi_w
        RSI_OVERSOLD = rsi_os
        RSI_OVERBOUGHT = rsi_ob
        REVERSE_STRATEGY = rev

        engine = TradingEngine(symbol, deepcopy(df))
        engine.MEAN_STD_WINDOW = MEAN_STD_WINDOW
        engine.MA30_WINDOW = MA30_WINDOW
        engine.MA50_WINDOW = MA50_WINDOW
        engine.MA100_WINDOW = MA100_WINDOW
        engine.RSI_WINDOW = RSI_WINDOW
        engine.RSI_OVERSOLD = RSI_OVERSOLD
        engine.RSI_OVERBOUGHT = RSI_OVERBOUGHT
        engine.REVERSE_STRATEGY = REVERSE_STRATEGY

        trades = engine.virtual_trade()
        metrics = evaluate_trades(trades)
        score = metrics['total_pnl'] - 0.5 * metrics['max_dd']
        best_score = best_metrics['total_pnl'] - 0.5 * best_metrics.get('max_dd', 0)

        if score > best_score:
            best_metrics.update(metrics)
            best_metrics['params'] = {
                'MEAN_STD_WINDOW': MEAN_STD_WINDOW,
                'MA30_WINDOW': MA30_WINDOW,
                'MA50_WINDOW': MA50_WINDOW,
                'MA100_WINDOW': MA100_WINDOW,
                'RSI_WINDOW': RSI_WINDOW,
                'RSI_OVERSOLD': RSI_OVERSOLD,
                'RSI_OVERBOUGHT': RSI_OVERBOUGHT,
                'REVERSE_STRATEGY': REVERSE_STRATEGY
            }

    logging.info(f"[{symbol}] Best Params: {best_metrics['params']}, Metrics: {best_metrics}")
    return best_metrics

#finished
