# trading_engine/engine.py

import pandas as pd
import numpy as np
import math
import datetime
import logging
from copy import deepcopy
from trading_engine.indicators import calculate_indicators
from ml_module.scorer import MLScorer
from config import (
    LOSS_PER_TRADE, MAX_TRADES_PER_DAY, GLOBAL_MAX_ACTIVE_TRADES,
    TRADE_ENTRY_START, TRADE_ENTRY_END, SQUARE_OFF_TIME, MAX_QTY
)

class TradingEngine:
    def __init__(self, symbol, df, ml_model=None):
        self.symbol = symbol
        self.df = df
        self.ml_model = ml_model or MLScorer()
        self.trades = []
        self.active_trade = None
        self.daily_trades = 0
        self.MEAN_STD_WINDOW = 25
        self.MA30_WINDOW, self.MA50_WINDOW, self.MA100_WINDOW = 30, 50, 100
        self.RSI_WINDOW = 14
        self.RSI_OVERSOLD = 30
        self.RSI_OVERBOUGHT = 70
        self.REVERSE_STRATEGY = True

    def check_trade_signal(self, df):
        if len(df) < self.MEAN_STD_WINDOW + 2:
            return None
        signal_candle = df.iloc[-2]
        breakout_candle = df.iloc[-1]
        if pd.isnull(signal_candle['rolling_mean']) or pd.isnull(signal_candle['rolling_std']):
            return None

        candidate = {'score': 0}
        # Mean reversion
        if (signal_candle['High'] > signal_candle['rolling_mean'] + signal_candle['rolling_std'] or
            signal_candle['Low'] < signal_candle['rolling_mean'] - signal_candle['rolling_std']):
            candidate['mr_signal'] = True
            candidate['score'] += 1
            candidate['orig_trade'] = "Long" if signal_candle['Low'] < signal_candle['rolling_mean'] else "Short"
            candidate['signal_price'] = breakout_candle['Close']
            candidate['signal_std'] = signal_candle['rolling_std']

        # MA breakout
        if (signal_candle['MA30'] > signal_candle['MA50'] > signal_candle['MA100'] and
            breakout_candle['Close'] > signal_candle['MA30']):
            candidate['ma_signal'] = True
            candidate['score'] += 1
            candidate['orig_trade'] = "Long"
        elif (signal_candle['MA30'] < signal_candle['MA50'] < signal_candle['MA100'] and
              breakout_candle['Close'] < signal_candle['MA30']):
            candidate['ma_signal'] = True
            candidate['score'] += 1
            candidate['orig_trade'] = "Short"

        # RSI
        if signal_candle['RSI'] < self.RSI_OVERSOLD:
            candidate['rsi_signal'] = True
            candidate['score'] += 1
            candidate['orig_trade'] = "Long"
        elif signal_candle['RSI'] > self.RSI_OVERBOUGHT:
            candidate['rsi_signal'] = True
            candidate['score'] += 1
            candidate['orig_trade'] = "Short"

        if candidate['score'] == 0:
            return None

        # Reverse logic
        if self.REVERSE_STRATEGY:
            trade_type = "Short" if candidate['orig_trade'] == "Long" else "Long"
        else:
            trade_type = candidate['orig_trade']

        entry_price = breakout_candle['Close']
        std = signal_candle['rolling_std']
        stoploss = entry_price - std if trade_type == "Long" else entry_price + std
        target = entry_price + 2 * std if trade_type == "Long" else entry_price - 2 * std
        qty = min(math.floor(LOSS_PER_TRADE / std), MAX_QTY)

        return {'trade_type': trade_type, 'entry_price': entry_price, 'stoploss': stoploss, 'target': target, 'qty': qty}

    def check_exit_condition(self, trade, candle):
        if trade['trade_type'] == "Long":
            if candle['High'] >= trade['target'] or candle['Low'] <= trade['stoploss']:
                return True
        else:
            if candle['Low'] <= trade['target'] or candle['High'] >= trade['stoploss']:
                return True
        return False

    def virtual_trade(self):
        df = calculate_indicators(self.df)
        for i in range(self.MEAN_STD_WINDOW + 2, len(df)):
            now = df.iloc[i]['Datetime'].time()
            if not (TRADE_ENTRY_START <= now <= SQUARE_OFF_TIME):
                continue

            current_df = df.iloc[:i+1].copy()
            signal = self.check_trade_signal(current_df)

            if self.active_trade is None and signal and self.daily_trades < MAX_TRADES_PER_DAY:
                risk_score = self.ml_model.predict_risk(current_df.tail(100))
                if risk_score > 0.8:
                    continue  # skip risky trades

                self.active_trade = {
                    'symbol': self.symbol,
                    'entry_index': i,
                    'entry_time': df.iloc[i]['Datetime'],
                    'entry_price': signal['entry_price'],
                    'stoploss': signal['stoploss'],
                    'target': signal['target'],
                    'qty': signal['qty'],
                    'trade_type': signal['trade_type'],
                    'status': 'open'
                }
                self.daily_trades += 1
                logging.info(f"[{self.symbol}] Entry {signal['trade_type']} at {signal['entry_price']:.2f}")

            elif self.active_trade is not None:
                candle = df.iloc[i]
                if self.check_exit_condition(self.active_trade, candle) or now >= SQUARE_OFF_TIME:
                    exit_price = candle['Close']
                    pnl = (exit_price - self.active_trade['entry_price']) * self.active_trade['qty'] if self.active_trade['trade_type'] == "Long" else (self.active_trade['entry_price'] - exit_price) * self.active_trade['qty']
                    self.active_trade.update({
                        'exit_index': i,
                        'exit_time': candle['Datetime'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'status': 'closed'
                    })
                    self.trades.append(self.active_trade)
                    self.ml_model.learn_from_trade(self.active_trade)
                    logging.info(f"[{self.symbol}] Exit {self.active_trade['trade_type']} at {exit_price:.2f}, PnL: {pnl:.2f}")
                    self.active_trade = None

        if self.active_trade and self.active_trade['status'] == 'open':
            candle = df.iloc[-1]
            exit_price = candle['Close']
            pnl = (exit_price - self.active_trade['entry_price']) * self.active_trade['qty'] if self.active_trade['trade_type'] == "Long" else (self.active_trade['entry_price'] - exit_price) * self.active_trade['qty']
            self.active_trade.update({
                'exit_index': len(df)-1,
                'exit_time': candle['Datetime'],
                'exit_price': exit_price,
                'pnl': pnl,
                'status': 'closed'
            })
            self.trades.append(self.active_trade)
            self.ml_model.learn_from_trade(self.active_trade)
            logging.info(f"[{self.symbol}] Forced Exit {self.active_trade['trade_type']} at {exit_price:.2f}, PnL: {pnl:.2f}")
            self.active_trade = None

        return self.trades

#finished
