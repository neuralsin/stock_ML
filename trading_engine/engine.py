# trading_engine/engine.py
import pandas as pd
import numpy as np
import math
import logging
from trading_engine.indicators import calculate_indicators
from ml_module.trainer import MLTrainer
from config import (
    LOSS_PER_TRADE, MAX_QTY,
    TRADE_ENTRY_START, SQUARE_OFF_TIME
)

logging.basicConfig(level=logging.INFO)

class TradingEngine:
    def __init__(self, symbol, df, ml_trainer=None):
        self.symbol = symbol
        self.df = df.copy()
        self.ml_trainer = ml_trainer or MLTrainer()
        self.trades = []
        self.active_trade = None

        # Strategy parameters
        self.MEAN_STD_WINDOW = 25
        self.RSI_WINDOW = 14
        self.RSI_OVERSOLD = 30
        self.RSI_OVERBOUGHT = 70
        self.REVERSE_STRATEGY = True

    def execute_trade(self, prediction, price, qty=None):
        if self.active_trade:
            return  # Only one active trade at a time

        qty = qty or min(math.floor(LOSS_PER_TRADE / (price*0.01)), MAX_QTY)
        if prediction == "BUY":
            self.active_trade = {
                "entry_price": price,
                "type": "LONG",
                "stop_loss": price * 0.99,
                "target": price * 1.02,
                "qty": qty
            }
            logging.info(f"[+] Long trade opened at {price:.2f}")
        elif prediction == "SELL":
            self.active_trade = {
                "entry_price": price,
                "type": "SHORT",
                "stop_loss": price * 1.01,
                "target": price * 0.98,
                "qty": qty
            }
            logging.info(f"[-] Short trade opened at {price:.2f}")

    def update_trade(self, price):
        if not self.active_trade:
            return

        trade = self.active_trade
        pnl = 0
        if trade["type"] == "LONG":
            if price >= trade["target"] or price <= trade["stop_loss"]:
                pnl = (price - trade["entry_price"]) / trade["entry_price"]
                self.trades.append(pnl)
                logging.info(f"[{'✓' if price >= trade['target'] else 'x'}] Long closed. PnL: {pnl:.2%}")
                self.active_trade = None
        elif trade["type"] == "SHORT":
            if price <= trade["target"] or price >= trade["stop_loss"]:
                pnl = (trade["entry_price"] - price) / trade["entry_price"]
                self.trades.append(pnl)
                logging.info(f"[{'✓' if price <= trade['target'] else 'x'}] Short closed. PnL: {pnl:.2%}")
                self.active_trade = None

    def virtual_trade(self, show_progress=True):
        df = calculate_indicators(self.df)
        self.ml_trainer.add_stock(self.symbol)

        total_rows = len(df)
        for i in range(self.MEAN_STD_WINDOW + 2, total_rows):
            now = df.iloc[i]['Datetime'].time()
            if not (TRADE_ENTRY_START <= now <= SQUARE_OFF_TIME):
                continue

            current_df = df.iloc[:i+1].copy()
            recent_data = current_df.tail(100)

            # 1️⃣ Per-stock LSTM signal
            ml_signal = self.ml_trainer.predict_next_move(self.symbol, recent_data)

            # 2️⃣ Global Transformer trend signal
            transformer_signal = self.ml_trainer.predict_global_trend(current_df.tail(200))

            # Only execute if transformer confirms (accuracy threshold internally handled)
            final_signal = None
            if ml_signal in ["BUY", "SELL"]:
                if transformer_signal == ml_signal or transformer_signal == "HOLD":
                    final_signal = ml_signal

            if final_signal and not self.active_trade:
                self.execute_trade(final_signal, df.iloc[i]['Close'])

            if self.active_trade:
                self.update_trade(df.iloc[i]['Close'])

            if show_progress and i % 20 == 0:
                percent = (i / total_rows) * 100
                logging.info(f"[Progress] {percent:.1f}% analyzed for {self.symbol}")

        # Force-close at end of day
        if self.active_trade:
            self.update_trade(df.iloc[-1]['Close'])

        return self.trades

    def summary(self):
        if not self.trades:
            return {"total_trades": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0}

        wins = len([x for x in self.trades if x > 0])
        avg_pnl = np.mean(self.trades)
        win_rate = wins / len(self.trades)
        total_pnl = np.sum(self.trades)
        return {
            "total_trades": len(self.trades),
            "win_rate": round(win_rate * 100, 2),
            "avg_pnl": round(avg_pnl * 100, 2),
            "total_pnl": round(total_pnl * 100, 2)
        }


if __name__ == "__main__":
    import os
    from ml_module.trainer import MLTrainer

    DATA_FOLDER = "data"
    ml_trainer = MLTrainer()

    if os.path.exists(DATA_FOLDER):
        for file_name in os.listdir(DATA_FOLDER):
            if file_name.endswith(".csv"):
                symbol = file_name.replace(".csv", "")
                file_path = os.path.join(DATA_FOLDER, file_name)
                try:
                    df = pd.read_csv(file_path)
                    # Ensure 'Datetime' column exists
                    if 'Date' in df.columns:
                        df['Datetime'] = pd.to_datetime(df['Date'])
                    elif 'date' in df.columns:
                        df['Datetime'] = pd.to_datetime(df['date'])
                    else:
                        raise ValueError("CSV must have 'Date' or 'date' column")
                    logging.info(f"[✓] Loaded {symbol} ({len(df)} rows)")
                    ml_trainer.add_stock(symbol)
                    ml_trainer.update_data(symbol, df)
                except Exception as e:
                    logging.warning(f"[!] Failed to load {file_name}: {e}")
    else:
        logging.warning(f"[!] Data folder '{DATA_FOLDER}' not found!")

    logging.info("[✓] ML + Transformer training completed for all CSVs")
