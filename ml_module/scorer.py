# ml_module/scorer.py

import numpy as np
from ml_module.trainer import MLTrainer

class MLScorer:
    def __init__(self, symbols=None):
        self.symbols = symbols or []
        self.trainer = MLTrainer(self.symbols)
        self.prediction_history = {symbol: [] for symbol in self.symbols}

    def add_symbols(self, symbols):
        """Add new symbols dynamically"""
        for s in symbols:
            if s not in self.symbols:
                self.symbols.append(s)
                self.trainer.models[s] = self.trainer.models.get(s) or self.trainer.models.setdefault(s, None)
                self.prediction_history[s] = []

    def predict_risk(self, df, symbol=None):
        """
        Returns a risk score between 0 and 1.
        Higher score = higher risk.
        Can use ML prediction vs current price or volatility
        """
        if symbol is None:
            return 0.5  # default moderate risk

        predicted_close = self.trainer.predict_next(symbol, df)
        if predicted_close is None:
            return 0.5  # default moderate risk

        last_close = df['Close'].iloc[-1]
        diff = abs(predicted_close - last_close) / last_close

        # Risk score: higher if deviation is large
        risk_score = min(diff * 10, 1)  # cap at 1
        self.prediction_history[symbol].append(predicted_close)
        return risk_score

    def learn_from_trade(self, trade):
        """
        Placeholder for feedback loop.
        Compare predicted vs actual and adjust model if needed.
        """
        symbol = trade['symbol']
        pnl = trade.get('pnl', 0)
        if pnl > 0:
            # Positive trade, keep logic
            pass
        else:
            # Negative trade, could retrain / adjust model
            self.trainer.update_model(symbol)
