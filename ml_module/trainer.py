# ml_module/trainer.py

import pandas as pd
from ml_module.lstm_model import LSTMModel

class Trainer:
    def __init__(self, model=None):
        self.model = model or LSTMModel()
    
    def prepare_data(self, df, feature_cols=['Close', 'High', 'Low', 'Volume'], target_col='Close'):
        df = df.copy()
        df = df.dropna()
        X = df[feature_cols].values
        y = df[target_col].values
        return X, y

    def train(self, df, epochs=10, batch_size=32):
        X, y = self.prepare_data(df)
        self.model.train(X, y, epochs=epochs, batch_size=batch_size)

    def update_model(self, trade_history_df):
        """
        Incremental learning from new virtual trades
        """
        # Extract features & targets from trade history
        if trade_history_df.empty:
            return
        X, y = self.prepare_data(trade_history_df)
        self.model.train(X, y, epochs=5, batch_size=16)

# finished
