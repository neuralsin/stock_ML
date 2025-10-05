# ml_module/trainer.py

import os
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ---------------- Transformer Block ----------------
class TransformerBlock:
    def __init__(self, embed_dim, num_heads=4, ff_dim=32, rate=0.1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def __call__(self, inputs):
        attn = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)(inputs, inputs)
        attn = Dropout(self.rate)(attn)
        out1 = LayerNormalization(epsilon=1e-6)(inputs + attn)
        ffn = Sequential([
            Dense(self.ff_dim, activation='relu'),
            Dense(self.embed_dim)
        ])(out1)
        ffn = Dropout(self.rate)(ffn)
        return LayerNormalization(epsilon=1e-6)(out1 + ffn)

# ---------------- LSTM Model ----------------
class MLModel:
    def __init__(self, seq_len=50, feature_dim=5):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.model = self.build_model()
        self.scaler = MinMaxScaler()

    def build_model(self):
        model = Sequential([
            Input(shape=(self.seq_len, self.feature_dim)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1, activation='tanh')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X, y, epochs=10, batch_size=16):
        if len(X)==0 or len(y)==0:
            return None
        es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        lr_reduce = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=0)
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es, lr_reduce])
        return history

    def predict_next_move(self, X_seq):
        if X_seq.shape[0] != self.seq_len:
            return 0
        X_seq = np.expand_dims(X_seq, axis=0)
        pred = self.model.predict(X_seq, verbose=0)[0][0]
        threshold = 0.05
        if abs(pred) < threshold:
            return 0
        return int(np.sign(pred))

    def preprocess(self, df):
        df = df[['Open','High','Low','Close','Volume']].fillna(method='ffill')
        scaled = self.scaler.fit_transform(df.values)
        X = []
        for i in range(len(scaled)-self.seq_len):
            X.append(scaled[i:i+self.seq_len])
        return np.array(X), scaled[self.seq_len:]

# ---------------- Transformer Model ----------------
class TransformerModel:
    def __init__(self, seq_len=50, feature_dim=5):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.model = self.build_transformer()

    def build_transformer(self):
        inputs = Input(shape=(self.seq_len, self.feature_dim))
        x = TransformerBlock(embed_dim=self.feature_dim)(inputs)
        x = Dense(1, activation='tanh')(x[:, -1, :])
        model = Model(inputs, x)
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X, y, epochs=10, batch_size=16):
        if len(X)==0 or len(y)==0:
            return None
        es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        lr_reduce = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=0)
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es, lr_reduce])
        return history

    def predict_next_move(self, X_seq):
        if X_seq.shape[0] != self.seq_len:
            return 0
        X_seq = np.expand_dims(X_seq, axis=0)
        pred = self.model.predict(X_seq, verbose=0)[0][0]
        threshold = 0.05
        if abs(pred) < threshold:
            return 0
        return int(np.sign(pred))

# ---------------- ML Trainer ----------------
class MLTrainer:
    def __init__(self, model_path="ml_model", log_file="ml_stats.csv", global_model=False):
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
        self.models = {}
        self.stocks_data = {}
        self.total_pnl = {}
        self.session_pnl = {}
        self.paper_trades_file = "paper_trades.csv"
        self.log_file = log_file
        self.global_model_enabled = global_model
        self.global_model = TransformerModel() if global_model else None
        self.combined_dataset = pd.DataFrame()

        # initialize files
        if not os.path.exists(self.log_file):
            pd.DataFrame(columns=["symbol","rows_trained","epochs_run","final_loss",
                                  "dataset_start","dataset_end","session_pnl","total_pnl"]).to_csv(self.log_file,index=False)
        if not os.path.exists(self.paper_trades_file):
            pd.DataFrame(columns=["Symbol","Date","Action","Price","Next_Close","PnL"]).to_csv(self.paper_trades_file,index=False)

        # Load global model if available
        if self.global_model_enabled:
            global_path = os.path.join(self.model_path, "global_transformer.h5")
            if os.path.exists(global_path):
                self.global_model.model = load_model(global_path)
                print("[✓] Loaded saved global Transformer model")

    def add_stock(self, symbol):
        if symbol not in self.stocks_data:
            self.stocks_data[symbol] = pd.DataFrame()
            self.models[symbol] = MLModel()

            model_file = os.path.join(self.model_path, f"{symbol}.h5")
            if os.path.exists(model_file):
                self.models[symbol].model = load_model(model_file)
                print(f"[✓] Loaded saved model for {symbol}")

            self.total_pnl[symbol] = 0.0
            self.session_pnl[symbol] = 0.0

    def update_data(self, symbol, new_data: pd.DataFrame):
        if new_data.empty:
            return
        self.add_stock(symbol)

        for col in ['Open','High','Low','Close','Volume']:
            if col not in new_data.columns:
                new_data[col] = 0.0
        if 'Datetime' not in new_data.columns:
            new_data['Datetime'] = pd.to_datetime(np.arange(len(new_data)))

        self.stocks_data[symbol] = pd.concat([self.stocks_data[symbol], new_data]) \
            .drop_duplicates(subset='Datetime') \
            .sort_values('Datetime').reset_index(drop=True)

        if self.global_model_enabled:
            self.combined_dataset = pd.concat([self.combined_dataset, new_data]) \
                .drop_duplicates(subset='Datetime') \
                .sort_values('Datetime').reset_index(drop=True)

        self.simulate_confident_trades(symbol)
        self.train(symbol)

    def prepare_sequences(self, df, seq_len=50):
        df = df[['Open','High','Low','Close','Volume']].astype(float)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df.values)
        X, y = [], []
        for i in range(seq_len, len(scaled)):
            X.append(scaled[i-seq_len:i])
            y.append(scaled[i][3]-scaled[i-1][3])
        return np.array(X), np.array(y)

    def train(self, symbol):
        df = self.stocks_data[symbol]
        if len(df) < 51:
            print(f"[!] Not enough rows for {symbol}")
            return

        X, y = self.prepare_sequences(df)
        model = self.models[symbol]
        history = model.train(X, y, epochs=10)

        final_loss = history.history['loss'][-1] if history else 0
        session_profit = self.session_pnl[symbol]
        self.total_pnl[symbol] += session_profit

        # Save the trained model
        model.model.save(os.path.join(self.model_path, f"{symbol}.h5"))

        # Train + save global transformer
        if self.global_model_enabled and len(self.combined_dataset) >= 51:
            Xg, yg = self.prepare_sequences(self.combined_dataset)
            self.global_model.train(Xg, yg, epochs=10)
            self.global_model.model.save(os.path.join(self.model_path, "global_transformer.h5"))

        pd.DataFrame([{
            "symbol": symbol,
            "rows_trained": len(df),
            "epochs_run": len(history.history['loss']) if history else 0,
            "final_loss": round(final_loss,6),
            "dataset_start": df['Datetime'].iloc[0],
            "dataset_end": df['Datetime'].iloc[-1],
            "session_pnl": round(session_profit,4),
            "total_pnl": round(self.total_pnl[symbol],4)
        }]).to_csv(self.log_file, mode='a', header=False, index=False)

        print(f"[✓] Trained {symbol}, final_loss={round(final_loss,6)}, session_pnl={round(session_profit,4)}")

    # ---------------- Paper Trades ----------------
    def simulate_confident_trades(self, symbol):
        df = self.stocks_data[symbol].copy()
        trades = []
        if len(df) < 51:
            return trades

        seq_len = 50
        for i in range(seq_len, len(df)-1):
            window = df.iloc[i-seq_len:i+1]
            X_seq = window[['Open','High','Low','Close','Volume']].astype(float).to_numpy()
            signal = self.models[symbol].predict_next_move(X_seq)
            next_close = df['Close'].iloc[i+1]
            today_close = df['Close'].iloc[i]
            pnl = 0
            action = "HOLD"

            if signal == 1:
                pnl = next_close - today_close
                action = "BUY"
            elif signal == -1:
                pnl = today_close - next_close
                action = "SELL"

            trades.append(pnl)

            pd.DataFrame([{
                "Symbol": symbol,
                "Date": df['Datetime'].iloc[i+1],
                "Action": action,
                "Price": today_close,
                "Next_Close": next_close,
                "PnL": pnl
            }]).to_csv(self.paper_trades_file, mode='a', header=False, index=False)

        self.session_pnl[symbol] = sum(trades)
        return trades
