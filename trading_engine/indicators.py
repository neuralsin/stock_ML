# trading_engine/indicators.py

import pandas as pd
import numpy as np

def calculate_indicators(df):
    """Add common technical indicators used by the trading engine"""
    df = df.copy()

    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['Datetime']):
        df['Datetime'] = pd.to_datetime(df['Datetime'])

    # Rolling mean & std for mean reversion
    df['rolling_mean'] = df['Close'].rolling(window=25).mean()
    df['rolling_std'] = df['Close'].rolling(window=25).std()

    # Moving Averages
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR (Average True Range)
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # Drop temporary columns
    df.drop(columns=['H-L', 'H-PC', 'L-PC', 'TR'], inplace=True)

    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    return df

#finished