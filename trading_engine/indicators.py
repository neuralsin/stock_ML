# trading_engine/indicators.py

import pandas as pd
import ta  # pip install ta

def calculate_indicators(df, MEAN_STD_WINDOW=25, MA30_WINDOW=30, MA50_WINDOW=50, MA100_WINDOW=100, RSI_WINDOW=14):
    """
    Calculate technical indicators required for trading engine.
    """
    # Rolling mean & std
    df['rolling_mean'] = df['Close'].rolling(window=MEAN_STD_WINDOW).mean()
    df['rolling_std'] = df['Close'].rolling(window=MEAN_STD_WINDOW).std()
    
    # Moving averages
    df['MA30'] = df['Close'].rolling(window=MA30_WINDOW).mean()
    df['MA50'] = df['Close'].rolling(window=MA50_WINDOW).mean()
    df['MA100'] = df['Close'].rolling(window=MA100_WINDOW).mean()
    
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=RSI_WINDOW).rsi()
    
    # ATR
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    
    return df

#finished
