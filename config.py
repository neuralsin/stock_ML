# config.py

import datetime

# -------------------------------
# GLOBAL SETTINGS
# -------------------------------
# Trade & risk
LOSS_PER_TRADE = 200
MAX_TRADES_PER_DAY = 3
GLOBAL_MAX_ACTIVE_TRADES = 3
TRADE_ENTRY_START = datetime.time(9, 15)
TRADE_ENTRY_END = datetime.time(14, 45)
SQUARE_OFF_TIME = datetime.time(15, 15)
DEFAULT_FUNDS = 10000
MAX_QTY = 1000

# Parameter ranges for optimizer
MEAN_STD_WINDOW_RANGE = [15, 25, 35]
MA_WINDOWS = [(20, 50, 100), (10, 30, 60)]
RSI_WINDOWS = [10, 14, 20]
RSI_OVERSOLD_RANGE = [25, 30, 35]
RSI_OVERBOUGHT_RANGE = [65, 70, 75]
REVERSE_STRATEGY_OPTIONS = [True, False]

# ML Settings
ML_TRAIN_INTERVAL = "1d"  # Interval to retrain ML model automatically
ML_MIN_TRAIN_TRADES = 50  # Minimum trades before retraining ML
ML_ACCEPTABLE_SUCCESS_RATE = 0.8  # Threshold to push learned logic to main engine

#finished
