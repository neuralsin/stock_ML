# ------------------------------- config.py -------------------------------
import datetime

# ------------------------------- KITE API -------------------------------
# replace with your own API keys (use env vars for security if deploying)
KITE_API_KEY = "your_kite_api_key"
KITE_API_SECRET = "your_kite_api_secret"
KITE_ACCESS_TOKEN = "your_access_token"  # or generate dynamically

# ------------------------------- FLASK SETTINGS -------------------------------
DEBUG_MODE = True
HOST = "0.0.0.0"
PORT = 5000

# ------------------------------- TRADING PARAMETERS -------------------------------
REVERSE_STRATEGY = True
MEAN_STD_WINDOW = 25
MA30_WINDOW = 30
MA50_WINDOW = 50
MA100_WINDOW = 100
LOSS_PER_TRADE = 200
MAX_TRADES_PER_DAY = 3
GLOBAL_MAX_ACTIVE_TRADES = 3
TRADE_ENTRY_START = datetime.time(9, 15)
TRADE_ENTRY_END = datetime.time(14, 45)
SQUARE_OFF_TIME = datetime.time(15, 15)
DEFAULT_FUNDS = 10000

# ------------------------------- ML SETTINGS -------------------------------
# training parameters for LSTM/ML hybrid model
ML_MODEL_PATH = "models/lstm_trading_model.h5"
TRAINING_BATCH_SIZE = 64
TRAINING_EPOCHS = 50
LEARNING_RATE = 0.001
ACCURACY_THRESHOLD = 0.85
PROFIT_THRESHOLD = 0.8

# ------------------------------- FILE PATHS -------------------------------
DATA_DIR = "data/"
VIRTUAL_TRADE_SUMMARY = "outputs/virtual_trade_summary.csv"
MODEL_LOGS = "logs/ml_training.log"

# ------------------------------- RISK CONTROL -------------------------------
MAX_RISK_PERCENT = 0.02  # max risk per trade (2%)
RISK_REWARD_RATIO = 2.0  # target = risk * ratio
MAX_OPEN_POSITIONS = 3

# ------------------------------- INDICATOR SETTINGS -------------------------------
RSI_PERIOD = 14
ATR_PERIOD = 14
BOLLINGER_BAND_WINDOW = 20

# ------------------------------- OTHER SETTINGS -------------------------------
UPDATE_INTERVAL = 300  # in seconds (5 min refresh)
USE_GPU = True  # for RTX 4050 based LSTM training
SEED = 42

# ------------------------------- LOGGING -------------------------------
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = "INFO"

