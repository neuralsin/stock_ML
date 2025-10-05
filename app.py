# app.py
import os
import warnings
import logging
from datetime import datetime

# Suppress TF & warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, jsonify
import pandas as pd

from trading_engine.engine import TradingEngine
from ml_module.trainer import MLTrainer
from utils.kite_api import fetch_ohlc

# -------------------
# Flask & Logging Setup
# -------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------
# Global Storage
# -------------------
stocks = {}
ml_trainer = MLTrainer(global_model=True)  # Enable global Transformer
DATA_FOLDER = "data"
RESULTS_FILE = "ml_results.csv"

# -------------------
# Load All CSVs and Train ML + Transformer
# -------------------
def load_and_train_all():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        logging.warning(f"[!] Data folder '{DATA_FOLDER}' not found. Created empty folder.")
        return

    csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]
    if not csv_files:
        logging.warning("[!] No CSV files found in 'data/' folder.")
        return

    for file_name in csv_files:
        symbol = file_name.replace(".csv", "")
        file_path = os.path.join(DATA_FOLDER, file_name)

        # Fetch OHLC dynamically (live fallback to CSV)
        df = fetch_ohlc(symbol, csv_fallback=file_path)
        if df.empty:
            logging.warning(f"[!] No data found for {symbol}")
            continue

        stocks[symbol] = {"symbol": symbol}
        ml_trainer.add_stock(symbol)
        ml_trainer.update_data(symbol, df)
        logging.info(f"[✓] Trained ML & Transformer on {symbol} ({len(df)} rows)")

load_and_train_all()
logging.info("[✓] All ML + Transformer training completed")

# -------------------
# CSV Logging Utility
# -------------------
def log_results(symbol, trades, summary):
    now = datetime.now()
    record = {
        "Timestamp": now,
        "Symbol": symbol,
        "Total_Trades": summary.get("total_trades", 0),
        "Win_Rate_%": summary.get("win_rate", 0),
        "Avg_PnL_%": summary.get("avg_pnl", 0),
        "Total_PnL_%": summary.get("total_pnl", 0)
    }
    df = pd.DataFrame([record])
    if os.path.exists(RESULTS_FILE):
        df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(RESULTS_FILE, index=False)

# -------------------
# Flask Routes
# -------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/add_stock', methods=['POST'])
def add_stock():
    data = request.get_json()
    symbol = data.get('symbol')
    if not symbol:
        return jsonify({"message": "No symbol provided!"}), 400

    if symbol not in stocks:
        stocks[symbol] = {"symbol": symbol}
        ml_trainer.add_stock(symbol)

        # Try loading CSV dynamically if exists
        file_path = os.path.join(DATA_FOLDER, f"{symbol}.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                ml_trainer.update_data(symbol, df)
                logging.info(f"[✓] Stock {symbol} added and paper-traded from CSV.")
            except Exception as e:
                logging.error(f"[!] Failed to load CSV for {symbol}: {e}")
                return jsonify({"message": f"Error loading CSV for {symbol}."}), 500

        return jsonify({"message": f"Stock {symbol} added successfully."})
    else:
        return jsonify({"message": f"Stock {symbol} already exists."})


@app.route('/run_trading', methods=['POST'])
def run_trading():
    data = request.get_json()
    symbol = data.get('symbol')
    interval = data.get('interval', '5minute')

    if not symbol:
        return jsonify({"success": False, "message": "No symbol provided."})

    try:
        file_path = os.path.join(DATA_FOLDER, f"{symbol}.csv")
        df = fetch_ohlc(symbol, interval=interval, csv_fallback=file_path)
        if df.empty:
            return jsonify({"success": False, "message": f"No data fetched for {symbol}."})

        # Update ML models & simulate confident paper trades
        ml_trainer.update_data(symbol, df)

        # Initialize trading engine with updated ML trainer
        engine = TradingEngine(symbol, df, ml_trainer=ml_trainer)
        trades = engine.virtual_trade(show_progress=True)
        summary = engine.summary()

        # Log session results
        log_results(symbol, trades, summary)

        return jsonify({"success": True, "trades": trades, "summary": summary})

    except Exception as e:
        logging.error(f"[!] Error running trading for {symbol}: {e}")
        return jsonify({"success": False, "message": str(e)})


# -------------------
# Run Flask App
# -------------------
if __name__ == "__main__":
    app.run(debug=True)
