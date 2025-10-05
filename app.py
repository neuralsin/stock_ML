# app.py
from flask import Flask, render_template, request, jsonify
from trading_engine.engine import TradingEngine
from ml_module.trainer import MLModule
from utils.kite_api import fetch_ohlc_data
import pandas as pd
import logging

# ------------------------------- Flask App -------------------------------
app = Flask(__name__)

# ------------------------------- Logging -------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ------------------------------- Initialize Modules -------------------------------
trading_engine = TradingEngine()
ml_module = MLModule()

# ------------------------------- Routes -------------------------------

@app.route('/')
def index():
    # Frontend HTML page
    return render_template('index.html')


@app.route('/run_trades', methods=['POST'])
def run_trades():
    """
    Expects JSON payload:
    {
        "symbols": ["RELIANCE", "TCS", "INFY"]
    }
    """
    data = request.get_json()
    symbols = data.get('symbols', [])

    all_trades = []
    report = []

    for symbol in symbols:
        try:
            # Fetch OHLC data from Kite API
            df = fetch_ohlc_data(symbol)
            if df.empty:
                logging.warning(f"No data for {symbol}")
                continue

            # Run virtual trading engine
            trades = trading_engine.run(df, symbol)

            # ML Module predicts high/low & risk factor for each trade
            for trade in trades:
                trade_features = trading_engine.extract_features(trade, df)
                ml_preds = ml_module.predict(trade_features)
                trade.update(ml_preds)

            all_trades.extend(trades)

        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")

    # Generate summary report
    if all_trades:
        report_df = trading_engine.generate_report(all_trades)
        report = report_df.to_dict('records')

    return jsonify({"trades": all_trades, "report": report})


@app.route('/update_ml', methods=['POST'])
def update_ml():
    """
    Trigger ML model retraining with new virtual trades
    """
    try:
        ml_module.train_on_virtual_trades()
        return jsonify({"status": "ML model updated successfully"})
    except Exception as e:
        logging.error(f"ML update error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ------------------------------- Run Flask -------------------------------
if __name__ == "__main__":
    import os

    # Custom port setup (if running multiple services)
    PORT = int(os.environ.get("PORT", 5000))

    # Enable threaded requests (so multiple users/symbols don’t block)
    app.run(
        host="0.0.0.0", 
        port=PORT,
        debug=True,          # set to False in production
        threaded=True, 
        use_reloader=True    # auto reloads when code changes
    )

    # Optional: gracefully close Kite session or ML resources on exit
    try:
        trading_engine.close()
        ml_module.cleanup()
        print("Clean shutdown complete. 🖤")
    except Exception as e:
        print(f"Error while closing resources: {e}")
