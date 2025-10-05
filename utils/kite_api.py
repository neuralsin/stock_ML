# utils/kite_api.py

from kiteconnect import KiteConnect
import pandas as pd
import logging
from config import KITE_API_KEY, KITE_API_SECRET, KITE_ACCESS_TOKEN

class KiteAPI:
    def __init__(self):
        self.kite = KiteConnect(api_key=KITE_API_KEY)
        self.kite.set_access_token(KITE_ACCESS_TOKEN)

    def fetch_ohlc(self, symbol, interval="15minute", from_date=None, to_date=None):
        """
        Fetch OHLC data for a symbol.
        interval: 'minute', '15minute', 'day', etc.
        from_date, to_date: string 'YYYY-MM-DD'
        """
        try:
            from_date = from_date or pd.Timestamp.now() - pd.Timedelta(days=30)
            to_date = to_date or pd.Timestamp.now()
            data = self.kite.historical_data(symbol, from_date, to_date, interval)
            df = pd.DataFrame(data)
            if df.empty:
                logging.warning(f"No data returned for {symbol}")
            return df
        except Exception as e:
            logging.error(f"Error fetching OHLC for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_ltp(self, symbol):
        """Fetch latest price"""
        try:
            ltp_data = self.kite.ltp(symbol)
            return ltp_data[symbol]['last_price']
        except Exception as e:
            logging.error(f"Error fetching LTP for {symbol}: {e}")
            return None

# finished
# utils/kite_api.py

from kiteconnect import KiteConnect
import pandas as pd
import logging
from config import KITE_API_KEY, KITE_API_SECRET, KITE_ACCESS_TOKEN

class KiteAPI:
    def __init__(self):
        self.kite = KiteConnect(api_key=KITE_API_KEY)
        self.kite.set_access_token(KITE_ACCESS_TOKEN)

    def fetch_ohlc(self, symbol, interval="15minute", from_date=None, to_date=None):
        """
        Fetch OHLC data for a symbol.
        interval: 'minute', '15minute', 'day', etc.
        from_date, to_date: string 'YYYY-MM-DD'
        """
        try:
            from_date = from_date or pd.Timestamp.now() - pd.Timedelta(days=30)
            to_date = to_date or pd.Timestamp.now()
            data = self.kite.historical_data(symbol, from_date, to_date, interval)
            df = pd.DataFrame(data)
            if df.empty:
                logging.warning(f"No data returned for {symbol}")
            return df
        except Exception as e:
            logging.error(f"Error fetching OHLC for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_ltp(self, symbol):
        """Fetch latest price"""
        try:
            ltp_data = self.kite.ltp(symbol)
            return ltp_data[symbol]['last_price']
        except Exception as e:
            logging.error(f"Error fetching LTP for {symbol}: {e}")
            return None

# finished