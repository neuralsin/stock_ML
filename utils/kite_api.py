# utils/kite_api.py
from kiteconnect import KiteConnect
import pandas as pd
import datetime
import os

API_KEY = "zhy0s5sn5c2mvn5a"
API_SECRET = "eq4iglxw8zqjgqac73eua5yefchl17is"
ACCESS_TOKEN = "f1TzW1FoDH1pE9lDYW0xd45iVAND5tzK"

# Initialize KiteConnect
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

def fetch_ohlc(symbol, interval="5minute", duration_days=5, csv_fallback=None):
    """
    Fetch historical OHLC data for a given symbol.
    If live data cannot be fetched, fall back to CSV.
    Automatically parses CSVs in the 'DateOpenHighLowClose' format.
    """
    to_date = datetime.datetime.now()
    from_date = to_date - datetime.timedelta(days=duration_days)

    try:
        # Try fetching live data
        instrument_token = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]['instrument_token']
        data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval
        )
        if not data:
            raise ValueError("No data returned from Kite API")
        
        df = pd.DataFrame(data)
        df['Datetime'] = pd.to_datetime(df['date'])
        df.rename(columns={'date':'Datetime'}, inplace=True)
        print(f"[✓] Live data fetched for {symbol}")
        return df

    except Exception as e:
        print(f"[!] Error fetching live OHLC for {symbol}: {e}")

        # Fallback to CSV if provided
        if csv_fallback and os.path.exists(csv_fallback):
            try:
                # Try parsing custom CSV format
                df = pd.read_csv(csv_fallback)
                if len(df.columns) == 1:
                    # split single column into Date/Open/High/Low/Close
                    df = df[df.iloc[:,0].str.strip() != ""]  # remove empty lines
                    df = df[~df.iloc[:,0].str.contains("Date", na=False)]  # remove header rows
                    df = df[0].str.extract(r'(?P<Date>\d{2}/\d{2}) (?P<Time>\d{2}:\d{2})(?P<Open>\d+\.\d+)(?P<High>\d+\.\d+)(?P<Low>\d+\.\d+)(?P<Close>\d+\.\d+)')
                    df['Datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'], format="%d/%m %H:%M")
                    df = df[['Datetime','Open','High','Low','Close']].astype(float, errors='ignore')
                
                print(f"[✓] Data loaded from CSV fallback: {csv_fallback}")
                return df
            except Exception as ce:
                print(f"[!] Error reading CSV fallback: {ce}")
                return pd.DataFrame()
        else:
            print("[!] No CSV fallback provided or file does not exist.")
            return pd.DataFrame()
