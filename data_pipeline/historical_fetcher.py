import os
import time
import requests
import json
from datetime import datetime, timezone, timedelta

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)
BTC_SPOT_FILE = os.path.join(OUTPUT_DIR, 'btc_spot_1m.csv')

def fetch_binance_klines(symbol='BTCUSDT', interval='1m', start_time_ms=None, end_time_ms=None):
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    current_start = start_time_ms
    limit = 1000

    print(f"Fetching Binance {interval} klines for {symbol}...")
    while current_start < end_time_ms:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_time_ms,
            'limit': limit
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            all_klines.extend(data)
            # The last element's 0th index is the open time
            current_start = data[-1][0] + 1  # Add 1ms to avoid fetching the same candle
            
            # Print progress implicitly
            last_date = datetime.fromtimestamp(data[-1][0] / 1000, tz=timezone.utc)
            print(f"Fetched up to {last_date.isoformat()} ({len(all_klines)} candles total)")
            
            time.sleep(0.1) # Be nice to Binance API API
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(2) # Wait a bit before retrying

    return all_klines

def fetch_and_save_1_year_data():
    now = datetime.now(timezone.utc)
    # 1 year ago
    one_year_ago = now - timedelta(days=365)
    
    start_time_ms = int(one_year_ago.timestamp() * 1000)
    end_time_ms = int(now.timestamp() * 1000)
    
    # We'll write to a temporary file first, or just append
    # But since it's only 525,600 minutes in a year, it fits in memory easily natively.
    # We will do chunks and write directly to avoid massive memory footprint during script run.
    print(f"Fetching data from {one_year_ago.isoformat()} to {now.isoformat()}...")
    
    with open(BTC_SPOT_FILE, 'w') as f:
        # Header: timestamp, open, high, low, close, volume, close_time
        f.write("timestamp_ms,open,high,low,close,volume,close_time_ms\n")
        
        current_start = start_time_ms
        limit = 1000
        url = "https://api.binance.com/api/v3/klines"
        total_rows = 0
        
        while current_start < end_time_ms:
            params = {
                'symbol': 'BTCUSDT',
                'interval': '1m',
                'startTime': current_start,
                'endTime': end_time_ms,
                'limit': limit
            }
            try:
                response = requests.get(url, params=params)
                if response.status_code == 429:
                    print("Rate limit hit, sleeping...")
                    time.sleep(5)
                    continue
                    
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                    
                for row in data:
                    # Kline format: [Open time, Open, High, Low, Close, Volume, Close time, ...]
                    t_ms, o, h, l, c, v, ct_ms = row[0], row[1], row[2], row[3], row[4], row[5], row[6]
                    f.write(f"{t_ms},{o},{h},{l},{c},{v},{ct_ms}\n")
                    
                total_rows += len(data)
                current_start = data[-1][0] + 1
                
                last_date = datetime.fromtimestamp(data[-1][0] / 1000, tz=timezone.utc)
                print(f"Persisted up to {last_date.isoformat()} | Total rows: {total_rows}")
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                time.sleep(2)

    print(f"Finished fetching 1 year of BTC spot data: {BTC_SPOT_FILE}")

if __name__ == "__main__":
    fetch_and_save_1_year_data()
