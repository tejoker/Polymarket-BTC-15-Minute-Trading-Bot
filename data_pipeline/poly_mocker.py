import os
import csv
import math
from datetime import datetime, timezone

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
BTC_SPOT_FILE = os.path.join(OUTPUT_DIR, 'btc_spot_1m.csv')
COMBINED_BACKTEST_FILE = os.path.join(OUTPUT_DIR, 'historical_backtest_data.csv')

def generate_poly_mock_data():
    """
    Reads the binance 1m spot file and generates corresponding 
    historical Polymarket probabilities and Order Flow metrics to build 
    a highly realistic simulated backtesting engine dataset.
    """
    if not os.path.exists(BTC_SPOT_FILE):
        print(f"Error: {BTC_SPOT_FILE} not found. Run historical_fetcher.py first.")
        return

    print("Generating Polymarket simulation data based on historical BTC spot limits...")
    
    with open(BTC_SPOT_FILE, 'r') as infile, open(COMBINED_BACKTEST_FILE, 'w') as outfile:
        reader = csv.DictReader(infile)
        
        fieldnames = [
            'timestamp_ms', 'spot_price', 'spot_volume',
            'poly_15m_prob', 'poly_15m_bids', 'poly_15m_asks',
            'poly_5m_prob', 'poly_5m_bids', 'poly_5m_asks',
            'poly_1h_prob', 'poly_1h_bids', 'poly_1h_asks',
            'interval_15m_start_price', 'interval_5m_start_price', 'interval_1h_start_price'
        ]
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Track starting prices for intervals to calculate deterministic probabilities
        start_15m, start_5m, start_1h = None, None, None
        
        row_count = 0
        
        for row in reader:
            ts_ms = int(row['timestamp_ms'])
            unix_sec = ts_ms // 1000
            
            spot = float(row['close'])
            vol = float(row['volume'])
            
            # Identify interval boundaries
            is_new_5m = (unix_sec % 300) == 0
            is_new_15m = (unix_sec % 900) == 0
            is_new_1h = (unix_sec % 3600) == 0
            
            if is_new_5m or start_5m is None: start_5m = spot
            if is_new_15m or start_15m is None: start_15m = spot
            if is_new_1h or start_1h is None: start_1h = spot
            
            # Simple mathematical model: Probability = Sigmoid of price delta from start
            # The closer to the end of the interval, the steeper the sigmoid
            
            def calc_prob(current, start, elapsed_sec, total_sec):
                delta = current - start
                # Volatility scaling
                time_remaining = max(1, total_sec - elapsed_sec)
                z_score = delta / (math.sqrt(time_remaining) * 0.5) 
                
                # Sigmoid curve
                prob = 1 / (1 + math.exp(-max(min(z_score, 10), -10)))
                return max(0.01, min(0.99, prob))
            
            p_5m = calc_prob(spot, start_5m, unix_sec % 300, 300)
            p_15m = calc_prob(spot, start_15m, unix_sec % 900, 900)
            p_1h = calc_prob(spot, start_1h, unix_sec % 3600, 3600)
            
            # Generate deterministic order book pressures based on volume and probability gradients
            # High volume + rising prob = bid-heavy orderbook
            
            writer.writerow({
                'timestamp_ms': ts_ms,
                'spot_price': spot,
                'spot_volume': vol,
                
                'poly_15m_prob': round(p_15m, 4),
                'poly_15m_bids': int(vol * 50 * p_15m),
                'poly_15m_asks': int(vol * 50 * (1 - p_15m)),
                
                'poly_5m_prob': round(p_5m, 4),
                'poly_5m_bids': int(vol * 30 * p_5m),
                'poly_5m_asks': int(vol * 30 * (1 - p_5m)),
                
                'poly_1h_prob': round(p_1h, 4),
                'poly_1h_bids': int(vol * 100 * p_1h),
                'poly_1h_asks': int(vol * 100 * (1 - p_1h)),
                
                'interval_15m_start_price': start_15m,
                'interval_5m_start_price': start_5m,
                'interval_1h_start_price': start_1h
            })
            
            row_count += 1
            if row_count % 50000 == 0:
                print(f"Generated {row_count} synthetic historical ticks...")

    print(f"Success. Wrote {row_count} simulated ticks to {COMBINED_BACKTEST_FILE}")

if __name__ == "__main__":
    generate_poly_mock_data()
