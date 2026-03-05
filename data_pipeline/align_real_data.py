import csv
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
btc_spot_file = os.path.join(DATA_DIR, 'btc_spot_1m.csv')
real_poly_file = os.path.join(DATA_DIR, 'real_polymarket_history.csv')
output_file = os.path.join(DATA_DIR, 'real_aligned_backtest.csv')

def align_data():
    print("Loading BTC Spot prices...")
    btc_spot = {}
    with open(btc_spot_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = int(row['timestamp_ms'])
            minute_ts = ts // 60000 * 60000
            btc_spot[minute_ts] = float(row['close'])
            
    print("Parsing 1-Month Real Polymarket Tick Flow...")
    data_rows = []
    
    # We will track interval start prices dynamically
    interval_starts = {}
    
    with open(real_poly_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['interval_type'] == '15m':
                ts = int(row['timestamp_ms'])
                price = float(row['price'])
                slug = row['slug']
                
                minute_ts = ts // 60000 * 60000
                spot = btc_spot.get(minute_ts)
                
                # Try fallback contiguous minute if spot gap
                if spot is None:
                    spot = btc_spot.get(minute_ts - 60000)
                if spot is None:
                    spot = btc_spot.get(minute_ts + 60000)
                
                if spot is not None:
                    # Capture the start price for momentum derivation
                    if slug not in interval_starts:
                        interval_starts[slug] = spot
                        
                    data_rows.append({
                        'timestamp_ms': ts,
                        'poly_15m_prob': price,
                        'spot_price': spot,
                        'interval_15m_start_price': interval_starts[slug],
                        # We must mock the orderbook depth here, since Polymarket deleted it.
                        # We provide an ambient $50,000 baseline liquidity volume so LiT does not divide by zero.
                        'poly_15m_bids': 50000.0, 
                        'poly_15m_asks': 50000.0,
                        'slug': slug
                    })
                    
    print(f"Sorting {len(data_rows)} fully correlated ticks...")
    data_rows.sort(key=lambda x: x['timestamp_ms'])
    
    print("Writing aligned structural file...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestamp_ms', 'poly_15m_prob', 'spot_price', 
            'interval_15m_start_price', 'poly_15m_bids', 'poly_15m_asks', 'slug'
        ])
        writer.writeheader()
        writer.writerows(data_rows)
        
    print(f"Success! Generated {output_file}")

if __name__ == "__main__":
    align_data()
