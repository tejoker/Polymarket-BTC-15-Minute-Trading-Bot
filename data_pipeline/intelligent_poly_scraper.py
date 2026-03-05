import os
import time
import json
import csv
import requests
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)
REAL_POLY_FILE = os.path.join(OUTPUT_DIR, 'real_polymarket_history.csv')

# Use a session for connection pooling
session = requests.Session()

def get_market_history(slug):
    """
    Combines getting the token ID and fetching the history into one threadable function.
    """
    url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    try:
        res = session.get(url, timeout=10)
        if res.ok and res.json():
            data = res.json()[0]
            markets = data.get('markets', [])
            if markets:
                clob_token_ids_str = markets[0].get('clobTokenIds', '[]')
                if clob_token_ids_str:
                    clob_token_ids = json.loads(clob_token_ids_str)
                    if clob_token_ids:
                        token_id = clob_token_ids[0]
                        # Fetch price history immediately
                        history_url = f"https://clob.polymarket.com/prices-history?market={token_id}&interval=1m&fidelity=10"
                        hist_res = session.get(history_url, timeout=10)
                        if hist_res.ok:
                            return slug, hist_res.json().get('history', [])
    except Exception as e:
        pass
    return slug, []

def scrape_day(target_date):
    """
    Scrapes the 15m, 5m, and 1h markets for a specific UTC date concurrently.
    """
    print(f"\n--- Scraping Polymarket data for {target_date.strftime('%Y-%m-%d')} ---")
    
    start_of_day = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
    start_ts = int(start_of_day.timestamp())
    
    # Generate slugs
    slugs_1h = [f"btc-updown-1h-{start_ts + (i * 3600)}" for i in range(24)]
    slugs_15m = [f"btc-updown-15m-{start_ts + (i * 900)}" for i in range(96)]
    slugs_5m = [f"btc-updown-5m-{start_ts + (i * 300)}" for i in range(288)]
    
    all_slugs = slugs_1h + slugs_15m + slugs_5m
    
    found_count = 0
    results_to_write = []
    
    # Parallel fetch to bypass sequential wait times
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_slug = {executor.submit(get_market_history, slug): slug for slug in all_slugs}
        
        count = 0
        for future in as_completed(future_to_slug):
            slug, history = future.result()
            count += 1
            if history:
                interval_type = '1h' if '1h' in slug else ('15m' if '15m' in slug else '5m')
                for pt in history:
                    t_ms = int(pt.get('t', 0)) * 1000
                    p = pt.get('p', 0.0)
                    results_to_write.append([t_ms, slug, interval_type, p])
                found_count += 1
            
            if count % 50 == 0:
                print(f"Processed {count}/{len(all_slugs)} potential markets. Found {found_count} so far.")
            
            # Sub-millisecond sleep to space out thread execution slightly against Cloudflare
            time.sleep(0.05)

    # Write to CSV in bulk block to save disk I/O latency
    file_exists = os.path.exists(REAL_POLY_FILE)
    with open(REAL_POLY_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp_ms', 'slug', 'interval_type', 'price'])
        writer.writerows(results_to_write)
                
    print(f"Finished {target_date.strftime('%Y-%m-%d')}. Successfully fetched real histories for {found_count} markets.")

def run_intelligent_scraper(days_back=30):
    """
    Runs backwards day by day (from yesterday).
    """
    now = datetime.now(timezone.utc)
    # Start from yesterday as today is not fully resolved
    for i in range(1, days_back + 1):
        target_date = now - timedelta(days=i)
        scrape_day(target_date)
        print("Sleeping 3 seconds before next historical day to reset Polymarket API rate buckets...")
        time.sleep(3)

if __name__ == "__main__":
    # Fetch 30 days as a strong starter dataset (allows user to run the script longer without timeout)
    run_intelligent_scraper(days_back=30)
