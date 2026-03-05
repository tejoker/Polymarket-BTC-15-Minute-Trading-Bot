import os
import json
import time
import asyncio
import sqlite3
from datetime import datetime, timezone
import httpx

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)
DB_PATH = os.path.join(OUTPUT_DIR, 'live_orderbook.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS orderbooks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_ms INTEGER,
            asset TEXT,
            slug TEXT,
            best_bid REAL,
            best_ask REAL,
            bid_volume REAL,
            ask_volume REAL
        )
    ''')

    # Backward-compatible migration for existing DBs created before `asset` existed
    cursor.execute("PRAGMA table_info(orderbooks)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if "asset" not in existing_columns:
        cursor.execute("ALTER TABLE orderbooks ADD COLUMN asset TEXT")

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_orderbooks_asset_ts ON orderbooks(asset, timestamp_ms)"
    )

    conn.commit()
    conn.close()

def save_orderbook(timestamp_ms, asset, slug, best_bid, best_ask, bid_volume, ask_volume):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO orderbooks (timestamp_ms, asset, slug, best_bid, best_ask, bid_volume, ask_volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp_ms, asset, slug, best_bid, best_ask, bid_volume, ask_volume))
    conn.commit()
    conn.close()

async def get_active_crypto_slugs():
    """Builds current BTC/ETH/SOL updown slugs for 5m/15m/1h intervals."""
    url = "https://gamma-api.polymarket.com/events?slug=bitcoin-up-or-down-march-2-12pm-et"
    # We'll just look for btc-updown
    # Instead of pulling specific events, a more robust way is querying for active markets by underlying.
    # We will approximate this for the recorder.
    
    # Let's dynamically construct the current active slugs like the bot does.
    now = datetime.now(timezone.utc)
    # This is a simplifed continuous polling approach for the recorder based on the Bot's logic.
    unix_sec = int(now.timestamp())
    
    interval_5m_end = ((unix_sec // 300) + 1) * 300
    interval_15m_end = ((unix_sec // 900) + 1) * 900
    interval_1h_end = ((unix_sec // 3600) + 1) * 3600
    
    assets = ["btc", "eth", "sol"]
    return [
        *[f"{asset}-updown-5m-{interval_5m_end}" for asset in assets],
        *[f"{asset}-updown-15m-{interval_15m_end}" for asset in assets],
        *[f"{asset}-updown-1h-{interval_1h_end}" for asset in assets],
    ]

async def record_loop():
    print(f"Starting Live Orderbook Recorder. Database: {DB_PATH}")
    init_db()
    
    # Note: For production we would use Nautilus adapter websockets.
    # To keep this script self-contained and stable without needing Redis/Env vars
    # We will poll the REST API for orderbook depth at 1-second intervals.
    
    async with httpx.AsyncClient() as client:
        while True:
            try:
                active_slugs = await get_active_crypto_slugs()
                
                for slug in active_slugs:
                    asset = slug.split('-')[0].lower() if '-' in slug else 'unknown'

                    # 1. Get token ID
                    res = await client.get(f"https://gamma-api.polymarket.com/events?slug={slug}")
                    if res.status_code == 200 and res.json():
                        data = res.json()[0]
                        markets = data.get('markets', [])
                        if markets:
                            clob_token_ids_str = markets[0].get('clobTokenIds', '[]')
                            if clob_token_ids_str:
                                clob_token_ids = json.loads(clob_token_ids_str)
                                if clob_token_ids:
                                    token_id = clob_token_ids[0]
                                    
                                    # 2. Get Orderbook
                                    ob_res = await client.get(f"https://clob.polymarket.com/book?token_id={token_id}")
                                    if ob_res.status_code == 200:
                                        ob_data = ob_res.json()
                                        
                                        bids = ob_data.get('bids', [])
                                        asks = ob_data.get('asks', [])
                                        
                                        best_bid = float(bids[0]['price']) if bids else 0.0
                                        best_ask = float(asks[0]['price']) if asks else 0.0
                                        
                                        bid_volume = sum(float(b['size']) for b in bids)
                                        ask_volume = sum(float(a['size']) for a in asks)
                                        
                                        save_orderbook(
                                            int(time.time() * 1000), 
                                            asset,
                                            slug, 
                                            best_bid, 
                                            best_ask, 
                                            bid_volume, 
                                            ask_volume
                                        )
                                        
                await asyncio.sleep(2) # Record every 2 seconds
            
            except Exception as e:
                print(f"Error in recording loop: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(record_loop())
