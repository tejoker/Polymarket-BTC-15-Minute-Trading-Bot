import requests
import json
import time

def fetch_pm_history():
    # 1. Get recent closed 15m BTC markets
    gamma_url = "https://gamma-api.polymarket.com/markets?active=false&closed=true&limit=10"
    markets = requests.get(gamma_url).json()
    
    btc_15m = [m for m in markets if 'btc' in m.get('slug', '').lower() and '15m' in m.get('slug', '').lower()]
    print(f"Found {len(btc_15m)} recent closed 15m BTC markets.")
    
    if not btc_15m:
        print("No recent markets found.")
        return
        
    m = btc_15m[0]
    print(f"Target: {m['slug']} (closed: {m.get('endDate')})")
    
    condition_id = m.get('conditionId')
    tokens = m.get('tokens', [])
    yes_token = tokens[0]['token_id'] if tokens and 'token_id' in tokens[0] else None
    
    print(f"Condition ID: {condition_id}, YES Token: {yes_token}")
    
    # 2. Try to get history from CLOB
    if yes_token:
        clob_url = f"https://clob.polymarket.com/prices-history?token_id={yes_token}"  # Might need fidelity/interval
        try:
            res = requests.get(clob_url)
            print(f"Prices History Status: {res.status_code}")
            if res.status_code == 200:
                data = res.json()
                print(f"History elements: {len(data.get('history', []))}")
                if len(data.get('history', [])) > 0:
                    print("Sample:", data['history'][:3])
        except Exception as e:
            print("Error", e)

if __name__ == "__main__":
    fetch_pm_history()
