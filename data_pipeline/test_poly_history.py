import requests
import json

# Let's query gamma API first to get a token ID for a historical market.
res = requests.get('https://gamma-api.polymarket.com/events?slug=bitcoin-up-or-down-march-2-12pm-et')
if res.ok and res.json():
    data = res.json()[0]
    markets = data.get('markets', [])
    if markets:
        clob_token_ids = json.loads(markets[0].get('clobTokenIds', '[]'))
        if clob_token_ids:
            token_id = clob_token_ids[0]
            print(f"Token ID: {token_id}")
            
            # Now fetch price history
            history_url = f"https://clob.polymarket.com/prices-history?market={token_id}&interval=1m&fidelity=10"
            history_res = requests.get(history_url)
            if history_res.ok:
                print("Got history points:", len(history_res.json().get('history', [])))
            else:
                print("Failed to get history:", history_res.status_code, history_res.text)
