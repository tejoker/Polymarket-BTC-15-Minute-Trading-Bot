import asyncio
import httpx
from decimal import Decimal
import sys
import os
import time
from datetime import datetime

# Append project root
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.divergence_processor import PriceDivergenceProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.signal_processors.orderbook_processor import OrderBookImbalanceProcessor
from core.strategy_brain.fusion_engine.signal_fusion import SignalFusionEngine
from loguru import logger
logger.remove()

async def backtest_live_polymarket():
    print("=" * 80)
    print(" POLYMARKET LIVE DATA MATH PIPELINE TEST ")
    print("=" * 80)
    
    spike = SpikeDetectionProcessor()
    div = PriceDivergenceProcessor()
    vel = TickVelocityProcessor()
    ob = OrderBookImbalanceProcessor()
    fusion = SignalFusionEngine()
    
    async with httpx.AsyncClient() as client:
        # Get live BTC Spot from Coinbase
        cb_resp = await client.get("https://api.coinbase.com/v2/prices/BTC-USD/spot")
        spot_price = float(cb_resp.json()['data']['amount'])
        print(f"[{datetime.now().strftime('%H:%M:%S')}] BTC Spot: ${spot_price:,.2f}")
        
        # Get active 15m BTC Market
        gamma_url = "https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=100"
        m_resp = await client.get(gamma_url)
        markets = m_resp.json()
        
        btc_15m = [m for m in markets if 'btc' in m.get('slug', '').lower() and '15m' in m.get('slug', '').lower()]
        
        if not btc_15m:
            print("No active 15m BTC markets found to test against. Using synthetic data.")
            btc_15m = [{'question': 'Will BTC go up?', 'tokens': [{'price': 0.45}, {'price': 0.55}]}]
            book_state = {'bids': [[0.44, 100], [0.43, 200]], 'asks': [[0.46, 100], [0.47, 500]]}
        else:
            market = btc_15m[0]
            print(f"Testing against real market: {market['question']}")
            # Mock a book state since CLOB is deeper
            book_state = {'bids': [[0.49, 5000], [0.48, 2000]], 'asks': [[0.51, 6000], [0.52, 1000]]}
            
        print("\nFiring data through Math Processors...")
        poly_prob = Decimal("0.45") # Base injection
        tick_history = [Decimal(str(0.45 + (i * 0.001))) for i in range(20)]
        
        metadata = {
            'spot_price': spot_price,
            'momentum': 0.002,
            'bids_volume': sum(b[1] for b in book_state['bids']),
            'asks_volume': sum(a[1] for a in book_state['asks']),
            'bids_max_size': max(b[1] for b in book_state['bids']),
            'asks_max_size': max(a[1] for a in book_state['asks']),
            'sentiment_score': 65, # Neutral-greed
        }
        
        signals = []
        signals.append(spike.process(poly_prob, tick_history, metadata))
        signals.append(div.process(poly_prob, tick_history, metadata))
        signals.append(vel.process(poly_prob, tick_history, metadata))
        signals.append(ob.process(poly_prob, tick_history, metadata))
        
        signals = [s for s in signals if s is not None]
        
        print("\nGenerated Probabilistic Signals:")
        for s in signals:
            print(f" -> [{s.source}] Dir: {s.direction.value}, Conf: {s.confidence:.2f}, Score: {s.score:.2f}")
            
        final = fusion.fuse_signals(signals, poly_prob)
        if final:
            print(f"\nDempster-Shafer Consensus: {final.direction.value} (Score: {final.score:.2f} / Conf: {final.confidence:.2f}%)")
        else:
            print("\nDempster-Shafer Fusion rejected signal due to math uncertainty bounds.")
            
        print("=" * 80)

if __name__ == "__main__":
    asyncio.run(backtest_live_polymarket())
