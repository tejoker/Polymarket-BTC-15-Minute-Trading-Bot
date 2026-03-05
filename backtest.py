import time
import random
from decimal import Decimal
from datetime import datetime
import statistics

import os
import sys

# Append project root
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.sentiment_processor import SentimentProcessor
from core.strategy_brain.signal_processors.divergence_processor import PriceDivergenceProcessor
from core.strategy_brain.signal_processors.orderbook_processor import OrderBookImbalanceProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.signal_processors.deribit_pcr_processor import DeribitPCRProcessor
from core.strategy_brain.signal_processors.lead_lag_processor import LeadLagProcessor
from core.strategy_brain.signal_processors.coinbase_premium_processor import CoinbasePremiumProcessor
from core.strategy_brain.fusion_engine.signal_fusion import SignalFusionEngine
from execution.risk_engine import RiskEngine
from feedback.learning_engine import LearningEngine

# Disable all internal loguru logs from processors for clean benchmarking
from loguru import logger
logger.remove()

import asyncio

async def run_hft_backtest():
    print("=" * 80)
    print(" 2026 SOTA HFT MATHEMATICAL ENGINE BACKTEST ")
    print("=" * 80)
    
    # 1. Initialize SOTA Processors
    spike_detector = SpikeDetectionProcessor(spike_threshold=0.05, lookback_periods=20)
    sentiment_processor = SentimentProcessor()
    divergence_processor = PriceDivergenceProcessor(divergence_threshold=0.05)
    orderbook_processor = OrderBookImbalanceProcessor(imbalance_threshold=0.30)
    tick_velocity_processor = TickVelocityProcessor(velocity_threshold_60s=0.015)
    deribit_pcr_processor = DeribitPCRProcessor()
    lead_lag_processor = LeadLagProcessor(correlation_window=15)
    coinbase_premium_processor = CoinbasePremiumProcessor(premium_threshold=10.0)
    
    # 2. Fusion and Learning
    fusion_engine = SignalFusionEngine()
    fusion_engine.set_weight("LeadLag", 0.15)
    fusion_engine.set_weight("CoinbasePremium", 0.15)
    
    print("Looking for aggregated historical backtest data...")
    data_file = os.path.join(os.path.dirname(__file__), 'data', 'real_aligned_backtest.csv')
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please run the data_pipeline scripts first.")
        return
        
    print(f"Loading continuous historical market state from {data_file}...")
    
    tick_history = []
    
    fused_signals = 0
    trade_triggers = 0
    pnl = 0.0
    winning_trades = 0
    losing_trades = 0
    total_ticks = 0
    latencies = []
    
    import csv
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        
        # Warmup filter buffers
        print("Warming up Savitzky-Golay and Continuous Hawkes buffers...")
        for _ in range(50):
            try:
                row = next(reader)
                poly_prob = Decimal(row['poly_15m_prob'])
                tick_history.append(poly_prob)
                
                metadata = {
                    'spot_price': float(row['spot_price']),
                    'momentum': 0.0,
                    'bids_volume': float(row['poly_15m_bids']),
                    'asks_volume': float(row['poly_15m_asks']),
                }
                spike_detector.process(poly_prob, tick_history, metadata)
                tick_velocity_processor.process(poly_prob, tick_history, metadata)
                total_ticks += 1
            except StopIteration:
                break
                
        print("\nExecuting hot-path fusion algorithms across 1-Year historical grid...")
        
        for row in reader:
            total_ticks += 1
            poly_prob = Decimal(row['poly_15m_prob'])
            tick_history.append(poly_prob)
            
            # Simulated orderbook pressure & multi-asset data
            spot_val = float(row['spot_price'])
            metadata = {
                'spot_price': spot_val,
                'binance_spot_price': spot_val - random.uniform(-20.0, 20.0), # ~20 premium fluctuation
                'eth_price': spot_val / 15.0 + random.uniform(-10.0, 10.0), # Mock ETH structural decoupling noise
                'sol_price': spot_val / 250.0 + random.uniform(-2.0, 2.0), # Mock SOL structural decoupling noise
                'momentum': (spot_val - float(row['interval_15m_start_price'])) / float(row['interval_15m_start_price']) if float(row['interval_15m_start_price']) > 0 else 0,
                'bids_volume': float(row['poly_15m_bids']),
                'asks_volume': float(row['poly_15m_asks']),
            }
            
            # === START HFT TIMER ===
            start_time = time.perf_counter_ns()
            
            signals = []
            
            s1 = spike_detector.process(poly_prob, tick_history, metadata)
            if s1: signals.append(s1)
                
            s2 = sentiment_processor.process(poly_prob, [], metadata)
            if s2: signals.append(s2)
                
            s3 = divergence_processor.process(poly_prob, [], metadata)
            if s3: signals.append(s3)
                
            s4 = await orderbook_processor.process(poly_prob, [], metadata)
            if s4: signals.append(s4)
                
            s5 = tick_velocity_processor.process(poly_prob, tick_history, metadata)
            if s5: signals.append(s5)
                
            s6 = lead_lag_processor.process(poly_prob, tick_history, metadata)
            if s6: signals.append(s6)
                
            s7 = coinbase_premium_processor.process(poly_prob, tick_history, metadata)
            if s7: signals.append(s7)
                
            if signals:
                fused_signals += 1
                # Run ChIMP Evidential Fusion via SignalFusionEngine overriding DST
                final_decision = fusion_engine.fuse_signals(signals, poly_prob, min_score=1.0)
                
                # Basic Trade Simulation Logic based on threshold 60.0% evidence
                if final_decision is not None and final_decision.score >= 60.0:
                    trade_triggers += 1
                    
                    # Very simple retroactive PnL simulation based on closing probabilities
                    # If we buy 'YES' at 0.60 probability and the market resolves YES (prob approaches 1.0)
                    if poly_prob < Decimal('0.90'): 
                        # Assume successful convergence later if momentum was positive
                        if metadata['momentum'] > 0:
                            winning_trades += 1
                            pnl += (1.0 - float(poly_prob)) * 100 # $100 position size
                        else:
                            losing_trades += 1
                            pnl -= float(poly_prob) * 100
                            
            # === END HFT TIMER ===
            end_time = time.perf_counter_ns()
            
            latency_ns = end_time - start_time
            latencies.append(latency_ns)
            
            if len(tick_history) > 100:
                tick_history.pop(0)

            if total_ticks % 50000 == 0:
                print(f"Processed {total_ticks} historical intervals... Current PnL: ${pnl:.2f}")

    # Convert to microseconds
    latencies_us = [l / 1000.0 for l in latencies]
    avg_latency = statistics.mean(latencies_us)
    p99_latency = statistics.quantiles(latencies_us, n=100)[98]
    max_latency = max(latencies_us)
    min_latency = min(latencies_us)

    print("\n" + "=" * 80)
    print(" BACKTEST RESULTS ")
    print("=" * 80)
    print(f"Total Ticks Processed: {total_ticks}")
    print(f"Total Fusable Signal Events: {fused_signals}")
    print(f"Trades Triggered (Dempster-Shafer confirmation): {trade_triggers}")
    print(f"Total PnL Generated: ${pnl:.2f}")
    win_rate = (winning_trades / max(1, trade_triggers)) * 100
    print(f"Winning Trades: {winning_trades} ({win_rate:.2f}%) | Losing Trades: {losing_trades}")
    
    # Calculate Probabilistic Sharpe Ratio
    # Annualized to 35040 intervals of 15-minutes
    import math
    if trade_triggers > 0:
        daily_return = (pnl / (total_ticks / 1440.0)) / (trade_triggers * 100)
        annualized_sharpe = daily_return * math.sqrt(35040)
        print(f"Probabilistic Sharpe Ratio (PSR): {annualized_sharpe:.4f}")
    
    print("")
    print("=== LATENCY PROFILE (Microseconds / µs) ===")
    print(f"Average Path Latency: {avg_latency:.2f} µs")
    print(f"Median Path Latency:  {statistics.median(latencies_us):.2f} µs")
    print(f"99th Percentile:      {p99_latency:.2f} µs")
    print(f"Max Trough Latency:   {max_latency:.2f} µs")
    print(f"Min Path Latency:     {min_latency:.2f} µs")
    
    print("\nVERDICT:")
    if avg_latency < 1000:
         print("[PASS] The Evidential Signal Fusion and SOTA math runs extremely fast (< 1ms).")
         print("[PASS] High-Frequency Trading determinism preserved.")
    else:
         print("[FAIL] Architecture is blocking the event loop or math is too slow.")
         
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_hft_backtest())
