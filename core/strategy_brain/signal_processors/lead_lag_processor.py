"""
Lead-Lag Processor (Cross-Asset Correlation)
Ingests BTC, ETH, and SOL prices to detect localized structural decoupling and
use the leading asset (e.g. SOL) as a predictive oracle for BTC.
"""
from typing import List, Dict, Any, Optional
from decimal import Decimal
import datetime
from loguru import logger
import collections

try:
    import numpy as np
except ImportError:
    np = None

from .base_processor import BaseSignalProcessor, TradingSignal, SignalDirection, SignalStrength, SignalType

class LeadLagProcessor(BaseSignalProcessor):
    def __init__(
        self,
        correlation_window: int = 20,    # 20 ticks
        decoupling_threshold: float = 0.5, # correlation below this = decoupled
    ):
        super().__init__("LeadLagProcessor")
        self.correlation_window = correlation_window
        self.decoupling_threshold = decoupling_threshold
        
        self.btc_history = collections.deque(maxlen=correlation_window)
        self.eth_history = collections.deque(maxlen=correlation_window)
        self.sol_history = collections.deque(maxlen=correlation_window)
        
    def process(
        self,
        current_price: Decimal,
        historical_prices: List[Decimal],
        metadata: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        
        if np is None:
            logger.warning("LeadLagProcessor requires numpy.")
            return None
            
        eth_price = metadata.get('eth_price')
        sol_price = metadata.get('sol_price')
        btc_spot = metadata.get('binance_spot_price')
        
        if eth_price is None or sol_price is None or btc_spot is None:
            return None
            
        self.btc_history.append(float(btc_spot))
        self.eth_history.append(float(eth_price))
        self.sol_history.append(float(sol_price))
        
        if len(self.btc_history) < self.correlation_window:
            return None
            
        btc_arr = np.array(self.btc_history)
        eth_arr = np.array(self.eth_history)
        sol_arr = np.array(self.sol_history)
        
        # Calculate returns
        btc_ret = np.diff(btc_arr) / btc_arr[:-1]
        eth_ret = np.diff(eth_arr) / eth_arr[:-1]
        sol_ret = np.diff(sol_arr) / sol_arr[:-1]
        
        if len(btc_ret) < 2:
            return None
            
        # Pearson correlation (add small epsilon to avoid div by zero if flat)
        corr_btc_eth = np.corrcoef(btc_ret, eth_ret + 1e-9)[0, 1]
        corr_btc_sol = np.corrcoef(btc_ret, sol_ret + 1e-9)[0, 1]
        
        # Determine lead-lag momentum
        eth_momentum = eth_ret[-1]
        sol_momentum = sol_ret[-1]
        
        confidence = 0.0
        score = 0.0
        direction = SignalDirection.NEUTRAL
        
        # If SOL is structurally decoupling (correlation drops) and moving aggressively, it might lead
        if corr_btc_sol < self.decoupling_threshold and abs(sol_momentum) > 0.002:
            if sol_momentum > 0:
                direction = SignalDirection.BULLISH
                score = 3.0 + (sol_momentum * 500)
            else:
                direction = SignalDirection.BEARISH
                score = -3.0 + (sol_momentum * 500)
            confidence = min(0.9, 1.0 - corr_btc_sol + abs(sol_momentum)*10)
            
        # Or if ETH is leading
        elif corr_btc_eth < self.decoupling_threshold and abs(eth_momentum) > 0.002:
            if eth_momentum > 0:
                direction = SignalDirection.BULLISH
                score = 2.0 + (eth_momentum * 500)
            else:
                direction = SignalDirection.BEARISH
                score = -2.0 + (eth_momentum * 500)
            confidence = min(0.85, 1.0 - corr_btc_eth + abs(eth_momentum)*10)
            
        if direction == SignalDirection.NEUTRAL:
            return None
            
        # Determine strict strength
        strength = SignalStrength.MODERATE
        if abs(score) > 4.0 and confidence > 0.7:
            strength = SignalStrength.STRONG
            
        signal = TradingSignal(
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            source=self.name,
            signal_type=SignalType.MOMENTUM,
            direction=direction,
            strength=strength,
            confidence=max(0.1, confidence),
            current_price=current_price,
            metadata={
                "corr_btc_eth": float(corr_btc_eth),
                "corr_btc_sol": float(corr_btc_sol),
                "eth_momentum": float(eth_momentum),
                "sol_momentum": float(sol_momentum),
                "custom_raw_score": float(max(-5.0, min(5.0, score)))
            }
        )
        self._record_signal(signal)
        return signal
