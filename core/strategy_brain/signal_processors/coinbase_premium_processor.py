"""
Coinbase Premium Processor (Institutional Arbitrage)
Compares Binance BTC/USDT spot book against Coinbase BTC/USD institutional book.
A persistent positive premium indicates mass US spot buying (ETF flow).
"""
from typing import List, Dict, Any, Optional
from decimal import Decimal

import datetime
from .base_processor import BaseSignalProcessor, TradingSignal, SignalDirection, SignalStrength, SignalType

class CoinbasePremiumProcessor(BaseSignalProcessor):
    def __init__(
        self,
        premium_threshold: float = 10.0, # $10 USD premium
    ):
        super().__init__("CoinbasePremiumProcessor")
        self.premium_threshold = premium_threshold
        
    def process(
        self,
        current_price: Decimal, # Binance BTC price
        historical_prices: List[Decimal],
        metadata: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        
        cb_spot = metadata.get('spot_price') # Coinbase spot
        binance_spot = metadata.get('binance_spot_price') 
        
        if cb_spot is None or binance_spot is None:
            return None
            
        premium = float(cb_spot) - float(binance_spot)
        
        if abs(premium) < self.premium_threshold:
            return None
            
        if premium > self.premium_threshold:
            direction = SignalDirection.BULLISH
            score = min(5.0, premium / 10.0) # $50 premium = 5.0 score
            confidence = min(0.95, 0.4 + (premium / 100.0))
        else:
            direction = SignalDirection.BEARISH
            score = max(-5.0, abs(premium) / 10.0)
            confidence = min(0.95, 0.4 + (abs(premium) / 100.0))
            
        strength = SignalStrength.STRONG if abs(score) > 3.0 else SignalStrength.MODERATE
            
        signal = TradingSignal(
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            source=self.name,
            signal_type=SignalType.ANOMALY,
            direction=direction,
            strength=strength,
            confidence=confidence,
            current_price=current_price,
            metadata={
                "premium_usd": premium,
                "coinbase_spot": float(cb_spot),
                "binance_spot": float(binance_spot),
                "custom_raw_score": score
            }
        )
        self._record_signal(signal)
        return signal
