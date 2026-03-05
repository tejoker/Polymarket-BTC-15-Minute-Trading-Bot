import math
import time
from typing import List, Tuple, Dict, Any
from enum import Enum
from dataclasses import dataclass
from loguru import logger
from decimal import Decimal

class SignalDirection(Enum):
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0

@dataclass
class FusedSignal:
    timestamp: float
    direction: SignalDirection
    confidence: float
    score: float
    signals: List[Any]
    weights: Dict[str, float]
    metadata: Dict[str, Any]

class ChIMPFusion:
    """
    Neural / Fuzzy Fusion (ChIMP)
    A lightweight, deterministic forward-pass heuristic mimicking an ONNX-exported
    regime classifier. This evades the Dempster Paradox by assigning dynamic,
    regime-aware weight multipliers rather than relying on orthogonal mass summation.
    """
    def __init__(self, base_weights: Dict[str, float] = None):
        if base_weights is None:
            self.base_weights = {
                "SpikeDetectionProcessor": 1.2,
                "SentimentProcessor": 0.8,
                "PriceDivergenceProcessor": 1.5,
                "OrderBookImbalanceProcessor": 1.0, # Will be ignored during momentum regimes
                "TickVelocityProcessor": 2.0,
                "LeadLagProcessor": 1.5,
                "CoinbasePremiumProcessor": 1.8
            }
        else:
            self.base_weights = base_weights
            
    def _detect_regime(self, signals: List[Any], poly_prob: Decimal) -> str:
        """
        Simulates a fast neural regime classifier.
        Evaluates the contextual state of the market to decide which signals to trust.
        """
        premium_score = sum(abs(s.metadata.get("custom_raw_score", s.score / 20.0)) for s in signals if "Coinbase" in s.source)
        ll_score = sum(abs(s.metadata.get("custom_raw_score", s.score / 20.0)) for s in signals if "LeadLag" in s.source)
        velocity_score = sum(abs(s.metadata.get("custom_raw_score", s.score / 20.0)) for s in signals if "Velocity" in s.source)
        ob_score = sum(abs(s.metadata.get("custom_raw_score", s.score / 20.0)) for s in signals if "OrderBook" in s.source)
        
        if premium_score > 3.0:
            return "institutional_accumulation"
        if velocity_score > 3.0 or ll_score > 3.0:
            return "high_momentum_breakout"
        if poly_prob > 0.85 or poly_prob < 0.15:
            return "overbought_oversold_extreme"
        if velocity_score < 0.5 and ob_score > 1.0:
            return "structural_mean_reversion"
        return "chop"
        
    def _apply_fuzzy_weights(self, regime: str) -> Dict[str, float]:
        """
        Applies non-linear dynamic tuning to the base weights based on the regime.
        """
        dynamic_weights = self.base_weights.copy()
        
        if regime == "institutional_accumulation":
            dynamic_weights["CoinbasePremiumProcessor"] = dynamic_weights.get("CoinbasePremiumProcessor", 1.0) * 2.0
            dynamic_weights["OrderBookImbalanceProcessor"] = dynamic_weights.get("OrderBookImbalanceProcessor", 1.0) * 0.2
            dynamic_weights["LeadLagProcessor"] = dynamic_weights.get("LeadLagProcessor", 1.0) * 0.5
        elif regime == "high_momentum_breakout":
            # Ignore the static order book depth (which causes the DST conflict paradox)
            dynamic_weights["OrderBookImbalanceProcessor"] = 0.0
            # Hyper-focus on momentum and spikes
            dynamic_weights["TickVelocityProcessor"] = dynamic_weights.get("TickVelocityProcessor", 1.0) * 1.5
            dynamic_weights["SpikeDetectionProcessor"] = dynamic_weights.get("SpikeDetectionProcessor", 1.0) * 1.2
            dynamic_weights["LeadLagProcessor"] = dynamic_weights.get("LeadLagProcessor", 1.0) * 1.5
        elif regime == "structural_mean_reversion":
            dynamic_weights["OrderBookImbalanceProcessor"] = dynamic_weights.get("OrderBookImbalanceProcessor", 1.0) * 1.5
            dynamic_weights["TickVelocityProcessor"] = dynamic_weights.get("TickVelocityProcessor", 1.0) * 0.5
            dynamic_weights["PriceDivergenceProcessor"] = dynamic_weights.get("PriceDivergenceProcessor", 1.0) * 1.3
        elif regime == "overbought_oversold_extreme":
            dynamic_weights["TickVelocityProcessor"] = dynamic_weights.get("TickVelocityProcessor", 1.0) * 0.2
            dynamic_weights["PriceDivergenceProcessor"] = dynamic_weights.get("PriceDivergenceProcessor", 1.0) * 1.5
        elif regime == "chop":
            # Flat scaling, reduce trust in everything
            dynamic_weights = {k: v * 0.5 for k, v in dynamic_weights.items()}
            
        return dynamic_weights

    def get_decision(self, signals: List[Any], poly_prob: Decimal) -> Tuple[int, float]:
        """
        Fuzzy forward pass.
        Returns: (Direction_Vector, Confidence_Score)
        """
        if not signals:
            return 0, 0.0
            
        regime = self._detect_regime(signals, poly_prob)
        dynamic_weights = self._apply_fuzzy_weights(regime)
        
        bull_accumulator = 0.0
        bear_accumulator = 0.0
        total_weight = 0.0
        
        for s in signals:
            weight = dynamic_weights.get(s.source, 1.0)
            if weight == 0.0:
                continue # Fuzzy logic ignored this signal vector specifically
                
            # Score usually -5 to +5. Normalize it roughly 0 to 1
            intensity = min(1.0, abs(s.score) / 5.0) 
            weighted_impact = intensity * weight
            
            if "BULLISH" in str(s.direction).upper() or s.score > 0:
                bull_accumulator += weighted_impact
            elif "BEARISH" in str(s.direction).upper() or s.score < 0:
                bear_accumulator += weighted_impact
            
            total_weight += weight
            
        if total_weight == 0:
            return 0, 0.0
            
        # Neural-style activation (Sigmoid over the net difference)
        net_difference = (bull_accumulator - bear_accumulator) / total_weight
        
        # Sigmoid activation to simulate the final ONNX output node
        # 1 / (1 + e^-x), but scaled to -1 to +1 natively via tanh
        activation = math.tanh(net_difference * 2.5) 
        
        if activation > 0.15: # Breakthrough Threshold
            return 1, abs(activation)
        elif activation < -0.15:
            return -1, abs(activation)
        else:
            return 0, 0.0
