"""
A Predictability-Aware Fusion Engine integrating Dempster-Shafer Evidence
Theory (APTF) or ChIMP Neural/Fuzzy logic.
"""
from typing import List, Dict, Optional, Any
from datetime import datetime
from loguru import logger
import time

import os 
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from .chimp_fusion import ChIMPFusion, FusedSignal, SignalDirection
from core.strategy_brain.signal_processors.base_processor import (
    TradingSignal,
    SignalStrength,
)

class SignalFusionEngine:
    def __init__(self):
        self.weights = {
            "SpikeDetectionProcessor": 1.2,
            "SentimentProcessor": 0.8,
            "PriceDivergenceProcessor": 1.5,
            "OrderBookImbalanceProcessor": 1.0,
            "TickVelocityProcessor": 2.0,
            "DeribitPCRProcessor": 0.5,
            "LeadLagProcessor": 1.5,
            "CoinbasePremiumProcessor": 1.8,
            "default": 1.0
        }
        
        self._signal_history: List[FusedSignal] = []
        self._max_history = 100
        self._fusions_performed = 0
        self._chimp_engine = ChIMPFusion(self.weights)
        
        logger.info("Initialized Signal Fusion Engine with ChIMP")
        
    def set_weight(self, processor_name: str, weight: float) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Weight must be between 0.0 and 1.0")
        self.weights[processor_name] = weight
        self._chimp_engine.base_weights[processor_name] = weight
        logger.info(f"Set weight for {processor_name}: {weight:.2f}")
    
    def fuse_signals(
        self, 
        recent_signals: List[TradingSignal],
        poly_prob: float,
        min_score: float = 60.0
    ) -> Optional[FusedSignal]:
        
        current_time = time.time() if hasattr(time, 'time') else datetime.now().timestamp()
        
        if not recent_signals or len(recent_signals) < 2:
            logger.debug(f"Not enough recent signals: {len(recent_signals)}")
            return None
        
        # Bypass Dempster-Shafer Theory to evade the Epistemic Uncertainty Paradox.
        # Utilize the ChIMP Neural / Fuzzy heuristic fast-path.
        direction_val, confidence = self._chimp_engine.get_decision(recent_signals, poly_prob)
        
        if direction_val == 0:
            logger.debug(f"ChIMP Neural Fusion resolved to CHOP (0 conviction). No execution.")
            return None
            
        direction = SignalDirection.BULLISH if direction_val == 1 else SignalDirection.BEARISH
        consensus_score = confidence * 100.0
        
        if consensus_score < min_score:
            logger.debug(f"ChIMP Consensus score too low: {consensus_score:.1f} < {min_score}")
            return None
        
        fused = FusedSignal(
            timestamp=current_time,
            direction=direction,
            confidence=confidence,
            score=consensus_score,
            signals=recent_signals,
            weights=self.weights.copy(),
            metadata={
                "chimp_activation": confidence
            }
        )
        
        self._fusions_performed += 1
        self._signal_history.append(fused)
        if len(self._signal_history) > self._max_history:
            self._signal_history.pop(0)
        
        logger.info(
            f"ChIMP Fused {len(recent_signals)} signals → {direction} "
            f"(score={consensus_score:.1f})"
        )
        
        return fused
    
    def get_recent_fusions(self, limit: int = 10) -> List[FusedSignal]:
        return self._signal_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        if not self._signal_history:
            return {
                "total_fusions": self._fusions_performed,
                "recent_fusions": 0,
            }
            
        return {
            "total_fusions": self._fusions_performed,
            "recent_fusions": len(self._signal_history),
            "bullish_ratio": sum(1 for s in self._signal_history if s.direction == SignalDirection.BULLISH) / len(self._signal_history),
            "avg_confidence": sum(s.confidence for s in self._signal_history) / len(self._signal_history)
        }

_fusion_engine_instance = None

def get_fusion_engine() -> SignalFusionEngine:
    global _fusion_engine_instance
    if _fusion_engine_instance is None:
        _fusion_engine_instance = SignalFusionEngine()
    return _fusion_engine_instance