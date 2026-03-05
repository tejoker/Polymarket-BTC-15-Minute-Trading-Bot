"""
Learning Engine
Learns from trading performance to optimize strategy weights
"""
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from loguru import logger

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.math.ftrl import FTRLProximal
from core.math.gmadl import compute_source_gradient
from monitoring.performance_tracker import get_performance_tracker, Trade
from core.strategy_brain.fusion_engine.signal_fusion import get_fusion_engine


@dataclass
class SignalPerformance:
    """Performance metrics for a signal source."""
    source_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_pnl: Decimal
    total_pnl: Decimal
    avg_confidence: float
    avg_score: float
    last_updated: datetime


class LearningEngine:
    """
    Learning engine that optimizes strategy based on performance.
    
    Features:
    - Analyzes signal source performance
    - Adjusts fusion weights
    - Identifies winning patterns
    - Improves over time
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        min_trades_for_learning: int = 10,
    ):
        """
        Initialize learning engine.
        
        Args:
            learning_rate: How quickly to adjust weights (0-1)
            min_trades_for_learning: Minimum trades before adjusting
        """
        self.learning_rate = learning_rate
        self.min_trades = min_trades_for_learning
        
        # Components
        self.performance = get_performance_tracker()
        self.fusion = get_fusion_engine()
        
        # Signal performance tracking
        self._signal_performance: Dict[str, SignalPerformance] = {}
        
        # Learning history
        self._weight_adjustments: List[Dict[str, Any]] = []
        
        # 2026 SOTA FTRL-Proximal Optimizer
        self.ftrl = FTRLProximal(alpha=self.learning_rate)
        
        logger.info(
            f"Initialized Learning Engine "
            f"(learning_rate={learning_rate}, min_trades={min_trades_for_learning})"
        )
    
    def analyze_signal_performance(
        self,
        lookback_days: int = 7,
    ) -> Dict[str, SignalPerformance]:
        """
        Analyze performance of each signal source.
        
        Args:
            lookback_days: Number of days to analyze
            
        Returns:
            Performance metrics per signal source
        """
        cutoff = datetime.now() - timedelta(days=lookback_days)
        trades = self.performance.get_trade_history(
            limit=1000,
            start_date=cutoff,
        )
        
        # Group trades by signal source
        source_trades: Dict[str, List[Trade]] = {}
        
        for trade in trades:
            # Extract signal source from metadata
            # This assumes trades store which signal triggered them
            sources = trade.metadata.get("signal_sources", [])
            
            for source in sources:
                if source not in source_trades:
                    source_trades[source] = []
                
                source_trades[source].append(trade)
        
        # Calculate performance per source
        performances = {}
        
        for source, source_trade_list in source_trades.items():
            wins = [t for t in source_trade_list if t.pnl > 0]
            losses = [t for t in source_trade_list if t.pnl < 0]
            
            total = len(source_trade_list)
            win_count = len(wins)
            loss_count = len(losses)
            
            win_rate = win_count / total if total > 0 else 0.0
            
            avg_pnl = sum(t.pnl for t in source_trade_list) / total if total > 0 else Decimal("0")
            total_pnl = sum(t.pnl for t in source_trade_list)
            
            avg_conf = sum(t.signal_confidence for t in source_trade_list) / total if total > 0 else 0.0
            avg_score = sum(t.signal_score for t in source_trade_list) / total if total > 0 else 0.0
            
            perf = SignalPerformance(
                source_name=source,
                total_trades=total,
                winning_trades=win_count,
                losing_trades=loss_count,
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                total_pnl=total_pnl,
                avg_confidence=avg_conf,
                avg_score=avg_score,
                last_updated=datetime.now(),
            )
            
            performances[source] = perf
            self._signal_performance[source] = perf
        
        logger.info(f"Analyzed performance for {len(performances)} signal sources")
        
        return performances
    
    def calculate_optimal_weights(
        self,
        performances: Dict[str, SignalPerformance],
    ) -> Dict[str, float]:
        """
        Calculate optimal weights using GMADL-weighted FTRL gradients.

        For each signal source, attempts to build (r_t, y_hat_t) pairs from
        trade metadata and compute a GMADL gradient that magnitude-weights the
        directional accuracy of the signal.

        GMADL gradient interpretation:
          - Negative gradient → signal performed well on large moves → weight boosted
          - Positive gradient → signal was wrong on large moves → weight suppressed
          - Small-magnitude moves contribute near-zero gradient → noise filtered

        Falls back to the original flat win-rate / PnL gradient for sources that
        lack per-trade direction metadata (trades recorded before this upgrade).
        """
        # Fetch individual trades once for GMADL pair construction.
        # The per-source aggregate (performances) doesn't carry per-trade direction.
        recent_trades = self.performance.get_trade_history(limit=500)

        for source, perf in performances.items():
            if perf.total_trades < self.min_trades:
                continue

            # Build (realized_return, predicted_score) pairs for GMADL.
            #
            # r_t   = BTC-frame return: pnl_pct for LONG, -pnl_pct for SHORT.
            #         Positive when BTC went up, negative when BTC went down.
            # y_hat = direction_sign × avg_confidence.
            #         Encodes both the direction the source voted and how strongly.
            trade_pairs = []
            for trade in recent_trades:
                dirs = trade.metadata.get("signal_directions", {})
                if source not in dirs:
                    continue  # This trade predates GMADL tracking

                direction_sign = dirs[source]  # +1 BULLISH, -1 BEARISH

                # Convert trade return to BTC-frame (unsigned by trade direction)
                if trade.direction == "long":
                    r_t = trade.pnl_pct  # positive if BTC went up
                else:
                    r_t = -trade.pnl_pct  # short: pnl_pct positive = BTC went down

                y_hat = direction_sign * perf.avg_confidence
                trade_pairs.append((r_t, y_hat))

            if trade_pairs:
                # GMADL magnitude-weighted directional gradient
                total_gradient = compute_source_gradient(
                    trade_pairs, k=10.0, p=1.0
                )
                logger.debug(
                    f"GMADL gradient [{source}]: {total_gradient:.5f} "
                    f"({len(trade_pairs)} directional trades)"
                )
            else:
                # Fallback: original flat win-rate / PnL gradient.
                # Used for sources without per-trade direction data (old trades).
                win_rate_loss = 0.50 - perf.win_rate
                pnl_loss = -float(perf.total_pnl) / 100.0
                total_gradient = (win_rate_loss * 0.6) + (pnl_loss * 0.4)
                logger.debug(
                    f"Fallback gradient [{source}]: {total_gradient:.5f} "
                    f"(no GMADL trades yet — using win-rate/PnL)"
                )

            # Step the FTRL optimizer
            self.ftrl.update(source, total_gradient)
            
        feature_ids = list(performances.keys())
        
        # Ensure all existing features have at least a baseline if unseen
        for source in self.fusion.weights.keys():
            if source not in feature_ids:
                feature_ids.append(source)
                
        new_weights = self.ftrl.get_normalized_weights(feature_ids)
        
        # In case FTRL collapses to zero for everything, maintain defaults
        if not new_weights:
            return self.fusion.weights.copy()
            
        return new_weights
    
    async def optimize_weights(self) -> Dict[str, float]:
        """
        Optimize signal fusion weights based on performance.
        
        Returns:
            New weights
        """
        logger.info("=" * 60)
        logger.info("OPTIMIZING SIGNAL WEIGHTS")
        logger.info("=" * 60)
        
        # Analyze performance
        performances = self.analyze_signal_performance(lookback_days=7)
        
        if not performances:
            logger.warning("No performance data available for optimization")
            return self.fusion.weights.copy()
        
        # Calculate optimal weights
        new_weights = self.calculate_optimal_weights(performances)
        
        # Log changes
        logger.info("Weight adjustments:")
        for source, new_weight in new_weights.items():
            old_weight = self.fusion.weights.get(source, 0.0)
            change = new_weight - old_weight
            
            logger.info(
                f"  {source}: {old_weight:.3f} → {new_weight:.3f} "
                f"({change:+.3f})"
            )
        
        # Apply new weights
        for source, weight in new_weights.items():
            self.fusion.set_weight(source, weight)
        
        # Record adjustment
        self._weight_adjustments.append({
            "timestamp": datetime.now(),
            "old_weights": self.fusion.weights.copy(),
            "new_weights": new_weights.copy(),
            "performances": {
                source: {
                    "win_rate": perf.win_rate,
                    "total_pnl": float(perf.total_pnl),
                    "trades": perf.total_trades,
                }
                for source, perf in performances.items()
            },
        })
        
        logger.info("✓ Weights optimized successfully")
        
        return new_weights
    
    def get_signal_rankings(self) -> List[Dict[str, Any]]:
        """
        Get signals ranked by performance.
        
        Returns:
            List of signals sorted by performance
        """
        rankings = []
        
        for source, perf in self._signal_performance.items():
            rankings.append({
                "source": source,
                "win_rate": perf.win_rate,
                "total_pnl": float(perf.total_pnl),
                "avg_pnl": float(perf.avg_pnl),
                "total_trades": perf.total_trades,
                "current_weight": self.fusion.weights.get(source, 0.0),
            })
        
        # Sort by total P&L
        rankings.sort(key=lambda x: x["total_pnl"], reverse=True)
        
        return rankings
    
    def get_learning_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get history of weight adjustments.
        
        Args:
            limit: Max adjustments to return
            
        Returns:
            List of weight adjustments
        """
        return self._weight_adjustments[-limit:]
    
    def export_insights(self) -> Dict[str, Any]:
        """
        Export learning insights.
        
        Returns:
            Insights dict
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "signal_performance": {
                source: {
                    "win_rate": perf.win_rate,
                    "total_pnl": float(perf.total_pnl),
                    "total_trades": perf.total_trades,
                    "current_weight": self.fusion.weights.get(source, 0.0),
                }
                for source, perf in self._signal_performance.items()
            },
            "signal_rankings": self.get_signal_rankings(),
            "recent_adjustments": self.get_learning_history(5),
        }


# Singleton instance
_learning_engine_instance = None

def get_learning_engine() -> LearningEngine:
    """Get singleton learning engine."""
    global _learning_engine_instance
    if _learning_engine_instance is None:
        _learning_engine_instance = LearningEngine()
    return _learning_engine_instance