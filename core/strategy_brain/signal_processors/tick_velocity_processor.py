"""
Tick Velocity Signal Processor
Measures how fast the Polymarket UP probability is moving in the
last 60 seconds before the trade window opens.

WHY THIS WORKS:
  If the "Up" probability moves from 0.50 → 0.57 in the last 45 seconds,
  BEFORE your bot even looks at it, that reflects real order flow from
  other traders reacting to BTC spot movement. The probability is being
  pushed by real money, and it often continues in the same direction
  for at least another 30–60 seconds.

  This is "price action" on the Polymarket itself — not a lagging
  external indicator. It's the most direct signal available.

HOW IT WORKS:
  The strategy stores a rolling tick buffer:
    self._tick_buffer = deque of {'ts': datetime, 'price': Decimal}

  This processor receives that buffer via metadata['tick_buffer'].

  It computes:
    1. 60s velocity  = (now_price - price_60s_ago) / price_60s_ago
    2. 30s velocity  = (now_price - price_30s_ago) / price_30s_ago
    3. Acceleration  = 30s_velocity - (60s_velocity - 30s_velocity)
                       (is it speeding up or slowing down?)

  Signal thresholds (for 0–1 probability prices):
    velocity > +1.5%  in 60s → BULLISH
    velocity < -1.5%  in 60s → BEARISH
    acceleration bonus: if move is accelerating → higher confidence

INTEGRATION:
  In bot.py on_quote_tick(), add:
    self._tick_buffer.append({'ts': now, 'price': mid_price})

  In _fetch_market_context(), add:
    metadata['tick_buffer'] = list(self._tick_buffer)
"""
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from collections import deque
from typing import Optional, Dict, Any, List
from loguru import logger

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.strategy_brain.signal_processors.base_processor import (
    BaseSignalProcessor,
    TradingSignal,
    SignalType,
    SignalDirection,
    SignalStrength,
)


class TickVelocityProcessor(BaseSignalProcessor):
    """
    Measures Polymarket probability velocity over the last 60 seconds.

    Fast moves in probability = real order flow = tradeable signal.
    """

    def __init__(
        self,
        velocity_threshold_60s: float = 0.015,   # 1.5% move in 60s
        velocity_threshold_30s: float = 0.010,   # 1.0% move in 30s
        min_ticks: int = 5,                       # need at least 5 ticks in window
        min_confidence: float = 0.55,
    ):
        super().__init__("TickVelocity")

        self.velocity_threshold_60s = velocity_threshold_60s
        self.velocity_threshold_30s = velocity_threshold_30s
        self.min_ticks = min_ticks
        self.min_confidence = min_confidence

        logger.info(
            f"Initialized Tick Velocity Processor: "
            f"60s_threshold={velocity_threshold_60s:.1%}, "
            f"30s_threshold={velocity_threshold_30s:.1%}"
        )

    def _get_price_at(
        self,
        tick_buffer,
        seconds_ago: float,
        now: datetime,
    ) -> Optional[float]:
        """Find the tick price closest to `seconds_ago` seconds before now."""
        target_ts = now.timestamp() - seconds_ago
        return tick_buffer.get_price_at_time(target_ts, tolerance=15.0)

    def process(
        self,
        current_price: Decimal,
        historical_prices: list,
        metadata: Dict[str, Any] = None,
    ) -> Optional[TradingSignal]:
        if not self.is_enabled or not metadata:
            return None

        tick_buffer = metadata.get("tick_buffer")
        if not tick_buffer or len(tick_buffer) < self.min_ticks:
            logger.debug(
                f"TickVelocity: insufficient ticks "
                f"({len(tick_buffer) if tick_buffer else 0} < {self.min_ticks})"
            )
            return None

        now = datetime.now(timezone.utc)
        curr = float(current_price)

        # Get historical prices from the buffer
        # Compute velocities utilizing Savitzky-Golay (per-tick momentum)
        recent_ticks = tick_buffer.get_recent_prices(15)
        if len(recent_ticks) < 15:
            return None
            
        smoothed_price, sg_vel, sg_accel = self.sg_filter.get_kinematics(recent_ticks)

        logger.info(
            f"TickVelocity(SG): curr={curr:.4f}, "
            f"smoothed={smoothed_price:.4f}, "
            f"vel={sg_vel*100:+.3f}%, "
            f"accel={sg_accel*100:+.4f}%"
        )

        # SG Velocity threshold (re-calibrated for per-tick convolution)
        threshold = 0.005  # 0.5% per tick density

        if abs(sg_vel) < threshold:
            logger.debug(
                f"TickVelocity(SG): {sg_vel*100:+.3f}% below threshold "
                f"{threshold*100:.1f}% — no signal"
            )
            return None

        direction = SignalDirection.BULLISH if sg_vel > 0 else SignalDirection.BEARISH
        abs_vel = abs(sg_vel)

        # Strength by velocity magnitude
        if abs_vel >= 0.02:      # >2%
            strength = SignalStrength.VERY_STRONG
        elif abs_vel >= 0.01:    # >1%
            strength = SignalStrength.STRONG
        elif abs_vel >= 0.007:   # >0.7%
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        # Base confidence from velocity magnitude (utilizing logistic growth)
        import math
        confidence = 1.0 / (1.0 + math.exp(-200.0 * (abs_vel - threshold)))
        confidence = min(0.90, max(self.min_confidence, confidence))

        # Acceleration bonus
        accel_same_direction = (
            (sg_accel > 0 and sg_vel > 0) or
            (sg_accel < 0 and sg_vel < 0)
        )
        if accel_same_direction and abs(sg_accel) > 0.001:
            confidence = min(0.95, confidence + 0.08)
            logger.info(f"TickVelocity: acceleration bonus applied ({sg_accel*100:+.4f}%)")

        # If 60s velocity conflicts with 30s velocity — reduce confidence
        if vel_60s is not None and vel_30s is not None:
            if (vel_60s > 0) != (vel_30s > 0):
                confidence *= 0.80
                logger.info("TickVelocity: velocity reversal — confidence reduced")

        if confidence < self.min_confidence:
            return None

        signal = TradingSignal(
            timestamp=datetime.now(),
            source=self.name,
            signal_type=SignalType.MOMENTUM,
            direction=direction,
            strength=strength,
            confidence=confidence,
            current_price=current_price,
            metadata={
                "velocity_sg": round(sg_vel, 6),
                "acceleration_sg": round(sg_accel, 6),
                "smoothed_price": round(smoothed_price, 6),
                "ticks_in_buffer": len(tick_buffer),
            }
        )

        self._record_signal(signal)

        logger.info(
            f"Generated {direction.value.upper()} signal (TickVelocity SG): "
            f"vel={sg_vel*100:+.3f}%, accel={sg_accel*100:+.4f}%, "
            f"confidence={confidence:.2%}, score={signal.score:.1f}"
        )

        return signal