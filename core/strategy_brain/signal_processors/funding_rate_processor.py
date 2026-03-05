"""
Binance Perpetual Funding Rate Signal Processor

Detects structural fragility from crowded perpetual futures positioning.

THEORY:
  Perpetual futures funding rates are the cost of carrying leveraged positions.
  Positive funding = longs pay shorts = long-heavy positioning.
  Negative funding = shorts pay longs = short-heavy positioning.

  When funding is extreme AND open interest is elevated, the market is in a
  pre-cascade regime: a minor adverse move forces margin calls, which forces
  liquidations, which causes further adverse moves (cascade).

  For a Polymarket binary options taker, this matters because a 3-5% BTC move
  in 15 minutes (caused by a cascade) can flip the binary resolution against
  an otherwise-correct directional thesis.

ENDPOINT (free, no auth required):
  GET https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT
  Returns: lastFundingRate, markPrice, indexPrice, nextFundingTime

SIGNAL THRESHOLDS:
  fundingRate > +0.001  (+0.1%/8h ≈ +110% annualized): longs crowded → contrarian BEARISH
  fundingRate < -0.0005 (−0.05%/8h): shorts crowded → contrarian BULLISH
  abs(fundingRate) > 0.003 (>0.3%/8h): FRAGILE — cascade shield activates in risk engine
"""
import httpx
from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from loguru import logger

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.math.sigmoid import calculate_sigmoid_confidence
from core.strategy_brain.signal_processors.base_processor import (
    BaseSignalProcessor,
    TradingSignal,
    SignalType,
    SignalDirection,
    SignalStrength,
)

BINANCE_PREMIUM_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"


class FundingRateProcessor(BaseSignalProcessor):
    """
    Binance BTCUSDT perpetual funding rate signal processor.

    Identifies crowded-positioning regimes where structural fragility
    makes the market susceptible to liquidation cascades, and exposes
    an is_fragile flag consumed by the risk engine cascade shield.
    """

    # Crowding thresholds
    LONG_CROWDED_THRESHOLD = 0.001    # +0.1%/8h: clearly crowded longs
    SHORT_CROWDED_THRESHOLD = -0.0005 # −0.05%/8h: crowded shorts

    # Fragility threshold: activates cascade shield in risk engine
    FRAGILITY_THRESHOLD = 0.003       # ±0.3%/8h: structural fragility

    def __init__(
        self,
        cache_seconds: int = 300,
        min_confidence: float = 0.55,
    ):
        super().__init__("FundingRateProcessor")
        self.cache_seconds = cache_seconds
        self.min_confidence = min_confidence

        self._cached_data: Optional[Dict] = None
        self._cache_time: Optional[datetime] = None
        self._async_client: Optional[httpx.AsyncClient] = None
        self._is_fragile: bool = False

        logger.info(
            f"Initialized Funding Rate Processor: "
            f"long_crowded>{self.LONG_CROWDED_THRESHOLD*100:.3f}%/8h, "
            f"fragile>±{self.FRAGILITY_THRESHOLD*100:.3f}%/8h"
        )

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(timeout=6.0)
        return self._async_client

    @property
    def is_fragile(self) -> bool:
        """True when funding rate is extreme enough to warrant cascade shield."""
        return self._is_fragile

    async def _fetch_funding(self) -> Optional[Dict]:
        try:
            client = self._ensure_client()
            resp = await client.get(BINANCE_PREMIUM_URL, params={"symbol": "BTCUSDT"})
            resp.raise_for_status()
            data = resp.json()

            funding_rate = float(data.get("lastFundingRate", 0.0))
            mark_price = float(data.get("markPrice", 0.0))
            index_price = float(data.get("indexPrice", 0.0))
            basis_pct = (mark_price - index_price) / index_price if index_price > 0 else 0.0

            result = {
                "funding_rate": funding_rate,
                "mark_price": mark_price,
                "index_price": index_price,
                "basis_pct": basis_pct,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(
                f"Binance funding: rate={funding_rate*100:.4f}%/8h "
                f"(mark=${mark_price:,.2f}, basis={basis_pct*100:.3f}%)"
            )
            return result

        except Exception as e:
            logger.warning(f"Funding rate fetch failed: {e}")
            return None

    async def process(
        self,
        current_price: Decimal,
        historical_prices: list,
        metadata: Dict[str, Any] = None,
    ) -> Optional[TradingSignal]:
        if not self.is_enabled:
            return None

        now = datetime.now(timezone.utc)
        cache_valid = (
            self._cached_data is not None
            and self._cache_time is not None
            and (now - self._cache_time).total_seconds() < self.cache_seconds
        )

        if cache_valid:
            data = self._cached_data
        else:
            data = await self._fetch_funding()
            if data is None:
                return None
            self._cached_data = data
            self._cache_time = now

        return self._generate_signal(current_price, data)

    def _generate_signal(
        self, current_price: Decimal, data: Dict
    ) -> Optional[TradingSignal]:
        fr = data["funding_rate"]

        # Update fragility flag for risk engine consumption
        self._is_fragile = abs(fr) >= self.FRAGILITY_THRESHOLD
        if self._is_fragile:
            logger.warning(
                f"FundingRate: FRAGILE REGIME — rate={fr*100:.4f}%/8h "
                f"(threshold=±{self.FRAGILITY_THRESHOLD*100:.3f}%/8h). "
                f"Cascade shield will activate."
            )

        if fr >= self.LONG_CROWDED_THRESHOLD:
            direction = SignalDirection.BEARISH
            extremeness = (fr - self.LONG_CROWDED_THRESHOLD) / self.LONG_CROWDED_THRESHOLD
            confidence = calculate_sigmoid_confidence(
                extremeness=extremeness,
                steepness=4.0,
                midpoint=0.5,
                max_confidence=0.80,
                min_confidence_floor=self.min_confidence,
            )
            if fr >= self.FRAGILITY_THRESHOLD:
                strength = SignalStrength.VERY_STRONG
            elif fr >= 0.002:
                strength = SignalStrength.STRONG
            else:
                strength = SignalStrength.MODERATE
            interpretation = "crowded_longs_contrarian_bearish"
            logger.info(
                f"FundingRate: HIGH rate={fr*100:.4f}%/8h → longs crowded → contrarian BEARISH"
            )

        elif fr <= self.SHORT_CROWDED_THRESHOLD:
            direction = SignalDirection.BULLISH
            extremeness = (
                (self.SHORT_CROWDED_THRESHOLD - fr) / abs(self.SHORT_CROWDED_THRESHOLD)
            )
            confidence = calculate_sigmoid_confidence(
                extremeness=extremeness,
                steepness=4.0,
                midpoint=0.5,
                max_confidence=0.80,
                min_confidence_floor=self.min_confidence,
            )
            if fr <= -self.FRAGILITY_THRESHOLD:
                strength = SignalStrength.VERY_STRONG
            elif fr <= -0.001:
                strength = SignalStrength.STRONG
            else:
                strength = SignalStrength.MODERATE
            interpretation = "crowded_shorts_contrarian_bullish"
            logger.info(
                f"FundingRate: LOW rate={fr*100:.4f}%/8h → shorts crowded → contrarian BULLISH"
            )

        else:
            logger.debug(
                f"FundingRate: balanced rate={fr*100:.4f}%/8h — no crowding signal"
            )
            return None

        if confidence < self.min_confidence:
            return None

        signal = TradingSignal(
            timestamp=datetime.now(),
            source=self.name,
            signal_type=SignalType.SENTIMENT_SHIFT,
            direction=direction,
            strength=strength,
            confidence=confidence,
            current_price=current_price,
            metadata={
                "funding_rate": round(fr, 6),
                "funding_rate_pct_8h": round(fr * 100, 4),
                "basis_pct": round(data.get("basis_pct", 0.0), 4),
                "is_fragile": self._is_fragile,
                "interpretation": interpretation,
            },
        )

        self._record_signal(signal)

        logger.info(
            f"Generated {direction.value.upper()} signal (FundingRate): "
            f"rate={fr*100:.4f}%/8h, confidence={confidence:.2%}, score={signal.score:.1f}"
        )

        return signal
