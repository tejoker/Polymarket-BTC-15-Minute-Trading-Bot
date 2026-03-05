"""
Binance Perpetual Open Interest Change Processor

Tracks the rate of change in BTC perpetual futures open interest to detect
structural fragility (rapid leverage buildup) and post-cascade opportunities.

THEORY:
  Open Interest (OI) = total notional value of all open positions.
  Rising OI means new positions are being opened — new money entering.
  Falling OI means positions are closing — deleveraging or liquidation.

  Rapid OI increase (+3% in 15 min) combined with positive funding signals
  that new leveraged longs are entering a crowded market. This is the
  textbook pre-cascade setup: the market is structurally fragile.

  Rapid OI decrease (−5% in 15 min) signals forced deleveraging — either
  a cascade is underway or just completed. Post-flush OI collapse often
  creates a mean-reversion opportunity (oversold bounce).

ENDPOINT (free, no auth required):
  GET https://fapi.binance.com/futures/data/openInterestHist
    ?symbol=BTCUSDT&period=5m&limit=4
  Returns the last 4 × 5-minute OI snapshots (covers the last 15 minutes).
  Records are returned oldest-first.

SIGNAL LOGIC:
  ΔOI > +3% over 15 min: OI surge → leverage buildup → contrarian BEARISH
  ΔOI > +5% over 15 min: FRAGILE flag → cascade shield activates
  ΔOI < −5% over 15 min: OI flush → post-cascade → contrarian BULLISH
"""
import httpx
from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
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

BINANCE_OI_HIST_URL = "https://fapi.binance.com/futures/data/openInterestHist"


class PerpOIChangeProcessor(BaseSignalProcessor):
    """
    Tracks Binance BTCUSDT perpetual OI rate-of-change over 15 minutes.
    Flags rapid leverage buildup as structural fragility for the cascade shield.
    """

    OI_SURGE_THRESHOLD = 0.03      # +3%: leverage building
    OI_FRAGILITY_THRESHOLD = 0.05  # +5%: structural fragility, cascade risk
    OI_FLUSH_THRESHOLD = -0.05     # −5%: mass deleveraging / post-cascade

    def __init__(
        self,
        cache_seconds: int = 300,
        min_confidence: float = 0.55,
    ):
        super().__init__("PerpOIChangeProcessor")
        self.cache_seconds = cache_seconds
        self.min_confidence = min_confidence

        self._cached_data: Optional[Dict] = None
        self._cache_time: Optional[datetime] = None
        self._async_client: Optional[httpx.AsyncClient] = None
        self._is_fragile: bool = False

        logger.info(
            f"Initialized Perp OI Change Processor: "
            f"surge>{self.OI_SURGE_THRESHOLD*100:.0f}%, "
            f"fragile>{self.OI_FRAGILITY_THRESHOLD*100:.0f}%, "
            f"flush<{self.OI_FLUSH_THRESHOLD*100:.0f}%"
        )

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(timeout=6.0)
        return self._async_client

    @property
    def is_fragile(self) -> bool:
        """True when OI surge crosses the fragility threshold."""
        return self._is_fragile

    async def _fetch_oi_history(self) -> Optional[Dict]:
        try:
            client = self._ensure_client()
            resp = await client.get(
                BINANCE_OI_HIST_URL,
                params={"symbol": "BTCUSDT", "period": "5m", "limit": 4},
            )
            resp.raise_for_status()
            records: List[Dict] = resp.json()

            if len(records) < 2:
                logger.warning("OI history returned too few records")
                return None

            # Records are oldest-first
            oldest_oi = float(records[0]["sumOpenInterest"])
            latest_oi = float(records[-1]["sumOpenInterest"])
            oldest_usd = float(records[0]["sumOpenInterestValue"])
            latest_usd = float(records[-1]["sumOpenInterestValue"])

            oi_change_pct = (
                (latest_oi - oldest_oi) / oldest_oi if oldest_oi > 0 else 0.0
            )
            usd_change_pct = (
                (latest_usd - oldest_usd) / oldest_usd if oldest_usd > 0 else 0.0
            )
            window_minutes = (len(records) - 1) * 5

            result = {
                "oi_change_pct": oi_change_pct,
                "usd_change_pct": usd_change_pct,
                "latest_oi": latest_oi,
                "latest_usd_bn": round(latest_usd / 1e9, 3),
                "window_minutes": window_minutes,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(
                f"Perp OI: Δ={oi_change_pct*100:+.2f}% over {window_minutes}min "
                f"(notional ${latest_usd/1e9:.2f}B)"
            )
            return result

        except Exception as e:
            logger.warning(f"Perp OI fetch failed: {e}")
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
            data = await self._fetch_oi_history()
            if data is None:
                return None
            self._cached_data = data
            self._cache_time = now

        return self._generate_signal(current_price, data)

    def _generate_signal(
        self, current_price: Decimal, data: Dict
    ) -> Optional[TradingSignal]:
        oi_chg = data["oi_change_pct"]

        # Update fragility flag for risk engine consumption
        self._is_fragile = oi_chg >= self.OI_FRAGILITY_THRESHOLD
        if self._is_fragile:
            logger.warning(
                f"PerpOI: FRAGILE — OI surge +{oi_chg*100:.2f}% over "
                f"{data.get('window_minutes', 15)}min "
                f"(threshold={self.OI_FRAGILITY_THRESHOLD*100:.0f}%). "
                f"Cascade shield will activate."
            )

        if oi_chg >= self.OI_SURGE_THRESHOLD:
            # Rapid new-position buildup: crowded leverage = fragile upside
            direction = SignalDirection.BEARISH
            extremeness = (oi_chg - self.OI_SURGE_THRESHOLD) / self.OI_SURGE_THRESHOLD
            confidence = calculate_sigmoid_confidence(
                extremeness=extremeness,
                steepness=3.0,
                midpoint=0.5,
                max_confidence=0.75,
                min_confidence_floor=self.min_confidence,
            )
            if oi_chg >= self.OI_FRAGILITY_THRESHOLD:
                strength = SignalStrength.VERY_STRONG
            elif oi_chg >= 0.04:
                strength = SignalStrength.STRONG
            else:
                strength = SignalStrength.MODERATE
            interpretation = "oi_surge_leverage_buildup_bearish"
            logger.info(
                f"PerpOI: OI surge +{oi_chg*100:.2f}% → leverage buildup → contrarian BEARISH"
            )

        elif oi_chg <= self.OI_FLUSH_THRESHOLD:
            # Mass deleveraging: cascade in progress or just completed → oversold bounce
            direction = SignalDirection.BULLISH
            extremeness = abs(oi_chg - self.OI_FLUSH_THRESHOLD) / abs(
                self.OI_FLUSH_THRESHOLD
            )
            confidence = calculate_sigmoid_confidence(
                extremeness=extremeness,
                steepness=3.0,
                midpoint=0.5,
                max_confidence=0.70,
                min_confidence_floor=self.min_confidence,
            )
            strength = (
                SignalStrength.STRONG if oi_chg <= -0.08 else SignalStrength.MODERATE
            )
            interpretation = "oi_flush_post_cascade_bullish"
            logger.info(
                f"PerpOI: OI flush {oi_chg*100:.2f}% → deleveraging → contrarian BULLISH"
            )

        else:
            logger.debug(
                f"PerpOI: neutral OI change {oi_chg*100:+.2f}% — no signal"
            )
            return None

        if confidence < self.min_confidence:
            return None

        signal = TradingSignal(
            timestamp=datetime.now(),
            source=self.name,
            signal_type=SignalType.VOLUME_SURGE,
            direction=direction,
            strength=strength,
            confidence=confidence,
            current_price=current_price,
            metadata={
                "oi_change_pct": round(oi_chg, 4),
                "oi_change_display": f"{oi_chg*100:+.2f}%",
                "latest_oi_bn_usd": data.get("latest_usd_bn"),
                "window_minutes": data.get("window_minutes", 15),
                "is_fragile": self._is_fragile,
                "interpretation": interpretation,
            },
        )

        self._record_signal(signal)

        logger.info(
            f"Generated {direction.value.upper()} signal (PerpOI): "
            f"Δoi={oi_chg*100:+.2f}%, confidence={confidence:.2%}, score={signal.score:.1f}"
        )

        return signal
