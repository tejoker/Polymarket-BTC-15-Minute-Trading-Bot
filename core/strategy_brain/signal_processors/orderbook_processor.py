"""
Order Book Imbalance Signal Processor
Reads the Polymarket CLOB order book for the current YES token and
detects when buy-side or sell-side pressure is heavily skewed.

WHY THIS WORKS:
  The Polymarket CLOB (Central Limit Order Book) shows exactly how many
  dollars are queued to buy "Up" vs sell "Up" at various price levels.

  If $800 is sitting on the bid (buy side) and only $200 on the ask
  (sell side), someone large expects BTC to go UP → follow them BULLISH.

  This is real-time, forward-looking information that reflects what
  sophisticated market participants are actually doing RIGHT NOW —
  not a lagging indicator.

API USED:
  GET https://clob.polymarket.com/book?token_id=<YES_token_id>

  Returns:
    {
      "bids": [{"price": "0.52", "size": "150"}, ...],  ← buyers of YES
      "asks": [{"price": "0.54", "size": "80"},  ...],  ← sellers of YES
    }

SIGNAL LOGIC:
  imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
  Range: -1.0 (all sellers) to +1.0 (all buyers)

  imbalance > +0.30  → BULLISH  (heavy buy pressure)
  imbalance < -0.30  → BEARISH  (heavy sell pressure)
  |imbalance| < 0.30 → no signal (balanced book)

  We also check WALL detection: a single order > 20% of total book
  volume indicates a large player taking a strong position.
"""
import httpx
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, Any, List
from loguru import logger

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.math.lit_ofi import LobTransformerAttention, OrderFlowImbalance

from core.strategy_brain.signal_processors.base_processor import (
    BaseSignalProcessor,
    TradingSignal,
    SignalType,
    SignalDirection,
    SignalStrength,
)

CLOB_BASE = "https://clob.polymarket.com"


class OrderBookImbalanceProcessor(BaseSignalProcessor):
    """
    Detects order book imbalance on the Polymarket CLOB.

    Wired into the strategy by passing the YES token_id via metadata:
      metadata['yes_token_id'] = <token id string>

    This is set once per market in _load_all_btc_instruments and stored
    on the strategy as self._yes_token_id.
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.30,   # 30% skew to signal
        wall_threshold: float = 0.20,         # single order > 20% of book = wall
        min_book_volume: float = 50.0,        # ignore books with < $50 total (illiquid)
        min_confidence: float = 0.55,
        top_levels: int = 10,                 # how many price levels to consider
    ):
        super().__init__("OrderBookImbalance")

        self.imbalance_threshold = imbalance_threshold
        self.wall_threshold = wall_threshold
        self.min_book_volume = min_book_volume
        self.min_confidence = min_confidence
        self.top_levels = top_levels

        # 2026 SOTA Liquidity Topology
        self.lit_attention = LobTransformerAttention(depth_levels=self.top_levels, embedding_dim=4)
        self.ofi = OrderFlowImbalance()

        # Persistent async HTTP client (set externally or created on first use)
        self._async_client: Optional[httpx.AsyncClient] = None
        self._cache: Optional[Dict] = None

        logger.info(
            f"Initialized Order Book Imbalance Processor: "
            f"imbalance_threshold={imbalance_threshold:.0%}, "
            f"wall_threshold={wall_threshold:.0%}, "
            f"min_book_volume=${min_book_volume:.0f}"
        )

    async def _ensure_async_client(self) -> httpx.AsyncClient:
        """Return persistent async HTTP client, creating if needed."""
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(timeout=5.0)
        return self._async_client

    async def fetch_order_book_async(self, token_id: str) -> Optional[Dict]:
        """Fetch order book from Polymarket CLOB asynchronously."""
        try:
            client = await self._ensure_async_client()
            resp = await client.get(
                f"{CLOB_BASE}/book",
                params={"token_id": token_id},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"OrderBook fetch failed for {token_id[:16]}...: {e}")
            return None

    def fetch_order_book(self, token_id: str) -> Optional[Dict]:
        """Sync fallback -- only used if called outside async context."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(
                    f"{CLOB_BASE}/book",
                    params={"token_id": token_id},
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning(f"OrderBook fetch failed for {token_id[:16]}...: {e}")
            return None

    def _parse_levels(self, levels: List[Dict]) -> float:
        """Sum total volume across price levels (returns USD volume)."""
        total = 0.0
        for level in levels[:self.top_levels]:
            try:
                price = float(level.get("price", 0))
                size = float(level.get("size", 0))
                total += price * size   # USD value at each level
            except (ValueError, TypeError):
                continue
        return total

    def _detect_wall(self, levels: List[Dict], total_volume: float) -> Optional[float]:
        """Return the size of the largest single order if it's a wall, else None."""
        if total_volume <= 0:
            return None
        for level in levels[:self.top_levels]:
            try:
                price = float(level.get("price", 0))
                size = float(level.get("size", 0))
                order_usd = price * size
                if order_usd / total_volume >= self.wall_threshold:
                    return order_usd
            except (ValueError, TypeError):
                continue
        return None

    async def process(
        self,
        current_price: Decimal,
        historical_prices: list,
        metadata: Dict[str, Any] = None,
    ) -> Optional[TradingSignal]:
        """Fetch order book asynchronously and generate signal."""
        if not self.is_enabled or not metadata:
            return None

        token_id = metadata.get("yes_token_id")
        if not token_id:
            return None

        try:
            book = await self.fetch_order_book_async(token_id)
            if not book:
                return None

            bids = book.get("bids", [])
            asks = book.get("asks", [])

            # We sum simply for volume logging, but use LiT for signal
            bid_volume = self._parse_levels(bids)
            ask_volume = self._parse_levels(asks)
            total_volume = bid_volume + ask_volume

            if total_volume < self.min_book_volume:
                logger.debug(
                    f"OrderBook too thin: ${total_volume:.1f} < ${self.min_book_volume:.1f} — skipping"
                )
                return None

            # --- Extract structured depth patches ---
            def get_arrays(levels, max_depth):
                p_arr, v_arr = [], []
                for level in levels[:max_depth]:
                    try:
                        p, v = float(level.get("price", 0)), float(level.get("size", 0))
                        if p > 0 and v > 0:
                            p_arr.append(p)
                            v_arr.append(v)
                    except: pass
                return p_arr, v_arr

            b_p, b_v = get_arrays(bids, self.top_levels)
            a_p, a_v = get_arrays(asks, self.top_levels)
            mid = float(current_price)

            # Limit Order Book Transformer (LiT) Directional Pressure ([-1, 1])
            lit_imbalance = self.lit_attention.compute_attention(
                bid_prices=b_p, bid_vols=b_v,
                ask_prices=a_p, ask_vols=a_v,
                mid_price=mid
            )

            # Strict Order Flow Imbalance (OFI) update
            best_bid_p, best_bid_v = (b_p[0], b_v[0]) if b_p else (0.0, 0.0)
            best_ask_p, best_ask_v = (a_p[0], a_v[0]) if a_p else (1.0, 0.0)
            ofi_diff = self.ofi.update(best_bid_p, best_bid_v, best_ask_p, best_ask_v)

            logger.info(
                f"OrderBook(LiT): bids=${bid_volume:.1f}, asks=${ask_volume:.1f}, "
                f"lit_pressure={lit_imbalance:+.3f}, OFI={ofi_diff:+.1f}"
            )

            bid_wall = self._detect_wall(bids, total_volume)
            ask_wall = self._detect_wall(asks, total_volume)

            # Use LiT pressure directly instead of scalar imbalance
            if abs(lit_imbalance) < self.imbalance_threshold:
                logger.debug(f"OrderBook(LiT) balanced (pressure={lit_imbalance:+.3f}) — no signal")
                return None

            direction = SignalDirection.BULLISH if lit_imbalance > 0 else SignalDirection.BEARISH
            abs_imb = abs(lit_imbalance)

            # Recalibrate magnitude bounds for LiT output space
            if abs_imb >= 0.60:
                strength = SignalStrength.VERY_STRONG
            elif abs_imb >= 0.45:
                strength = SignalStrength.STRONG
            elif abs_imb >= 0.30:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK

            # Base Confidence from mathematical pressure
            confidence = min(0.85, 0.55 + abs_imb * 0.40)
            
            # Incorporate OFI consistency bonus
            if (direction == SignalDirection.BULLISH and ofi_diff > 0) or \
               (direction == SignalDirection.BEARISH and ofi_diff < 0):
                confidence = min(0.92, confidence + 0.05)
                logger.debug(f"OrderBook(LiT): OFI confirms direction -> confidence {confidence:.2%}")

            wall_side = bid_wall if direction == SignalDirection.BULLISH else ask_wall
            if wall_side:
                # We verify the wall aligns with our LiT pressure to prevent spoof falling
                confidence = min(0.95, confidence + 0.05)
                logger.info(
                    f"Genuine Wall confirmed on {'bid' if direction == SignalDirection.BULLISH else 'ask'} "
                    f"side: ${wall_side:.1f}"
                )

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
                    "bid_volume_usd": round(bid_volume, 2),
                    "ask_volume_usd": round(ask_volume, 2),
                    "total_volume_usd": round(total_volume, 2),
                    "lit_imbalance": round(lit_imbalance, 4),
                    "ofi": round(ofi_diff, 2),
                    "bid_wall_usd": round(bid_wall, 2) if bid_wall else None,
                    "ask_wall_usd": round(ask_wall, 2) if ask_wall else None,
                }
            )

            self._record_signal(signal)
            logger.info(
                f"Generated {direction.value.upper()} signal (OrderBook LiT): "
                f"pressure={lit_imbalance:+.3f}, confidence={confidence:.2%}, score={signal.score:.1f}"
            )
            return signal

        except Exception as e:
            logger.warning(f"OrderBookImbalance process error: {e}")
            return None