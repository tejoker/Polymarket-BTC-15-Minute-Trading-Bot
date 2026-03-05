"""
Binance Liquidation Feed Client

Streams forced liquidation events from Binance to feed the HawkesExecutionKernel.

WHY THIS MATTERS:
  Each forced liquidation on Binance represents a cascade event: a leveraged
  position that crossed the maintenance margin threshold was automatically
  closed by the exchange as a market order. These forced market orders:
    1. Deplete order book liquidity
    2. Move the spot price adversely for the liquidated side
    3. Can trigger further liquidations (cascade)

  The HawkesExecutionKernel already models self-exciting processes with
  exponential decay (λ(t) = μ + Σ α·e^(-β·Δt)). By feeding real liquidation
  events as the excitation source instead of generic "trade signals", the
  kernel becomes a genuine real-time liquidation cascade detector.

  When kernel intensity is high (many recent liquidations), the bot is in a
  cascade regime. Entering new positions during high intensity risks entering
  right as BTC flushes through a liquidation waterfall.

WEBSOCKET (free, no auth):
  wss://fstream.binance.com/ws/btcusdt@forceOrder
  Binance streams every forced liquidation for BTCUSDT perps in real-time.

INTEGRATION:
  client = LiquidationFeedClient(execution_kernel)
  asyncio.ensure_future(client.run())

  After startup, execution_kernel.get_intensity(time.time()) reflects
  real-time cascade risk. Values above 1.0 indicate active cascade conditions.

REQUIRES: websockets >= 10.0 (pip install websockets)
"""
import asyncio
import json
import time
from typing import Optional
from loguru import logger

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    logger.warning(
        "LiquidationFeedClient: 'websockets' not installed. "
        "Run: pip install websockets"
    )

BINANCE_LIQUIDATION_WS = "wss://fstream.binance.com/ws/btcusdt@forceOrder"

_INITIAL_RECONNECT_DELAY = 2.0
_MAX_RECONNECT_DELAY = 60.0

# Only register liquidations above this BTC size — filters tiny forced orders
# that don't materially move the market. 0.5 BTC ≈ $40k at $80k/BTC.
_MIN_BTC_QTY = 0.5

# Scale intensity by BTC qty, capped to prevent one huge liquidation from
# overwhelming the Hawkes kernel's decay dynamics.
_MAX_INTENSITY_PER_EVENT = 5.0


class LiquidationFeedClient:
    """
    Background asyncio task that streams Binance forced liquidation events
    and feeds them into the HawkesExecutionKernel as intensity events.

    The kernel's get_intensity() method then reflects current cascade risk.
    Higher intensity = more recent liquidations = higher cascade probability.

    Lifecycle:
      client = LiquidationFeedClient(execution_kernel)
      asyncio.ensure_future(client.run())
    """

    def __init__(self, hawkes_kernel, min_btc_qty: float = _MIN_BTC_QTY):
        """
        Args:
            hawkes_kernel: HawkesExecutionKernel instance (execution_kernel global)
            min_btc_qty: Minimum BTC size to register as a Hawkes event.
        """
        self.kernel = hawkes_kernel
        self.min_btc_qty = min_btc_qty
        self._running = False
        self._reconnect_delay = _INITIAL_RECONNECT_DELAY

        # Telemetry
        self.events_received = 0
        self.total_btc_liquidated = 0.0

    async def run(self) -> None:
        """Run the WebSocket stream with exponential-backoff reconnection."""
        if not HAS_WEBSOCKETS:
            logger.warning(
                "LiquidationFeedClient: websockets missing, Hawkes kernel "
                "will use trade signals only (no liquidation enrichment)"
            )
            return

        self._running = True
        logger.info(
            "LiquidationFeedClient: starting Binance @forceOrder stream "
            f"(min_btc_qty={self.min_btc_qty})"
        )

        while self._running:
            try:
                await self._stream()
                # Clean exit → reset backoff
                self._reconnect_delay = _INITIAL_RECONNECT_DELAY

            except asyncio.CancelledError:
                logger.info("LiquidationFeedClient: stream cancelled")
                break

            except Exception as e:
                logger.warning(
                    f"LiquidationFeedClient: stream error ({e}), "
                    f"reconnecting in {self._reconnect_delay:.0f}s"
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, _MAX_RECONNECT_DELAY
                )

    async def _stream(self) -> None:
        async with websockets.connect(
            BINANCE_LIQUIDATION_WS,
            ping_interval=20,
            ping_timeout=10,
        ) as ws:
            logger.info(
                "LiquidationFeedClient: connected to Binance @forceOrder stream"
            )
            self._reconnect_delay = _INITIAL_RECONNECT_DELAY

            async for raw_msg in ws:
                try:
                    self._handle_message(raw_msg)
                except Exception as e:
                    logger.debug(f"LiquidationFeedClient: message error: {e}")

    def _handle_message(self, raw: str) -> None:
        msg = json.loads(raw)
        if msg.get("e") != "forceOrder":
            return

        order = msg.get("o", {})

        # "q" = original quantity, "l" = last filled quantity
        qty_str = order.get("q") or order.get("l")
        side = order.get("S", "UNKNOWN")  # SELL = long liquidation, BUY = short liquidation

        if not qty_str:
            return

        btc_qty = float(qty_str)
        if btc_qty < self.min_btc_qty:
            return

        now_ts = time.time()

        # Scale intensity: 1.0 base + proportional to size, capped
        # 10 BTC threshold for max intensity (large single liquidation)
        intensity = min(1.0 + (btc_qty / 10.0), _MAX_INTENSITY_PER_EVENT)

        self.kernel.add_event(timestamp=now_ts, intensity_multiplier=intensity)
        self.events_received += 1
        self.total_btc_liquidated += btc_qty

        current_intensity = self.kernel.get_intensity(now_ts)

        logger.info(
            f"Liquidation: {side} {btc_qty:.3f} BTC forced → "
            f"Hawkes λ={current_intensity:.3f} "
            f"(total events: {self.events_received}, "
            f"total BTC: {self.total_btc_liquidated:.1f})"
        )

        if current_intensity >= 2.0:
            logger.warning(
                f"LiquidationFeedClient: HIGH CASCADE INTENSITY λ={current_intensity:.2f} "
                f"— new entries carry elevated adverse selection risk"
            )

    def stop(self) -> None:
        self._running = False

    def get_stats(self) -> dict:
        return {
            "events_received": self.events_received,
            "total_btc_liquidated": round(self.total_btc_liquidated, 2),
            "current_hawkes_intensity": round(
                self.kernel.get_intensity(time.time()), 4
            ),
            "running": self._running,
        }
