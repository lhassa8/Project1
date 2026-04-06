"""Binance WebSocket feed for real-time BTC/USD price data.

Connects to the public trade stream — no API key needed.
Provides sub-second price updates that serve as "ground truth"
for the latency arbitrage strategy.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import websockets

logger = logging.getLogger("bot.feeds.binance")


@dataclass
class BinanceTick:
    """A single trade tick from Binance."""
    price: float
    quantity: float
    timestamp_ms: int  # Binance server timestamp
    local_timestamp: float  # When we received it

    @property
    def age_ms(self) -> float:
        return (time.time() - self.local_timestamp) * 1000


class BinanceFeed:
    """WebSocket client for Binance real-time trade stream.

    Subscribes to the individual trade stream (@trade) which fires
    on every single trade — lower latency than @kline or @ticker.
    """

    def __init__(self, ws_url: str, symbol: str = "btcusdt"):
        self.ws_url = ws_url
        self.symbol = symbol.lower()
        self.stream_url = f"{ws_url}/{self.symbol}@trade"

        self.latest_tick: BinanceTick | None = None
        self._ws = None
        self._running = False
        self._callbacks: list = []

        # Track price changes for latency arb signal
        self._price_window: list[tuple[float, float]] = []  # (timestamp, price)
        self._window_size = 30  # seconds of history

    def on_tick(self, callback) -> None:
        """Register a callback for each new tick: callback(tick: BinanceTick)."""
        self._callbacks.append(callback)

    async def start(self) -> None:
        """Connect and begin streaming. Reconnects on failure."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_stream()
            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                if self._running:
                    logger.warning(f"Binance WS disconnected: {e}. Reconnecting in 2s...")
                    await asyncio.sleep(2)

    async def _connect_and_stream(self) -> None:
        logger.info(f"Connecting to Binance: {self.stream_url}")
        async with websockets.connect(self.stream_url, ping_interval=20) as ws:
            self._ws = ws
            logger.info("Binance feed connected")
            async for raw in ws:
                msg = json.loads(raw)
                tick = BinanceTick(
                    price=float(msg["p"]),
                    quantity=float(msg["q"]),
                    timestamp_ms=msg["T"],
                    local_timestamp=time.time(),
                )
                self.latest_tick = tick
                self._update_price_window(tick)

                for cb in self._callbacks:
                    try:
                        result = cb(tick)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        logger.exception("Error in Binance tick callback")

    def _update_price_window(self, tick: BinanceTick) -> None:
        """Maintain a rolling window of recent prices."""
        now = time.time()
        self._price_window.append((now, tick.price))
        # Prune old entries
        cutoff = now - self._window_size
        self._price_window = [(t, p) for t, p in self._price_window if t >= cutoff]

    def recent_price_change_pct(self, lookback_sec: float = 5.0) -> float | None:
        """Calculate price change % over the last N seconds.

        Returns None if insufficient data.
        """
        if len(self._price_window) < 2:
            return None
        now = time.time()
        cutoff = now - lookback_sec
        old_prices = [p for t, p in self._price_window if t <= cutoff]
        if not old_prices:
            # Use oldest available
            old_price = self._price_window[0][1]
        else:
            old_price = old_prices[-1]
        current_price = self._price_window[-1][1]
        if old_price == 0:
            return None
        return ((current_price - old_price) / old_price) * 100

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
