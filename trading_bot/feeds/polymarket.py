"""Polymarket CLOB feed for real-time order book and market data.

Connects to the Polymarket WebSocket for live order book updates
and monitors for new 5-minute BTC up/down markets.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import aiohttp
import websockets

logger = logging.getLogger("bot.feeds.polymarket")

# 5-minute markets follow fixed 300-second Unix epoch intervals
MARKET_INTERVAL_SEC = 300


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class MarketSnapshot:
    """Current state of a Polymarket binary market."""
    condition_id: str
    token_id_yes: str
    token_id_no: str
    best_bid_yes: float = 0.0
    best_ask_yes: float = 1.0
    best_bid_no: float = 0.0
    best_ask_no: float = 1.0
    bids_yes: list[OrderBookLevel] = field(default_factory=list)
    asks_yes: list[OrderBookLevel] = field(default_factory=list)
    total_bid_depth_usd: float = 0.0
    total_ask_depth_usd: float = 0.0
    last_update: float = field(default_factory=time.time)
    resolution_time: float = 0.0  # Unix timestamp when market resolves

    @property
    def spread(self) -> float:
        return self.best_ask_yes - self.best_bid_yes

    @property
    def mid_price(self) -> float:
        return (self.best_bid_yes + self.best_ask_yes) / 2

    @property
    def age_ms(self) -> float:
        return (time.time() - self.last_update) * 1000

    @property
    def is_thin(self) -> bool:
        """Market has very low liquidity — potential vacuum opportunity."""
        return self.total_bid_depth_usd < 50 or self.total_ask_depth_usd < 50


def current_market_window() -> tuple[int, int]:
    """Return (start, end) Unix timestamps for the current 5-min market."""
    now = int(time.time())
    start = now - (now % MARKET_INTERVAL_SEC)
    end = start + MARKET_INTERVAL_SEC
    return start, end


def next_market_window() -> tuple[int, int]:
    """Return (start, end) Unix timestamps for the next 5-min market."""
    start, _ = current_market_window()
    next_start = start + MARKET_INTERVAL_SEC
    return next_start, next_start + MARKET_INTERVAL_SEC


def seconds_until_next_market() -> float:
    """Seconds until the next 5-minute market spawns."""
    _, end = current_market_window()
    return max(0, end - time.time())


class PolymarketFeed:
    """WebSocket + REST client for Polymarket CLOB data.

    Discovers active 5-minute BTC up/down markets via REST,
    then subscribes to live order book updates via WebSocket.
    """

    def __init__(self, rest_url: str, ws_url: str):
        self.rest_url = rest_url
        self.ws_url = ws_url
        self._ws = None
        self._running = False
        self._callbacks: list = []
        self.active_market: MarketSnapshot | None = None

        # Cache discovered market IDs
        self._known_markets: dict[str, MarketSnapshot] = {}

    def on_market_update(self, callback) -> None:
        """Register callback for order book updates: callback(market: MarketSnapshot)."""
        self._callbacks.append(callback)

    async def discover_active_markets(self) -> list[dict]:
        """Find currently active 5-minute BTC up/down markets via REST API.

        Polls the Polymarket API for markets matching the current time window.
        """
        start, end = current_market_window()
        url = f"{self.rest_url}/markets"
        params = {
            "active": "true",
            "closed": "false",
            "tag": "crypto",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        logger.warning(f"Market discovery failed: HTTP {resp.status}")
                        return []
                    data = await resp.json()

            # Filter for 5-minute BTC markets
            btc_markets = []
            for market in data if isinstance(data, list) else data.get("data", []):
                question = (market.get("question") or market.get("description", "")).lower()
                if "bitcoin" in question or "btc" in question:
                    if "5" in question and ("minute" in question or "min" in question):
                        btc_markets.append(market)

            logger.info(f"Discovered {len(btc_markets)} active BTC 5-min markets")
            return btc_markets

        except Exception:
            logger.exception("Error discovering markets")
            return []

    async def fetch_order_book(self, token_id: str) -> dict:
        """Fetch current order book for a specific token."""
        url = f"{self.rest_url}/book"
        params = {"token_id": token_id}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status != 200:
                        return {}
                    return await resp.json()
        except Exception:
            logger.exception("Error fetching order book")
            return {}

    def parse_order_book(self, book_data: dict, market: MarketSnapshot) -> MarketSnapshot:
        """Parse raw order book data into a MarketSnapshot."""
        bids = book_data.get("bids", [])
        asks = book_data.get("asks", [])

        market.bids_yes = [
            OrderBookLevel(price=float(b["price"]), size=float(b["size"]))
            for b in sorted(bids, key=lambda x: -float(x["price"]))
        ]
        market.asks_yes = [
            OrderBookLevel(price=float(a["price"]), size=float(a["size"]))
            for a in sorted(asks, key=lambda x: float(x["price"]))
        ]

        if market.bids_yes:
            market.best_bid_yes = market.bids_yes[0].price
        if market.asks_yes:
            market.best_ask_yes = market.asks_yes[0].price

        market.total_bid_depth_usd = sum(b.price * b.size for b in market.bids_yes)
        market.total_ask_depth_usd = sum(a.price * a.size for a in market.asks_yes)
        market.last_update = time.time()

        return market

    async def start(self) -> None:
        """Start polling for markets and streaming order book updates."""
        self._running = True
        while self._running:
            try:
                await self._poll_loop()
            except Exception:
                if self._running:
                    logger.exception("Polymarket feed error. Retrying in 3s...")
                    await asyncio.sleep(3)

    async def _poll_loop(self) -> None:
        """Main loop: discover markets, poll order books, fire callbacks."""
        while self._running:
            # Discover markets
            markets = await self.discover_active_markets()

            if markets:
                # Track the first matching market
                m = markets[0]
                tokens = m.get("tokens", [])
                yes_token = next((t for t in tokens if t.get("outcome") == "Yes"), None)
                no_token = next((t for t in tokens if t.get("outcome") == "No"), None)

                if yes_token and no_token:
                    snapshot = MarketSnapshot(
                        condition_id=m.get("condition_id", m.get("id", "")),
                        token_id_yes=yes_token.get("token_id", ""),
                        token_id_no=no_token.get("token_id", ""),
                    )

                    # Fetch and parse order book
                    book = await self.fetch_order_book(snapshot.token_id_yes)
                    if book:
                        snapshot = self.parse_order_book(book, snapshot)
                        self.active_market = snapshot

                        for cb in self._callbacks:
                            try:
                                result = cb(snapshot)
                                if asyncio.iscoroutine(result):
                                    await result
                            except Exception:
                                logger.exception("Error in market update callback")

            # Poll every 500ms for low-latency updates
            await asyncio.sleep(0.5)

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
