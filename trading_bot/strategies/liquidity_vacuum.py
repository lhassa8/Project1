"""Strategy 1: Liquidity Vacuum Exploitation

Exploits the empty order book that exists for a few seconds when a new
5-minute BTC up/down market spawns on Polymarket.

Mechanism:
  1. Monitor for new market creation (every 300s aligned to Unix epoch)
  2. In the first 1-5 seconds, the order book is empty or has only
     stale limit orders at extreme prices ($0.01-$0.03)
  3. Sweep available shares at near-zero prices
  4. Wait for market makers to arrive and reprice to ~$0.50
  5. Either sell at fair value or hold to resolution

Edge: 100:1 price-to-payout ratio means a single win covers many losses.
Risk: Market may not reprice (stays illiquid), or BTC goes against you.
"""

from __future__ import annotations

import logging
import time

from trading_bot.config import LiquidityVacuumConfig
from trading_bot.execution.executor import BaseExecutor, Order
from trading_bot.feeds.binance import BinanceFeed
from trading_bot.feeds.polymarket import (
    MarketSnapshot,
    current_market_window,
    seconds_until_next_market,
)
from trading_bot.strategies.base import BaseStrategy

logger = logging.getLogger("bot.strategy.vacuum")


class LiquidityVacuumStrategy(BaseStrategy):
    """Buy shares at near-zero prices during the liquidity vacuum."""

    def __init__(self, executor: BaseExecutor, config: LiquidityVacuumConfig):
        super().__init__("liquidity_vacuum", executor)
        self.config = config

        # Track per-market state
        self._last_market_start: int = 0
        self._accumulated_shares: float = 0
        self._entry_cost: float = 0
        self._has_position: bool = False

    async def evaluate(
        self,
        market: MarketSnapshot,
        binance: BinanceFeed,
    ) -> list[Order]:
        """Look for vacuum opportunities at market creation."""
        if not self.enabled:
            return []

        start, end = current_market_window()
        elapsed = time.time() - start

        # Only act in the first 10 seconds of a new market
        if elapsed > 10:
            return []

        # Don't re-enter the same market window
        if start == self._last_market_start and self._has_position:
            return []

        # Check for vacuum conditions
        if not market.is_thin:
            logger.debug("Market not thin enough for vacuum strategy")
            return []

        # Look for asks at near-zero prices
        cheap_asks = [
            ask for ask in market.asks_yes
            if ask.price <= self.config.max_entry_price
        ]

        if not cheap_asks:
            logger.debug(
                f"No asks below ${self.config.max_entry_price}. "
                f"Best ask: ${market.best_ask_yes:.4f}"
            )
            return []

        # Calculate how many shares we can buy within budget
        orders = []
        budget_remaining = self.config.max_position_size_usd

        for ask in cheap_asks:
            if budget_remaining <= 0:
                break

            affordable_shares = budget_remaining / ask.price
            shares_to_buy = min(ask.size, affordable_shares)

            if shares_to_buy >= self.config.min_shares:
                orders.append(Order(
                    token_id=market.token_id_yes,
                    side="BUY",
                    price=ask.price,
                    size=shares_to_buy,
                    order_type="LIMIT",
                ))
                budget_remaining -= ask.price * shares_to_buy

                logger.info(
                    f"VACUUM SIGNAL: {shares_to_buy:.0f} shares @ ${ask.price:.4f} "
                    f"(cost: ${ask.price * shares_to_buy:.2f}, "
                    f"potential payout: ${shares_to_buy:.2f})"
                )

        if orders:
            self._last_market_start = start
            self._has_position = True
            self._accumulated_shares = sum(o.size for o in orders)
            self._entry_cost = sum(o.price * o.size for o in orders)

        return orders

    def should_exit(self, market: MarketSnapshot) -> list[Order]:
        """Exit when price reaches target or hold to resolution."""
        if not self._has_position:
            return []

        if self.config.hold_to_resolution:
            # Let the position ride to market resolution
            return []

        # Sell if price has risen to exit target
        if market.best_bid_yes >= self.config.exit_price:
            logger.info(
                f"VACUUM EXIT: Selling {self._accumulated_shares:.0f} shares "
                f"@ ${market.best_bid_yes:.4f} "
                f"(entry cost: ${self._entry_cost:.2f}, "
                f"exit value: ${self._accumulated_shares * market.best_bid_yes:.2f})"
            )
            self._has_position = False
            return [Order(
                token_id=market.token_id_yes,
                side="SELL",
                price=market.best_bid_yes,
                size=self._accumulated_shares,
                order_type="LIMIT",
            )]

        return []
