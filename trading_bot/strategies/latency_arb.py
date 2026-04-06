"""Strategy 2: Cross-Exchange Latency Arbitrage

Exploits the 30-90 second lag between Binance spot price movements
and Polymarket binary contract repricing.

Mechanism:
  1. Monitor Binance BTC/USD in real-time (sub-second updates)
  2. Monitor Polymarket order book for active 5-min contract
  3. When Binance shows a clear directional move but Polymarket
     hasn't repriced yet → edge exists
  4. Buy the underpriced outcome before the order book catches up
  5. Either sell after repricing or hold to resolution

Edge: Information advantage — you know BTC moved before Polymarket does.
Risk: Dynamic taker fees (3.15%) eat into margins. Need >4% edge to profit.

NOTE: As of Jan 2026, Polymarket's dynamic fees have significantly
reduced this edge. The bot accounts for fees in its edge calculation.
"""

from __future__ import annotations

import logging
import time

from trading_bot.config import LatencyArbConfig
from trading_bot.execution.executor import BaseExecutor, Order
from trading_bot.feeds.binance import BinanceFeed
from trading_bot.feeds.polymarket import MarketSnapshot
from trading_bot.strategies.base import BaseStrategy

logger = logging.getLogger("bot.strategy.latency_arb")


class LatencyArbStrategy(BaseStrategy):
    """Buy underpriced outcomes when Binance moves but Polymarket lags."""

    def __init__(self, executor: BaseExecutor, config: LatencyArbConfig):
        super().__init__("latency_arb", executor)
        self.config = config

        self._last_signal_time: float = 0
        self._cooldown_sec: float = 10  # Don't re-signal too fast
        self._has_position: bool = False
        self._position_side: str = ""  # "up" or "down"
        self._position_shares: float = 0
        self._position_token_id: str = ""

    def _estimate_true_probability(self, btc_change_pct: float) -> float:
        """Estimate true probability of BTC being up at resolution.

        Based on the magnitude and direction of the recent move on Binance.
        A +0.1% move in the last few seconds strongly predicts the 5-min
        outcome because momentum tends to persist at these timescales.

        This is a simplified model. A production bot would use more
        sophisticated features (volume, order flow, volatility regime).
        """
        # Logistic-like mapping: bigger moves → higher confidence
        # At 0.08% move: ~65% confidence
        # At 0.15% move: ~75% confidence
        # At 0.30% move: ~85% confidence
        import math
        confidence = 1 / (1 + math.exp(-btc_change_pct * 30))
        # Clamp to reasonable range
        return max(0.15, min(0.85, confidence))

    async def evaluate(
        self,
        market: MarketSnapshot,
        binance: BinanceFeed,
    ) -> list[Order]:
        """Check for latency arbitrage opportunity."""
        if not self.enabled or self._has_position:
            return []

        # Cooldown check
        if time.time() - self._last_signal_time < self._cooldown_sec:
            return []

        # Need fresh Binance data
        if not binance.latest_tick:
            return []

        # Check if Polymarket quote is stale enough to exploit
        if market.age_ms < 500:
            # Market is too fresh — no lag to exploit
            return []

        # Get recent BTC price change from Binance
        btc_change = binance.recent_price_change_pct(lookback_sec=5.0)
        if btc_change is None:
            return []

        abs_change = abs(btc_change)

        # Need a meaningful move
        if abs_change < self.config.price_move_threshold_pct:
            return []

        # Estimate true probability based on Binance data
        if btc_change > 0:
            # BTC is going UP
            true_prob_yes = self._estimate_true_probability(btc_change)
            polymarket_price = market.best_ask_yes

            # Edge = true probability - price we'd pay - fees
            gross_edge = (true_prob_yes - polymarket_price) * 100
            net_edge = gross_edge - self.config.taker_fee_pct

            if net_edge >= self.config.min_edge_pct:
                # BUY YES (betting Up)
                shares = self.config.max_position_size_usd / polymarket_price
                self._last_signal_time = time.time()

                logger.info(
                    f"LATENCY ARB SIGNAL: BTC +{btc_change:.3f}% on Binance, "
                    f"Polymarket YES @ ${polymarket_price:.4f} "
                    f"(true prob: {true_prob_yes:.1%}, "
                    f"gross edge: {gross_edge:.1f}%, net edge: {net_edge:.1f}%)"
                )

                self._has_position = True
                self._position_side = "up"
                self._position_shares = shares
                self._position_token_id = market.token_id_yes

                return [Order(
                    token_id=market.token_id_yes,
                    side="BUY",
                    price=polymarket_price,
                    size=shares,
                )]

        else:
            # BTC is going DOWN
            true_prob_no = self._estimate_true_probability(-btc_change)
            polymarket_price = market.best_ask_no

            gross_edge = (true_prob_no - polymarket_price) * 100
            net_edge = gross_edge - self.config.taker_fee_pct

            if net_edge >= self.config.min_edge_pct:
                # BUY NO (betting Down)
                shares = self.config.max_position_size_usd / polymarket_price
                self._last_signal_time = time.time()

                logger.info(
                    f"LATENCY ARB SIGNAL: BTC {btc_change:.3f}% on Binance, "
                    f"Polymarket NO @ ${polymarket_price:.4f} "
                    f"(true prob: {true_prob_no:.1%}, "
                    f"gross edge: {gross_edge:.1f}%, net edge: {net_edge:.1f}%)"
                )

                self._has_position = True
                self._position_side = "down"
                self._position_shares = shares
                self._position_token_id = market.token_id_no

                return [Order(
                    token_id=market.token_id_no,
                    side="BUY",
                    price=polymarket_price,
                    size=shares,
                )]

        return []

    def should_exit(self, market: MarketSnapshot) -> list[Order]:
        """Exit when the market reprices toward true value."""
        if not self._has_position:
            return []

        # Check if the market has caught up (repriced)
        if self._position_side == "up" and market.best_bid_yes >= 0.60:
            logger.info(
                f"LATENCY ARB EXIT: YES repriced to ${market.best_bid_yes:.4f}, "
                f"selling {self._position_shares:.0f} shares"
            )
            self._has_position = False
            return [Order(
                token_id=self._position_token_id,
                side="SELL",
                price=market.best_bid_yes,
                size=self._position_shares,
            )]

        if self._position_side == "down" and market.best_bid_no >= 0.60:
            logger.info(
                f"LATENCY ARB EXIT: NO repriced to ${market.best_bid_no:.4f}, "
                f"selling {self._position_shares:.0f} shares"
            )
            self._has_position = False
            return [Order(
                token_id=self._position_token_id,
                side="SELL",
                price=market.best_bid_no,
                size=self._position_shares,
            )]

        return []
