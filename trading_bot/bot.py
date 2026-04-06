"""Main bot orchestrator — ties together feeds, strategies, risk, and execution.

This is the core event loop that:
  1. Starts data feeds (Binance + Polymarket) as concurrent async tasks
  2. On each market update, evaluates all enabled strategies
  3. Passes resulting orders through risk management
  4. Executes approved orders via the configured executor (paper or live)
  5. Logs all activity and tracks P&L
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time

from trading_bot.config import BotConfig
from trading_bot.execution.executor import BaseExecutor, PolymarketExecutor
from trading_bot.execution.paper_trader import PaperTrader
from trading_bot.feeds.binance import BinanceFeed
from trading_bot.feeds.polymarket import (
    MarketSnapshot,
    PolymarketFeed,
    seconds_until_next_market,
)
from trading_bot.risk.manager import RiskManager
from trading_bot.strategies.base import BaseStrategy
from trading_bot.strategies.latency_arb import LatencyArbStrategy
from trading_bot.strategies.liquidity_vacuum import LiquidityVacuumStrategy
from trading_bot.utils.logger import TradeLogger, setup_logger
from trading_bot.utils.metrics import MetricsTracker, Trade

logger = logging.getLogger("bot")


class TradingBot:
    """The main trading bot — coordinates all components."""

    def __init__(self, config: BotConfig):
        self.config = config
        self._running = False

        # Set up logging
        setup_logger("bot", config.logging.level, config.logging.file)
        self.trade_logger = TradeLogger(config.logging.trade_log)
        self.metrics = MetricsTracker()

        # Set up executor
        self.executor: BaseExecutor
        if config.mode == "live":
            if not config.polymarket.api_key:
                raise ValueError(
                    "Live mode requires Polymarket API credentials. "
                    "Set POLY_API_KEY, POLY_API_SECRET, POLY_PASSPHRASE env vars."
                )
            self.executor = PolymarketExecutor(
                rest_url=config.polymarket.rest_url,
                api_key=config.polymarket.api_key,
                api_secret=config.polymarket.api_secret,
                passphrase=config.polymarket.passphrase,
                chain_id=config.polymarket.chain_id,
            )
            logger.warning("=" * 60)
            logger.warning("  LIVE TRADING MODE — REAL MONEY AT RISK")
            logger.warning("=" * 60)
        else:
            self.executor = PaperTrader(
                starting_balance=config.paper.starting_balance_usd,
                slippage_pct=config.paper.simulated_fill_slippage_pct,
                latency_ms=config.paper.simulated_latency_ms,
            )
            logger.info("Paper trading mode — no real money at risk")

        # Set up risk manager
        self.risk = RiskManager(config.risk)

        # Set up data feeds
        self.binance_feed = BinanceFeed(
            ws_url=config.binance.ws_url,
            symbol=config.binance.symbol,
        )
        self.polymarket_feed = PolymarketFeed(
            rest_url=config.polymarket.rest_url,
            ws_url=config.polymarket.ws_url,
        )

        # Set up strategies
        self.strategies: list[BaseStrategy] = []
        if config.liquidity_vacuum.enabled:
            self.strategies.append(
                LiquidityVacuumStrategy(self.executor, config.liquidity_vacuum)
            )
        if config.latency_arb.enabled:
            self.strategies.append(
                LatencyArbStrategy(self.executor, config.latency_arb)
            )

        logger.info(
            f"Bot initialized: mode={config.mode}, "
            f"strategies={[s.name for s in self.strategies]}"
        )

    async def run(self) -> None:
        """Start the bot — runs until interrupted."""
        self._running = True

        # Register signal handlers for clean shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

        logger.info("Starting trading bot...")
        logger.info(f"Next market in {seconds_until_next_market():.1f}s")

        # Register market update callback
        self.polymarket_feed.on_market_update(self._on_market_update)

        # Run feeds and strategy loop concurrently
        try:
            await asyncio.gather(
                self.binance_feed.start(),
                self.polymarket_feed.start(),
                self._status_loop(),
            )
        except asyncio.CancelledError:
            pass
        finally:
            await self._shutdown()

    async def _on_market_update(self, market: MarketSnapshot) -> None:
        """Called on every Polymarket order book update — evaluates strategies."""
        if not self._running:
            return

        for strategy in self.strategies:
            try:
                # Check for exit signals first
                exit_orders = strategy.should_exit(market)
                for order in exit_orders:
                    await self._execute_order(order, strategy.name, is_exit=True)

                # Then check for entry signals
                entry_orders = await strategy.evaluate(market, self.binance_feed)
                for order in entry_orders:
                    await self._execute_order(order, strategy.name, is_exit=False)

            except Exception:
                logger.exception(f"Error in strategy {strategy.name}")

    async def _execute_order(self, order, strategy_name: str, is_exit: bool) -> None:
        """Execute an order through the risk manager and executor."""
        # Risk check
        allowed, reason = self.risk.check_order(order, strategy_name)
        if not allowed:
            logger.warning(f"Order blocked by risk manager: {reason}")
            return

        # Execute
        fill = await self.executor.place_order(order)
        if fill is None:
            logger.warning(f"Order failed to fill: {order.side} {order.size} @ ${order.price}")
            return

        # Update risk tracking
        if order.side == "BUY":
            self.risk.record_entry(order, strategy_name)
            pnl = 0.0
        else:
            pnl = self.risk.record_exit(order.token_id, fill.price, fill.size)

        # Record metrics
        trade = Trade(
            strategy=strategy_name,
            market_id=order.token_id,
            side=order.side,
            price=fill.price,
            shares=fill.size,
            cost_usd=fill.price * fill.size,
            pnl_usd=pnl,
        )
        self.metrics.record_trade(trade)

        # Log trade
        self.trade_logger.log_trade(
            strategy=strategy_name,
            side=order.side,
            market_id=order.token_id,
            price=fill.price,
            shares=fill.size,
            cost_usd=fill.price * fill.size,
            pnl_usd=pnl,
            metadata={"fee": fill.fee_usd, "is_exit": is_exit},
        )

    async def _status_loop(self) -> None:
        """Periodically log bot status."""
        while self._running:
            await asyncio.sleep(60)  # Every minute
            if not self._running:
                break

            btc_price = (
                f"${self.binance_feed.latest_tick.price:,.2f}"
                if self.binance_feed.latest_tick
                else "N/A"
            )

            poly_status = "N/A"
            if self.polymarket_feed.active_market:
                m = self.polymarket_feed.active_market
                poly_status = (
                    f"YES=${m.best_bid_yes:.2f}/{m.best_ask_yes:.2f} "
                    f"spread={m.spread:.3f} "
                    f"depth=${m.total_bid_depth_usd:.0f}/${m.total_ask_depth_usd:.0f}"
                )

            metrics = self.metrics.summary()
            risk_status = self.risk.status()

            logger.info(
                f"STATUS | BTC: {btc_price} | Poly: {poly_status} | "
                f"PnL: ${metrics['total_pnl']} | "
                f"Trades: {metrics['trade_count']} | "
                f"Win: {metrics['win_rate']}% | "
                f"Positions: {risk_status['open_positions']} | "
                f"Exposure: ${risk_status['total_exposure']}"
            )

            # Paper trader specific summary
            if isinstance(self.executor, PaperTrader):
                ps = self.executor.summary()
                logger.info(
                    f"PAPER  | Balance: ${ps['balance']} | "
                    f"PnL: ${ps['total_pnl']} ({ps['total_pnl_pct']}%) | "
                    f"Fees: ${ps['fees_paid']}"
                )

    async def stop(self) -> None:
        """Gracefully stop the bot."""
        logger.info("Stopping bot...")
        self._running = False

    async def _shutdown(self) -> None:
        """Clean shutdown — close feeds and log final stats."""
        await asyncio.gather(
            self.binance_feed.stop(),
            self.polymarket_feed.stop(),
            return_exceptions=True,
        )

        # Final summary
        metrics = self.metrics.summary()
        logger.info("=" * 60)
        logger.info("  FINAL SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Total P&L:        ${metrics['total_pnl']}")
        logger.info(f"  Total Trades:     {metrics['trade_count']}")
        logger.info(f"  Win Rate:         {metrics['win_rate']}%")
        logger.info(f"  Vacuum P&L:       ${metrics['liquidity_vacuum_pnl']}")
        logger.info(f"  Latency Arb P&L:  ${metrics['latency_arb_pnl']}")

        if isinstance(self.executor, PaperTrader):
            ps = self.executor.summary()
            logger.info(f"  Paper Balance:    ${ps['balance']}")
            logger.info(f"  Fees Paid:        ${ps['fees_paid']}")

        logger.info("=" * 60)
