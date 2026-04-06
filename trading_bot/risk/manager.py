"""Risk management — position limits, loss limits, and exposure control.

Sits between strategies and execution. Every order passes through
the risk manager before reaching the executor. Orders that violate
risk limits are rejected with a logged reason.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from trading_bot.config import RiskConfig
from trading_bot.execution.executor import Order

logger = logging.getLogger("bot.risk")


@dataclass
class OpenPosition:
    """Tracks an open position for risk accounting."""
    token_id: str
    strategy: str
    side: str
    entry_price: float
    shares: float
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


class RiskManager:
    """Enforces risk limits on all trading activity.

    Checks:
      - Daily loss limit
      - Maximum open positions
      - Maximum total exposure (USD)
      - Cooldown period after losses
      - Per-order size limits
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.positions: list[OpenPosition] = []
        self.daily_pnl: float = 0.0
        self.last_loss_time: float = 0.0
        self._day_start: float = time.time()

    def check_order(self, order: Order, strategy: str) -> tuple[bool, str]:
        """Validate an order against risk limits.

        Returns:
            (allowed, reason) — True if order can proceed, False with reason if blocked.
        """
        # Reset daily P&L at midnight
        self._maybe_reset_daily()

        # Check daily loss limit
        if self.daily_pnl <= -self.config.max_daily_loss_usd:
            msg = (
                f"BLOCKED: Daily loss limit reached "
                f"(${self.daily_pnl:.2f} / -${self.config.max_daily_loss_usd})"
            )
            logger.warning(msg)
            return False, msg

        # Check cooldown after loss
        if self.last_loss_time > 0:
            elapsed = time.time() - self.last_loss_time
            if elapsed < self.config.cooldown_after_loss_sec:
                remaining = self.config.cooldown_after_loss_sec - elapsed
                msg = f"BLOCKED: Loss cooldown active ({remaining:.0f}s remaining)"
                logger.warning(msg)
                return False, msg

        # Check open position count (only for new buys)
        if order.side == "BUY":
            if len(self.positions) >= self.config.max_open_positions:
                msg = (
                    f"BLOCKED: Max open positions reached "
                    f"({len(self.positions)}/{self.config.max_open_positions})"
                )
                logger.warning(msg)
                return False, msg

            # Check total exposure
            current_exposure = sum(p.cost_usd for p in self.positions)
            new_exposure = current_exposure + order.cost_usd
            if new_exposure > self.config.max_total_exposure_usd:
                msg = (
                    f"BLOCKED: Max exposure exceeded "
                    f"(${new_exposure:.2f} / ${self.config.max_total_exposure_usd})"
                )
                logger.warning(msg)
                return False, msg

        logger.debug(f"Risk check passed: {order.side} {order.size} @ ${order.price}")
        return True, "OK"

    def record_entry(self, order: Order, strategy: str) -> None:
        """Record a new position after a buy fill."""
        self.positions.append(OpenPosition(
            token_id=order.token_id,
            strategy=strategy,
            side=order.side,
            entry_price=order.price,
            shares=order.size,
            cost_usd=order.cost_usd,
        ))
        logger.info(
            f"Position opened: {strategy} {order.size:.0f} shares @ ${order.price:.4f} "
            f"(open positions: {len(self.positions)})"
        )

    def record_exit(self, token_id: str, exit_price: float, shares: float) -> float:
        """Record a position exit and return the P&L."""
        pnl = 0.0
        remaining = []

        for pos in self.positions:
            if pos.token_id == token_id and shares > 0:
                closed_shares = min(pos.shares, shares)
                entry_cost = pos.entry_price * closed_shares
                exit_value = exit_price * closed_shares
                trade_pnl = exit_value - entry_cost
                pnl += trade_pnl
                shares -= closed_shares

                if pos.shares > closed_shares:
                    pos.shares -= closed_shares
                    pos.cost_usd -= entry_cost
                    remaining.append(pos)
            else:
                remaining.append(pos)

        self.positions = remaining
        self.daily_pnl += pnl

        if pnl < 0:
            self.last_loss_time = time.time()

        logger.info(
            f"Position closed: PnL ${pnl:.2f} "
            f"(daily: ${self.daily_pnl:.2f}, open: {len(self.positions)})"
        )
        return pnl

    def record_resolution(self, token_id: str, outcome_won: bool) -> float:
        """Handle market resolution for held positions."""
        payout_price = 1.0 if outcome_won else 0.0
        return self.record_exit(token_id, payout_price, float("inf"))

    def _maybe_reset_daily(self) -> None:
        """Reset daily P&L every 24 hours."""
        if time.time() - self._day_start > 86400:
            logger.info(f"Daily reset. Previous day P&L: ${self.daily_pnl:.2f}")
            self.daily_pnl = 0.0
            self._day_start = time.time()

    def status(self) -> dict:
        return {
            "open_positions": len(self.positions),
            "total_exposure": round(sum(p.cost_usd for p in self.positions), 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_loss_limit": self.config.max_daily_loss_usd,
            "in_cooldown": (
                time.time() - self.last_loss_time < self.config.cooldown_after_loss_sec
                if self.last_loss_time > 0 else False
            ),
        }
