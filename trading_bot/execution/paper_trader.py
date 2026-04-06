"""Paper trading executor — simulates order fills without real money.

Simulates realistic fill behavior including:
- Configurable slippage
- Simulated latency
- Position tracking
- P&L calculation on market resolution
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

from trading_bot.execution.executor import BaseExecutor, Fill, Order

logger = logging.getLogger("bot.execution.paper")


@dataclass
class PaperPosition:
    """Tracks an open paper trading position."""
    market_id: str
    token_id: str
    side: str
    entry_price: float
    shares: float
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


class PaperTrader(BaseExecutor):
    """Simulated executor for paper trading.

    Tracks positions, simulates fills with slippage, and calculates P&L
    when positions are closed or markets resolve.
    """

    def __init__(
        self,
        starting_balance: float = 10000.0,
        slippage_pct: float = 0.5,
        latency_ms: int = 100,
        taker_fee_pct: float = 3.15,
    ):
        self.balance = starting_balance
        self.starting_balance = starting_balance
        self.slippage_pct = slippage_pct
        self.latency_ms = latency_ms
        self.taker_fee_pct = taker_fee_pct

        self.positions: list[PaperPosition] = []
        self.closed_trades: list[dict] = []
        self.total_fees_paid: float = 0.0

    async def place_order(self, order: Order) -> Fill | None:
        """Simulate an order fill with slippage and latency."""
        # Simulate network/exchange latency
        await asyncio.sleep(self.latency_ms / 1000)

        # Apply slippage
        if order.side == "BUY":
            fill_price = order.price * (1 + self.slippage_pct / 100)
        else:
            fill_price = order.price * (1 - self.slippage_pct / 100)

        fill_price = max(0.001, min(0.999, fill_price))  # Clamp to valid range
        cost = fill_price * order.size
        fee = cost * (self.taker_fee_pct / 100)

        # Check balance
        if order.side == "BUY" and (cost + fee) > self.balance:
            logger.warning(
                f"Paper: Insufficient balance. Need ${cost + fee:.2f}, have ${self.balance:.2f}"
            )
            return None

        # Execute
        fill_id = str(uuid.uuid4())[:8]

        if order.side == "BUY":
            self.balance -= (cost + fee)
            self.positions.append(PaperPosition(
                market_id="",
                token_id=order.token_id,
                side="BUY",
                entry_price=fill_price,
                shares=order.size,
                cost_usd=cost,
            ))
        else:
            # Find matching position to close
            self.balance += (cost - fee)

        self.total_fees_paid += fee

        logger.info(
            f"Paper fill: {order.side} {order.size:.0f} shares @ ${fill_price:.4f} "
            f"(cost: ${cost:.2f}, fee: ${fee:.2f}, balance: ${self.balance:.2f})"
        )

        return Fill(
            order_id=fill_id,
            token_id=order.token_id,
            side=order.side,
            price=fill_price,
            size=order.size,
            fee_usd=fee,
            timestamp=time.time(),
        )

    async def cancel_order(self, order_id: str) -> bool:
        logger.info(f"Paper: Cancelled order {order_id}")
        return True

    async def get_balance(self) -> float:
        return self.balance

    def resolve_position(self, token_id: str, outcome: bool) -> float:
        """Resolve a position when the market settles.

        Args:
            token_id: The YES token ID
            outcome: True if YES wins, False if NO wins

        Returns:
            P&L for this position
        """
        pnl = 0.0
        remaining = []

        for pos in self.positions:
            if pos.token_id == token_id:
                if outcome:
                    # YES wins: each share pays $1.00
                    revenue = pos.shares * 1.0
                else:
                    # NO wins: YES shares worth $0
                    revenue = 0.0

                trade_pnl = revenue - pos.cost_usd
                pnl += trade_pnl
                self.balance += revenue

                self.closed_trades.append({
                    "token_id": token_id,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "shares": pos.shares,
                    "cost": pos.cost_usd,
                    "revenue": revenue,
                    "pnl": trade_pnl,
                    "outcome": "YES" if outcome else "NO",
                    "timestamp": time.time(),
                })

                logger.info(
                    f"Paper resolved: {pos.shares:.0f} shares @ ${pos.entry_price:.4f} → "
                    f"{'WON' if trade_pnl > 0 else 'LOST'} ${abs(trade_pnl):.2f}"
                )
            else:
                remaining.append(pos)

        self.positions = remaining
        return pnl

    def summary(self) -> dict:
        total_pnl = self.balance - self.starting_balance
        wins = sum(1 for t in self.closed_trades if t["pnl"] > 0)
        total = len(self.closed_trades)

        return {
            "balance": round(self.balance, 2),
            "starting_balance": self.starting_balance,
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round((total_pnl / self.starting_balance) * 100, 1),
            "trades_closed": total,
            "win_rate": round((wins / total * 100), 1) if total > 0 else 0,
            "open_positions": len(self.positions),
            "fees_paid": round(self.total_fees_paid, 2),
        }
