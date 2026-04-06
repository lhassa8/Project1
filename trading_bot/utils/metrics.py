"""Real-time P&L and performance metrics tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Trade:
    strategy: str
    market_id: str
    side: str  # "buy" or "sell"
    price: float
    shares: float
    cost_usd: float
    pnl_usd: float
    timestamp: float = field(default_factory=time.time)


class MetricsTracker:
    """Track P&L, win rate, and per-strategy performance."""

    def __init__(self) -> None:
        self.trades: list[Trade] = []
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.day_start: float = time.time()

    def record_trade(self, trade: Trade) -> None:
        self.trades.append(trade)
        self.total_pnl += trade.pnl_usd
        self.daily_pnl += trade.pnl_usd

    def reset_daily(self) -> None:
        self.daily_pnl = 0.0
        self.day_start = time.time()

    @property
    def win_rate(self) -> float:
        closed = [t for t in self.trades if t.pnl_usd != 0]
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.pnl_usd > 0)
        return wins / len(closed)

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    def strategy_pnl(self, strategy: str) -> float:
        return sum(t.pnl_usd for t in self.trades if t.strategy == strategy)

    def summary(self) -> dict:
        return {
            "total_pnl": round(self.total_pnl, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "trade_count": self.trade_count,
            "win_rate": round(self.win_rate * 100, 1),
            "liquidity_vacuum_pnl": round(self.strategy_pnl("liquidity_vacuum"), 2),
            "latency_arb_pnl": round(self.strategy_pnl("latency_arb"), 2),
        }
