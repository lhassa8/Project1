"""Base strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from trading_bot.execution.executor import BaseExecutor, Order
from trading_bot.feeds.binance import BinanceFeed
from trading_bot.feeds.polymarket import MarketSnapshot


class BaseStrategy(ABC):
    """Interface for trading strategies.

    Each strategy evaluates market conditions and returns orders to execute.
    """

    def __init__(self, name: str, executor: BaseExecutor):
        self.name = name
        self.executor = executor
        self.enabled = True

    @abstractmethod
    async def evaluate(
        self,
        market: MarketSnapshot,
        binance: BinanceFeed,
    ) -> list[Order]:
        """Evaluate current conditions and return orders to place.

        Returns an empty list if no trade signal is present.
        """

    @abstractmethod
    def should_exit(self, market: MarketSnapshot) -> list[Order]:
        """Check if any open positions should be exited.

        Returns sell orders for positions that should be closed.
        """
