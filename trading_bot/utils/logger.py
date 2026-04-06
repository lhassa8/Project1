"""Structured logging for the trading bot."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


def setup_logger(name: str, level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Create a logger with console + optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class TradeLogger:
    """Append-only JSONL logger for trade records."""

    def __init__(self, path: str = "trades.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_trade(
        self,
        strategy: str,
        side: str,
        market_id: str,
        price: float,
        shares: float,
        cost_usd: float,
        pnl_usd: float | None = None,
        metadata: dict | None = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy,
            "side": side,
            "market_id": market_id,
            "price": price,
            "shares": shares,
            "cost_usd": cost_usd,
            "pnl_usd": pnl_usd,
            "metadata": metadata or {},
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
