"""Configuration loader for the trading bot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PolymarketConfig:
    rest_url: str = "https://clob.polymarket.com"
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    chain_id: int = 137
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""
    private_key: str = ""  # Ethereum private key for signing orders


@dataclass
class BinanceConfig:
    ws_url: str = "wss://stream.binance.com:9443/ws"
    symbol: str = "btcusdt"


@dataclass
class LiquidityVacuumConfig:
    enabled: bool = True
    max_entry_price: float = 0.03
    max_position_size_usd: float = 50.0
    min_shares: int = 100
    exit_price: float = 0.40
    hold_to_resolution: bool = False


@dataclass
class LatencyArbConfig:
    enabled: bool = True
    min_edge_pct: float = 4.0
    taker_fee_pct: float = 3.15
    max_position_size_usd: float = 200.0
    price_move_threshold_pct: float = 0.08
    max_staleness_ms: int = 2000


@dataclass
class RiskConfig:
    max_daily_loss_usd: float = 500.0
    max_open_positions: int = 5
    max_total_exposure_usd: float = 1000.0
    cooldown_after_loss_sec: int = 30


@dataclass
class PaperConfig:
    starting_balance_usd: float = 10000.0
    simulated_fill_slippage_pct: float = 0.5
    simulated_latency_ms: int = 100


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "trading_bot.log"
    trade_log: str = "trades.jsonl"


@dataclass
class BotConfig:
    mode: str = "paper"  # "paper" or "live"
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    liquidity_vacuum: LiquidityVacuumConfig = field(default_factory=LiquidityVacuumConfig)
    latency_arb: LatencyArbConfig = field(default_factory=LatencyArbConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str | Path = "config.yaml") -> BotConfig:
    """Load bot configuration from YAML file, with env var overrides."""
    path = Path(path)
    if not path.exists():
        return BotConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = BotConfig(mode=raw.get("mode", "paper"))

    # Load each section
    if "polymarket" in raw:
        cfg.polymarket = PolymarketConfig(**raw["polymarket"])
    if "binance" in raw:
        cfg.binance = BinanceConfig(**raw["binance"])
    if "liquidity_vacuum" in raw:
        cfg.liquidity_vacuum = LiquidityVacuumConfig(**raw["liquidity_vacuum"])
    if "latency_arb" in raw:
        cfg.latency_arb = LatencyArbConfig(**raw["latency_arb"])
    if "risk" in raw:
        cfg.risk = RiskConfig(**raw["risk"])
    if "paper" in raw:
        cfg.paper = PaperConfig(**raw["paper"])
    if "logging" in raw:
        cfg.logging = LoggingConfig(**raw["logging"])

    # Environment variable overrides for secrets (never put keys in YAML)
    cfg.polymarket.api_key = os.getenv("POLY_API_KEY", cfg.polymarket.api_key)
    cfg.polymarket.api_secret = os.getenv("POLY_API_SECRET", cfg.polymarket.api_secret)
    cfg.polymarket.passphrase = os.getenv("POLY_PASSPHRASE", cfg.polymarket.passphrase)
    cfg.polymarket.private_key = os.getenv("POLY_PRIVATE_KEY", cfg.polymarket.private_key)

    return cfg
