"""Entry point for the Polymarket BTC Trading Bot.

Usage:
    # Paper trading (default — safe, no real money)
    python -m trading_bot.main

    # Paper trading with custom config
    python -m trading_bot.main --config my_config.yaml

    # Live trading (REAL MONEY — requires API keys in env vars)
    python -m trading_bot.main --mode live

Environment Variables (for live trading):
    POLY_API_KEY       - Polymarket API key
    POLY_API_SECRET    - Polymarket API secret
    POLY_PASSPHRASE    - Polymarket API passphrase
    POLY_PRIVATE_KEY   - Ethereum private key for signing orders
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from trading_bot.bot import TradingBot
from trading_bot.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polymarket BTC 5-Minute Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["paper", "live"],
        default=None,
        help="Override trading mode (default: from config)",
    )
    parser.add_argument(
        "--no-vacuum",
        action="store_true",
        help="Disable liquidity vacuum strategy",
    )
    parser.add_argument(
        "--no-arb",
        action="store_true",
        help="Disable latency arbitrage strategy",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.mode:
        config.mode = args.mode
    if args.no_vacuum:
        config.liquidity_vacuum.enabled = False
    if args.no_arb:
        config.latency_arb.enabled = False

    # Safety check for live mode
    if config.mode == "live":
        print("\n" + "=" * 60)
        print("  WARNING: LIVE TRADING MODE")
        print("  Real money will be at risk.")
        print("  Make sure you understand the strategies and risks.")
        print("=" * 60)
        confirm = input("\nType 'YES I UNDERSTAND' to continue: ")
        if confirm != "YES I UNDERSTAND":
            print("Aborted.")
            sys.exit(0)

    # Run bot
    bot = TradingBot(config)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")


if __name__ == "__main__":
    main()
