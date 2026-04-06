"""
Polymarket Bitcoin Trading Bot

Strategies:
  1. Liquidity Vacuum - Exploit empty order books at 5-min market creation
  2. Latency Arbitrage - Exploit price lag between Binance and Polymarket

Default mode: PAPER TRADING. Real trading requires explicit configuration
and API keys. Never risk more than you can afford to lose.
"""

__version__ = "0.1.0"
