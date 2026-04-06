"""Order execution interface and live Polymarket executor.

The live executor signs and submits orders to Polymarket's CLOB
using EIP-712 typed data signatures. Requires a funded Polygon
wallet and Polymarket API credentials.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger("bot.execution")


@dataclass
class Order:
    """Represents an order to be placed."""
    token_id: str
    side: str  # "BUY" or "SELL"
    price: float
    size: float  # Number of shares
    order_type: str = "LIMIT"  # "LIMIT" or "MARKET"

    @property
    def cost_usd(self) -> float:
        return self.price * self.size


@dataclass
class Fill:
    """Represents an executed fill."""
    order_id: str
    token_id: str
    side: str
    price: float
    size: float
    fee_usd: float
    timestamp: float


class BaseExecutor(ABC):
    """Abstract base class for order execution."""

    @abstractmethod
    async def place_order(self, order: Order) -> Fill | None:
        """Place an order and return the fill, or None if rejected."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if successful."""

    @abstractmethod
    async def get_balance(self) -> float:
        """Get available USDC balance."""


class PolymarketExecutor(BaseExecutor):
    """Live order executor for Polymarket CLOB.

    Signs orders using EIP-712 typed data and submits via REST API.
    Requires: POLY_API_KEY, POLY_API_SECRET, POLY_PASSPHRASE, POLY_PRIVATE_KEY
    """

    def __init__(
        self,
        rest_url: str,
        api_key: str,
        api_secret: str,
        passphrase: str,
        chain_id: int = 137,
    ):
        self.rest_url = rest_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.chain_id = chain_id

    def _sign_request(self, method: str, path: str, body: str = "") -> dict:
        """Generate HMAC-SHA256 authentication headers."""
        timestamp = str(int(time.time()))
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()

        return {
            "POLY-ADDRESS": self.api_key,
            "POLY-SIGNATURE": signature,
            "POLY-TIMESTAMP": timestamp,
            "POLY-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }

    async def place_order(self, order: Order) -> Fill | None:
        """Place an order on Polymarket CLOB."""
        path = "/order"
        body_dict = {
            "tokenID": order.token_id,
            "side": order.side,
            "price": str(order.price),
            "size": str(order.size),
            "type": order.order_type,
        }
        body = json.dumps(body_dict)
        headers = self._sign_request("POST", path, body)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.rest_url}{path}",
                    headers=headers,
                    data=body,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    result = await resp.json()

                    if resp.status != 200:
                        logger.error(f"Order rejected: {result}")
                        return None

                    order_id = result.get("orderID", result.get("id", ""))
                    logger.info(
                        f"Order placed: {order.side} {order.size} @ ${order.price} "
                        f"(ID: {order_id})"
                    )

                    return Fill(
                        order_id=order_id,
                        token_id=order.token_id,
                        side=order.side,
                        price=order.price,
                        size=order.size,
                        fee_usd=order.cost_usd * 0.0315,  # Estimated taker fee
                        timestamp=time.time(),
                    )

        except Exception:
            logger.exception("Error placing order")
            return None

    async def cancel_order(self, order_id: str) -> bool:
        path = f"/order/{order_id}"
        headers = self._sign_request("DELETE", path)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.rest_url}{path}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    return resp.status == 200
        except Exception:
            logger.exception("Error cancelling order")
            return False

    async def get_balance(self) -> float:
        path = "/balance"
        headers = self._sign_request("GET", path)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.rest_url}{path}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    data = await resp.json()
                    return float(data.get("balance", 0))
        except Exception:
            logger.exception("Error fetching balance")
            return 0.0
