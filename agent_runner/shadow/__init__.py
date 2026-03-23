"""Shadow mode — virtual filesystem overlay and transaction replay."""

from agent_runner.shadow.state import ShadowState
from agent_runner.shadow.diff import ShadowDiff
from agent_runner.shadow.replay import TransactionReplay

__all__ = ["ShadowState", "ShadowDiff", "TransactionReplay"]
