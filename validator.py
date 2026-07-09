"""
validator.py
Safety checks for the future MT5 demo bot.

These checks are intentionally conservative. Real trading remains blocked unless
the operator changes config.py and sets the explicit environment confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import CAPITAL, INSTRUMENTS, TRADING_MODE
from risk_manager import RiskManager, PositionPlan


@dataclass
class SafetyDecision:
    allowed: bool
    reason: str


class TradingSafetyValidator:
    def __init__(self, symbol: str, risk_manager: RiskManager | None = None):
        self.symbol = symbol
        self.spec = INSTRUMENTS[symbol]
        self.risk_manager = risk_manager or RiskManager(symbol)

    def validate_mode(self, account_info: Any | None = None) -> SafetyDecision:
        if TRADING_MODE.get("real_mode", False):
            confirmed = (
                TRADING_MODE.get("allow_real_trading", False)
                and TRADING_MODE.get("real_confirmation")
                == TRADING_MODE.get("require_real_confirmation")
            )
            if not confirmed:
                return SafetyDecision(False, "real trading blocked by config safety gate")

        if account_info is not None:
            trade_mode = getattr(account_info, "trade_mode", None)
            is_demo = trade_mode == 0
            if not is_demo and TRADING_MODE.get("demo_mode", True):
                return SafetyDecision(False, "account is not demo while demo_mode=True")

        return SafetyDecision(True, "mode allowed")

    def validate_spread(self, spread_pips: float | None) -> SafetyDecision:
        if spread_pips is None:
            return SafetyDecision(True, "spread unavailable; continue only in dry-run")
        max_spread = float(self.spec.get("max_spread_pips", CAPITAL.get("spread_pips", 1.5)))
        if spread_pips > max_spread:
            return SafetyDecision(False, f"spread {spread_pips:.2f} pips exceeds max {max_spread:.2f}")
        return SafetyDecision(True, "spread allowed")

    def validate_position(self, plan: PositionPlan) -> SafetyDecision:
        if not plan.can_trade:
            return SafetyDecision(False, plan.reason)
        return SafetyDecision(True, "position allowed")

    def validate_risk_state(
        self,
        initial_capital: float,
        current_capital: float,
        daily_pnl: float = 0.0,
        weekly_pnl: float = 0.0,
        loss_streak: int = 0,
    ) -> SafetyDecision:
        ok, reason = self.risk_manager.can_continue(
            initial_capital=initial_capital,
            current_capital=current_capital,
            daily_pnl=daily_pnl,
            weekly_pnl=weekly_pnl,
            loss_streak=loss_streak,
        )
        return SafetyDecision(ok, reason)
