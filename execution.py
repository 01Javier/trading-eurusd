"""
execution.py
MT5 execution shell for demo trading.

Default behavior is dry-run. The class refuses real-account execution unless the
explicit config safety gate is opened. This file is a safe base, not a finished
production execution engine.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from config import CAPITAL, INSTRUMENTS
from logger import CsvLogger
from risk_manager import PositionPlan
from validator import TradingSafetyValidator


class MT5DemoExecutor:
    def __init__(self, symbol: str, dry_run: bool = True, logger: CsvLogger | None = None):
        self.symbol = symbol
        self.spec = INSTRUMENTS[symbol]
        self.dry_run = dry_run
        self.logger = logger or CsvLogger()
        self.validator = TradingSafetyValidator(symbol)

    def estimate_spread_pips(self, tick: Any) -> float | None:
        bid = getattr(tick, "bid", None)
        ask = getattr(tick, "ask", None)
        if bid is None or ask is None:
            return None
        pip_size = float(self.spec.get("pip_size", 0.0001))
        return float((ask - bid) / pip_size) if pip_size else None

    def place_market_order(
        self,
        mt5: Any,
        account_info: Any,
        signal: str,
        plan: PositionPlan,
        sl_price: float,
        tp_price: float | None = None,
        comment: str = "codex-demo-bot",
    ) -> dict:
        mode_decision = self.validator.validate_mode(account_info)
        if not mode_decision.allowed:
            return self._blocked(mode_decision.reason, signal, plan)

        position_decision = self.validator.validate_position(plan)
        if not position_decision.allowed:
            return self._blocked(position_decision.reason, signal, plan)

        tick = mt5.symbol_info_tick(self.symbol)
        spread_decision = self.validator.validate_spread(self.estimate_spread_pips(tick))
        if not spread_decision.allowed:
            return self._blocked(spread_decision.reason, signal, plan)

        order = {
            "action": getattr(mt5, "TRADE_ACTION_DEAL", 1),
            "symbol": self.symbol,
            "volume": plan.lots,
            "type": getattr(mt5, "ORDER_TYPE_BUY", 0) if signal == "BUY" else getattr(mt5, "ORDER_TYPE_SELL", 1),
            "sl": sl_price,
            "tp": tp_price or 0.0,
            "deviation": int(CAPITAL.get("slippage_pips", 0.2) * 10),
            "comment": comment,
        }

        if self.dry_run:
            self.logger.write("dry_run_order", {"signal": signal, **asdict(plan), "sl": sl_price, "tp": tp_price})
            return {"sent": False, "dry_run": True, "reason": "dry-run only", "order": order}

        result = mt5.order_send(order)
        self.logger.write("order_send", {"signal": signal, **asdict(plan), "result": str(result)})
        return {"sent": True, "dry_run": False, "result": result, "order": order}

    def _blocked(self, reason: str, signal: str, plan: PositionPlan) -> dict:
        self.logger.write("order_blocked", {"signal": signal, "reason": reason, **asdict(plan)})
        return {"sent": False, "dry_run": self.dry_run, "blocked": True, "reason": reason}
