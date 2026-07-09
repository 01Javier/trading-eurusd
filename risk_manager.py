"""
risk_manager.py
Gestion conservadora de lotaje, limites de perdida y viabilidad de cuenta.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import CAPITAL, INSTRUMENTS
from backtest import round_lot_size


@dataclass(frozen=True)
class PositionPlan:
    symbol: str
    lots: float
    risk_usd: float
    risk_pct: float
    sl_pips: float
    min_lot_risk_usd: float
    affordable: bool
    reason: str


class RiskManager:
    """Calcula lotaje y bloquea operaciones que violen reglas de supervivencia."""

    def __init__(self, capital_config: dict[str, Any] | None = None):
        self.capital = capital_config or CAPITAL

    def position_size(self, symbol: str, capital: float, sl_distance: float) -> PositionPlan:
        spec = INSTRUMENTS[symbol]
        pip_size = float(spec.get("pip_size", 0.0001))
        pip_value = float(spec.get("pip_value_per_lot", 0.10))
        min_lot = float(spec.get("min_lot", 0.01))
        risk_pct = float(self.capital.get("risk_pct", 0.015))
        risk_usd = capital * risk_pct
        sl_pips = sl_distance / pip_size if pip_size > 0 else 0.0
        raw_lots = risk_usd / (sl_pips * pip_value) if sl_pips > 0 and pip_value > 0 else 0.0
        lots = round_lot_size(raw_lots, spec)
        min_lot_risk = sl_pips * pip_value * min_lot
        affordable = lots > 0 and min_lot_risk <= risk_usd
        reason = "OK" if affordable else (
            f"Min lot {min_lot} arriesga ${min_lot_risk:.2f}; presupuesto ${risk_usd:.2f}"
        )
        return PositionPlan(
            symbol=symbol,
            lots=lots,
            risk_usd=round(risk_usd, 2),
            risk_pct=risk_pct,
            sl_pips=round(sl_pips, 1),
            min_lot_risk_usd=round(min_lot_risk, 2),
            affordable=affordable,
            reason=reason,
        )

    def can_continue(self, initial_capital: float, current_capital: float,
                     daily_pnl: float = 0.0, weekly_pnl: float = 0.0,
                     loss_streak: int = 0) -> tuple[bool, str]:
        max_dd = float(self.capital.get("max_drawdown", 0.15))
        max_daily = float(self.capital.get("max_daily_loss_pct", 0.03))
        max_weekly = float(self.capital.get("max_weekly_loss_pct", 0.06))
        max_streak = int(self.capital.get("max_loss_streak_pause", 4))
        if current_capital <= initial_capital * (1 - max_dd):
            return False, "Max drawdown alcanzado"
        if daily_pnl <= -initial_capital * max_daily:
            return False, "Limite de perdida diaria alcanzado"
        if weekly_pnl <= -initial_capital * max_weekly:
            return False, "Limite de perdida semanal alcanzado"
        if loss_streak >= max_streak:
            return False, "Pausa por racha de perdidas"
        return True, "OK"
