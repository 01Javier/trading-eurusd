"""
strategy.py
Thin strategy layer for the frozen pullback system.

Research remains in indicators.py/backtest.py. This module gives the future demo
bot a stable interface: prepare signals, inspect the latest actionable signal,
and keep M15 disabled until coverage is sufficient.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import INDICATORS, SESSION_MODE
from indicators import generate_signals


@dataclass
class SignalDecision:
    action: str
    direction: int
    timestamp: pd.Timestamp | None
    price: float | None
    reason: str


class FrozenPullbackStrategy:
    def __init__(self, mode: str | None = None, session_mode: str | None = None):
        self.mode = mode or INDICATORS.get("strategy_mode", "pullback")
        self.session_mode = session_mode or SESSION_MODE

    def prepare(self, df_h4: pd.DataFrame) -> pd.DataFrame:
        return generate_signals(
            df_h4,
            use_session=True,
            session_mode=self.session_mode,
            use_adx=True,
            strict_entries=True,
            use_m15=False,
            mode=self.mode,
        )

    def latest(self, df_h4: pd.DataFrame) -> SignalDecision:
        if df_h4.empty:
            return SignalDecision("WAIT", 0, None, None, "no data")

        df_sig = self.prepare(df_h4)
        if len(df_sig) < 2:
            return SignalDecision("WAIT", 0, df_sig.index[-1], None, "not enough bars")

        # Backtest enters on the next bar after the previous completed signal.
        row = df_sig.iloc[-2]
        sig = int(row.get("signal", 0))
        if sig == 0:
            return SignalDecision("WAIT", 0, row.name, float(row["close"]), "no frozen-system signal")

        return SignalDecision(
            action="BUY" if sig == 1 else "SELL",
            direction=sig,
            timestamp=row.name,
            price=float(row["close"]),
            reason=f"{self.mode} signal on completed H4 bar; M15 OFF",
        )
