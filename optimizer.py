"""
optimizer.py
Conservative research helpers.

This module does not auto-apply parameters. It is for comparing risk profiles or
candidate settings and then deciding manually from stability, drawdown and
out-of-sample evidence.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from backtest import BacktestEngine, compute_metrics
from config import RISK_PROFILES


def evaluate_risk_profiles(
    df_sig: pd.DataFrame,
    symbol: str,
    profiles: dict[str, float] | None = None,
    initial_capital: float | None = None,
) -> list[dict]:
    rows = []
    for name, risk_pct in (profiles or RISK_PROFILES).items():
        engine = BacktestEngine(
            initial_capital=initial_capital,
            risk_pct=float(risk_pct),
            instrument=symbol,
            trailing_stop=True,
            partial_tp=True,
        )
        res = engine.run(df_sig)
        metrics = compute_metrics(res)
        rows.append({
            "profile": name,
            "risk_pct": float(risk_pct),
            **metrics,
        })
    return rows


def conservative_rank(rows: Iterable[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (
            r.get("net_pnl_usd", 0) > 0,
            -r.get("max_drawdown_pct", 999),
            r.get("profit_factor", 0),
            r.get("sharpe_ratio", 0),
        ),
        reverse=True,
    )
