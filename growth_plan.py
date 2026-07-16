"""
growth_plan.py
Conservative account-growth simulations from observed backtest equity.

The module does not promise targets. It bootstraps observed monthly returns and
reports whether targets are statistically defensible under the current evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from config import CAPITAL, MONTE_CARLO


@dataclass
class GrowthScenario:
    name: str
    monthly_return_pct: float
    projected_12m_capital: float
    months_to_250: float | None
    months_to_recover_150_profit: float | None
    months_to_double: float | None


def monthly_returns_from_equity(equity: pd.Series, initial_capital: float) -> pd.DataFrame:
    if equity is None or equity.empty:
        return pd.DataFrame(columns=["month", "capital", "pnl", "return_pct", "positive"])

    monthly = equity.resample("ME").last().dropna()
    if monthly.empty:
        return pd.DataFrame(columns=["month", "capital", "pnl", "return_pct", "positive"])

    pnl = monthly.diff()
    pnl.iloc[0] = monthly.iloc[0] - initial_capital
    prev = monthly.shift(1)
    prev.iloc[0] = initial_capital
    ret = pnl / prev.replace(0, np.nan)
    return pd.DataFrame({
        "month": monthly.index.astype(str),
        "capital": monthly.values.astype(float),
        "pnl": pnl.values.astype(float),
        "return_pct": (ret.fillna(0).values.astype(float) * 100),
        "positive": pnl.values.astype(float) > 0,
    })


def _months_to_target(monthly_return_pct: float, initial: float, target_capital: float) -> float | None:
    if monthly_return_pct <= 0 or initial <= 0 or target_capital <= initial:
        return None
    monthly = monthly_return_pct / 100
    return float(np.log(target_capital / initial) / np.log(1 + monthly))


def deterministic_scenarios(monthly_returns_pct: Iterable[float], initial: float) -> list[GrowthScenario]:
    values = np.array(list(monthly_returns_pct), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        values = np.array([0.0])

    scenario_defs = [
        ("conservador_p25", float(np.percentile(values, 25))),
        ("base_mediana", float(np.percentile(values, 50))),
        ("optimista_p75", float(np.percentile(values, 75))),
    ]
    rows = []
    for name, monthly_ret in scenario_defs:
        projected = initial * ((1 + monthly_ret / 100) ** 12) if monthly_ret > -100 else 0.0
        rows.append(GrowthScenario(
            name=name,
            monthly_return_pct=round(monthly_ret, 3),
            projected_12m_capital=round(float(projected), 2),
            months_to_250=(
                round(_months_to_target(monthly_ret, initial, 250.0), 1)
                if _months_to_target(monthly_ret, initial, 250.0) is not None else None
            ),
            months_to_recover_150_profit=(
                round(_months_to_target(monthly_ret, initial, initial + 150.0), 1)
                if _months_to_target(monthly_ret, initial, initial + 150.0) is not None else None
            ),
            months_to_double=(
                round(_months_to_target(monthly_ret, initial, initial * 2), 1)
                if _months_to_target(monthly_ret, initial, initial * 2) is not None else None
            ),
        ))
    return rows


def _max_negative_streak(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def bootstrap_growth(
    monthly_returns_pct: Iterable[float],
    initial: float | None = None,
    n_months: int = 60,
    n_sim: int = 10_000,
    seed: int = 42,
) -> dict:
    initial = float(initial or CAPITAL["initial"])
    returns = np.array(list(monthly_returns_pct), dtype=float) / 100.0
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        returns = np.array([0.0])

    rng = np.random.default_rng(seed)
    targets = {
        "target_250": 250.0,
        "recover_150_profit": initial + 150.0,
        "double_account": initial * 2,
    }
    hit_months = {key: [] for key in targets}
    final_caps = []
    min_caps = []
    positive_month_counts = []
    negative_month_counts = []
    max_negative_streaks = []
    ruin_threshold = float(MONTE_CARLO.get("ruin_threshold", 0.50))
    ruin_count = 0

    for _ in range(n_sim):
        sample = rng.choice(returns, size=n_months, replace=True)
        cap = initial
        hit = {key: None for key in targets}
        caps = []
        for month_idx, monthly_ret in enumerate(sample, start=1):
            cap = max(cap * (1 + monthly_ret), 0.0)
            caps.append(cap)
            for key, target in targets.items():
                if hit[key] is None and cap >= target:
                    hit[key] = month_idx
        for key, value in hit.items():
            if value is not None:
                hit_months[key].append(value)
        min_cap = min(caps) if caps else initial
        final_caps.append(cap)
        min_caps.append(min_cap)
        positive_month_counts.append(int((sample > 0).sum()))
        negative_month_counts.append(int((sample < 0).sum()))
        max_negative_streaks.append(_max_negative_streak(sample))
        if min_cap <= initial * (1 - ruin_threshold):
            ruin_count += 1

    def hit_summary(key: str) -> dict:
        arr = np.array(hit_months[key], dtype=float)
        if len(arr) == 0:
            return {"prob_hit_pct": 0.0, "median_months": None, "p25_months": None, "p75_months": None}
        return {
            "prob_hit_pct": round(float(len(arr) / n_sim * 100), 1),
            "median_months": round(float(np.median(arr)), 1),
            "p25_months": round(float(np.percentile(arr, 25)), 1),
            "p75_months": round(float(np.percentile(arr, 75)), 1),
        }

    finals = np.array(final_caps, dtype=float)
    return {
        "n_months": n_months,
        "n_sim": n_sim,
        "monthly_observations": int(len(returns)),
        "avg_monthly_return_pct": round(float(np.mean(returns) * 100), 3),
        "median_monthly_return_pct": round(float(np.median(returns) * 100), 3),
        "positive_month_probability_pct": round(float((returns > 0).mean() * 100), 1),
        "expected_negative_months_per_year": round(float((returns < 0).mean() * 12), 1),
        "sim_final_capital_median": round(float(np.median(finals)), 2),
        "sim_final_capital_p25": round(float(np.percentile(finals, 25)), 2),
        "sim_final_capital_p75": round(float(np.percentile(finals, 75)), 2),
        "sim_ruin_prob_pct": round(float(ruin_count / n_sim * 100), 2),
        "expected_max_negative_month_streak": round(float(np.mean(max_negative_streaks)), 1),
        "p95_max_negative_month_streak": round(float(np.percentile(max_negative_streaks, 95)), 1),
        "target_250": hit_summary("target_250"),
        "recover_150_profit": hit_summary("recover_150_profit"),
        "double_account": hit_summary("double_account"),
    }


def build_growth_plan(equity: pd.Series, initial_capital: float) -> dict:
    monthly = monthly_returns_from_equity(equity, initial_capital)
    returns = monthly["return_pct"].tolist() if not monthly.empty else []
    return {
        "monthly_table": monthly,
        "scenarios": deterministic_scenarios(returns, initial_capital),
        "bootstrap": bootstrap_growth(returns, initial_capital),
    }
