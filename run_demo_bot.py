"""
run_demo_bot.py
Safe demo-bot dry run.

This script reads the latest H4 data, evaluates the frozen pullback strategy and
prints/logs the order plan. It does not send orders by default.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CAPITAL, SYMBOL
from logger import CsvLogger
from mt5_data import load_csv
from risk_manager import RiskManager
from strategy import FrozenPullbackStrategy
from validator import TradingSafetyValidator


def main() -> int:
    parser = argparse.ArgumentParser(description="Safe MT5 demo-bot dry run")
    parser.add_argument("--symbol", default=SYMBOL, help="Canonical symbol, e.g. EURUSD")
    parser.add_argument("--send-demo-order", action="store_true", help="Reserved for future demo order_send integration")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    log = CsvLogger()
    df = load_csv(symbol, "H4")
    if df is None or df.empty:
        print(f"No H4 data for {symbol}. Run: python -X utf8 run.py --download-asset {symbol}")
        return 1

    strategy = FrozenPullbackStrategy()
    df_sig = strategy.prepare(df)
    decision = strategy.latest(df)
    if decision.direction == 0:
        print(f"{symbol}: WAIT - {decision.reason}")
        log.write("demo_wait", {"symbol": symbol, "reason": decision.reason})
        return 0

    row = df_sig.loc[decision.timestamp]
    sl_distance = float(row["atr"]) * CAPITAL.get("atr_sl_mult", 1.5)
    manager = RiskManager(symbol)
    plan = manager.position_size(symbol=symbol, capital=CAPITAL["initial"], sl_distance=sl_distance)
    validator = TradingSafetyValidator(symbol, manager)
    risk_decision = validator.validate_position(plan)

    print(f"{symbol}: {decision.action} candidate at {decision.timestamp}")
    print(f"Reason: {decision.reason}")
    print(f"Lot plan: {plan.lots:.4f} lots | risk ${plan.risk_usd:.2f} ({plan.risk_pct:.2f}%)")
    print(f"Safety: {'ALLOW' if risk_decision.allowed else 'BLOCK'} - {risk_decision.reason}")

    log.write("demo_signal", {
        "symbol": symbol,
        "action": decision.action,
        "timestamp": str(decision.timestamp),
        "lots": plan.lots,
        "risk_usd": plan.risk_usd,
        "risk_pct": plan.risk_pct,
        "allowed": risk_decision.allowed,
        "reason": risk_decision.reason,
    })

    if args.send_demo_order:
        print("Order sending is intentionally not wired yet. Keep this script dry-run until demo QA is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
