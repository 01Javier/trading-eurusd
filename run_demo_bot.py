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

from config import CAPITAL, RESULTS_DIR, SYMBOL
from logger import CsvLogger
from mt5_data import load_csv
from risk_manager import RiskManager
from strategy import FrozenPullbackStrategy
from validator import TradingSafetyValidator


def _signals_log() -> CsvLogger:
    return CsvLogger(os.path.join(RESULTS_DIR, "demo_signals_log.csv"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Safe MT5 demo-bot dry run")
    parser.add_argument("--symbol", default=SYMBOL, help="Canonical symbol, e.g. EURUSD")
    parser.add_argument("--send-demo-order", action="store_true", help="Reserved for future demo order_send integration")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    log = CsvLogger()
    signals_log = _signals_log()
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
        signals_log.write("demo_signal", {
            "symbol": symbol,
            "action": "WAIT",
            "signal_time": str(decision.timestamp) if decision.timestamp is not None else "",
            "decision": "REJECTED",
            "reason": decision.reason,
            "spread_pips": "",
            "lots": 0,
            "risk_usd": 0,
            "risk_pct": 0,
            "sl": "",
            "tp1": "",
            "tp2": "",
            "backtest_mode": f"{strategy.mode}|H4|M15_OFF",
            "result": "",
            "pnl_usd": "",
            "closed": False,
        })
        return 0

    row = df_sig.loc[decision.timestamp]
    sl_distance = float(row["atr"]) * CAPITAL.get("atr_sl_mult", 1.5)
    tp1_distance = float(row["atr"]) * CAPITAL.get("atr_tp1_mult", 2.0)
    tp2_distance = float(row["atr"]) * CAPITAL.get("atr_tp2_mult", 4.0)
    entry_ref = float(decision.price or row["close"])
    sl_price = entry_ref - decision.direction * sl_distance
    tp1_price = entry_ref + decision.direction * tp1_distance
    tp2_price = entry_ref + decision.direction * tp2_distance
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
    signals_log.write("demo_signal", {
        "symbol": symbol,
        "action": decision.action,
        "signal_time": str(decision.timestamp),
        "decision": "ACCEPTED" if risk_decision.allowed else "REJECTED",
        "reason": risk_decision.reason,
        "spread_pips": "",
        "lots": plan.lots,
        "risk_usd": plan.risk_usd,
        "risk_pct": plan.risk_pct,
        "sl": round(sl_price, 5),
        "tp1": round(tp1_price, 5),
        "tp2": round(tp2_price, 5),
        "backtest_mode": f"{strategy.mode}|H4|M15_OFF",
        "result": "",
        "pnl_usd": "",
        "closed": False,
    })

    if args.send_demo_order:
        print("Order sending is intentionally not wired yet. Keep this script dry-run until demo QA is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
