"""
run.py v5.4 — Pipeline completo con validación estadística
Comandos:
  --all         : Backtest + Monte Carlo + Walk-Forward
  --backtest    : Solo backtest de las 4 estrategias
  --walkforward : Solo Walk-Forward
  --monte-carlo : Solo Monte Carlo
  --validate    : Bootstrap + Holdout + Resumen de confianza (NUEVO v5.4)
  --compare-assets: comparacion multi-activo con analisis de metas
  --diagnose    : Diagnóstico de filtros por estrategia
  --download    : Descarga datos de MT5
  --download-asset SYMBOL : Descarga un activo canonico, ej. XAUUSD
  --download-assets: Descarga todos los activos configurados
  --compare-timeframes: compara M5/M15/H1/H4 sin reoptimizar
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SYMBOL, CAPITAL, INDICATORS, SESSION_MODE,
    DATA_DIR, RESULTS_DIR, HOLDOUT_MONTHS, BOOTSTRAP,
    INSTRUMENTS, SYMBOLS,
)
from mt5_data   import download_all, download_many, load_csv, synthetic_data
from indicators import generate_signals
from backtest   import (
    BacktestEngine, compute_metrics,
    trade_returns, annualized_trades_per_year,
)
from analysis   import (
    monte_carlo, monte_carlo_trades, walk_forward,
    robustness, bootstrap_trades, holdout_test
)
from charts import plot_backtest, plot_monte_carlo, plot_walkforward
from optimizer import evaluate_risk_profiles, conservative_rank
from report_generator import write_compare_assets_report

try:
    from tracker import save_best_run, export_trades_csv
    TRACKER_OK = True
except ImportError:
    TRACKER_OK = False
    save_best_run = None  # type: ignore[assignment]
    export_trades_csv = None  # type: ignore[assignment]

BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║   SISTEMA TRADING v5.4 — 7 INDICADORES DE LA GUÍA                  ║
║   ADX/DM · RSI · MACD  |  Momentum · Cruce EMA · Pullback         ║
║   EMA {fast}/{slow}  |  ADX>{adx}  |  Riesgo: {risk:.1f}%  |  Scoring: {score}/5       ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def _eng() -> BacktestEngine:
    return BacktestEngine(trailing_stop=True, partial_tp=True)


def _system_mode() -> str:
    return INDICATORS.get("strategy_mode", "pullback")


def _resolve_m15_usage(df_h4, df_m15, requested: bool = True,
                       min_coverage: float = 0.80) -> tuple[bool, str]:
    if not requested:
        return False, "confirmación M15 desactivada"
    if df_m15 is None or df_m15.empty:
        return False, "sin datos M15"

    h4_start, h4_end = df_h4.index[0], df_h4.index[-1]
    m15_start, m15_end = df_m15.index[0], df_m15.index[-1]
    overlap_start = max(h4_start, m15_start)
    overlap_end = min(h4_end, m15_end)

    if overlap_end <= overlap_start:
        return False, (f"sin solapamiento entre H4 ({h4_start.date()} → {h4_end.date()}) "
                       f"y M15 ({m15_start.date()} → {m15_end.date()})")

    total_span = max((h4_end - h4_start).total_seconds(), 1.0)
    overlap_span = max((overlap_end - overlap_start).total_seconds(), 0.0)
    coverage = overlap_span / total_span
    detail = (f"cobertura M15 {coverage*100:.1f}% "
              f"({m15_start.date()} → {m15_end.date()})")
    if coverage < min_coverage:
        return False, detail
    return True, detail


def _row(name: str, m: dict, best: bool = False) -> str:
    pnl = m.get("net_pnl_usd", 0)
    tag = " ★" if best else ""
    ps  = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
    return (
        f"  {name:<32} {m.get('total_trades',0):>7} "
        f"{m.get('win_rate',0):>5.0f}% "
        f"{m.get('sharpe_ratio',0):>7.2f} "
        f"{m.get('sortino_ratio',0):>7.2f} "
        f"{m.get('max_drawdown_pct',0):>6.1f}%  "
        f"${m.get('final_capital',150):>8.2f} {ps:>8}{tag}"
    )


def _fmt_money(value) -> str:
    if value is None:
        return "N/A"
    return f"${float(value):,.2f}"


def _fmt_pct(value) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.1f}%"


def _data_path(symbol: str, timeframe: str = "H4") -> str:
    return os.path.join(DATA_DIR, f"{symbol}_{timeframe}.csv")


def _months_in_data(df) -> float:
    if df is None or len(df) < 2:
        return 1.0
    days = max((df.index[-1] - df.index[0]).days, 1)
    return max(days / 30.4375, 1.0)


def _monthly_stats(equity, initial_capital: float) -> dict:
    if equity is None or equity.empty:
        return {
            "avg_monthly_return_pct": 0.0,
            "avg_monthly_pnl": 0.0,
            "best_month_pnl": 0.0,
            "worst_month_pnl": 0.0,
        }

    monthly = equity.resample("ME").last().dropna()
    if monthly.empty:
        return {
            "avg_monthly_return_pct": 0.0,
            "avg_monthly_pnl": 0.0,
            "best_month_pnl": 0.0,
            "worst_month_pnl": 0.0,
        }

    monthly_pnl = monthly.diff()
    monthly_pnl.iloc[0] = monthly.iloc[0] - initial_capital
    prev_capital = monthly.shift(1)
    prev_capital.iloc[0] = initial_capital
    monthly_returns = []
    for pnl, prev in zip(monthly_pnl.tolist(), prev_capital.tolist()):
        monthly_returns.append((float(pnl) / float(prev) * 100) if float(prev) else 0.0)

    return {
        "avg_monthly_return_pct": round(sum(monthly_returns) / len(monthly_returns), 2),
        "avg_monthly_pnl": round(float(monthly_pnl.mean()), 2),
        "best_month_pnl": round(float(monthly_pnl.max()), 2),
        "worst_month_pnl": round(float(monthly_pnl.min()), 2),
    }


def _position_feasibility(symbol: str, df_h4) -> dict:
    spec = INSTRUMENTS[symbol]
    pip_size = float(spec.get("pip_size", 0.0001))
    pip_value = float(spec.get("pip_value_per_lot", 0.10))
    min_lot = float(spec.get("min_lot", 0.01))
    risk_budget = CAPITAL["initial"] * CAPITAL["risk_pct"]

    atr_series = df_h4["atr"] if "atr" in df_h4.columns else None
    if atr_series is None:
        from indicators import atr
        atr_series = atr(df_h4, INDICATORS.get("atr_period", 14))

    median_atr = float(atr_series.replace([float("inf"), -float("inf")], 0).dropna().median())
    sl_dist = median_atr * CAPITAL.get("atr_sl_mult", 1.5)
    sl_pips = sl_dist / pip_size if pip_size > 0 else 0.0
    min_lot_risk = sl_pips * pip_value * min_lot
    required_lot = risk_budget / (sl_pips * pip_value) if sl_pips > 0 and pip_value > 0 else 0.0
    min_capital = min_lot_risk / CAPITAL["risk_pct"] if CAPITAL["risk_pct"] > 0 else 0.0
    risk_pct_at_min_lot = min_lot_risk / CAPITAL["initial"] * 100 if CAPITAL["initial"] > 0 else 0.0

    return {
        "median_atr": round(median_atr, 5),
        "median_sl_pips": round(sl_pips, 1),
        "risk_budget_usd": round(risk_budget, 2),
        "min_lot_risk_usd": round(min_lot_risk, 2),
        "risk_pct_at_min_lot": round(risk_pct_at_min_lot, 2),
        "required_lot_for_risk": round(required_lot, 4),
        "min_lot": min_lot,
        "min_capital_required": round(min_capital, 2),
        "is_affordable": min_lot_risk <= risk_budget,
    }


def _goal_analysis(avg_monthly_return_pct: float) -> dict:
    account = CAPITAL["initial"]
    current_risk_pct = CAPITAL["risk_pct"] * 100
    dd_limit_pct = CAPITAL["max_drawdown"] * 100
    weekly_required_pct = 100 / account * 100
    monthly_required_pct = 600 / account * 100
    avg_weekly_return_pct = avg_monthly_return_pct / 4.345

    def capital_required(target_usd: float, observed_pct: float):
        if observed_pct <= 0:
            return None
        return target_usd / (observed_pct / 100)

    def risk_required(required_pct: float, observed_pct: float):
        if observed_pct <= 0:
            return None
        return current_risk_pct * (required_pct / observed_pct)

    weekly_risk = risk_required(weekly_required_pct, avg_weekly_return_pct)
    monthly_risk = risk_required(monthly_required_pct, avg_monthly_return_pct)
    expected_weekly_usd = account * (avg_weekly_return_pct / 100)
    expected_monthly_usd = account * (avg_monthly_return_pct / 100)
    return {
        "weekly_required_pct": round(weekly_required_pct, 1),
        "monthly_required_pct": round(monthly_required_pct, 1),
        "avg_weekly_return_pct": round(avg_weekly_return_pct, 2),
        "avg_monthly_return_pct": round(avg_monthly_return_pct, 2),
        "expected_weekly_usd": round(expected_weekly_usd, 2),
        "expected_monthly_usd": round(expected_monthly_usd, 2),
        "weekly_capital_required": capital_required(100, avg_weekly_return_pct),
        "monthly_capital_required": capital_required(600, avg_monthly_return_pct),
        "weekly_risk_required_pct": weekly_risk,
        "monthly_risk_required_pct": monthly_risk,
        "weekly_is_conservative": (
            weekly_required_pct <= dd_limit_pct and
            weekly_risk is not None and weekly_risk <= current_risk_pct * 2
        ),
        "monthly_is_conservative": (
            monthly_required_pct <= dd_limit_pct and
            monthly_risk is not None and monthly_risk <= current_risk_pct * 2
        ),
    }


def _robustness_rank(verdict: str) -> int:
    text = (verdict or "").upper()
    if "NO ROBUSTO" in text:
        return 0
    if "MODERADO" in text:
        return 1
    if "ROBUSTO" in text:
        return 2
    return 0


def _evaluate_asset(symbol: str) -> dict:
    spec = INSTRUMENTS[symbol]
    path_h4 = _data_path(symbol, "H4")
    if not os.path.exists(path_h4):
        return {
            "symbol": symbol,
            "label": spec.get("label", symbol),
            "status": "SIN DATOS",
            "reason": f"falta importar o descargar {path_h4}",
        }

    df_h4 = load_csv(symbol, "H4")
    if df_h4 is None or len(df_h4) < 500:
        return {
            "symbol": symbol,
            "label": spec.get("label", symbol),
            "status": "SIN DATOS",
            "reason": "H4 insuficiente para comparar con robustez",
        }

    feasibility = _position_feasibility(symbol, df_h4)
    df_m15 = load_csv(symbol, "M15") if os.path.exists(_data_path(symbol, "M15")) else None
    _, m15_detail = _resolve_m15_usage(df_h4, df_m15, requested=(df_m15 is not None))
    m15_note = f"OFF por fase conservadora; {m15_detail}" if df_m15 is not None else \
               "OFF por fase conservadora; sin datos M15 comparables"

    mode = _system_mode()
    df_sig = generate_signals(
        df_h4,
        use_session=True,
        session_mode=SESSION_MODE,
        use_adx=True,
        strict_entries=True,
        use_m15=False,
        mode=mode,
    )
    signal_count = int((df_sig["signal"] != 0).sum())
    engine = BacktestEngine(instrument=symbol, trailing_stop=True, partial_tp=True)
    res = engine.run(df_sig)
    metrics = compute_metrics(res)
    months = _months_in_data(df_h4)
    monthly = _monthly_stats(res["equity"], res["initial_capital"])
    goals = _goal_analysis(monthly["avg_monthly_return_pct"])

    if signal_count > 0 and metrics.get("total_trades", 0) == 0 and not feasibility["is_affordable"]:
        return {
            "symbol": symbol,
            "label": spec.get("label", symbol),
            "status": "NO OPERABLE",
            "reason": (
                f"min_lot {feasibility['min_lot']} arriesga "
                f"${feasibility['min_lot_risk_usd']:.2f} con SL mediano; "
                f"presupuesto actual ${feasibility['risk_budget_usd']:.2f}"
            ),
            "contract_confirmed": bool(spec.get("contract_confirmed", True)),
            "contract_note": spec.get("contract_note", ""),
            "m15_note": m15_note,
            "bars": len(df_h4),
            "start": df_h4.index[0].date(),
            "end": df_h4.index[-1].date(),
            "signals": signal_count,
            "metrics": metrics,
            "monthly": monthly,
            "goals": goals,
            "feasibility": feasibility,
        }

    trade_ret = trade_returns(res["trades"], res["initial_capital"])
    periods_per_year = annualized_trades_per_year(len(trade_ret), res["equity"].index)
    if len(trade_ret) >= BOOTSTRAP["min_trades_valid"]:
        bs = bootstrap_trades(
            trade_ret.values,
            periods_per_year=periods_per_year,
            initial_capital=res["initial_capital"],
        )
    else:
        bs = {
            "valid": False,
            "promising": False,
            "reason": f"Solo {len(trade_ret)} trades para bootstrap",
            "sharpe_ci_low": 0.0,
            "sharpe_ci_high": 0.0,
            "ruin_prob_pct": 100.0,
        }

    ruin_real = None
    if not res["trades"].empty:
        pnl_by_trade = res["trades"].groupby("entry_dt")["pnl_usd"].sum().values
        if len(pnl_by_trade) >= 10:
            _, ruin_real = monte_carlo_trades(
                pnl_by_trade,
                initial=res["initial_capital"],
            )

    wf_df, _ = walk_forward(df_h4, instrument=symbol, verbose=False)
    rob = robustness(wf_df)

    return {
        "symbol": symbol,
        "label": spec.get("label", symbol),
        "status": "OK",
        "contract_confirmed": bool(spec.get("contract_confirmed", True)),
        "contract_note": spec.get("contract_note", ""),
        "m15_note": m15_note,
        "bars": len(df_h4),
        "start": df_h4.index[0].date(),
        "end": df_h4.index[-1].date(),
        "metrics": metrics,
        "signals": signal_count,
        "feasibility": feasibility,
        "trades_per_month": metrics.get("total_trades", 0) / months,
        "monthly": monthly,
        "goals": goals,
        "bootstrap": bs,
        "ruin_real_pct": (round(float(ruin_real * 100), 3)
                          if ruin_real is not None else None),
        "walkforward": rob,
        "robustness_rank": _robustness_rank(rob.get("verdict", "")),
    }


def _choose_recommended_asset(results: list[dict]) -> dict | None:
    candidates = []
    dd_limit = CAPITAL["max_drawdown"] * 100
    for item in results:
        if item.get("status") != "OK":
            continue
        metrics = item["metrics"]
        bs = item["bootstrap"]
        if not item.get("contract_confirmed", True):
            continue
        if metrics.get("net_pnl_usd", 0) <= 0:
            continue
        if metrics.get("max_drawdown_pct", 999) > dd_limit:
            continue
        if bs.get("sharpe_ci_low", -999) <= 0:
            continue
        candidates.append(item)

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: (
            item.get("robustness_rank", 0),
            -item["metrics"].get("max_drawdown_pct", 999),
            item["metrics"].get("net_pnl_usd", 0),
            item.get("trades_per_month", 0),
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────────────────

def cmd_download(symbol: str | None = None, all_assets: bool = False) -> None:
    print("\n📡 Descargando datos MT5...")
    if all_assets:
        data = download_many(list(INSTRUMENTS.keys()))
    else:
        target = (symbol or SYMBOL).upper()
        data = download_all(target)

    if not data:
        print("⚠️  MT5 no disponible.")
        target = (symbol or SYMBOL).upper()
        if target == SYMBOL and not os.path.exists(os.path.join(DATA_DIR, f"{SYMBOL}_H4.csv")):
            df = synthetic_data()
            df.to_csv(os.path.join(DATA_DIR, f"{SYMBOL}_H4.csv"))
            print("   Datos sintéticos generados como fallback solo para EURUSD.")
        elif target != SYMBOL:
            print(f"   No se generan datos sintéticos para {target}; importa datos reales.")
        return

    if all_assets:
        for asset, frames in data.items():
            if not frames:
                continue
            for tf, df in frames.items():
                print(f"   {asset} {tf}: {len(df)} barras "
                      f"({df.index[0].date()} → {df.index[-1].date()})")
    else:
        for tf, df in data.items():
            print(f"   {tf}: {len(df)} barras "
                  f"({df.index[0].date()} → {df.index[-1].date()})")


# ─────────────────────────────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────────────────────────────

def cmd_backtest(show_comparison: bool = True) -> dict:
    print("\n" + "=" * 62)
    print("  BACKTEST — 3 ESTRATEGIAS + TODAS")
    print("=" * 62)

    df_h4  = load_csv(SYMBOL, "H4")
    df_m15 = load_csv(SYMBOL, "M15")
    if df_h4 is None:
        print("⚠️  Sin datos H4. Ejecuta: python run.py --download")
        return {}

    print(f"\n  H4 : {len(df_h4)} barras "
          f"({df_h4.index[0].date()} → {df_h4.index[-1].date()})")
    use_m15, m15_note = _resolve_m15_usage(df_h4, df_m15, requested=(df_m15 is not None))
    if df_m15 is not None:
        status = "ON" if use_m15 else "OFF"
        print(f"  M15: {len(df_m15)} barras | Confirmación: {status} | {m15_note}")
    else:
        print("  M15: no disponible")
    print(f"  ADX:{INDICATORS['adx_min']} | EMA:{INDICATORS['ema_fast']}/{INDICATORS['ema_slow']} "
          f"| Score≥{INDICATORS['min_score']} | Riesgo:{CAPITAL['risk_pct']*100:.1f}%")
    print(f"  Sistema base: {_system_mode()} | Sesión: {SESSION_MODE}")

    eng      = _eng()
    # v5.4: 3 estrategias probadas (eliminadas Reversión y Breakout por sin edge)
    strategies = [
        ("1. Momentum   (RSI+MACD+SMA200)", "momentum"),
        ("2. Cruce EMA  (EMA+MACD+ADX)",   "ema_crossover"),
        ("3. Pullback   (Retroceso EMA)",   "pullback"),
        ("4. TODAS      (combinadas)",      "all"),
    ]

    configs: dict = {}
    print()
    for label, mode in strategies:
        print(f"  {label}...", end=" ", flush=True)
        df_sig = generate_signals(
            df_h4, df_m15=df_m15,
            use_session=True, session_mode=SESSION_MODE, use_adx=True,
            strict_entries=True, use_m15=use_m15,
            mode=mode,
        )
        res = eng.run(df_sig)
        m   = compute_metrics(res)
        configs[label] = (df_sig, res, m)
        print(f"Trades:{m['total_trades']:>3}  "
              f"WR:{m['win_rate']:>4.0f}%  "
              f"Sharpe:{m['sharpe_ratio']:>6.2f}  "
              f"Sortino:{m['sortino_ratio']:>6.2f}")

    if show_comparison:
        valid   = {k: v for k, v in configs.items() if v[2].get("total_trades", 0) >= 10}
        best_key = max(valid or configs,
                       key=lambda k: (configs[k][2].get("sharpe_ratio", -999)))

        print(f"\n  {'Config':<32} {'Trades':>7} {'WR%':>5} "
              f"{'Sharpe':>7} {'Sortino':>7} {'DD%':>6}  {'Capital':>9} {'P&L':>8}")
        print("  " + "─" * 84)
        for name, (_, _, m) in configs.items():
            print(_row(name, m, best=(name == best_key)))

        df_all = configs["4. TODAS      (combinadas)"][0]
        if "signal_strategy" in df_all.columns:
            counts = df_all[df_all["signal"] != 0]["signal_strategy"].value_counts()
            if not counts.empty:
                print(f"\n  Origen de señales:")
                for strat, cnt in counts.items():
                    print(f"    {strat:<22}: {cnt}")

        months = max(len(df_h4) * 4 / 24 / 30, 1)
        best_m = configs[best_key][2]
        tpm    = best_m["total_trades"] / months
        print(f"\n  Trades/mes (mejor estrategia): ~{tpm:.1f}")
        if best_m["total_trades"] < 20:
            print(f"  ⚠️  {best_m['total_trades']} trades — ejecuta --validate "
                  f"para confirmar validez estadística")

    # Graficar mejor con trades > 0
    valid_configs = {k: v for k, v in configs.items() if v[2].get("total_trades", 0) >= 10}
    if valid_configs:
        best_key  = max(valid_configs,
                        key=lambda k: valid_configs[k][2].get("sharpe_ratio", -999))
        best_df, best_res, best_m = configs[best_key]
        print(f"\n  Gráfico: {best_key.strip()}...")
        plot_backtest(
            best_df, best_res, best_m,
            title=f"EUR/USD | {best_key.strip()} | v5.4",
            save=os.path.join(RESULTS_DIR, "backtest_final.png"),
        )
        print(f"\n  {'─'*54}")
        print(f"  Mejor: {best_key.strip()}")
        for k, v in [
            ("Trades",        best_m["total_trades"]),
            ("Win Rate",      f"{best_m['win_rate']:.1f}%"),
            ("Sharpe",        f"{best_m['sharpe_ratio']:.3f}"),
            ("Sortino",       f"{best_m['sortino_ratio']:.3f}"),
            ("R:R Real",      f"{best_m['actual_rr']:.2f}"),
            ("Max DD",        f"{best_m['max_drawdown_pct']:.1f}%"),
            ("Profit Factor", f"{best_m['profit_factor']:.2f}"),
            ("Capital Final", f"${best_m['final_capital']:.2f}"),
            ("P&L Neto",      f"${best_m['net_pnl_usd']:.2f}"),
        ]:
            print(f"  {k:<18}: {v}")

        s = best_m["sharpe_ratio"]
        print(f"\n  Sharpe: "
              f"{'✅ EXCELENTE (>2)' if s>2 else '✅ BUENO (1-2)' if s>1 else '⚠  DÉBIL (0-1)' if s>0 else '❌ NEGATIVO'}")

        if export_trades_csv is not None and not best_res["trades"].empty:
            csv_path = export_trades_csv(best_res["trades"], "best_backtest_trades.csv")
            print(f"  Trades CSV       : {csv_path}")

    if TRACKER_OK and save_best_run is not None:
        print(f"\n  Guardando en tracker...")
        save_best_run(configs, version="v5.4")

    return configs


# ─────────────────────────────────────────────────────────────────────
# MONTE CARLO
# ─────────────────────────────────────────────────────────────────────

def cmd_monte_carlo() -> tuple:
    print("\n🎲 SIMULACIÓN MONTE CARLO...")
    df_h4 = load_csv(SYMBOL, "H4")
    df_m15 = load_csv(SYMBOL, "M15")
    wr, avg_w = 0.60, 2.0
    pnl_array = None
    system_mode = _system_mode()

    if df_h4 is not None:
        use_m15, m15_note = _resolve_m15_usage(df_h4, df_m15, requested=(df_m15 is not None))
        print(f"  Sistema: {system_mode} | Sesión: {SESSION_MODE} | M15: {'ON' if use_m15 else 'OFF'}")
        if df_m15 is not None:
            print(f"  {m15_note}")
        df_sig = generate_signals(
            df_h4,
            df_m15=df_m15,
            use_session=True,
            session_mode=SESSION_MODE,
            use_adx=True,
            strict_entries=True,
            use_m15=use_m15,
            mode=system_mode,
        )
        res    = _eng().run(df_sig)
        m      = compute_metrics(res)
        if m.get("total_trades", 0) > 5:
            wr    = m["win_rate"] / 100
            avg_w = m["actual_rr"] if m.get("actual_rr", 0) > 0 else 2.0
            print(f"  WR real: {wr*100:.1f}%  R:R real: {avg_w:.2f}  "
                  f"Sortino: {m.get('sortino_ratio',0):.2f}")
            if avg_w < 1.5:
                print(f"  ⚠️  R:R {avg_w:.2f} < 1.5 — revisa trail_atr_mult en config.py")
            # Extraer P&L reales para MC con bootstrap
            if not res["trades"].empty:
                pnl_array = (res["trades"].groupby("entry_dt")["pnl_usd"]
                             .sum().values)

    # MC binomial (rápido)
    matrix, ruin_prob = monte_carlo(win_rate=wr, avg_win_r=avg_w)
    finals = matrix[:, -1]
    print(f"\n  Monte Carlo binomial ({len(matrix):,} simulaciones):")
    print(f"  Prob. ruina  : {ruin_prob*100:.3f}%")
    print(f"  Capital med. : ${finals.mean():.2f}")
    print(f"  Capital P25  : ${float(finals[finals.argsort()[len(finals)//4]]):.2f}")
    print(f"  Capital P75  : ${float(finals[finals.argsort()[len(finals)*3//4]]):.2f}")

    # MC con trades reales (más preciso, si hay suficientes datos)
    if pnl_array is not None and len(pnl_array) >= 10:
        print(f"\n  Monte Carlo con trades reales (bootstrap):")
        _, ruin_real = monte_carlo_trades(pnl_array)
        if ruin_real is not None:
            print(f"  Prob. ruina real: {ruin_real*100:.3f}%")
            verdict_real = "✅ Validado" if ruin_real < 0.05 else "❌ Ruina > 5%"
            print(f"  {verdict_real}")

    plot_monte_carlo(matrix, ruin_prob,
                     save=os.path.join(RESULTS_DIR, "monte_carlo.png"))

    verdict = "✅ Gestión de riesgo validada" if ruin_prob < 0.05 else \
              "❌ Ruina > 5% — reduce riesgo por trade en config.py"
    print(f"\n  {verdict}")
    return matrix, ruin_prob


# ─────────────────────────────────────────────────────────────────────
# WALK-FORWARD
# ─────────────────────────────────────────────────────────────────────

def cmd_walkforward() -> tuple:
    print("\n🔄 WALK-FORWARD TEST...")
    df_h4 = load_csv(SYMBOL, "H4")
    df_m15 = load_csv(SYMBOL, "M15")
    if df_h4 is None:
        print("⚠️  Sin datos H4.")
        return None, None  # type: ignore[return-value]

    system_mode = _system_mode()
    _, m15_note = _resolve_m15_usage(df_h4, df_m15, requested=(df_m15 is not None))
    print(f"  {len(df_h4)} barras "
          f"({df_h4.index[0].date()} → {df_h4.index[-1].date()})")
    print(f"  Sistema fijo: {system_mode} | Sesión: {SESSION_MODE} | M15: OFF")
    if df_m15 is not None:
        print(f"  {m15_note}")

    wf_df, wf_full = walk_forward(df_h4)
    rob = robustness(wf_df)

    print(f"\n  {'V':<5} {'TrSh':>8} {'TeSh':>8} {'TrTrd':>7} "
          f"{'TeTrd':>7} {'TePnL':>10} {'DD%':>7} {'WR%':>6}")
    print("  " + "─" * 74)
    for _, r in wf_df.iterrows():
        tag = "✅" if r["test_positive"] else "❌"
        print(f"  V{int(r['window']):<4} "
              f"{r['train_sharpe']:>8.2f} "
              f"{r['test_sharpe']:>8.2f} "
              f"{int(r.get('train_trades', 0)):>7} "
              f"{int(r.get('test_trades', 0)):>7} "
              f"${r.get('test_pnl', 0):>9.2f} "
              f"{r.get('test_max_dd', 0):>6.1f}% "
              f"{r.get('test_win_rate', 0):>5.0f}%  {tag}")

    print(f"\n  Ventanas positivas : {rob['positive_windows']}/{rob['total_windows']} "
          f"({rob['pct_positive_test']:.1f}%)")
    print(f"  DD TEST promedio   : {rob['avg_test_max_dd']:.2f}%")
    print(f"  {rob['verdict']}")

    plot_walkforward(wf_df, wf_full, rob,
                     save=os.path.join(RESULTS_DIR, "walkforward.png"))
    return wf_df, rob


# ─────────────────────────────────────────────────────────────────────
# VALIDATE — Bootstrap + Holdout (NUEVO v5.4)
# ─────────────────────────────────────────────────────────────────────

def cmd_validate() -> None:
    """
    Validación estadística completa.
    Responde: ¿es el sistema estadísticamente válido para trading real?

    1. Bootstrap de los trades reales → IC95% del Sharpe
    2. Holdout out-of-sample → comportamiento en datos nunca vistos
    3. Veredicto final de confianza
    """
    print("\n" + "=" * 62)
    print("  VALIDACIÓN ESTADÍSTICA — Bootstrap + Holdout")
    print("=" * 62)

    df_h4 = load_csv(SYMBOL, "H4")
    df_m15 = load_csv(SYMBOL, "M15")
    if df_h4 is None:
        print("⚠️  Sin datos H4.")
        return

    system_mode = _system_mode()
    use_m15, m15_note = _resolve_m15_usage(df_h4, df_m15, requested=(df_m15 is not None))

    # Obtener trades reales del sistema principal
    print(f"\n  Ejecutando sistema {system_mode} en datos completos...")
    print(f"  Sesión: {SESSION_MODE} | M15: {'ON' if use_m15 else 'OFF'}")
    if df_m15 is not None:
        print(f"  {m15_note}")
    df_sig  = generate_signals(
        df_h4,
        df_m15=df_m15,
        use_session=True,
        session_mode=SESSION_MODE,
        use_adx=True,
        strict_entries=True,
        use_m15=use_m15,
        mode=system_mode,
    )
    res     = _eng().run(df_sig)
    m       = compute_metrics(res)
    print(f"  Trades: {m['total_trades']} | WR: {m['win_rate']:.1f}% | "
          f"Sharpe: {m['sharpe_ratio']:.3f} | Sortino: {m['sortino_ratio']:.3f}")

    # ── 1. BOOTSTRAP ──────────────────────────────────────────────────
    print(f"\n  {'─'*54}")
    print(f"  BOOTSTRAP ({BOOTSTRAP['n_resample']:,} remuestreos, "
          f"IC {BOOTSTRAP['confidence']*100:.0f}%)")
    print(f"  {'─'*54}")

    if res["trades"].empty:
        print("  ⚠️  Sin trades para bootstrap.")
        return

    trade_ret = trade_returns(res["trades"], res["initial_capital"])
    periods_per_year = annualized_trades_per_year(len(trade_ret), res["equity"].index)
    bs = bootstrap_trades(
        trade_ret.values,
        periods_per_year=periods_per_year,
        initial_capital=res["initial_capital"],
    )

    if "reason" in bs:
        print(f"  {bs['reason']}")
    else:
        print(f"  N° trades         : {bs['n_trades']}")
        print(f"  Trades/año        : {bs['periods_per_year']:.2f}")
        print(f"  Sharpe medio      : {bs['sharpe_mean']:.3f}")
        print(f"  IC95% Sharpe      : [{bs['sharpe_ci_low']:.3f},  "
              f"{bs['sharpe_ci_high']:.3f}]")
        print(f"  Sortino medio     : {bs['sortino_mean']:.3f}")
        print(f"  Win Rate IC95%    : [{bs['wr_ci_low']:.1f}%, "
              f"{bs['wr_ci_high']:.1f}%]")
        print(f"  Capital mediano   : ${bs['capital_median']:.2f}")
        print(f"  Prob. ruina (BS)  : {bs['ruin_prob_pct']:.2f}%")
        print(f"\n  {bs['verdict']}")

        # Interpretación del IC
        lo = bs['sharpe_ci_low']
        if lo > 1.0:
            print(f"  → IC inferior {lo:.2f} > 1.0: edge sólido con alta confianza")
        elif lo > 0:
            print(f"  → IC inferior {lo:.2f} > 0: edge positivo pero con incertidumbre")
            print(f"     Recomendación: paper trading mínimo 3 meses antes de real")
        else:
            print(f"  → IC inferior {lo:.2f} ≤ 0: no es concluyente aún")
            print(f"     Necesitas más trades — continúa acumulando señales")

    # ── 2. HOLDOUT OUT-OF-SAMPLE ──────────────────────────────────────
    print(f"\n  {'─'*54}")
    print(f"  HOLDOUT OUT-OF-SAMPLE (últimos {HOLDOUT_MONTHS} meses)")
    print(f"  {'─'*54}")

    hd = holdout_test(
        df_h4,
        holdout_months=HOLDOUT_MONTHS,
        mode=system_mode,
        session_mode=SESSION_MODE,
        use_m15=use_m15,
        df_m15=df_m15,
    )
    if not hd.get("valid") and "reason" in hd:
        print(f"  ⚠️  {hd['reason']}")
    else:
        print(f"  Entrenamiento : {hd['train_start']} → {hd['train_end']} "
              f"({hd['train_bars']} barras)")
        print(f"  Holdout       : {hd['holdout_start']} → {hd['holdout_end']} "
              f"({hd['holdout_bars']} barras)")
        print(f"\n  {'Métrica':<20} {'Entrenamiento':>15} {'Holdout':>12}")
        print(f"  {'─'*50}")
        for label, tk, hk in [
            ("Trades",   "train_trades",  "holdout_trades"),
            ("Win Rate", "train_wr",      "holdout_wr"),
            ("Sharpe",   "train_sharpe",  "holdout_sharpe"),
            ("P&L $",    "train_pnl",     "holdout_pnl"),
        ]:
            tv = hd[tk]
            hv = hd[hk]
            tv_str = f"{tv:.1f}%" if "wr" in tk.lower() else \
                     f"{tv:.3f}" if "sharpe" in tk.lower() else \
                     f"${tv:.2f}" if "pnl" in tk.lower() else str(int(tv))
            hv_str = f"{hv:.1f}%" if "wr" in hk.lower() else \
                     f"{hv:.3f}" if "sharpe" in hk.lower() else \
                     f"${hv:.2f}" if "pnl" in hk.lower() else str(int(hv))
            print(f"  {label:<20} {tv_str:>15} {hv_str:>12}")

        print(f"\n  Degradación Sharpe: {hd['sharpe_degradation_pct']:+.1f}%")
        print(f"  {hd['verdict']}")

    # ── 3. VEREDICTO FINAL ────────────────────────────────────────────
    print(f"\n  {'═'*54}")
    print(f"  VEREDICTO FINAL — ¿Listo para paper trading?")
    print(f"  {'═'*54}")

    criteria = []
    if "n_trades" in bs:
        sharpe_ok   = bs.get("sharpe_ci_low", -1) > 0
        ruin_ok     = bs.get("ruin_prob_pct", 100) < 10
        trades_ok   = bs.get("n_trades", 0) >= 20
        criteria = [
            (sharpe_ok,   f"IC95% Sharpe inferior > 0:   {'✅' if sharpe_ok else '❌'}"),
            (ruin_ok,     f"Prob. ruina bootstrap < 10%: {'✅' if ruin_ok else '❌'} "
                          f"({bs.get('ruin_prob_pct',100):.1f}%)"),
            (trades_ok,   f"Trades ≥ 20 para validez:    {'✅' if trades_ok else '❌'} "
                          f"({bs.get('n_trades',0)} trades)"),
        ]
        if hd.get("valid"):
            holdout_ok = hd.get("holdout_pnl", -999) > -5
            criteria.append(
                (holdout_ok, f"Holdout sin pérdidas graves:  {'✅' if holdout_ok else '❌'} "
                             f"(${hd.get('holdout_pnl',0):.2f})")
            )
    else:
        criteria = [(False, bs.get("reason", "Bootstrap no calculable"))]

    for ok, text in criteria:
        print(f"  {text}")

    all_ok = all(ok for ok, _ in criteria) if criteria else False
    some_ok = sum(1 for ok, _ in criteria if ok) >= len(criteria) // 2 + 1

    print(f"\n  {'=' * 54}")
    if all_ok:
        print("  ✅ SISTEMA VALIDADO — Puedes iniciar paper trading")
        print("  Próximo paso: registrar trades en pestaña 'Trades Reales'")
    elif some_ok:
        print("  ⚠️  PROMETEDOR — Continúa acumulando trades")
        print(f"  Faltan {sum(1 for ok,_ in criteria if not ok)} criterio(s). "
              f"Meta: 30+ trades reales antes de capital real.")
    else:
        print("  ❌ NO VALIDADO — Sistema necesita más evidencia")
        print("  Recomendación: no operar con capital real todavía")
    print(f"  {'=' * 54}\n")


# ─────────────────────────────────────────────────────────────────────
# COMPARE ASSETS
# ─────────────────────────────────────────────────────────────────────

def cmd_compare_assets() -> list[dict]:
    print("\n" + "=" * 72)
    print("  COMPARACION MULTI-ACTIVO CONSERVADORA")
    print("=" * 72)
    print(f"  Cuenta base     : ${CAPITAL['initial']:.2f}")
    print(f"  Riesgo/trade    : {CAPITAL['risk_pct']*100:.1f}%")
    print(f"  Sistema fijo    : {_system_mode()} | Sesion: {SESSION_MODE} | M15: OFF")
    print("  '100% efectivas': alta probabilidad validada; no significa cero perdidas.")

    results = []
    for symbol in SYMBOLS:
        label = INSTRUMENTS[symbol].get("label", symbol)
        print(f"\n  Evaluando {symbol} ({label})...", flush=True)
        item = _evaluate_asset(symbol)
        results.append(item)
        if item.get("status") != "OK":
            print(f"    {item['status']}: {item['reason']}")
        else:
            m = item["metrics"]
            bs = item["bootstrap"]
            rob = item["walkforward"]
            print(f"    Trades: {m.get('total_trades',0)} | "
                  f"Sharpe: {m.get('sharpe_ratio',0):.3f} | "
                  f"P&L: {_fmt_money(m.get('net_pnl_usd',0))} | "
                  f"DD: {m.get('max_drawdown_pct',0):.1f}%")
            print(f"    Bootstrap Sharpe IC95%: "
                  f"[{bs.get('sharpe_ci_low',0):.3f}, {bs.get('sharpe_ci_high',0):.3f}] | "
                  f"Ruina BS: {bs.get('ruin_prob_pct',0):.2f}%")
            print(f"    Walk-Forward: {rob.get('positive_windows',0)}/{rob.get('total_windows',0)} "
                  f"ventanas positivas | DD prom. {rob.get('avg_test_max_dd',0):.2f}%")

    print(f"\n  {'Activo':<8} {'Estado':<10} {'Trades':>7} {'Tr/mes':>7} "
          f"{'WR%':>6} {'Sharpe':>8} {'Sortino':>8} {'PF':>6} "
          f"{'DD%':>6} {'P&L':>10} {'Ret/mes':>8} {'WF+':>7}")
    print("  " + "-" * 104)
    for item in results:
        if item.get("status") != "OK":
            print(f"  {item['symbol']:<8} {item['status']:<10} "
                  f"{'-':>7} {'-':>7} {'-':>6} {'-':>8} {'-':>8} {'-':>6} "
                  f"{'-':>6} {'-':>10} {'-':>8} {'-':>7}")
            continue
        m = item["metrics"]
        mon = item["monthly"]
        rob = item["walkforward"]
        wf = f"{rob.get('positive_windows',0)}/{rob.get('total_windows',0)}"
        print(f"  {item['symbol']:<8} {'OK':<10} "
              f"{m.get('total_trades',0):>7} "
              f"{item.get('trades_per_month',0):>7.1f} "
              f"{m.get('win_rate',0):>5.1f}% "
              f"{m.get('sharpe_ratio',0):>8.3f} "
              f"{m.get('sortino_ratio',0):>8.3f} "
              f"{m.get('profit_factor',0):>6.2f} "
              f"{m.get('max_drawdown_pct',0):>5.1f}% "
              f"{_fmt_money(m.get('net_pnl_usd',0)):>10} "
              f"{mon.get('avg_monthly_return_pct',0):>7.1f}% "
              f"{wf:>7}")

    print("\n  Detalle por activo:")
    for item in results:
        if item.get("status") != "OK":
            if item.get("status") == "NO OPERABLE":
                fz = item.get("feasibility", {})
                print(f"  - {item['symbol']}: No operable con gestion conservadora actual. "
                      f"{item['reason']}")
                print(f"    Señales detectadas: {item.get('signals', 0)} | "
                      f"Lotaje requerido aprox.: {fz.get('required_lot_for_risk', 0):.4f} | "
                      f"Min lot broker/config: {fz.get('min_lot', 0):.4f}")
                print(f"    Capital minimo estimado a 1.5% riesgo: "
                      f"{_fmt_money(fz.get('min_capital_required'))} | "
                      f"Riesgo usando min lot con $150: {fz.get('risk_pct_at_min_lot', 0):.2f}%")
            else:
                print(f"  - {item['symbol']}: Activo no evaluable por falta de datos. {item['reason']}")
            continue
        mon = item["monthly"]
        rob = item["walkforward"]
        print(f"  - {item['symbol']}: {item['bars']} barras H4 "
              f"({item['start']} -> {item['end']}); {item['m15_note']}")
        print(f"    Mejor mes: {_fmt_money(mon['best_month_pnl'])} | "
              f"Peor mes: {_fmt_money(mon['worst_month_pnl'])} | "
              f"MC real ruina: {_fmt_pct(item['ruin_real_pct'])}")
        print(f"    WF: {rob.get('verdict','N/A')}")
        if not item.get("contract_confirmed", True):
            print(f"    Contrato: pendiente de confirmar. {item.get('contract_note','')}")

    recommended = _choose_recommended_asset(results)
    ok_assets = [r for r in results if r.get("status") == "OK"]
    reference = recommended or (ok_assets[0] if ok_assets else None)

    print("\n  Recomendacion:")
    if recommended is None:
        print("  Activo recomendado hoy: ninguno con criterio conservador completo.")
    else:
        rob = recommended["walkforward"]
        print(f"  Activo recomendado hoy: {recommended['symbol']} "
              f"({recommended['label']})")
        print(f"  Motivo: DD bajo, bootstrap positivo y WF "
              f"{rob.get('positive_windows',0)}/{rob.get('total_windows',0)} "
              f"ventanas positivas.")

    missing = [r for r in results if r.get("status") == "SIN DATOS"]
    for item in missing:
        print(f"  Activo no evaluable por falta de datos: {item['symbol']} — {item['reason']}")

    blocked = [r for r in results if r.get("status") == "NO OPERABLE"]
    for item in blocked:
        fz = item.get("feasibility", {})
        print(f"  Activo con señales pero no operable por tamaño de cuenta/lote: {item['symbol']} "
              f"(capital minimo estimado {_fmt_money(fz.get('min_capital_required'))})")

    print("\n  Metas financieras con cuenta de $150:")
    if reference is None:
        print("  No hay activo evaluable para estimar capital recomendado.")
    else:
        goals = reference["goals"]
        weekly_label = "CONSERVADORA" if goals["weekly_is_conservative"] else "NO CONSERVADORA"
        monthly_label = "CONSERVADORA" if goals["monthly_is_conservative"] else "NO CONSERVADORA"
        print(f"  Referencia usada: {reference['symbol']} "
              f"(retorno mensual historico prom. {goals['avg_monthly_return_pct']:.2f}%)")
        print(f"  Objetivo realista estimado con $150 al rendimiento validado: "
              f"{_fmt_money(goals['expected_weekly_usd'])}/semana | "
              f"{_fmt_money(goals['expected_monthly_usd'])}/mes")
        print(f"  Meta $100/semana: requiere {goals['weekly_required_pct']:.1f}% semanal "
              f"-> {weekly_label}")
        print(f"    Capital recomendado sin subir riesgo: "
              f"{_fmt_money(goals['weekly_capital_required'])}")
        print(f"    Riesgo/trade requerido con $150: "
              f"{_fmt_pct(goals['weekly_risk_required_pct'])}")
        print(f"  Meta $600/mes: requiere {goals['monthly_required_pct']:.1f}% mensual "
              f"-> {monthly_label}")
        print(f"    Capital recomendado sin subir riesgo: "
              f"{_fmt_money(goals['monthly_capital_required'])}")
        print(f"    Riesgo/trade requerido con $150: "
              f"{_fmt_pct(goals['monthly_risk_required_pct'])}")

    if len(ok_assets) < 2:
        if missing:
            print("\n  Dashboard Data Analytics: no generado; falta data real para varios activos.")
            print("  Siguiente paso: python -X utf8 run.py --download-assets")
        else:
            print("\n  Dashboard Data Analytics: no generado; solo hay un activo operable "
                  "con la cuenta y lote actuales.")
    report_path = write_compare_assets_report(results)
    print(f"  Reporte Markdown: {report_path}")
    print("\n  Nota: --compare-assets no escribe resultados en el tracker.\n")
    return results


# ─────────────────────────────────────────────────────────────────────
# COMPARE TIMEFRAMES
# ─────────────────────────────────────────────────────────────────────

def _evaluate_timeframe(symbol: str, timeframe: str) -> dict:
    path = _data_path(symbol, timeframe)
    if not os.path.exists(path):
        return {"symbol": symbol, "timeframe": timeframe, "status": "SIN DATOS"}

    df = load_csv(symbol, timeframe)
    if df is None or len(df) < 500:
        return {"symbol": symbol, "timeframe": timeframe, "status": "INSUFICIENTE"}

    df_sig = generate_signals(
        df,
        use_session=True,
        session_mode=SESSION_MODE,
        use_adx=True,
        strict_entries=True,
        use_m15=False,
        mode=_system_mode(),
    )
    signals = int((df_sig["signal"] != 0).sum())
    res = BacktestEngine(instrument=symbol, trailing_stop=True, partial_tp=True).run(df_sig)
    metrics = compute_metrics(res)
    status = "OK"
    feasibility = _position_feasibility(symbol, df_sig)
    if signals > 0 and metrics.get("total_trades", 0) == 0 and not feasibility["is_affordable"]:
        status = "NO OPERABLE"

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "status": status,
        "bars": len(df),
        "start": df.index[0].date(),
        "end": df.index[-1].date(),
        "signals": signals,
        "metrics": metrics,
        "feasibility": feasibility,
        "contract_confirmed": bool(INSTRUMENTS[symbol].get("contract_confirmed", True)),
    }


def cmd_compare_timeframes(symbol: str | None = None) -> list[dict]:
    symbols = [symbol.upper()] if symbol else SYMBOLS
    timeframes = ["M5", "M15", "H1", "H4"]
    rows = []

    print("\n" + "=" * 84)
    print("  COMPARACION DE TEMPORALIDADES — SIN REOPTIMIZAR")
    print("=" * 84)
    print(f"  Sistema fijo: {_system_mode()} | Sesion: {SESSION_MODE} | Confirmacion M15: OFF")

    for sym in symbols:
        if sym not in INSTRUMENTS:
            print(f"\n  {sym}: simbolo no configurado")
            continue
        print(f"\n  {sym} ({INSTRUMENTS[sym].get('label', sym)})")
        for tf in timeframes:
            row = _evaluate_timeframe(sym, tf)
            rows.append(row)
            if row["status"] != "OK":
                print(f"    {tf:<3} {row['status']}")
                continue
            m = row["metrics"]
            print(f"    {tf:<3} bars:{row['bars']:>5} trades:{m.get('total_trades',0):>4} "
                  f"WR:{m.get('win_rate',0):>5.1f}% Sharpe:{m.get('sharpe_ratio',0):>6.2f} "
                  f"PF:{m.get('profit_factor',0):>5.2f} DD:{m.get('max_drawdown_pct',0):>5.1f}% "
                  f"P&L:{_fmt_money(m.get('net_pnl_usd',0)):>9}")

    print(f"\n  {'Activo':<8} {'TF':<4} {'Estado':<12} {'Trades':>7} {'WR%':>6} "
          f"{'Sharpe':>8} {'PF':>6} {'DD%':>7} {'P&L':>10}")
    print("  " + "-" * 82)
    for row in rows:
        if row["status"] != "OK":
            print(f"  {row['symbol']:<8} {row['timeframe']:<4} {row['status']:<12} "
                  f"{'-':>7} {'-':>6} {'-':>8} {'-':>6} {'-':>7} {'-':>10}")
            continue
        m = row["metrics"]
        print(f"  {row['symbol']:<8} {row['timeframe']:<4} {'OK':<12} "
              f"{m.get('total_trades',0):>7} "
              f"{m.get('win_rate',0):>5.1f}% "
              f"{m.get('sharpe_ratio',0):>8.3f} "
              f"{m.get('profit_factor',0):>6.2f} "
              f"{m.get('max_drawdown_pct',0):>6.1f}% "
              f"{_fmt_money(m.get('net_pnl_usd',0)):>10}")

    print("\n  Nota: esta comparacion no optimiza parametros por timeframe; es prueba de robustez.")
    return rows


# ─────────────────────────────────────────────────────────────────────
# RISK PROFILES
# ─────────────────────────────────────────────────────────────────────

def cmd_risk_profiles(symbol: str = SYMBOL) -> list[dict]:
    symbol = symbol.upper()
    print("\n" + "=" * 72)
    print(f"  PERFILES DE RIESGO — {symbol}")
    print("=" * 72)
    df_h4 = load_csv(symbol, "H4")
    if df_h4 is None:
        print(f"  Sin datos H4 para {symbol}.")
        return []

    df_sig = generate_signals(
        df_h4,
        use_session=True,
        session_mode=SESSION_MODE,
        use_adx=True,
        strict_entries=True,
        use_m15=False,
        mode=_system_mode(),
    )
    rows = evaluate_risk_profiles(df_sig, symbol=symbol)
    ranked = conservative_rank(rows)
    print(f"  Sistema fijo: {_system_mode()} | Sesion: {SESSION_MODE} | M15: OFF")
    print(f"\n  {'Perfil':<12} {'Risk%':>7} {'Trades':>7} {'WR%':>6} "
          f"{'PF':>6} {'Sharpe':>8} {'DD%':>7} {'DD$':>8} {'P&L':>10} {'StreakL':>8}")
    print("  " + "-" * 90)
    for row in ranked:
        print(f"  {row['profile']:<12} "
              f"{row['risk_pct']*100:>6.2f}% "
              f"{row.get('total_trades',0):>7} "
              f"{row.get('win_rate',0):>5.1f}% "
              f"{row.get('profit_factor',0):>6.2f} "
              f"{row.get('sharpe_ratio',0):>8.3f} "
              f"{row.get('max_drawdown_pct',0):>6.1f}% "
              f"${row.get('max_drawdown_usd',0):>7.2f} "
              f"{_fmt_money(row.get('net_pnl_usd',0)):>10} "
              f"{row.get('max_loss_streak',0):>8}")

    print("\n  Nota: perfiles agresivos solo son diagnostico; no habilitan trading real.")
    return rows


# ─────────────────────────────────────────────────────────────────────
# DIAGNOSE
# ─────────────────────────────────────────────────────────────────────

def cmd_diagnose() -> None:
    print("\n🔬 DIAGNÓSTICO DE FILTROS...")
    df_h4 = load_csv(SYMBOL, "H4")
    if df_h4 is None:
        print("⚠️  Sin datos H4.")
        return
    from indicators import diagnose_signals
    diagnose_signals(df_h4)


# ─────────────────────────────────────────────────────────────────────
# ALL
# ─────────────────────────────────────────────────────────────────────

def cmd_all() -> None:
    print(BANNER.format(
        fast  = INDICATORS["ema_fast"],
        slow  = INDICATORS["ema_slow"],
        adx   = INDICATORS["adx_min"],
        risk  = CAPITAL["risk_pct"] * 100,
        score = INDICATORS["min_score"],
    ))

    df_check = load_csv(SYMBOL, "H4")
    n = len(df_check) if df_check is not None else 0
    if n == 0:
        cmd_download()
    else:
        status = "✅" if n >= 5000 else "⚠️  descarga 8000 barras"
        print(f"📡 Datos: {n} barras H4 {status}")

    configs           = cmd_backtest(show_comparison=True)
    matrix, ruin_prob = cmd_monte_carlo()
    wf_df, rob        = cmd_walkforward()

    if not configs:
        return

    # Solo considerar configs con suficientes trades para el resumen
    valid_for_summary = {k: v for k, v in configs.items()
                         if v[2].get("total_trades", 0) >= 10}
    best_m = max((valid_for_summary or configs).values(),
                 key=lambda x: x[2].get("sharpe_ratio", -999))[2]

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║   RESUMEN FINAL v5.4                                     ║")
    print("╠" + "═" * 58 + "╣")
    rows = [
        ("Sharpe (mejor)",   f"{best_m.get('sharpe_ratio',0):.3f}"),
        ("Sortino (mejor)",  f"{best_m.get('sortino_ratio',0):.3f}"),
        ("Win Rate",         f"{best_m.get('win_rate',0):.1f}%"),
        ("R:R Real",         f"{best_m.get('actual_rr',0):.2f}"),
        ("Max Drawdown",     f"{best_m.get('max_drawdown_pct',0):.1f}%"),
        ("Capital Final",    f"${best_m.get('final_capital',150):.2f}"),
        ("P&L Neto",         f"${best_m.get('net_pnl_usd',0):.2f}"),
        ("Prob. Ruina MC",   f"{ruin_prob*100:.3f}%"),
        ("Walk-Forward",     (rob.get("verdict","N/A")[:35] if rob else "N/A")),
        ("WF ventanas pos.", (f"{rob.get('positive_windows',0)}/{rob.get('total_windows',0)} "
                              f"({rob.get('pct_positive_test',0):.1f}%)" if rob else "N/A")),
        ("WF DD promedio",   (f"{rob.get('avg_test_max_dd',0):.2f}%"
                              if rob else "N/A")),
    ]
    for k, v in rows:
        print(f"║  {k:<22} : {str(v):<32}║")
    print("╠" + "═" * 58 + "╣")
    print("║  Tip: ejecuta --validate para análisis estadístico      ║")
    print("╚" + "═" * 58 + "╝\n")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sistema Trading EUR/USD v5.4")
    p.add_argument("--download",    action="store_true", help="Descargar datos MT5")
    p.add_argument("--download-asset", metavar="SYMBOL",
                   help="Descargar un activo canonico configurado, ej. EURUSD o XAUUSD")
    p.add_argument("--download-assets", action="store_true",
                   help="Descargar todos los activos configurados")
    p.add_argument("--backtest",    action="store_true", help="Backtest estrategias")
    p.add_argument("--monte-carlo", action="store_true", dest="mc")
    p.add_argument("--walkforward", action="store_true")
    p.add_argument("--validate",    action="store_true",
                   help="Bootstrap + Holdout — validación estadística completa")
    p.add_argument("--compare-assets", action="store_true",
                   help="Comparar activos configurados y evaluar metas")
    p.add_argument("--compare-timeframes", nargs="?", const="", metavar="SYMBOL",
                   help="Comparar M5/M15/H1/H4 para un simbolo o todos")
    p.add_argument("--risk-profiles", nargs="?", const=SYMBOL, metavar="SYMBOL",
                   help="Comparar riesgo conservador/base/moderado/agresivo")
    p.add_argument("--diagnose",    action="store_true",
                   help="Diagnóstico de filtros por estrategia")
    p.add_argument("--all",         action="store_true", help="Pipeline completo")
    a = p.parse_args()

    if   a.download_assets: cmd_download(all_assets=True)
    elif a.download_asset: cmd_download(symbol=a.download_asset)
    elif a.download:    cmd_download()
    elif a.backtest:    cmd_backtest()
    elif a.mc:          cmd_monte_carlo()
    elif a.walkforward: cmd_walkforward()
    elif a.validate:    cmd_validate()
    elif a.compare_assets: cmd_compare_assets()
    elif a.compare_timeframes is not None: cmd_compare_timeframes(a.compare_timeframes or None)
    elif a.risk_profiles: cmd_risk_profiles(a.risk_profiles)
    elif a.diagnose:    cmd_diagnose()
    else:               cmd_all()
