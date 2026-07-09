"""
config.py v5.4
Filosofia: priorizar ganancias netas consistentes con una frecuencia util.
Base actual: pullback optimizado sobre H4 completo y validado en holdout.
"""
import os


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "y"}


MT5 = {
    "login": int(os.getenv("MT5_LOGIN", "0") or 0),
    "password": os.getenv("MT5_PASSWORD", ""),
    "server": os.getenv("MT5_SERVER", "MetaQuotes-Demo"),
    "path": os.getenv("MT5_PATH", ""),
}

TRADING_MODE = {
    "demo_mode": True,
    "real_mode": False,
    "allow_real_trading": _env_bool("ALLOW_REAL_TRADING", False),
    "require_real_confirmation": "I_UNDERSTAND_REAL_TRADING_RISK",
    "real_confirmation": os.getenv("REAL_TRADING_CONFIRMATION", ""),
}

CAPITAL = {
    "initial"        : 150.0,
    "risk_pct"       : 0.015,   # 1.5% validado con MC (WR 68%, R:R 2.0)
    "rr_ratio"       : 2.0,
    "max_drawdown"   : 0.15,
    "ruin_threshold" : 0.30,
    "max_trades_day" : 2,
    # Salidas
    "atr_sl_mult"    : 1.5,
    "atr_tp1_mult"   : 2.0,     # TP parcial 50% posición
    "atr_tp2_mult"   : 4.0,     # TP final — dejar correr
    "trail_atr_mult" : 2.5,     # deja respirar el pullback ganador antes de cerrar
    "spread_pips"    : 1.5,     # EUR/USD spread típico H4
    "slippage_pips"  : 0.2,
    "commission_per_lot_round_turn": 0.0,
    "max_daily_loss_pct": 0.03,
    "max_weekly_loss_pct": 0.06,
    "max_loss_streak_pause": 4,
}

RISK_PROFILES = {
    "conservador": 0.005,
    "base": 0.015,
    "moderado": 0.025,
    "agresivo": 0.050,
}

SYMBOLS = ["EURUSD", "XAUUSD", "GBPUSD", "USDJPY", "NAS100", "US30", "BTCUSD"]
SYMBOL  = "EURUSD"

# Economia por instrumento para backtesting multi-activo.
# Nota: EURUSD conserva la unidad historica del motor para no mover el baseline.
# XAUUSD queda preparado, pero debe validarse contra el contrato real del broker.
INSTRUMENTS = {
    "EURUSD": {
        "label": "EUR/USD",
        "aliases": ["EURUSD", "EURUSDm", "EURUSD.", "EURUSD_i"],
        "pip_size": 0.0001,
        "pip_value_per_lot": 0.10,
        "spread_pips": 1.5,
        "max_spread_pips": 3.0,
        "slippage_pips": 0.2,
        "commission_per_lot_round_turn": 0.0,
        "spread_value_per_lot": 0.001,
        "min_lot": 0.01,
        "max_lot": 200.0,
        "lot_step": 0.01,
        "required_timeframes": ["H4"],
        "contract_confirmed": True,
    },
    "XAUUSD": {
        "label": "Oro / XAUUSD",
        "aliases": ["XAUUSD", "XAUUSDm", "XAUUSD.", "XAUUSD_i", "GOLD", "GOLDm"],
        "pip_size": 0.01,
        "pip_value_per_lot": 1.00,
        "spread_pips": 30.0,
        "max_spread_pips": 60.0,
        "slippage_pips": 5.0,
        "commission_per_lot_round_turn": 0.0,
        "spread_value_per_lot": 1.00,
        "min_lot": 0.01,
        "max_lot": 50.0,
        "lot_step": 0.01,
        "required_timeframes": ["H4"],
        "contract_confirmed": False,
        "contract_note": "Validar pip_value_per_lot y spread_pips con el broker antes de operar.",
    },
    "GBPUSD": {
        "label": "GBP/USD",
        "aliases": ["GBPUSD", "GBPUSDm", "GBPUSD.", "GBPUSD_i"],
        "pip_size": 0.0001,
        "pip_value_per_lot": 0.10,
        "spread_pips": 2.0,
        "max_spread_pips": 4.0,
        "slippage_pips": 0.3,
        "commission_per_lot_round_turn": 0.0,
        "spread_value_per_lot": 0.001,
        "min_lot": 0.01,
        "max_lot": 200.0,
        "lot_step": 0.01,
        "required_timeframes": ["M5", "M15", "H1", "H4"],
        "contract_confirmed": False,
        "contract_note": "Validar valor de pip y spread con el broker antes de operar.",
    },
    "USDJPY": {
        "label": "USD/JPY",
        "aliases": ["USDJPY", "USDJPYm", "USDJPY.", "USDJPY_i"],
        "pip_size": 0.01,
        "pip_value_per_lot": 0.10,
        "spread_pips": 2.0,
        "max_spread_pips": 4.0,
        "slippage_pips": 0.3,
        "commission_per_lot_round_turn": 0.0,
        "spread_value_per_lot": 0.001,
        "min_lot": 0.01,
        "max_lot": 200.0,
        "lot_step": 0.01,
        "required_timeframes": ["M5", "M15", "H1", "H4"],
        "contract_confirmed": False,
        "contract_note": "Validar valor de pip JPY con el broker antes de operar.",
    },
    "NAS100": {
        "label": "NASDAQ 100",
        "aliases": ["NAS100", "NAS100m", "USTEC", "USTECm", "US100", "US100m"],
        "pip_size": 1.0,
        "pip_value_per_lot": 1.0,
        "spread_pips": 20.0,
        "max_spread_pips": 50.0,
        "slippage_pips": 2.0,
        "commission_per_lot_round_turn": 0.0,
        "spread_value_per_lot": 1.0,
        "min_lot": 0.01,
        "max_lot": 50.0,
        "lot_step": 0.01,
        "required_timeframes": ["M5", "M15", "H1", "H4"],
        "contract_confirmed": False,
        "contract_note": "Indice CFD: confirmar tick value, contrato y horario del broker.",
    },
    "US30": {
        "label": "Dow Jones / US30",
        "aliases": ["US30", "US30m", "DJ30", "DJ30m", "WS30", "WS30m"],
        "pip_size": 1.0,
        "pip_value_per_lot": 1.0,
        "spread_pips": 30.0,
        "max_spread_pips": 80.0,
        "slippage_pips": 3.0,
        "commission_per_lot_round_turn": 0.0,
        "spread_value_per_lot": 1.0,
        "min_lot": 0.01,
        "max_lot": 50.0,
        "lot_step": 0.01,
        "required_timeframes": ["M5", "M15", "H1", "H4"],
        "contract_confirmed": False,
        "contract_note": "Indice CFD: confirmar tick value, contrato y horario del broker.",
    },
    "BTCUSD": {
        "label": "Bitcoin / BTCUSD",
        "aliases": ["BTCUSD", "BTCUSDm", "BTCUSD.", "BTCUSD_i", "Bitcoin"],
        "pip_size": 1.0,
        "pip_value_per_lot": 1.0,
        "spread_pips": 100.0,
        "max_spread_pips": 300.0,
        "slippage_pips": 10.0,
        "commission_per_lot_round_turn": 0.0,
        "spread_value_per_lot": 1.0,
        "min_lot": 0.01,
        "max_lot": 10.0,
        "lot_step": 0.01,
        "required_timeframes": ["M5", "M15", "H1", "H4"],
        "contract_confirmed": False,
        "contract_note": "Cripto CFD: confirmar tick value, spread, comision y horario del broker.",
    },
}

BARS = {
    "M5" : 12000,
    "M15": 12000,
    "H1" : 10000,
    "H4" : 8000,
}

DATA_RANGE = {
    "date_from": os.getenv("DATA_DATE_FROM", ""),
    "date_to": os.getenv("DATA_DATE_TO", ""),
}

INDICATORS = {
    "ema_fast"         : 10,
    "ema_slow"         : 55,
    "sma_macro"        : 200,
    "rsi_period"       : 14,
    "rsi_low"          : 40,
    "rsi_high"         : 60,
    "rsi_momentum_min" : 52,
    "atr_period"       : 14,
    "atr_sl_mult"      : 1.5,
    "atr_tp_mult"      : 4.0,
    "use_fvg"          : True,
    "adx_min"          : 15,
    "slope_min_pips"   : 0.2,
    "stoch_k"          : 5,
    "stoch_d"          : 3,
    "stoch_oversold"   : 35,
    "stoch_overbought" : 65,
    "macd_fast"        : 12,
    "macd_slow"        : 26,
    "macd_signal"      : 9,
    "bb_period"        : 20,
    "bb_std"           : 2,
    "use_pullback"     : True,
    "pullback_atr"     : 1.1,
    # Scoring de confluencia (Camino A)
    "use_scoring"      : True,
    "min_score"        : 3,     # mínimo 3/5 indicadores confirmando
    "strategy_mode"    : "pullback",
}

SESSION_MODE = "london_ny"

MONTE_CARLO = {
    "n_simulations"  : 10_000,
    "n_trades"       : 200,
    "ruin_threshold" : 0.50,
}

WALKFORWARD = {
    "n_windows"  : 10,
    "train_pct"  : 0.70,
    "fast_range" : range(5, 25, 5),
    "slow_range" : range(20, 70, 10),
}

# ── HOLDOUT OUT-OF-SAMPLE (DeepSeek rec.) ────────────────────────────
# Los últimos HOLDOUT_MONTHS meses se reservan como prueba final.
# NUNCA se usan en optimización ni en el backtest principal.
# Solo se evalúan con --validate cuando el sistema está "congelado".
HOLDOUT_MONTHS = 18

BOOTSTRAP = {
    "n_resample"     : 10_000,   # remuestreos con reemplazo
    "confidence"     : 0.95,     # intervalo de confianza 95%
    "min_trades_valid": 10,      # mínimo trades para considerar válido
}

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

if __name__ == "__main__":
    print("=" * 58)
    print("  CONFIG v5.4 — CAMINO A | TRADING ALGORÍTMICO EUR/USD")
    print("=" * 58)
    for k, v in [
        ("Par",             SYMBOL),
        ("Capital",         f"${CAPITAL['initial']}"),
        ("Riesgo/trade",    f"{CAPITAL['risk_pct']*100:.1f}% = ${CAPITAL['initial']*CAPITAL['risk_pct']:.2f}"),
        ("EMA",             f"{INDICATORS['ema_fast']}/{INDICATORS['ema_slow']}"),
        ("ADX mínimo",      INDICATORS['adx_min']),
        ("Score mínimo",    f"{INDICATORS['min_score']}/5 indicadores"),
        ("TP1 / TP2",       f"ATR×{CAPITAL['atr_tp1_mult']} / ATR×{CAPITAL['atr_tp2_mult']}"),
        ("Trailing",        f"ATR×{CAPITAL['trail_atr_mult']}"),
        ("Spread modelado", f"{CAPITAL['spread_pips']} pips"),
        ("Holdout",         f"últimos {HOLDOUT_MONTHS} meses (fuera de muestra)"),
        ("Sesión",          SESSION_MODE),
    ]:
        print(f"  {k:<18}: {v}")
