"""
mt5_data.py — Conector MT5 + Carga de Datos
Descarga OHLCV reales de EUR/USD, XAU/USD y otros instrumentos configurados.
"""
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

from config import MT5, SYMBOL, SYMBOLS, BARS, DATA_DIR, INSTRUMENTS, DATA_RANGE


def connect():
    """Conecta a MT5. Retorna True si exitoso."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("❌ MetaTrader5 no instalado: pip install MetaTrader5")
        return False, None

    path_arg = {"path": MT5["path"]} if MT5["path"] else {}
    if not mt5.initialize(**path_arg):
        print(f"❌ MT5 no pudo inicializar: {mt5.last_error()}")
        print("   Asegúrate de tener MT5 abierto y conectado.")
        return False, mt5

    if MT5["login"] != 0:
        if not mt5.login(MT5["login"], MT5["password"], MT5["server"]):
            print(f"❌ Login fallido: {mt5.last_error()}")
            mt5.shutdown()
            return False, mt5

    acc = mt5.account_info()
    if acc is None:
        print("❌ No se pudo obtener info de cuenta.")
        mt5.shutdown()
        return False, mt5

    print(f"✅ MT5 conectado | {acc.company} | "
          f"Cuenta: {acc.login} | Balance: ${acc.balance:.2f} | "
          f"{'Demo' if acc.trade_mode == 0 else 'Real'}")
    return True, mt5


def _canonical_symbol(symbol=None):
    """Normaliza el simbolo interno usado para nombrar CSVs."""
    sym = (symbol or SYMBOL).upper()
    return sym if sym in INSTRUMENTS else sym


def find_symbol(mt5, symbol=None):
    """Busca el simbolo real del broker para el instrumento canonico."""
    canonical = _canonical_symbol(symbol)
    spec = INSTRUMENTS.get(canonical, {})
    candidates = list(dict.fromkeys(spec.get("aliases", []) + [canonical]))

    for sym in candidates:
        if mt5.symbol_info(sym) is not None:
            return sym

    all_syms = mt5.symbols_get()
    if not all_syms:
        return None

    if canonical == "XAUUSD":
        matches = [
            s.name for s in all_syms
            if (("XAU" in s.name.upper() and "USD" in s.name.upper()) or
                "GOLD" in s.name.upper())
        ]
    elif canonical == "EURUSD":
        matches = [
            s.name for s in all_syms
            if "EUR" in s.name.upper() and "USD" in s.name.upper()
        ]
    else:
        matches = [s.name for s in all_syms if canonical in s.name.upper()]

    return matches[0] if matches else None


def download_symbol(mt5, symbol, timeframe_str, n_bars):
    """Descarga n_bars de OHLCV y retorna DataFrame."""
    tf_map = {
        'M5' : mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'H1' : mt5.TIMEFRAME_H1,
        'H4' : mt5.TIMEFRAME_H4,
        'M1' : mt5.TIMEFRAME_M1,
    }
    tf    = tf_map[timeframe_str]
    info  = mt5.symbol_info(symbol)
    if info and not info.visible:
        mt5.symbol_select(symbol, True)
        time.sleep(0.3)

    date_from = DATA_RANGE.get("date_from") if isinstance(DATA_RANGE, dict) else ""
    date_to = DATA_RANGE.get("date_to") if isinstance(DATA_RANGE, dict) else ""
    if date_from and date_to:
        start = datetime.fromisoformat(date_from)
        end = datetime.fromisoformat(date_to)
        rates = mt5.copy_rates_range(symbol, tf, start, end)
    else:
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
    if rates is None or len(rates) == 0:
        print(f"   ❌ Sin datos para {symbol} {timeframe_str}")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('time')
    df = df.rename(columns={'tick_volume': 'volume'})
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df


def validate_ohlcv(df: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, dict]:
    """Valida OHLCV: duplicados, nulos, gaps y ultima vela incompleta."""
    freq_map = {"M1": "1min", "M5": "5min", "M15": "15min", "H1": "1h", "H4": "4h"}
    issues = {
        "rows_raw": int(len(df)),
        "duplicates_removed": 0,
        "null_rows_removed": 0,
        "gaps_detected": 0,
        "session_gaps_detected": 0,
        "unexpected_gaps_detected": 0,
        "last_incomplete_removed": False,
    }
    if df.empty:
        return df, issues

    df = df.sort_index()
    dupes = int(df.index.duplicated(keep="last").sum())
    if dupes:
        df = df[~df.index.duplicated(keep="last")]
    issues["duplicates_removed"] = dupes

    null_rows = int(df[['open', 'high', 'low', 'close']].isna().any(axis=1).sum())
    if null_rows:
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
    issues["null_rows_removed"] = null_rows

    freq = freq_map.get(timeframe)
    if freq and len(df) > 2:
        expected_delta = pd.Timedelta(freq)
        index_series = df.index.to_series()
        gaps = index_series.diff().dropna()
        gap_mask = gaps > expected_delta * 1.5
        issues["gaps_detected"] = int(gap_mask.sum())
        if issues["gaps_detected"]:
            gap_rows = pd.DataFrame({
                "prev": index_series.shift(1).loc[gap_mask.index],
                "curr": index_series.loc[gap_mask.index],
                "gap": gaps,
            }).loc[gap_mask]
            # Weekend, holiday and daily-session closures are expected in FX/CFD data.
            session_mask = gap_rows.apply(
                lambda r: (
                    r["prev"].date() != r["curr"].date()
                    or r["prev"].weekday() >= 4
                    or r["curr"].weekday() <= 0
                ),
                axis=1,
            )
            issues["session_gaps_detected"] = int(session_mask.sum())
            issues["unexpected_gaps_detected"] = int((~session_mask).sum())
        now = pd.Timestamp.utcnow().tz_localize(None)
        last_ts = pd.Timestamp(df.index[-1]).tz_localize(None) if df.index[-1].tzinfo else pd.Timestamp(df.index[-1])
        if now - last_ts < expected_delta:
            df = df.iloc[:-1]
            issues["last_incomplete_removed"] = True

    issues["rows_final"] = int(len(df))
    return df, issues


def _download_timeframes(mt5, canonical_symbol, broker_symbol):
    """Descarga timeframes y guarda con nombre canonico del sistema."""
    tick = mt5.symbol_info_tick(broker_symbol)
    bid = tick.bid if tick is not None else 0
    print(f"   Instrumento: {canonical_symbol} | Broker: {broker_symbol} | Bid: {bid:.5f}")

    data = {}
    for tf, n in BARS.items():
        print(f"   Descargando {broker_symbol} {tf} ({n} barras)...", end=" ")
        df = download_symbol(mt5, broker_symbol, tf, n)
        if df is not None:
            df, quality = validate_ohlcv(df, tf)
            data[tf] = df
            path = os.path.join(DATA_DIR, f"{canonical_symbol}_{tf}.csv")
            df.to_csv(path)
            print(f"✅ {len(df)} barras → {path}")
            qpath = os.path.join(DATA_DIR, f"{canonical_symbol}_{tf}_quality.txt")
            with open(qpath, "w", encoding="utf-8") as f:
                for key, value in quality.items():
                    f.write(f"{key}: {value}\n")
        else:
            print("❌")
    return data


def download_all(symbol=None):
    """
    Conecta a MT5, descarga todos los timeframes y guarda CSVs.
    Retorna dict con DataFrames o None si falla.
    """
    ok, mt5 = connect()
    if not ok:
        return None

    canonical = _canonical_symbol(symbol)
    broker_symbol = find_symbol(mt5, canonical)
    if broker_symbol is None:
        print(f"❌ Símbolo {canonical} no encontrado en el broker.")
        mt5.shutdown()
        return None

    data = _download_timeframes(mt5, canonical, broker_symbol)

    mt5.shutdown()
    print(f"✅ Desconectado de MT5.")
    return data


def download_many(symbols=None):
    """Descarga varios instrumentos configurados en una sola conexion MT5."""
    ok, mt5 = connect()
    if not ok:
        return None

    requested = symbols or SYMBOLS
    all_data = {}
    for symbol in requested:
        canonical = _canonical_symbol(symbol)
        broker_symbol = find_symbol(mt5, canonical)
        if broker_symbol is None:
            print(f"❌ Símbolo {canonical} no encontrado en el broker.")
            all_data[canonical] = None
            continue
        all_data[canonical] = _download_timeframes(mt5, canonical, broker_symbol)

    mt5.shutdown()
    print(f"✅ Desconectado de MT5.")
    return all_data


def load_csv(symbol=None, timeframe="H4"):
    """Carga datos desde CSV (sin necesitar MT5 abierto)."""
    sym  = symbol or SYMBOL
    path = os.path.join(DATA_DIR, f"{sym}_{timeframe}.csv")
    if not os.path.exists(path):
        print(f"❌ No existe: {path} — ejecuta primero: python run.py --download")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = 'time'
    return df


def load_all(symbol=None):
    """Carga todos los timeframes desde CSV."""
    return {tf: load_csv(symbol, tf) for tf in BARS.keys()}


def synthetic_data(n_bars=2000, seed=42):
    """Genera datos sintéticos EUR/USD para pruebas sin MT5."""
    np.random.seed(seed)
    pip   = 0.0001
    price = 1.0850
    dates = pd.date_range('2024-01-02', periods=n_bars, freq='4h')
    rows  = []
    for i in range(n_bars):
        vol   = pip * np.random.uniform(8, 20)
        price += np.random.normal(0, pip * 5)
        o = price + np.random.uniform(-vol*.3, vol*.3)
        h = max(o, price) + abs(np.random.normal(0, vol*.4))
        l = min(o, price) - abs(np.random.normal(0, vol*.4))
        rows.append({'open': round(o,5), 'high': round(h,5),
                     'low': round(l,5), 'close': round(price,5),
                     'volume': int(np.random.lognormal(10,.5))})
    return pd.DataFrame(rows, index=dates)
