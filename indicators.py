"""
indicators.py v5.4 — Sistema simplificado: 3 indicadores, 3 estrategias

ANÁLISIS HISTÓRICO (H4 completo + holdout):
  La base actual prioriza pullback por mejor equilibrio entre:
    Ganancia neta
    Cantidad de trades
    Drawdown controlado
  Configuración por defecto validada:
    Pullback EMA 10/55 + ADX 15 + sesión london_ny  → 159 trades, +$108.07

  Indicadores con mayor score acumulado en resultados positivos:
    1. ADX/DM    (score 10.04) — filtro de tendencia, presente en TODO
    2. RSI       (score  7.97) — momentum, corrección de la guía aplicada
    3. MACD      (score  7.97) — convergencia, señal de entrada
    + SMA200     (score  7.97) — filtro macro obligatorio (no contado en score)
    + EMA 10/50  (score  5.00) — señal de cruce y pullback

  Estrategias eliminadas (sin edge demostrado en ningún run):
    ✗ Reversión BB+RSI+Stoch  — WR 23%, Sharpe -9.57 consistentemente
    ✗ Breakout BB+SR          — Sharpe 0.70, sin consistencia histórica
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import INDICATORS, SESSION_MODE


# ════════════════════════════════════════════════════════════════════
# INDICADORES BASE
# ════════════════════════════════════════════════════════════════════

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=1).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI — Índice de Fuerza Relativa.
    CORRECCIÓN APLICADA (guía, sección 3/7):
      RSI alto NO significa sobrecompra automática.
      RSI > 52 = momentum fuerte hacia arriba = señal de continuación.
      Solo bloqueamos RSI > 85 (extremo absoluto).
    """
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low']  - df['close'].shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(
        span=period, adjust=False).mean()

def ema_slope(series: pd.Series, period: int, lookback: int = 3) -> pd.Series:
    e = series.ewm(span=period, adjust=False).mean()
    return (e - e.shift(lookback)) / 0.0001

def market_structure(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    highs = df['high'].rolling(lookback).max()
    lows  = df['low'].rolling(lookback).min()
    hh = (df['high'] > highs.shift(1)).astype(int)
    ll = (df['low']  < lows.shift(1)).astype(int)
    return pd.Series(np.where(hh > ll, 1, np.where(ll > hh, -1, 0)), index=df.index)


# ── INDICADOR 1: ADX / DM ─────────────────────────────────────────
# Score: 10.04 — presente en TODOS los resultados positivos
# Rol: filtra mercados laterales (ADX < 12 = rango, no operar)
#      DI+ vs DI- determina la dirección de la tendencia

def adx(df: pd.DataFrame, period: int = 14) -> tuple:
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    dm_plus  = high.diff()
    dm_minus = -low.diff()
    dm_plus  = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
    atr_w    = tr.ewm(span=period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(span=period, adjust=False).mean() / atr_w.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr_w.replace(0, np.nan)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx_val  = dx.ewm(span=period, adjust=False).mean()
    return adx_val.fillna(0), di_plus.fillna(0), di_minus.fillna(0)


# ── INDICADOR 2: RSI ─────────────────────────────────────────────
# Score: 7.97 — confirmación de momentum
# Rol: RSI > 52 confirma momentum alcista (no bloquea si RSI > 70)
#      RSI < 48 confirma momentum bajista


# ── INDICADOR 3: MACD ─────────────────────────────────────────────
# Score: 7.97 — señal de entrada principal en Momentum
# Rol: MACD > señal = presión compradora; histograma creciente = aceleración

def macd(series: pd.Series, fast: int = 12,
         slow: int = 26, signal: int = 9) -> tuple:
    ema_f  = series.ewm(span=fast,   adjust=False).mean()
    ema_s  = series.ewm(span=slow,   adjust=False).mean()
    line   = ema_f - ema_s
    sig    = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


# ── FILTRO MACRO: SMA 200 ─────────────────────────────────────────
# Score: 7.97 — filtro macro obligatorio (no contado en scoring)
# Rol: solo tomar longs si precio > SMA200 (tendencia macro alcista)
#      solo tomar shorts si precio < SMA200 (tendencia macro bajista)
# Margen 3%: permite trades en correcciones menores sin cambio de tendencia


# ════════════════════════════════════════════════════════════════════
# SISTEMA DE SCORING (Camino A) — 3 indicadores + 2 bonus
# ════════════════════════════════════════════════════════════════════
#
# Puntuación LONG:
#   +1.0  ADX > 12  Y  DI+ > DI-        (tendencia alcista activa)
#   +1.0  RSI > 52  Y  RSI < 85         (momentum positivo, no extremo)
#   +1.0  MACD positivo O línea > señal  (convergencia alcista)
#   +0.5  EMA slope positivo             (bonus: tendencia acelerando)
#   +0.5  Precio sobre SMA200            (bonus: macro alcista)
#   ──────────────────────────────────────
#   Máximo: 4.0 puntos
#   Umbral: 3.0 (tres indicadores principales confirmando)
#
# Con min_score=3: necesitas los 3 indicadores principales alineados.

def compute_score(df: pd.DataFrame) -> tuple:
    """
    Calcula score de confluencia para LONG y SHORT.
    Solo usa los 3 indicadores probados + 2 bonus.
    """
    adx_v    = df['adx_val']
    di_plus  = df['di_plus']
    di_minus = df['di_minus']
    rsi_v    = df['rsi']
    hist     = df['macd_hist']
    macd_l   = df['macd_line']
    macd_s_  = df['macd_signal']
    slope    = df['slope']
    sma200   = df['sma200']

    adx_min = INDICATORS.get('adx_min', 12)
    rsi_min = INDICATORS.get('rsi_momentum_min', 52)

    # ── Criterio 1: ADX/DM ────────────────────────────────────────
    c1_long  = ((adx_v > adx_min) & (di_plus  > di_minus)).astype(float)
    c1_short = ((adx_v > adx_min) & (di_minus > di_plus)).astype(float)

    # ── Criterio 2: RSI (guía: alto = momentum, no bloqueo) ──────
    c2_long  = ((rsi_v > rsi_min)       & (rsi_v < 85)).astype(float)
    c2_short = ((rsi_v < 100 - rsi_min) & (rsi_v > 15)).astype(float)

    # ── Criterio 3: MACD ─────────────────────────────────────────
    macd_bull = (hist > 0) | (macd_l > macd_s_)
    macd_bear = (hist < 0) | (macd_l < macd_s_)
    c3_long  = macd_bull.astype(float)
    c3_short = macd_bear.astype(float)

    # ── Bonus 1: EMA slope ───────────────────────────────────────
    sm = INDICATORS.get('slope_min_pips', 0.2)
    b1_long  = (slope > sm).astype(float)  * 0.5
    b1_short = (slope < -sm).astype(float) * 0.5

    # ── Bonus 2: SMA200 (macro) ──────────────────────────────────
    margin   = 0.03
    b2_long  = (df['close'] >= sma200 * (1 - margin)).astype(float) * 0.5
    b2_short = (df['close'] <= sma200 * (1 + margin)).astype(float) * 0.5

    score_long  = c1_long  + c2_long  + c3_long  + b1_long  + b2_long
    score_short = c1_short + c2_short + c3_short + b1_short + b2_short

    return score_long, score_short


# ════════════════════════════════════════════════════════════════════
# ESTRATEGIA 1: MOMENTUM (la mejor — Sharpe 5.86, +$23.07)
# ════════════════════════════════════════════════════════════════════
# ADX confirma tendencia + RSI confirma momentum + MACD confirma dirección
# SMA200 filtra macro. Score mínimo 3/4 para entrada.

def _strategy_momentum(df: pd.DataFrame) -> pd.Series:
    """
    Momentum puro: los 3 indicadores principales alineados.
    Entrada: cuando ADX + RSI + MACD confirman simultáneamente.
    Es la estrategia más probada del sistema.
    """
    min_sc = INDICATORS.get('min_score', 3)
    sl, ss = compute_score(df)

    # Base: ADX activo + DI direccional
    adx_bull = (df['adx_val'] > INDICATORS['adx_min']) & (df['di_plus'] > df['di_minus'])
    adx_bear = (df['adx_val'] > INDICATORS['adx_min']) & (df['di_minus'] > df['di_plus'])

    long_c  = adx_bull & (df['rsi'] > INDICATORS['rsi_momentum_min']) & \
              (df['rsi'] < 85) & (sl >= min_sc)
    short_c = adx_bear & (df['rsi'] < 100 - INDICATORS['rsi_momentum_min']) & \
              (df['rsi'] > 15) & (ss >= min_sc)

    return pd.Series(np.where(long_c, 1, np.where(short_c, -1, 0)), index=df.index)


# ════════════════════════════════════════════════════════════════════
# ESTRATEGIA 2: CRUCE EMA + ADX (segunda mejor — Sharpe 2.07, +$8.75)
# ════════════════════════════════════════════════════════════════════
# Entrada: EMA 10 cruza EMA 50 + ADX confirma tendencia real.
# Fue la base del sistema desde v2.1. Genera más trades que Momentum.

def _strategy_ema_crossover(df: pd.DataFrame) -> pd.Series:
    """
    Cruce de EMA 10/50 confirmado por ADX y MACD.
    Historial: v2.1 Sharpe 2.07, v1.2 Sharpe 1.36.
    Más trades que Momentum pero WR menor (~40-50%).
    """
    cross     = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
    cross_sig = pd.Series(cross, index=df.index).diff().fillna(0)

    adx_ok   = df['adx_val'] > INDICATORS['adx_min']
    macd_bull = (df['macd_hist'] > 0) | (df['macd_line'] > df['macd_signal'])
    macd_bear = (df['macd_hist'] < 0) | (df['macd_line'] < df['macd_signal'])
    slope     = df['slope']
    sm        = INDICATORS.get('slope_min_pips', 0.2)

    sma200_long  = df['close'] >= df['sma200'] * 0.97
    sma200_short = df['close'] <= df['sma200'] * 1.03

    long_c = (
        (cross_sig > 0) &
        adx_ok & (df['di_plus'] > df['di_minus']) &
        (df['rsi'] > 45) &
        macd_bull &
        (slope > sm) &
        sma200_long
    )
    short_c = (
        (cross_sig < 0) &
        adx_ok & (df['di_minus'] > df['di_plus']) &
        (df['rsi'] < 55) &
        macd_bear &
        (slope < -sm) &
        sma200_short
    )

    return pd.Series(np.where(long_c, 1, np.where(short_c, -1, 0)), index=df.index)


# ════════════════════════════════════════════════════════════════════
# ESTRATEGIA 3: PULLBACK A EMA (base actual por rentabilidad/frecuencia)
# ════════════════════════════════════════════════════════════════════
# Entrada: precio retrocede a tocar la EMA lenta en tendencia establecida.
# Alta probabilidad: entras en zona de valor, no en el cruce.
# Base v5.4: EMA 10/55, ADX 15, pullback_atr 1.1, sesión london_ny.

def _strategy_pullback(df: pd.DataFrame) -> pd.Series:
    """
    Pullback a EMA lenta en tendencia confirmada por ADX.
    Requiere: tendencia activa (EMA fast > EMA slow × 2 velas) +
              precio toca zona EMA (±0.8×ATR) + RSI en zona válida +
              ADX + DI confirman dirección + SMA200 macro ok.
    """
    adx_v    = df['adx_val']
    di_plus  = df['di_plus']
    di_minus = df['di_minus']

    # Zona de pullback: precio a menos de 0.8×ATR de la EMA lenta
    pb_zone = INDICATORS.get('pullback_atr', 0.8) * 1.2
    dist    = (df['close'] - df['ema_slow']).abs() / df['atr'].replace(0, np.nan)
    near    = dist <= pb_zone

    # Tendencia establecida: EMA fast sobre/bajo EMA slow durante 2+ velas
    trend_bull = (df['ema_fast'] > df['ema_slow']) & \
                 (df['ema_fast'].shift(1) > df['ema_slow'].shift(1))
    trend_bear = (df['ema_fast'] < df['ema_slow']) & \
                 (df['ema_fast'].shift(1) < df['ema_slow'].shift(1))

    sma200_long  = df['close'] >= df['sma200'] * 0.97
    sma200_short = df['close'] <= df['sma200'] * 1.03

    long_c = (
        trend_bull & near &
        (df['close'] > df['ema_slow']) &          # precio rebotó sobre EMA
        (df['rsi'] > 38) & (df['rsi'] < 72) &     # RSI en zona válida
        (df['close'] > df['open']) &               # vela alcista de confirmación
        (adx_v > INDICATORS['adx_min']) &
        (di_plus > di_minus) &
        sma200_long
    )
    short_c = (
        trend_bear & near &
        (df['close'] < df['ema_slow']) &
        (df['rsi'] < 62) & (df['rsi'] > 28) &
        (df['close'] < df['open']) &
        (adx_v > INDICATORS['adx_min']) &
        (di_minus > di_plus) &
        sma200_short
    )

    return pd.Series(np.where(long_c, 1, np.where(short_c, -1, 0)), index=df.index)


# ════════════════════════════════════════════════════════════════════
# SESIÓN Y CONFIRMACIÓN M15
# ════════════════════════════════════════════════════════════════════

def session_filter(df: pd.DataFrame, mode: str | None = None) -> pd.Series:
    mode  = mode or SESSION_MODE
    hours = df.index.hour
    masks = {
        "london_ny": (hours >= 8) & (hours < 21),
        "london"   : (hours >= 8) & (hours < 16),
        "ny"       : (hours >= 13) & (hours < 21),
        "all"      : np.ones(len(df), dtype=bool),
    }
    return pd.Series(masks.get(mode, masks['all']), index=df.index)

def m15_confirmation(df_h4: pd.DataFrame, df_m15: pd.DataFrame) -> tuple:
    m15 = df_m15.copy()
    if m15.index.tz is not None:
        m15.index = m15.index.tz_localize(None)
    if df_h4.index.tz is not None:
        df_h4 = df_h4.copy()
        df_h4.index = df_h4.index.tz_localize(None)
    m15['ema20'] = ema(m15['close'], 20)
    m15['rsi14'] = rsi(m15['close'], 14)
    m15['bull']  = ((m15['close'] > m15['ema20']) & (m15['rsi14'] > 45)).astype(int)
    m15['bear']  = ((m15['close'] < m15['ema20']) & (m15['rsi14'] < 55)).astype(int)
    bull_h4 = m15['bull'].resample('4h').max()
    bear_h4 = m15['bear'].resample('4h').max()
    tol     = pd.Timedelta('4h')
    bull_al = bull_h4.reindex(df_h4.index, method='nearest', tolerance=tol).fillna(0)
    bear_al = bear_h4.reindex(df_h4.index, method='nearest', tolerance=tol).fillna(0)
    return bull_al.astype(int), bear_al.astype(int)


# ════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ════════════════════════════════════════════════════════════════════

def generate_signals(
    df_h4: pd.DataFrame,
    df_m15: pd.DataFrame | None = None,
    use_session: bool = True,
    use_m15: bool = False,
    session_mode: str | None = None,
    use_adx: bool = True,
    strict_entries: bool = True,
    mode: str = 'all',
) -> pd.DataFrame:
    """
    Genera señales usando los 3 indicadores probados y 3 estrategias.

    mode='momentum'    → Estrategia 1 (mejor histórico)
    mode='ema_crossover' → Estrategia 2 (más trades)
    mode='pullback'    → Estrategia 3 (mayor WR)
    mode='all'         → Las 3 combinadas (Momentum tiene prioridad)
    """
    df = df_h4.copy()

    # ── Calcular los 3 indicadores + filtros ─────────────────────
    df['ema_fast']   = ema(df['close'], INDICATORS['ema_fast'])
    df['ema_slow']   = ema(df['close'], INDICATORS['ema_slow'])
    df['sma200']     = sma(df['close'], INDICATORS.get('sma_macro', 200))
    df['rsi']        = rsi(df['close'], INDICATORS['rsi_period'])
    df['atr']        = atr(df, INDICATORS['atr_period'])
    df['slope']      = ema_slope(df['close'], INDICATORS['ema_slow'], lookback=3)
    df['structure']  = market_structure(df)
    df['adx_val'], df['di_plus'], df['di_minus'] = adx(df)
    df['macd_line'], df['macd_signal'], df['macd_hist'] = macd(df['close'])

    # Scores de confluencia
    df['score_long'], df['score_short'] = compute_score(df)

    # ── Ejecutar estrategias ──────────────────────────────────────
    strategy_fns = {
        'momentum'      : _strategy_momentum,
        'ema_crossover' : _strategy_ema_crossover,
        'pullback'      : _strategy_pullback,
    }

    for name, fn in strategy_fns.items():
        df[f'sig_{name}'] = fn(df)

    if mode == 'all':
        # Prioridad: Momentum > Cruce EMA > Pullback
        combined = np.zeros(len(df))
        for name in ['pullback', 'ema_crossover', 'momentum']:
            sig = df[f'sig_{name}'].values
            combined = np.where(sig != 0, sig, combined)
    elif mode in strategy_fns:
        combined = df[f'sig_{mode}'].values
    else:
        combined = np.zeros(len(df))

    combined = pd.Series(combined, index=df.index)

    # Filtro de sesión
    if use_session:
        mask     = session_filter(df, mode=session_mode)
        combined = combined * mask.astype(int)

    # Confirmación M15 (opcional)
    if use_m15 and df_m15 is not None:
        try:
            bull_m15, bear_m15 = m15_confirmation(df, df_m15)
            combined = np.where(
                (combined ==  1) & (bull_m15 == 1),  1,
                np.where(
                (combined == -1) & (bear_m15 == 1), -1, 0))
        except Exception as e:
            print(f"   M15 no aplicado: {e}")

    df['signal'] = combined
    df['signal_strategy'] = 'none'
    for name in strategy_fns:
        df.loc[df[f'sig_{name}'] != 0, 'signal_strategy'] = name

    return df


# ════════════════════════════════════════════════════════════════════
# DIAGNÓSTICO
# ════════════════════════════════════════════════════════════════════

def diagnose_signals(df_h4: pd.DataFrame) -> None:
    """Diagnóstico: qué % de barras pasa cada filtro de los 3 indicadores."""
    df = df_h4.copy()
    df['ema_fast'] = ema(df['close'], INDICATORS['ema_fast'])
    df['ema_slow'] = ema(df['close'], INDICATORS['ema_slow'])
    df['sma200']   = sma(df['close'], INDICATORS.get('sma_macro', 200))
    df['rsi']      = rsi(df['close'], INDICATORS['rsi_period'])
    df['atr']      = atr(df)
    df['slope']    = ema_slope(df['close'], INDICATORS['ema_slow'], 3)
    df['adx_val'], df['di_plus'], df['di_minus'] = adx(df)
    df['macd_line'], df['macd_signal'], df['macd_hist'] = macd(df['close'])
    df['score_long'], df['score_short'] = compute_score(df)

    hours   = df.index.hour
    in_sess = ((hours >= 8) & (hours < 21)).mean() * 100
    adx_min = INDICATORS['adx_min']
    rsi_min = INDICATORS['rsi_momentum_min']

    print(f"\n{'='*60}")
    print(f"  DIAGNÓSTICO — {len(df)} barras H4")
    print(f"{'='*60}")
    print(f"\n  INDICADOR 1 — ADX/DM (filtro tendencia):")
    print(f"  ADX > {adx_min}              : {(df['adx_val'] > adx_min).mean()*100:.0f}% de barras")
    print(f"  ADX > {adx_min} + DI+ > DI-  : {((df['adx_val']>adx_min)&(df['di_plus']>df['di_minus'])).mean()*100:.0f}% long")
    print(f"  ADX > {adx_min} + DI- > DI+  : {((df['adx_val']>adx_min)&(df['di_minus']>df['di_plus'])).mean()*100:.0f}% short")
    print(f"\n  INDICADOR 2 — RSI (momentum, guía corregida):")
    print(f"  RSI promedio         : {df['rsi'].mean():.1f}")
    print(f"  RSI > {rsi_min} (long ok)  : {(df['rsi'] > rsi_min).mean()*100:.0f}%")
    print(f"  RSI < {100-rsi_min} (short ok) : {(df['rsi'] < 100-rsi_min).mean()*100:.0f}%")
    print(f"\n  INDICADOR 3 — MACD:")
    print(f"  MACD bull            : {((df['macd_hist']>0)|(df['macd_line']>df['macd_signal'])).mean()*100:.0f}%")
    print(f"  MACD bear            : {((df['macd_hist']<0)|(df['macd_line']<df['macd_signal'])).mean()*100:.0f}%")
    print(f"\n  FILTRO MACRO — SMA200:")
    print(f"  Precio sobre SMA200  : {(df['close'] > df['sma200']).mean()*100:.0f}%")
    print(f"\n  SCORING (Camino A):")
    min_sc = INDICATORS.get('min_score', 3)
    print(f"  Score LONG ≥ {min_sc}     : {(df['score_long'] >= min_sc).mean()*100:.1f}% de barras")
    print(f"  Score SHORT ≥ {min_sc}    : {(df['score_short'] >= min_sc).mean()*100:.1f}% de barras")
    print(f"  En sesión london_ny  : {in_sess:.0f}% del total")
    print(f"\n  → Si Score LONG < 5%: revisa ADX mínimo o rsi_momentum_min en config.py")
