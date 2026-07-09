"""
analysis.py v5.4
Nuevas funciones:
  - bootstrap_trades()  : intervalos de confianza reales para Sharpe/WR/ruina
  - monte_carlo_trades(): MC con remuestreo de trades reales (más preciso)
  - holdout_test()      : evaluación en datos completamente fuera de muestra
  - Walk-Forward v5.4   : rolling out-of-sample del sistema congelado
"""
import numpy as np
import pandas as pd
from config import CAPITAL, INDICATORS, SESSION_MODE, WALKFORWARD, MONTE_CARLO, BOOTSTRAP


# ════════════════════════════════════════════════════════════════════
# MONTE CARLO — distribución binomial (rápido, para estimaciones)
# ════════════════════════════════════════════════════════════════════

def monte_carlo(win_rate=0.52, avg_win_r=1.5, avg_loss_r=1.0,
                initial=None, n_trades=None, n_sim=None,
                ruin_threshold=None, seed=42):
    """Monte Carlo con distribución binomial. Rápido para exploración."""
    initial        = initial        or CAPITAL['initial']
    n_trades       = n_trades       or MONTE_CARLO['n_trades']
    n_sim          = n_sim          or MONTE_CARLO['n_simulations']
    ruin_threshold = ruin_threshold or MONTE_CARLO['ruin_threshold']

    np.random.seed(seed)
    risk_per_trade = initial * CAPITAL['risk_pct']
    matrix         = np.zeros((n_sim, n_trades + 1))
    matrix[:, 0]   = initial
    ruin_count     = 0

    for s in range(n_sim):
        cap = initial
        for t in range(n_trades):
            if cap <= initial * (1 - ruin_threshold):
                matrix[s, t + 1:] = cap
                ruin_count += 1
                break
            win = np.random.rand() < win_rate
            cap += risk_per_trade * avg_win_r if win else -risk_per_trade * avg_loss_r
            cap  = max(cap, 0)
            matrix[s, t + 1] = cap

    return matrix, ruin_count / n_sim


# ════════════════════════════════════════════════════════════════════
# MONTE CARLO — bootstrap de trades reales (preciso, DeepSeek rec.)
# ════════════════════════════════════════════════════════════════════

def monte_carlo_trades(pnl_series: np.ndarray,
                       initial: float = None,
                       n_trades: int = 200,
                       n_sim: int = 10_000,
                       ruin_threshold: float = None,
                       seed: int = 42) -> tuple:
    """
    Monte Carlo con remuestreo de los P&L reales (bootstrapping).

    En lugar de asumir una distribución binomial fija (win_rate, avg_win_r),
    remuestrea directamente de los trades históricos con reemplazo.
    Esto preserva la distribución real de ganancias y pérdidas, incluyendo
    rachas y outliers, produciendo estimaciones de ruina más precisas.

    Parámetros:
        pnl_series : array de P&L por trade (en USD, ya incluye spread)
        n_trades   : trades a simular en cada camino (típico: 200)
        n_sim      : número de simulaciones
    """
    initial        = initial        or CAPITAL['initial']
    ruin_threshold = ruin_threshold or MONTE_CARLO['ruin_threshold']

    if len(pnl_series) < 5:
        return None, None

    np.random.seed(seed)
    matrix     = np.zeros((n_sim, n_trades + 1))
    matrix[:, 0] = initial
    ruin_count = 0

    for s in range(n_sim):
        cap    = initial
        sample = np.random.choice(pnl_series, size=n_trades, replace=True)
        for t, pnl in enumerate(sample):
            if cap <= initial * (1 - ruin_threshold):
                matrix[s, t + 1:] = cap
                ruin_count += 1
                break
            # Escalar P&L según capital actual vs capital en backtest
            scaled_pnl = pnl * (cap / initial)
            cap = max(cap + scaled_pnl, 0)
            matrix[s, t + 1] = cap

    return matrix, ruin_count / n_sim


# ════════════════════════════════════════════════════════════════════
# BOOTSTRAP DE TRADES (DeepSeek rec. #1 — más importante)
# ════════════════════════════════════════════════════════════════════

def bootstrap_trades(return_series: np.ndarray,
                     periods_per_year: float,
                     n_resample: int = None,
                     initial_capital: float = None,
                     confidence: float = None,
                     seed: int = 42) -> dict:
    """
    Bootstrap de retornos por trade para obtener intervalos de confianza.

    Pregunta clave que responde: ¿El Sharpe 1.05 con 26 trades es real
    o puede deberse a azar? Un IC95% de [0.8, 3.5] es sólido;
    un IC95% de [-0.5, 4.8] indica que necesitamos más trades.

    Retorna:
        sharpe_mean, sharpe_ci_lower, sharpe_ci_upper,
        wr_mean, wr_ci, ruin_prob_bootstrap, is_statistically_valid
    """
    n_resample = n_resample or BOOTSTRAP['n_resample']
    confidence = confidence or BOOTSTRAP['confidence']
    initial_capital = initial_capital or CAPITAL['initial']

    if len(return_series) < BOOTSTRAP['min_trades_valid']:
        return {
            'valid': False,
            'reason': f'Solo {len(return_series)} trades — mínimo {BOOTSTRAP["min_trades_valid"]}',
        }

    np.random.seed(seed)
    sharpes   = []
    sortinos  = []
    win_rates = []
    finals    = []

    for _ in range(n_resample):
        sample = np.random.choice(return_series, size=len(return_series), replace=True)

        # Sharpe del remuestreo ajustado a la frecuencia real de trades.
        if len(sample) > 1 and sample.std(ddof=1) > 0 and periods_per_year > 0:
            sh = sample.mean() / sample.std(ddof=1) * np.sqrt(periods_per_year)
        else:
            sh = 0.0
        sharpes.append(sh)

        downside = np.minimum(sample, 0.0)
        downside_dev = np.sqrt(np.mean(downside ** 2))
        if downside_dev > 0 and periods_per_year > 0:
            so = sample.mean() / downside_dev * np.sqrt(periods_per_year)
        else:
            so = sh
        sortinos.append(so)

        # Win Rate del remuestreo
        wr = (sample > 0).mean()
        win_rates.append(wr)

        # Capital final compuesto con retornos por trade.
        cap = initial_capital
        for ret in sample:
            cap = max(cap * (1 + ret), 0)
        finals.append(cap)

    sharpes   = np.array(sharpes)
    sortinos  = np.array(sortinos)
    win_rates = np.array(win_rates)
    finals    = np.array(finals)

    alpha = 1 - confidence
    sharpe_lo = np.percentile(sharpes, alpha / 2 * 100)
    sharpe_hi = np.percentile(sharpes, (1 - alpha / 2) * 100)
    sortino_lo = np.percentile(sortinos, alpha / 2 * 100)
    sortino_hi = np.percentile(sortinos, (1 - alpha / 2) * 100)
    wr_lo     = np.percentile(win_rates, alpha / 2 * 100)
    wr_hi     = np.percentile(win_rates, (1 - alpha / 2) * 100)

    ruin_prob = (finals < initial_capital * (1 - MONTE_CARLO['ruin_threshold'])).mean()

    # Criterio de validez estadística
    is_valid = (
        sharpe_lo > 0 and          # IC inferior positivo
        sharpes.mean() > 1.0 and   # media > 1.0
        len(return_series) >= 20   # suficientes trades
    )

    is_promising = (
        sharpe_lo > 0 and          # al menos el límite inferior es positivo
        sharpes.mean() > 0.5
    )

    return {
        'valid'          : is_valid,
        'promising'      : is_promising,
        'n_trades'       : len(return_series),
        'n_resample'     : n_resample,
        'confidence'     : confidence,
        'periods_per_year': round(float(periods_per_year), 2),
        # Sharpe
        'sharpe_mean'    : round(float(sharpes.mean()), 3),
        'sharpe_median'  : round(float(np.median(sharpes)), 3),
        'sharpe_ci_low'  : round(float(sharpe_lo), 3),
        'sharpe_ci_high' : round(float(sharpe_hi), 3),
        # Sortino
        'sortino_mean'   : round(float(sortinos.mean()), 3),
        'sortino_ci_low' : round(float(sortino_lo), 3),
        'sortino_ci_high': round(float(sortino_hi), 3),
        # Win Rate
        'wr_mean'        : round(float(win_rates.mean() * 100), 1),
        'wr_ci_low'      : round(float(wr_lo * 100), 1),
        'wr_ci_high'     : round(float(wr_hi * 100), 1),
        # Capital
        'capital_mean'   : round(float(finals.mean()), 2),
        'capital_median' : round(float(np.median(finals)), 2),
        'ruin_prob_pct'  : round(float(ruin_prob * 100), 2),
        # Interpretación
        'verdict'        : (
            "✅ SÓLIDO — Edge estadístico confirmado"
                if is_valid else
            "⚠️  PROMETEDOR — Pocos trades para certeza total"
                if is_promising else
            "❌ NO CONCLUYENTE — Edge no confirmado por bootstrap"
        ),
    }


# ════════════════════════════════════════════════════════════════════
# HOLDOUT OUT-OF-SAMPLE (DeepSeek rec. #2)
# ════════════════════════════════════════════════════════════════════

def holdout_test(df_full: pd.DataFrame,
                 holdout_months: int = None,
                 mode: str | None = None,
                 session_mode: str | None = None,
                 use_m15: bool = False,
                 df_m15: pd.DataFrame | None = None) -> dict:
    """
    Evalúa el sistema en los últimos N meses de datos, completamente
    separados del proceso de optimización.

    Principio: estos datos NUNCA deben haber sido usados para ajustar
    parámetros. Son la prueba final de que el sistema no está sobreajustado.

    Retorna métricas del período holdout y comparación con el período
    de entrenamiento.
    """
    from indicators import generate_signals
    from backtest import BacktestEngine, compute_metrics

    holdout_months = holdout_months or __import__('config').HOLDOUT_MONTHS
    mode = mode or INDICATORS.get('strategy_mode', 'pullback')
    session_mode = session_mode or SESSION_MODE

    # Separar datos
    cutoff = df_full.index[-1] - pd.DateOffset(months=holdout_months)
    df_train   = df_full[df_full.index <= cutoff]
    df_holdout = df_full[df_full.index > cutoff]

    if len(df_holdout) < 50:
        return {
            'valid': False,
            'reason': f'Holdout muy pequeño ({len(df_holdout)} barras)',
        }

    engine = BacktestEngine(trailing_stop=True, partial_tp=True)

    # Evaluar en período de entrenamiento
    df_train_sig = generate_signals(
        df_train,
        df_m15=df_m15,
        use_session=True,
        session_mode=session_mode,
        use_adx=True,
        strict_entries=True,
        use_m15=use_m15,
        mode=mode,
    )
    res_train    = engine.run(df_train_sig)
    m_train      = compute_metrics(res_train)

    # Evaluar en holdout (datos nunca vistos)
    df_hold_sig  = generate_signals(
        df_holdout,
        df_m15=df_m15,
        use_session=True,
        session_mode=session_mode,
        use_adx=True,
        strict_entries=True,
        use_m15=use_m15,
        mode=mode,
    )
    res_hold     = engine.run(df_hold_sig)
    m_hold       = compute_metrics(res_hold)

    # ¿El sistema se sostiene fuera de muestra?
    is_valid = (
        m_hold.get('total_trades', 0) >= 3 and
        m_hold.get('net_pnl_usd', -999) > -5  # no pérdida catastrófica
    )

    degradation = 0.0
    train_sh = m_train.get('sharpe_ratio', 0)
    hold_sh  = m_hold.get('sharpe_ratio', 0)
    # Degradación solo tiene sentido cuando train_sharpe es positivo.
    # Si train < 0 y hold > 0, el sistema MEJORÓ out-of-sample.
    if train_sh > 0:
        degradation = (train_sh - hold_sh) / abs(train_sh) * 100
    elif train_sh < 0 and hold_sh > 0:
        degradation = -100.0  # mejora (negativo = bueno en este contexto)
    else:
        degradation = 0.0

    return {
        'valid'             : is_valid,
        'holdout_months'    : holdout_months,
        'mode'              : mode,
        'session_mode'      : session_mode,
        'use_m15'           : use_m15,
        'train_bars'        : len(df_train),
        'holdout_bars'      : len(df_holdout),
        'train_start'       : str(df_train.index[0].date()),
        'train_end'         : str(df_train.index[-1].date()),
        'holdout_start'     : str(df_holdout.index[0].date()),
        'holdout_end'       : str(df_holdout.index[-1].date()),
        # Métricas entrenamiento
        'train_trades'      : m_train.get('total_trades', 0),
        'train_sharpe'      : m_train.get('sharpe_ratio', 0),
        'train_wr'          : m_train.get('win_rate', 0),
        'train_pnl'         : m_train.get('net_pnl_usd', 0),
        # Métricas holdout (lo que importa)
        'holdout_trades'    : m_hold.get('total_trades', 0),
        'holdout_sharpe'    : m_hold.get('sharpe_ratio', 0),
        'holdout_wr'        : m_hold.get('win_rate', 0),
        'holdout_pnl'       : m_hold.get('net_pnl_usd', 0),
        'holdout_max_dd'    : m_hold.get('max_drawdown_pct', 0),
        # Degradación
        'sharpe_degradation_pct': round(degradation, 1),
        'verdict': (
            "✅ EXCELENTE — holdout mejor que entrenamiento"
                if (is_valid and hold_sh > train_sh) else
            "✅ PASA holdout — sistema generaliza bien"
                if (is_valid and m_hold.get('net_pnl_usd', -999) > 0) else
            "⚠️  HOLDOUT NEUTRO — sin pérdidas significativas"
                if is_valid else
            "❌ FALLA holdout — sobreajuste probable"
        ),
    }


# ════════════════════════════════════════════════════════════════════
# WALK-FORWARD v5.4 (sistema congelado TRAIN/TEST)
# ════════════════════════════════════════════════════════════════════

def _build_signals_frozen(df):
    """
    Construye señales usando exactamente la configuración congelada
    del sistema base.
    """
    from indicators import generate_signals
    return generate_signals(
        df,
        use_session=True,
        session_mode=SESSION_MODE,
        use_adx=True,
        strict_entries=True,
        use_m15=False,
        mode=INDICATORS.get('strategy_mode', 'pullback'),
    )


def _bounded_window_sharpe(value: float, n_trades: int) -> float:
    """Evita Sharpe explosivos en ventanas con poca varianza/trades."""
    if n_trades < 3 or not np.isfinite(value) or abs(value) > 10:
        return 0.0
    return float(value)


def walk_forward(df, instrument: str | None = None,
                 instrument_spec: dict | None = None,
                 verbose: bool = True):
    """Walk-Forward out-of-sample del sistema congelado."""
    from backtest import BacktestEngine, compute_metrics as _cm

    n_windows  = WALKFORWARD['n_windows']
    train_pct  = WALKFORWARD['train_pct']
    mode       = INDICATORS.get('strategy_mode', 'pullback')
    session    = SESSION_MODE
    n_bars      = len(df)
    window_size = n_bars // n_windows
    train_size  = int(window_size * train_pct)
    engine      = BacktestEngine(instrument=instrument, instrument_spec=instrument_spec)
    results     = []

    if verbose:
        print(f"\n  Barras totales : {n_bars}")
        print(f"  Ventanas       : {n_windows} | Train: {train_size} | Test: {window_size - train_size}")
        print(f"  Sistema fijo   : {mode} | Sesión: {session} | M15: OFF")

    for w in range(n_windows):
        s = w * window_size
        e = s + window_size
        if e > n_bars:
            break

        df_train = df.iloc[s            : s + train_size]
        df_test  = df.iloc[s + train_size : e]

        if verbose:
            print(f"\n  Ventana {w+1}/{n_windows} — ", end="", flush=True)

        df_train_sig = _build_signals_frozen(df_train)
        res_train    = engine.run(df_train_sig)
        m_train      = _cm(res_train)

        df_test_sig = _build_signals_frozen(df_test)
        res_test    = engine.run(df_test_sig)
        m_test      = _cm(res_test)
        train_trades = int(m_train['total_trades'])
        test_trades  = int(m_test['total_trades'])
        train_sharpe = _bounded_window_sharpe(float(m_train['sharpe_ratio']), train_trades)
        test_sharpe  = _bounded_window_sharpe(float(m_test['sharpe_ratio']), test_trades)
        test_positive = bool(m_test['net_pnl_usd'] > 0 and test_trades >= 3)

        if verbose:
            print(f"TRAIN Sharpe: {train_sharpe:.2f} | "
                  f"TEST Sharpe: {test_sharpe:.2f} | "
                  f"TEST P&L: ${m_test['net_pnl_usd']:.2f} | "
                  f"Trades: {test_trades}")

        results.append({
            'window'        : w + 1,
            'instrument'    : engine.instrument,
            'mode'          : mode,
            'session_mode'  : session,
            'use_m15'       : False,
            'train_start'   : df_train.index[0],
            'train_end'     : df_train.index[-1],
            'test_start'    : df_test.index[0],
            'test_end'      : df_test.index[-1],
            'train_sharpe'  : round(train_sharpe, 3),
            'test_sharpe'   : round(test_sharpe, 3),
            'train_trades'  : train_trades,
            'test_trades'   : test_trades,
            'train_pnl'     : m_train['net_pnl_usd'],
            'test_pnl'      : m_test['net_pnl_usd'],
            'test_max_dd'   : m_test['max_drawdown_pct'],
            'test_win_rate' : m_test['win_rate'],
            'test_positive' : test_positive,
            'test_capital'  : m_test['final_capital'],
            'equity'        : res_test['equity'],
        })

    return (
        pd.DataFrame([{k: v for k, v in r.items() if k != 'equity'} for r in results]),
        results
    )


def robustness(wf_df):
    """Evalúa robustez del sistema congelado en ventanas out-of-sample."""
    if wf_df.empty:
        return {
            'mode'             : INDICATORS.get('strategy_mode', 'pullback'),
            'session_mode'     : SESSION_MODE,
            'use_m15'          : False,
            'total_windows'    : 0,
            'positive_windows' : 0,
            'avg_train_sharpe' : 0.0,
            'avg_test_sharpe'  : 0.0,
            'avg_test_pnl'     : 0.0,
            'pct_positive_test': 0.0,
            'avg_test_max_dd'  : 0.0,
            'verdict'          : "❌ NO ROBUSTO — Sin ventanas suficientes para evaluar",
        }

    avg_tr   = wf_df['train_sharpe'].mean()
    avg_te   = wf_df['test_sharpe'].mean()
    avg_pnl  = wf_df['test_pnl'].mean()
    avg_dd   = wf_df['test_max_dd'].mean()
    pct_pos  = wf_df['test_positive'].mean() * 100
    pos_wins = int(wf_df['test_positive'].sum())
    total_w  = int(len(wf_df))
    dd_limit = CAPITAL['max_drawdown'] * 100

    if pct_pos >= 60 and avg_te > 0 and avg_dd <= dd_limit:
        verdict = "✅ ROBUSTO — Sistema fijo generaliza bien out-of-sample"
    elif pct_pos >= 40 and avg_pnl >= 0:
        verdict = "⚠️  MODERADO — Señal positiva pero irregular entre ventanas"
    else:
        verdict = "❌ NO ROBUSTO — Walk-forward inconsistente out-of-sample"

    return {
        'mode'             : str(wf_df['mode'].iloc[0]),
        'session_mode'     : str(wf_df['session_mode'].iloc[0]),
        'use_m15'          : bool(wf_df['use_m15'].iloc[0]),
        'total_windows'    : total_w,
        'positive_windows' : pos_wins,
        'avg_train_sharpe' : round(avg_tr, 3),
        'avg_test_sharpe'  : round(avg_te, 3),
        'avg_test_pnl'     : round(avg_pnl, 3),
        'pct_positive_test': round(pct_pos, 1),
        'avg_test_max_dd'  : round(avg_dd, 2),
        'verdict'          : verdict,
    }
