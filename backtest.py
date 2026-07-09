"""
backtest.py v5.3
Mejoras sobre v5.1:
  ✅ Sortino Ratio   — penaliza solo downside (DeepSeek rec.)
  ✅ Spread/costos   — descuenta spread EUR/USD en cada entrada
  ✅ Trailing 1.5    — fix v5.2: 0.8 cerraba demasiado pronto
  ✅ TP1=2.0, TP2=4.0 — parámetros ajustados
"""
import numpy as np
import pandas as pd
from config import CAPITAL, INSTRUMENTS, SYMBOL


def get_instrument_spec(instrument: str | None = None,
                        instrument_spec: dict | None = None) -> dict:
    """Resuelve la economia del instrumento sin mutar la configuracion global."""
    if instrument_spec is not None:
        return dict(instrument_spec)
    name = instrument or SYMBOL
    return dict(INSTRUMENTS.get(name, INSTRUMENTS["EURUSD"]))


def round_lot_size(lots: float, spec: dict) -> float:
    """Ajusta el tamano de posicion a min/max/step del instrumento."""
    step = float(spec.get("lot_step", 0.01))
    min_lot = float(spec.get("min_lot", 0.01))
    max_lot = float(spec.get("max_lot", 200.0))
    if lots <= 0 or step <= 0:
        return 0.0
    rounded = np.floor(lots / step) * step
    if rounded < min_lot or rounded > max_lot:
        return 0.0
    return round(float(rounded), 4)


def trade_pnls_by_entry(trades: pd.DataFrame) -> pd.Series:
    """Agrupa P&L por trade completo usando la fecha de entrada."""
    if trades.empty:
        return pd.Series(dtype=float)
    return trades.groupby('entry_dt')['pnl_usd'].sum().sort_index()


def trade_returns(trades: pd.DataFrame, initial_capital: float) -> pd.Series:
    """
    Convierte P&L por trade a retornos sobre el capital disponible
    antes de cada operación.
    """
    pnls = trade_pnls_by_entry(trades)
    if pnls.empty:
        return pd.Series(dtype=float)

    capital_before = []
    capital = float(initial_capital)
    for pnl in pnls:
        capital_before.append(max(capital, 1e-9))
        capital += float(pnl)

    capital_before = pd.Series(capital_before, index=pnls.index, dtype=float)
    returns = pnls.astype(float) / capital_before
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def annualized_trades_per_year(n_trades: int, timeline_index: pd.Index) -> float:
    """Frecuencia anual observada de trades en el período evaluado."""
    if n_trades <= 0 or len(timeline_index) < 2:
        return 0.0

    start = pd.Timestamp(timeline_index[0])
    end = pd.Timestamp(timeline_index[-1])
    span_years = (end - start).total_seconds() / (365.25 * 24 * 3600)
    if span_years <= 0:
        return 0.0
    return float(n_trades / span_years)


class BacktestEngine:
    """
    Motor barra a barra con gestión avanzada de salidas.

    AVANZADO (trailing_stop=True):
      - TP1 parcial: cierra 50% en TP1, mueve SL a breakeven
      - Trailing Stop: sigue al precio post-TP1
      - Spread descontado en cada entrada
    """

    def __init__(self, initial_capital=None, risk_pct=None, rr_ratio=None,
                 trailing_stop=True, partial_tp=True,
                 instrument: str | None = None,
                 instrument_spec: dict | None = None):
        self.initial_capital = initial_capital or CAPITAL['initial']
        self.risk_pct        = risk_pct        or CAPITAL['risk_pct']
        self.rr_ratio        = rr_ratio        or CAPITAL['rr_ratio']
        self.trailing_stop   = trailing_stop
        self.partial_tp      = partial_tp
        self.atr_sl          = CAPITAL.get('atr_sl_mult',    1.5)
        self.atr_tp1         = CAPITAL.get('atr_tp1_mult',   2.0)
        self.atr_tp2         = CAPITAL.get('atr_tp2_mult',   4.0)
        self.trail_mult      = CAPITAL.get('trail_atr_mult', 1.5)
        self.instrument      = instrument or SYMBOL
        self.instrument_spec = get_instrument_spec(instrument, instrument_spec)
        self.pip_size        = float(self.instrument_spec.get('pip_size', 0.0001))
        self.pip_value       = float(self.instrument_spec.get('pip_value_per_lot', 0.10))
        self.spread_value    = float(self.instrument_spec.get(
            'spread_value_per_lot',
            self.pip_value,
        ))
        self.spread_pips     = float(self.instrument_spec.get(
            'spread_pips',
            CAPITAL.get('spread_pips', 1.5),
        ))
        self.slippage_pips   = float(self.instrument_spec.get(
            'slippage_pips',
            CAPITAL.get('slippage_pips', 0.0),
        ))
        self.commission_rt   = float(self.instrument_spec.get(
            'commission_per_lot_round_turn',
            CAPITAL.get('commission_per_lot_round_turn', 0.0),
        ))

    def _apply_exit_slippage(self, price: float, direction: int) -> float:
        """Aplica slippage adverso al cierre."""
        return price - direction * self.slippage_pips * self.pip_size

    def run(self, df):
        pip         = self.pip_size
        spread_pips = self.spread_pips
        pip_value   = self.pip_value
        spread_value = self.spread_value
        commission_rt = self.commission_rt
        capital     = self.initial_capital
        trades      = []
        equity      = [capital]

        in_trade = False
        trade_dir = entry_price = sl = tp1 = tp2 = 0.0
        lot_size = lot_remaining = trail_sl = entry_atr = 0.0
        tp1_hit  = False
        entry_dt = None

        for i in range(1, len(df)):
            row  = df.iloc[i]
            prev = df.iloc[i - 1]

            if in_trade:
                hit_sl = hit_tp1 = hit_tp2 = False

                if trade_dir == 1:
                    if self.trailing_stop and tp1_hit:
                        new_trail = row['close'] - entry_atr * self.trail_mult
                        if new_trail > trail_sl:
                            trail_sl = new_trail
                        if row['low'] <= trail_sl:
                            hit_sl = True; close_price = self._apply_exit_slippage(trail_sl, trade_dir)
                    elif row['low'] <= sl:
                        hit_sl = True; close_price = self._apply_exit_slippage(sl, trade_dir)
                    if not hit_sl:
                        if self.partial_tp and not tp1_hit and row['high'] >= tp1:
                            hit_tp1 = True; close_price = self._apply_exit_slippage(tp1, trade_dir)
                        elif row['high'] >= tp2:
                            hit_tp2 = True; close_price = self._apply_exit_slippage(tp2, trade_dir)
                else:
                    if self.trailing_stop and tp1_hit:
                        new_trail = row['close'] + entry_atr * self.trail_mult
                        if new_trail < trail_sl:
                            trail_sl = new_trail
                        if row['high'] >= trail_sl:
                            hit_sl = True; close_price = self._apply_exit_slippage(trail_sl, trade_dir)
                    elif row['high'] >= sl:
                        hit_sl = True; close_price = self._apply_exit_slippage(sl, trade_dir)
                    if not hit_sl:
                        if self.partial_tp and not tp1_hit and row['low'] <= tp1:
                            hit_tp1 = True; close_price = self._apply_exit_slippage(tp1, trade_dir)
                        elif row['low'] <= tp2:
                            hit_tp2 = True; close_price = self._apply_exit_slippage(tp2, trade_dir)

                if hit_tp1 and not tp1_hit:
                    half_lots = lot_size * 0.50
                    pips      = (close_price - entry_price) * trade_dir / pip
                    pnl       = pips * half_lots * pip_value
                    commission = half_lots * commission_rt * 0.50
                    pnl      -= commission
                    capital  += pnl
                    equity.append(capital)
                    trades.append({
                        'entry_dt': entry_dt, 'exit_dt': row.name,
                        'direction': 'LONG' if trade_dir == 1 else 'SHORT',
                        'entry_price': entry_price, 'exit_price': close_price,
                        'sl': sl, 'tp': tp1,
                        'pnl_usd': round(pnl, 4), 'result': 'WIN_PARTIAL',
                        'capital': round(capital, 4), 'pips': round(pips, 1),
                        'spread_pips': spread_pips,
                        'slippage_pips': self.slippage_pips,
                        'commission_usd': round(commission, 4),
                        'type': 'TP1',
                    })
                    tp1_hit       = True
                    lot_remaining = lot_size * 0.50
                    buf = entry_atr * 0.2
                    sl  = entry_price + buf if trade_dir == 1 else entry_price - buf
                    trail_sl = sl
                    continue

                if hit_sl or hit_tp2:
                    active_lots = lot_remaining if tp1_hit else lot_size
                    pips        = (close_price - entry_price) * trade_dir / pip
                    pnl         = pips * active_lots * pip_value
                    commission  = active_lots * commission_rt * 0.50
                    pnl        -= commission
                    capital    += pnl
                    equity.append(capital)
                    result = 'WIN' if hit_tp2 else ('BREAKEVEN' if tp1_hit else 'LOSS')
                    trades.append({
                        'entry_dt': entry_dt, 'exit_dt': row.name,
                        'direction': 'LONG' if trade_dir == 1 else 'SHORT',
                        'entry_price': entry_price, 'exit_price': close_price,
                        'sl': sl, 'tp': tp2,
                        'pnl_usd': round(pnl, 4), 'result': result,
                        'capital': round(capital, 4), 'pips': round(pips, 1),
                        'spread_pips': spread_pips,
                        'slippage_pips': self.slippage_pips,
                        'commission_usd': round(commission, 4),
                        'type': 'TP2' if hit_tp2 else 'SL',
                    })
                    in_trade = tp1_hit = False
                    continue

            if not in_trade and prev['signal'] != 0 and capital > 50:
                sig      = int(prev['signal'])
                entry    = row['open'] + sig * self.slippage_pips * pip
                atrv     = prev['atr']
                sl_dist  = atrv * self.atr_sl
                tp1_dist = atrv * self.atr_tp1
                tp2_dist = atrv * self.atr_tp2

                if tp2_dist / max(sl_dist, 0.00001) < self.rr_ratio:
                    equity.append(capital); continue

                risk_usd = capital * self.risk_pct
                sl_pips  = sl_dist / pip
                lots_raw = risk_usd / (sl_pips * pip_value) if sl_pips > 0 and pip_value > 0 else 0
                lots     = round_lot_size(lots_raw, self.instrument_spec)

                if lots <= 0:
                    equity.append(capital); continue

                # Descontar spread en cada entrada
                entry_cost = spread_pips * lots * spread_value
                entry_cost += lots * commission_rt * 0.50
                capital -= entry_cost

                in_trade = True; trade_dir = sig
                entry_price = entry; entry_atr = atrv
                lot_size = lot_remaining = lots
                tp1_hit = False; entry_dt = row.name

                trades.append({
                    'entry_dt': entry_dt, 'exit_dt': row.name,
                    'direction': 'LONG' if trade_dir == 1 else 'SHORT',
                    'entry_price': entry_price, 'exit_price': entry_price,
                    'sl': 0.0, 'tp': 0.0,
                    'pnl_usd': round(-entry_cost, 4), 'result': 'COST',
                    'capital': round(capital, 4), 'pips': 0.0,
                    'spread_pips': spread_pips,
                    'slippage_pips': self.slippage_pips,
                    'commission_usd': round(lots * commission_rt * 0.50, 4),
                    'type': 'COST',
                })

                if sig == 1:
                    sl = entry - sl_dist; tp1 = entry + tp1_dist
                    tp2 = entry + tp2_dist; trail_sl = entry - sl_dist
                else:
                    sl = entry + sl_dist; tp1 = entry - tp1_dist
                    tp2 = entry - tp2_dist; trail_sl = entry + sl_dist

            equity.append(capital)

        if in_trade:
            close_p     = df.iloc[-1]['close']
            close_p     = self._apply_exit_slippage(close_p, trade_dir)
            active_lots = lot_remaining if tp1_hit else lot_size
            pips        = (close_p - entry_price) * trade_dir / pip
            pnl         = pips * active_lots * pip_value
            commission  = active_lots * commission_rt * 0.50
            pnl        -= commission
            capital    += pnl
            trades.append({
                'entry_dt': entry_dt, 'exit_dt': df.index[-1],
                'direction': 'LONG' if trade_dir == 1 else 'SHORT',
                'entry_price': entry_price, 'exit_price': close_p,
                'sl': sl, 'tp': tp2,
                'pnl_usd': round(pnl, 4),
                'result': 'WIN' if pnl > 0 else 'LOSS',
                'capital': round(capital, 4), 'pips': round(pips, 1),
                'spread_pips': spread_pips,
                'slippage_pips': self.slippage_pips,
                'commission_usd': round(commission, 4),
                'type': 'CLOSE',
            })

        return {
            'trades'         : pd.DataFrame(trades),
            'equity'         : pd.Series(equity, index=df.index[:len(equity)]),
            'final_capital'  : capital,
            'initial_capital': self.initial_capital,
            'instrument'     : self.instrument,
            'instrument_spec': self.instrument_spec,
        }


def _longest_streak(values: np.ndarray, predicate) -> int:
    longest = current = 0
    for value in values:
        if predicate(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def compute_metrics(results):
    """
    Métricas completas incluyendo Sharpe/Sortino corregidos.

    Importante:
      - Sharpe/Sortino se calculan sobre retornos por trade, no sobre P&L USD.
      - La anualización usa la frecuencia real de trades por año observada
        en el backtest, evitando inflar métricas con una raíz de 252 fija.
    """
    trades  = results['trades']
    equity  = results['equity']
    initial = results['initial_capital']
    final   = results['final_capital']

    empty = {k: 0 for k in [
        'total_trades', 'wins', 'losses', 'win_rate', 'expectancy_usd',
        'profit_factor', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown_pct',
        'max_drawdown_usd', 'calmar_ratio', 'avg_win_usd', 'avg_loss_usd',
        'actual_rr', 'trades_per_year', 'max_loss_streak', 'max_win_streak',
        'total_cost_usd', 'avg_cost_per_trade_usd',
        'total_return_pct', 'net_pnl_usd']}
    empty['final_capital'] = round(final, 2)
    if trades.empty or len(trades) < 2:
        return empty

    pnls_by_trade = trade_pnls_by_entry(trades)
    if pnls_by_trade.empty:
        return empty

    trade_ret = trade_returns(trades, initial)
    pnls   = pnls_by_trade.values
    wins   = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    wr     = len(wins) / len(pnls) if len(pnls) > 0 else 0

    avg_win  = wins.mean()        if len(wins)   > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    exp      = (avg_win * wr) - (avg_loss * (1 - wr))
    pf       = (wins.sum() / abs(losses.sum())
                if len(losses) > 0 and losses.sum() != 0 else 0)

    trades_per_year = annualized_trades_per_year(len(trade_ret), equity.index)

    if len(trade_ret) > 1 and trade_ret.std(ddof=1) > 0 and trades_per_year > 0:
        sharpe = (trade_ret.mean() / trade_ret.std(ddof=1) *
                  np.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    downside = trade_ret.clip(upper=0)
    downside_dev = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 0.0
    if downside_dev > 0 and trades_per_year > 0:
        sortino = trade_ret.mean() / downside_dev * np.sqrt(trades_per_year)
    else:
        sortino = sharpe

    eq    = equity.values
    peak  = np.maximum.accumulate(eq)
    dd_abs = peak - eq
    mdd_usd = float(dd_abs.max()) if len(dd_abs) else 0.0
    mdd   = abs(((eq - peak) / peak).min())
    total_ret = (final - initial) / initial
    calmar    = (total_ret / mdd) if mdd > 0 else 0
    total_cost = float(trades.loc[
        trades.get('type', pd.Series(index=trades.index, dtype=object)).eq('COST'),
        'pnl_usd',
    ].sum()) if 'type' in trades.columns else 0.0
    total_cost = abs(total_cost)

    return {
        'total_trades'    : len(pnls),
        'wins'            : len(wins),
        'losses'          : len(losses),
        'win_rate'        : round(wr * 100, 2),
        'expectancy_usd'  : round(exp, 4),
        'profit_factor'   : round(pf, 3),
        'sharpe_ratio'    : round(sharpe, 3),
        'sortino_ratio'   : round(sortino, 3),   # nuevo v5.3
        'max_drawdown_pct': round(mdd * 100, 2),
        'max_drawdown_usd': round(mdd_usd, 2),
        'calmar_ratio'    : round(calmar, 3),
        'avg_win_usd'     : round(avg_win, 4),
        'avg_loss_usd'    : round(avg_loss, 4),
        'actual_rr'       : round(avg_win / avg_loss, 2) if avg_loss > 0 else 0,
        'trades_per_year' : round(trades_per_year, 2),
        'max_loss_streak' : _longest_streak(pnls, lambda x: x < 0),
        'max_win_streak'  : _longest_streak(pnls, lambda x: x > 0),
        'total_cost_usd'  : round(total_cost, 4),
        'avg_cost_per_trade_usd': round(total_cost / len(pnls), 4) if len(pnls) else 0,
        'total_return_pct': round(total_ret * 100, 2),
        'final_capital'   : round(final, 2),
        'net_pnl_usd'     : round(final - initial, 4),
    }
