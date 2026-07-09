"""
charts.py — Todas las visualizaciones del sistema
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import os

from config import CAPITAL, INDICATORS, RESULTS_DIR

# Tema global
BG, CARD    = '#0d1117', '#161b22'
GREEN, RED  = '#00c853', '#ff1744'
BLUE, YELLOW= '#2979ff', '#ffd600'
ORANGE, PURPLE = '#ff9100', '#9c27b0'
GRAY, WHITE = '#8b949e', '#e6edf3'


def _style(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=GRAY, labelsize=8)
    for s in ax.spines.values():
        s.set_color('#30363d')
    ax.grid(alpha=0.1, color=GRAY)


def _usd(x, _): return f'${x:.0f}'
def _pct(x, _): return f'{x:.1f}%'


# ── BACKTEST DASHBOARD ───────────────────────────────────────────────

def plot_backtest(df, results, metrics, title="Backtest", save=None):
    trades = results['trades']
    equity = results['equity']

    fig = plt.figure(figsize=(20, 22), facecolor=BG)
    fig.suptitle(f'BACKTEST — {title}', fontsize=17,
                 color=WHITE, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.96, bottom=0.03)

    # Precio + EMAs
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)
    ax1.plot(df.index, df['close'], color=WHITE, lw=0.8, alpha=0.6)
    ax1.plot(df.index, df['ema_fast'], color=BLUE,   lw=1.5,
             label=f"EMA {INDICATORS['ema_fast']}")
    ax1.plot(df.index, df['ema_slow'], color=YELLOW, lw=1.5,
             label=f"EMA {INDICATORS['ema_slow']}")
    if not trades.empty:
        for _, t in trades.iterrows():
            c = GREEN if t['direction'] == 'LONG' else RED
            ax1.axvline(t['entry_dt'], color=c, alpha=0.3, lw=0.8, ls='--')
    ax1.set_title('EUR/USD — Precio, EMAs y Entradas', color=WHITE, pad=6)
    ax1.legend(fontsize=8, facecolor=CARD, labelcolor=WHITE)

    # RSI
    ax2 = fig.add_subplot(gs[1, :2])
    _style(ax2)
    ax2.plot(df.index, df['rsi'], color=YELLOW, lw=1.2)
    ax2.axhline(70, color=RED,   lw=1, ls='--', alpha=0.7)
    ax2.axhline(30, color=GREEN, lw=1, ls='--', alpha=0.7)
    ax2.axhline(50, color=GRAY,  lw=0.8, ls=':', alpha=0.5)
    ax2.fill_between(df.index, df['rsi'], 70, where=(df['rsi']>70),
                     color=RED,   alpha=0.1)
    ax2.fill_between(df.index, df['rsi'], 30, where=(df['rsi']<30),
                     color=GREEN, alpha=0.1)
    ax2.set_ylim(0, 100)
    ax2.set_title('RSI (14)', color=WHITE, pad=6)

    # ATR
    ax3 = fig.add_subplot(gs[1, 2])
    _style(ax3)
    atr_pips = df['atr'] / 0.0001
    ax3.fill_between(df.index, atr_pips, alpha=0.4, color=BLUE)
    ax3.plot(df.index, atr_pips, color=BLUE, lw=1)
    ax3.set_title('ATR (14) — Pips', color=WHITE, pad=6)

    # Equity
    ax4 = fig.add_subplot(gs[2, :2])
    _style(ax4)
    eq   = equity.values
    peak = np.maximum.accumulate(eq)
    ax4.fill_between(equity.index, eq, peak, where=(eq < peak),
                     color=RED, alpha=0.3, label='Drawdown')
    ax4.plot(equity.index, eq,   color=GREEN,  lw=2,   label='Equity')
    ax4.plot(equity.index, peak, color=YELLOW, lw=0.8, ls='--', alpha=0.6)
    ax4.axhline(CAPITAL['initial'], color=GRAY, lw=1, ls=':')
    ax4.set_title('Curva de Equity', color=WHITE, pad=6)
    ax4.legend(fontsize=8, facecolor=CARD, labelcolor=WHITE)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(_usd))

    # P&L por trade
    ax5 = fig.add_subplot(gs[2, 2])
    _style(ax5)
    if not trades.empty:
        pnls = trades['pnl_usd'].values
        cols = [GREEN if p > 0 else RED for p in pnls]
        ax5.bar(range(len(pnls)), pnls, color=cols, alpha=0.85)
        ax5.axhline(0, color=WHITE, lw=1)
    ax5.set_title('P&L por Trade', color=WHITE, pad=6)
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:.2f}'))

    # Tarjetas métricas
    ax6 = fig.add_subplot(gs[3, :])
    ax6.set_facecolor(BG)
    ax6.axis('off')
    ax6.set_title('📊 MÉTRICAS', color=WHITE, fontsize=13,
                  fontweight='bold', pad=8, loc='left')

    cards = [
        ('Trades',        str(metrics.get('total_trades', 0)),      WHITE),
        ('Win Rate',      f"{metrics.get('win_rate', 0):.1f}%",     GREEN if metrics.get('win_rate',0)>45 else RED),
        ('Expectancy',    f"${metrics.get('expectancy_usd', 0):.3f}",GREEN if metrics.get('expectancy_usd',0)>0 else RED),
        ('Sharpe',        f"{metrics.get('sharpe_ratio', 0):.2f}",  GREEN if metrics.get('sharpe_ratio',0)>1 else YELLOW),
        ('Max DD',        f"{metrics.get('max_drawdown_pct', 0):.1f}%", GREEN if metrics.get('max_drawdown_pct',0)<10 else RED),
        ('Profit Factor', f"{metrics.get('profit_factor', 0):.2f}", GREEN if metrics.get('profit_factor',0)>1.5 else RED),
        ('Retorno',       f"{metrics.get('total_return_pct', 0):.1f}%", GREEN if metrics.get('total_return_pct',0)>0 else RED),
        ('Capital Final', f"${metrics.get('final_capital', 150):.2f}", BLUE),
        ('R:R Real',      f"{metrics.get('actual_rr', 0):.2f}",     GREEN if metrics.get('actual_rr',0)>=2 else YELLOW),
        ('Calmar',        f"{metrics.get('calmar_ratio', 0):.2f}",  YELLOW),
    ]

    n_col = 5
    for i, (label, val, color) in enumerate(cards):
        col = i % n_col
        row = i // n_col
        x = 0.01 + col * 0.198
        y = 0.55 - row * 0.52

        rect = mpatches.FancyBboxPatch((x, y), 0.185, 0.40,
               boxstyle='round,pad=0.02', transform=ax6.transAxes,
               facecolor=CARD, edgecolor=color, lw=1.5, clip_on=False)
        ax6.add_patch(rect)
        ax6.text(x+0.0925, y+0.30, label, ha='center', va='center',
                 transform=ax6.transAxes, color=GRAY,  fontsize=8,  fontweight='bold')
        ax6.text(x+0.0925, y+0.11, val,   ha='center', va='center',
                 transform=ax6.transAxes, color=color, fontsize=13, fontweight='bold')

    # Histograma + Pie + Drawdown
    ax7 = fig.add_subplot(gs[4, 0])
    _style(ax7)
    if not trades.empty and len(trades) > 3:
        pnls = trades['pnl_usd'].values
        ax7.hist(pnls[pnls > 0],  bins=10, color=GREEN, alpha=0.7, density=True, label='Wins')
        ax7.hist(pnls[pnls <= 0], bins=10, color=RED,   alpha=0.7, density=True, label='Losses')
        ax7.axvline(pnls.mean(), color=YELLOW, lw=2, ls='--')
        ax7.legend(fontsize=7, facecolor=CARD, labelcolor=WHITE)
    ax7.set_title('Distribución P&L', color=WHITE, pad=6)

    ax8 = fig.add_subplot(gs[4, 1])
    ax8.set_facecolor(CARD)
    w_c = metrics.get('wins', 0)
    l_c = metrics.get('losses', 0)
    if w_c + l_c > 0:
        ax8.pie([w_c, l_c],
                labels=[f'Wins\n{w_c}', f'Losses\n{l_c}'],
                colors=[GREEN, RED], autopct='%1.0f%%',
                textprops={'color': WHITE, 'fontsize': 9},
                startangle=90,
                wedgeprops={'edgecolor': BG, 'linewidth': 2})
    ax8.set_title('Win vs Loss', color=WHITE, pad=6)

    ax9 = fig.add_subplot(gs[4, 2])
    _style(ax9)
    dd_pct = ((equity - np.maximum.accumulate(equity)) /
               np.maximum.accumulate(equity)) * 100
    ax9.fill_between(equity.index, dd_pct, 0, color=RED, alpha=0.6)
    ax9.plot(equity.index, dd_pct, color=RED, lw=1)
    ax9.set_title(f"Drawdown — Máx: {metrics.get('max_drawdown_pct',0):.1f}%",
                  color=WHITE, pad=6)
    ax9.yaxis.set_major_formatter(plt.FuncFormatter(_pct))

    save = save or os.path.join(RESULTS_DIR, 'backtest.png')
    plt.savefig(save, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"   📊 Gráfico → {save}")


# ── MONTE CARLO ───────────────────────────────────────────────────────

def plot_monte_carlo(matrix, ruin_prob, save=None):
    initial = CAPITAL['initial']
    n_sim   = matrix.shape[0]
    finals  = matrix[:, -1]
    x       = np.arange(matrix.shape[1])

    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.suptitle(f'SIMULACIÓN MONTE CARLO — {n_sim:,} escenarios',
                 fontsize=16, color=WHITE, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35,
                           top=0.93, bottom=0.06)

    ax1 = fig.add_subplot(gs[0, :2])
    _style(ax1)
    idx = np.random.choice(n_sim, min(300, n_sim), replace=False)
    for i in idx:
        c = RED if finals[i] < initial else GREEN
        ax1.plot(matrix[i], alpha=0.05, lw=0.5, color=c)

    for p, c, lbl in [(5, BLUE, 'P5-P95'), (25, BLUE, 'P25-P75')]:
        ax1.fill_between(x, np.percentile(matrix, p, axis=0),
                         np.percentile(matrix, 100-p, axis=0),
                         alpha=0.10 if p==5 else 0.20, color=c,
                         label=lbl if p==5 else None)

    ax1.plot(x, np.percentile(matrix, 50, axis=0),
             color=YELLOW, lw=2.5, label='Mediana')
    ax1.axhline(initial,        color=WHITE, lw=1.5, ls='--', alpha=0.7)
    ax1.axhline(initial * 0.5,  color=RED,   lw=1.5, ls='--', alpha=0.7,
                label='Umbral ruina 50%')
    ax1.legend(fontsize=8, facecolor=CARD, labelcolor=WHITE)
    ax1.set_title('Trayectorias de Equity', color=WHITE, pad=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(_usd))

    # KPIs
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(CARD)
    ax2.axis('off')
    median_f  = np.median(finals)
    pct_prof  = (finals > initial).mean() * 100
    var95     = np.percentile(finals, 5)
    best95    = np.percentile(finals, 95)

    kpis = [
        ('Prob. Ruina',         f'{ruin_prob*100:.2f}%',
         RED if ruin_prob > .05 else GREEN),
        ('Escenarios rentables',f'{pct_prof:.1f}%',
         GREEN if pct_prof > 50 else RED),
        ('Capital Mediano',     f'${median_f:.2f}',
         GREEN if median_f > initial else RED),
        ('VaR 95%',             f'${var95:.2f}',
         RED if var95 < initial*.8 else YELLOW),
        ('Mejor P95',           f'${best95:.2f}', BLUE),
        ('Simulaciones',        f'{n_sim:,}',      WHITE),
    ]
    ax2.set_title('Estadísticas MC', color=WHITE,
                  fontsize=11, fontweight='bold', pad=8)
    y = 0.88
    for label, val, color in kpis:
        ax2.text(0.05, y, label,  transform=ax2.transAxes,
                 color=GRAY, fontsize=9)
        ax2.text(0.95, y, val,    transform=ax2.transAxes,
                 color=color, fontsize=11, fontweight='bold', ha='right')
        y -= 0.12

    # Distribución
    ax3 = fig.add_subplot(gs[1, :2])
    _style(ax3)
    bins = np.linspace(finals.min(), finals.max(), 80)
    ax3.hist(finals[finals >= initial], bins=bins, color=GREEN, alpha=0.7,
             label=f'Rentable ({(finals>=initial).sum():,})')
    ax3.hist(finals[finals < initial],  bins=bins, color=RED,   alpha=0.7,
             label=f'En pérdida ({(finals<initial).sum():,})')
    ax3.axvline(initial,   color=WHITE,  lw=2, ls='--')
    ax3.axvline(median_f,  color=YELLOW, lw=2)
    ax3.axvline(var95,     color=RED,    lw=2, ls=':')
    ax3.legend(fontsize=8, facecolor=CARD, labelcolor=WHITE)
    ax3.set_title('Distribución del Capital Final', color=WHITE, pad=8)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(_usd))

    # Riesgo vs ruina
    ax4 = fig.add_subplot(gs[1, 2])
    _style(ax4)
    risks  = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    rps    = []
    from analysis import monte_carlo as mc_fn
    for r in risks:
        orig = CAPITAL['risk_pct']
        CAPITAL['risk_pct'] = r
        _, rp = mc_fn(n_sim=1000, n_trades=100)
        CAPITAL['risk_pct'] = orig
        rps.append(rp * 100)

    colors_b = [GREEN if rp<5 else (YELLOW if rp<15 else RED) for rp in rps]
    ax4.bar([f'{r*100:.1f}%' for r in risks], rps, color=colors_b, alpha=0.85)
    ax4.axhline(5, color=YELLOW, lw=2, ls='--', label='Límite 5%')
    ax4.legend(fontsize=7, facecolor=CARD, labelcolor=WHITE)
    ax4.set_title('Riesgo/Trade vs\nProb. Ruina', color=WHITE, pad=6)
    for bar, val in zip(ax4.patches, rps):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.2,
                 f'{val:.1f}%', ha='center', va='bottom',
                 color=WHITE, fontsize=8)

    save = save or os.path.join(RESULTS_DIR, 'monte_carlo.png')
    plt.savefig(save, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"   📊 Gráfico → {save}")


# ── WALK-FORWARD ──────────────────────────────────────────────────────

def plot_walkforward(wf_df, wf_full, rob, save=None):
    windows = wf_df['window'].values
    x       = np.arange(len(windows))
    mode_label = str(rob.get('mode', 'pullback')).upper()
    m15_label = 'ON' if rob.get('use_m15', False) else 'OFF'

    fig = plt.figure(figsize=(20, 20), facecolor=BG)
    fig.suptitle(f'WALK-FORWARD TEST — Sistema Congelado {mode_label}',
                 fontsize=17, color=WHITE, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.95, bottom=0.04)

    # Sharpe TRAIN vs TEST
    ax1 = fig.add_subplot(gs[0, :2])
    _style(ax1)
    w = 0.35
    b1 = ax1.bar(x - w/2, wf_df['train_sharpe'], w, color=BLUE,  alpha=0.85, label='TRAIN')
    b2 = ax1.bar(x + w/2, wf_df['test_sharpe'],  w,
                 color=[GREEN if s>0 else RED for s in wf_df['test_sharpe']],
                 alpha=0.85, label='TEST (out-of-sample)')
    ax1.axhline(0, color=WHITE, lw=1)
    ax1.axhline(1, color=GREEN, lw=1.5, ls='--', alpha=0.6, label='Sharpe objetivo 1.0')
    for bar, val in zip(b1, wf_df['train_sharpe']):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.05,
                 f'{val:.2f}', ha='center', va='bottom', color=BLUE, fontsize=9)
    for bar, val in zip(b2, wf_df['test_sharpe']):
        c = GREEN if val>0 else RED
        ax1.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+(0.05 if val>=0 else -0.25),
                 f'{val:.2f}', ha='center', va='bottom', color=c, fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Ventana {w}' for w in windows], color=GRAY)
    ax1.legend(fontsize=9, facecolor=CARD, labelcolor=WHITE)
    ax1.set_title('Sharpe TRAIN vs TEST por Ventana', color=WHITE, pad=8)

    # P&L TEST por ventana
    ax2 = fig.add_subplot(gs[0, 2])
    _style(ax2)
    pnl_colors = [GREEN if p > 0 else RED for p in wf_df['test_pnl']]
    bars = ax2.bar([f'V{w}' for w in windows],
                   wf_df['test_pnl'],
                   color=pnl_colors, alpha=0.85)
    ax2.axhline(0, color=WHITE, lw=1)
    for bar, val in zip(bars, wf_df['test_pnl']):
        yc = val + (0.6 if val >= 0 else -1.0)
        ax2.text(bar.get_x()+bar.get_width()/2, yc,
                 f'{val:+.1f}', ha='center', va='bottom',
                 color=(GREEN if val > 0 else RED), fontsize=8)
    ax2.set_title('P&L TEST por Ventana', color=WHITE, pad=6)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(_usd))

    # Trades TRAIN vs TEST
    ax3 = fig.add_subplot(gs[1, :2])
    _style(ax3)
    w = 0.35
    ax3.bar(x - w/2, wf_df['train_trades'], w, color=BLUE, alpha=0.85, label='TRAIN')
    ax3.bar(x + w/2, wf_df['test_trades'],  w, color=YELLOW, alpha=0.85, label='TEST')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'V{w}' for w in windows], color=GRAY)
    ax3.legend(fontsize=9, facecolor=CARD, labelcolor=WHITE)
    ax3.set_title('Frecuencia Operativa por Ventana',
                  color=WHITE, pad=8)

    # Win Rate y DD
    ax4 = fig.add_subplot(gs[1, 2])
    _style(ax4)
    ax4t = ax4.twinx()
    ax4.bar([f'V{w}' for w in windows], wf_df['test_win_rate'],
            color=BLUE, alpha=0.6, label='WR %')
    ax4t.plot([f'V{w}' for w in windows], wf_df['test_max_dd'],
              'o-', color=RED, lw=2, ms=6, label='DD %')
    ax4.axhline(50, color=WHITE, lw=1, ls='--', alpha=0.4)
    ax4.set_ylabel('Win Rate %', color=BLUE, fontsize=8)
    ax4t.set_ylabel('Max DD %',  color=RED,  fontsize=8)
    ax4.set_title('Win Rate vs Drawdown\nen TEST', color=WHITE, pad=6)

    # Equity TEST concatenada
    ax5 = fig.add_subplot(gs[2, :])
    _style(ax5)
    colors_eq = [BLUE, GREEN, YELLOW, ORANGE, PURPLE]
    offset = 0
    for i, r in enumerate(wf_full):
        eq   = r['equity'].values
        c    = colors_eq[i % len(colors_eq)]
        xs   = np.arange(len(eq)) + offset
        ax5.plot(xs, eq, color=c, lw=2,
                 label=f"V{r['window']} P&L:{r['test_pnl']:+.2f} "
                       f"Sharpe:{r['test_sharpe']:.2f}")
        ax5.axvline(offset, color='#30363d', lw=1.5, ls='--', alpha=0.6)
        offset += len(eq)
    ax5.axhline(CAPITAL['initial'], color=WHITE, lw=1, ls=':', alpha=0.4)
    ax5.legend(fontsize=8, facecolor=CARD, labelcolor=WHITE, ncol=3)
    ax5.set_title('Equity TEST por Ventana — Sistema congelado out-of-sample',
                  color=WHITE, pad=8)
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(_usd))

    # Veredicto
    ax6 = fig.add_subplot(gs[3, :])
    ax6.set_facecolor('#0a1628')
    ax6.axis('off')
    vc = GREEN if '✅' in rob['verdict'] else (YELLOW if '⚠️' in rob['verdict'] else RED)
    ax6.set_title('🔬  VEREDICTO DE ROBUSTEZ', color=vc,
                  fontsize=13, fontweight='bold', pad=10, loc='left')

    txt = (f"  Sharpe TRAIN promedio : {rob['avg_train_sharpe']:.3f}    "
           f"Sharpe TEST promedio  : {rob['avg_test_sharpe']:.3f}\n"
           f"  P&L TEST promedio    : ${rob['avg_test_pnl']:.2f}    "
           f"DD TEST promedio      : {rob['avg_test_max_dd']:.2f}%\n"
           f"  Ventanas positivas   : {rob['positive_windows']}/{rob['total_windows']} "
           f"({rob['pct_positive_test']:.0f}%)\n"
           f"  Sistema evaluado     : {rob['mode']} | {rob['session_mode']} | M15 {m15_label}\n\n"
           f"  {rob['verdict']}")
    ax6.text(0.02, 0.55, txt, transform=ax6.transAxes,
             color=WHITE, fontsize=11, va='center',
             fontfamily='monospace', linespacing=1.8)
    ax6.text(0.02, 0.08,
             f"  → Criterio robusto: >=60% ventanas positivas, "
             f"Sharpe TEST promedio > 0 y DD promedio <= {CAPITAL['max_drawdown']*100:.1f}%",
             transform=ax6.transAxes,
             color=YELLOW, fontsize=10, style='italic')

    save = save or os.path.join(RESULTS_DIR, 'walkforward.png')
    plt.savefig(save, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"   📊 Gráfico → {save}")
