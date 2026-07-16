"""
report_generator.py
Markdown reports for asset comparison and risk feasibility.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Sequence

from config import CAPITAL, REPORTS_DIR


def _money(value) -> str:
    if value is None:
        return "N/A"
    sign = "+" if float(value) > 0 else ""
    return f"{sign}${float(value):,.2f}"


def _capital(value) -> str:
    if value is None:
        return "N/A"
    return f"${float(value):,.2f}"


def _pct(value) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.1f}%"


def _months(value) -> str:
    if value is None:
        return "No defendible"
    return f"{float(value):.1f} meses"


def asset_verdict(item: dict) -> str:
    if item.get("status") == "NO OPERABLE":
        return "NO OPERABLE"
    if item.get("status") != "OK":
        return "NO VALIDADO"

    metrics = item.get("metrics", {})
    bootstrap = item.get("bootstrap", {})
    wf = item.get("walkforward", {})
    wf_total = max(float(wf.get("total_windows", 0)), 1.0)
    wf_pct = float(wf.get("positive_windows", 0)) / wf_total * 100
    dd_ok = metrics.get("max_drawdown_pct", 999) <= CAPITAL["max_drawdown"] * 100
    pf_ok = metrics.get("profit_factor", 0) >= 1.20
    pnl_ok = metrics.get("net_pnl_usd", 0) > 0
    wf_ok = wf_pct >= 70
    bootstrap_ok = bootstrap.get("sharpe_ci_low", -999) > 0

    if pnl_ok and dd_ok and pf_ok and wf_ok and not bootstrap_ok:
        return "APTO SOLO PAPER"
    if pnl_ok and dd_ok and wf_pct >= 50:
        return "PROMETEDOR"
    return "NO VALIDADO"


def write_compare_assets_report(results: Sequence[dict], path: str | None = None) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = path or os.path.join(REPORTS_DIR, "compare_assets_report.md")

    ok_assets = [r for r in results if r.get("status") == "OK"]
    dd_limit = CAPITAL["max_drawdown"] * 100
    strict_candidates = [
        r for r in ok_assets
        if bool(r.get("contract_confirmed", True))
        and r.get("metrics", {}).get("net_pnl_usd", 0) > 0
        and r.get("metrics", {}).get("max_drawdown_pct", 999) <= dd_limit
        and r.get("bootstrap", {}).get("sharpe_ci_low", -999) > 0
    ]
    recommended = sorted(
        strict_candidates,
        key=lambda r: (
            r.get("robustness_rank", 0),
            -r.get("metrics", {}).get("max_drawdown_pct", 999),
            r.get("metrics", {}).get("net_pnl_usd", 0),
        ),
        reverse=True,
    )[0] if strict_candidates else None
    reference = sorted(
        ok_assets,
        key=lambda r: (
            r.get("robustness_rank", 0),
            -r.get("metrics", {}).get("max_drawdown_pct", 999),
            r.get("metrics", {}).get("net_pnl_usd", 0),
        ),
        reverse=True,
    )[0] if ok_assets else None

    lines = [
        "# Reporte comparativo multi-activo",
        "",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Cuenta base: ${CAPITAL['initial']:.2f}",
        f"Riesgo base: {CAPITAL['risk_pct'] * 100:.2f}% por operacion",
        "",
        "## Conclusión",
    ]
    if recommended:
        lines.append(
            f"Activo recomendado hoy: **{recommended['symbol']}**, sujeto a paper trading y monitoreo de costos reales."
        )
    else:
        if reference:
            lines.append(
                "No hay activo recomendado con criterio conservador completo. "
                f"El mejor candidato para seguir en demo/paper es **{reference['symbol']}**, "
                "pero aun no confirma edge por bootstrap."
            )
        else:
            lines.append("No hay activo recomendado con criterio conservador completo.")
    lines.extend([
        "",
        "No se promete una tasa de acierto de 100%. El objetivo es supervivencia, bajo drawdown y validacion estadistica.",
        "",
        "## Tabla comparativa",
        "",
        "| Activo | Veredicto | Trades | WR | Sharpe | Sortino | PF | DD | P&L | Mes+ | WF+ | Bootstrap IC95% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for item in results:
        if item.get("status") != "OK":
            lines.append(f"| {item['symbol']} | {asset_verdict(item)} | - | - | - | - | - | - | - | - | - | - |")
            continue
        m = item["metrics"]
        wf = item["walkforward"]
        mon = item.get("monthly", {})
        bs = item.get("bootstrap", {})
        lines.append(
            f"| {item['symbol']} | {asset_verdict(item)} | {m.get('total_trades', 0)} | "
            f"{m.get('win_rate', 0):.1f}% | {m.get('sharpe_ratio', 0):.3f} | "
            f"{m.get('sortino_ratio', 0):.3f} | {m.get('profit_factor', 0):.2f} | "
            f"{m.get('max_drawdown_pct', 0):.1f}% | {_money(m.get('net_pnl_usd', 0))} | "
            f"{mon.get('positive_month_pct', 0):.1f}% | "
            f"{wf.get('positive_windows', 0)}/{wf.get('total_windows', 0)} | "
            f"[{bs.get('sharpe_ci_low', 0):.3f}, {bs.get('sharpe_ci_high', 0):.3f}] |"
        )

    lines.extend(["", "## Riesgos y gaps"])
    for item in results:
        if item.get("status") == "SIN DATOS":
            lines.append(f"- {item['symbol']}: sin datos H4 suficientes; descargar/importar antes de comparar.")
        elif item.get("status") == "NO OPERABLE":
            fz = item.get("feasibility", {})
            lines.append(
                f"- {item['symbol']}: no operable con ${CAPITAL['initial']:.0f}; "
                f"min lot arriesga {fz.get('risk_pct_at_min_lot', 0):.2f}% aprox."
            )
        elif not item.get("contract_confirmed", True):
            lines.append(f"- {item['symbol']}: contrato no confirmado; validar tick value, spread y comision.")

    lines.extend([
        "",
        "## Meta financiera",
        "",
        "$100/semana sobre $150 exige 66.7% semanal. $600/mes exige 400% mensual. "
        "Ambas metas son no conservadoras salvo que el capital aumente mucho o se acepte un riesgo incompatible con la supervivencia.",
        "",
    ])

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out


def write_growth_plan_report(symbol: str, metrics: dict, growth: dict,
                             path: str | None = None) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = path or os.path.join(REPORTS_DIR, "growth_plan_report.md")
    bs = growth["bootstrap"]
    lines = [
        "# Growth plan conservador",
        "",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Activo/base: {symbol} H4 Pullback",
        f"Capital inicial: ${CAPITAL['initial']:.2f}",
        "",
        "## Resumen ejecutivo",
        "",
        "NO VALIDADO PARA DINERO REAL TODAVIA. El plan usa backtest y bootstrap mensual para estimar crecimiento; no promete recuperar capital en pocos meses.",
        f"Backtest base: {metrics.get('total_trades', 0)} trades, PF {metrics.get('profit_factor', 0):.2f}, DD {_pct(metrics.get('max_drawdown_pct', 0))}, P&L {_money(metrics.get('net_pnl_usd', 0))}.",
        f"Probabilidad historica de mes positivo: {_pct(bs.get('positive_month_probability_pct', 0))}.",
        "",
        "## Escenarios deterministas",
        "",
        "| Escenario | Retorno mensual usado | Capital 12m | Llegar a $250 | Recuperar $150 de ganancia | Duplicar cuenta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in growth["scenarios"]:
        lines.append(
            f"| {scenario.name} | {scenario.monthly_return_pct:.3f}% | "
            f"{_capital(scenario.projected_12m_capital)} | {_months(scenario.months_to_250)} | "
            f"{_months(scenario.months_to_recover_150_profit)} | {_months(scenario.months_to_double)} |"
        )

    lines.extend([
        "",
        "## Bootstrap mensual",
        "",
        f"Simulaciones: {bs.get('n_sim', 0):,} caminos de {bs.get('n_months', 0)} meses.",
        f"Capital final mediano simulado: {_capital(bs.get('sim_final_capital_median'))} (P25 {_capital(bs.get('sim_final_capital_p25'))}, P75 {_capital(bs.get('sim_final_capital_p75'))}).",
        f"Riesgo de ruina simulado: {float(bs.get('sim_ruin_prob_pct', 0)):.2f}%.",
        f"Meses negativos esperados por ano: {bs.get('expected_negative_months_per_year', 0):.1f}.",
        f"Peor racha mensual negativa esperada: {bs.get('expected_max_negative_month_streak', 0):.1f} meses (P95 {bs.get('p95_max_negative_month_streak', 0):.1f}).",
        "",
        "## Probabilidad de alcanzar metas",
        "",
        "| Meta | Probabilidad en 60 meses | Mediana meses si se alcanza |",
        "|---|---:|---:|",
        f"| Llegar a $250 | {_pct(bs['target_250']['prob_hit_pct'])} | {_months(bs['target_250']['median_months'])} |",
        f"| Recuperar $150 de ganancia | {_pct(bs['recover_150_profit']['prob_hit_pct'])} | {_months(bs['recover_150_profit']['median_months'])} |",
        f"| Duplicar cuenta | {_pct(bs['double_account']['prob_hit_pct'])} | {_months(bs['double_account']['median_months'])} |",
        "",
        "## Lectura profesional",
        "",
        "Si el escenario conservador o base no llega a $250/duplicar en un horizonte razonable, no se debe forzar riesgo. La forma sana de acercarse al objetivo es acumular demo real, reducir errores de ejecucion y subir capital solo si el comportamiento demo replica el backtest.",
    ])

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out


def write_risk_profiles_report(symbol: str, rows: Sequence[dict],
                               path: str | None = None) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = path or os.path.join(REPORTS_DIR, "risk_profiles_report.md")
    lines = [
        "# Reporte de perfiles de riesgo",
        "",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Activo/base: {symbol} H4 Pullback",
        "",
        "## Recomendacion",
        "",
        "Para una cuenta de $150, el perfil defendible para demo/paper es conservador o base bajo. El perfil agresivo 3.0% es solo simulacion y no recomendado para dinero real.",
        "",
        "| Perfil | Riesgo/trade | Trades | PF | Sharpe | DD | P&L | Mes+ | Ruina sim | Recup. $150 | Capital mediano | Politica |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        risk = row.get("risk_pct", 0) * 100
        if risk <= 1.0:
            verdict = "Preferible para demo"
        elif risk <= 1.5:
            verdict = "Base maximo razonable"
        elif risk <= 2.5:
            verdict = "Solo si demo confirma"
        else:
            verdict = "No recomendado real"
        lines.append(
            f"| {row.get('profile')} | {risk:.2f}% | {row.get('total_trades', 0)} | "
            f"{row.get('profit_factor', 0):.2f} | {row.get('sharpe_ratio', 0):.3f} | "
            f"{row.get('max_drawdown_pct', 0):.1f}% | {_money(row.get('net_pnl_usd', 0))} | "
            f"{_pct(row.get('positive_month_probability_pct', 0))} | "
            f"{float(row.get('sim_ruin_prob_pct', 0)):.2f}% | "
            f"{_months(row.get('months_to_recover_150_profit'))} | "
            f"{_capital(row.get('sim_final_capital_median', 0))} | "
            f"{verdict}; {row.get('reinvestment_policy', 'N/A')} |"
        )

    lines.extend([
        "",
        "## Politica",
        "",
        "No subir riesgo por frustracion. Si hay dos semanas negativas seguidas, reducir riesgo. Si hay cuatro perdidas consecutivas, pausar y revisar spread, horario y ejecucion.",
    ])
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out


def write_demo_validation_plan(status: dict, path: str | None = None) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = path or os.path.join(REPORTS_DIR, "demo_validation_plan.md")
    lines = [
        "# Plan de validacion demo",
        "",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Estado actual",
        "",
        f"Senales demo registradas: {status.get('signals', 0)}.",
        f"Trades demo cerrados: {status.get('closed_trades', 0)}.",
        f"Profit factor demo: {status.get('profit_factor', 'N/A')}.",
        f"Max drawdown demo: {status.get('max_drawdown_pct', 'N/A')}.",
        "",
        "NO VALIDADO PARA DINERO REAL TODAVIA.",
        "",
        "## Reglas para considerar subir de $150 a $250",
        "",
        "- Minimo 30-50 senales demo registradas.",
        "- 3 meses consecutivos positivos en demo.",
        "- Drawdown menor a 10%-12%.",
        "- Profit factor demo mayor a 1.20.",
        "- No mas de 4 perdidas consecutivas sin pausa.",
        "- La distribucion demo debe parecerse al backtest: frecuencia, PF, DD y rachas.",
        "",
        "## Reglas de pausa",
        "",
        "- Si hay un mes con drawdown fuerte, detener escalamiento.",
        "- Si hay 2 semanas negativas seguidas, reducir riesgo.",
        "- Si hay 4 perdidas consecutivas, pausar operaciones.",
    ]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out


def write_final_recommendation(context: dict, path: str | None = None) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = path or os.path.join(REPORTS_DIR, "final_recommendation.md")
    lines = [
        "# Recomendacion final",
        "",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Decision",
        "",
        "NO VALIDADO PARA DINERO REAL TODAVIA.",
        "",
        f"Mejor activo actual: {context.get('best_asset', 'EURUSD')}.",
        f"Mejor temporalidad actual: {context.get('best_timeframe', 'H4')}.",
        f"Riesgo sugerido para $150: {context.get('suggested_risk', '0.5%-1.0% en demo/paper')}.",
        "",
        "## Que si hacer",
        "",
        "- Continuar con EURUSD H4 Pullback en demo/paper.",
        "- Registrar cada senal y cada rechazo en CSV.",
        "- Comparar resultados demo contra backtest despues de 30-50 senales.",
        "- Considerar $250 solo si demo cumple reglas de escalamiento.",
        "",
        "## Que NO hacer todavia",
        "",
        "- No operar dinero real.",
        "- No perseguir $100/semana ni $600/mes con $150.",
        "- No usar XAUUSD con $150 mientras el lote minimo arriesgue demasiado.",
        "- No subir a 2.5%-3.0% por trade sin evidencia demo.",
        "- No optimizar parametros globalmente solo para mejorar el backtest.",
    ]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out
