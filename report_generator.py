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
        "| Activo | Estado | Trades | WR | Sharpe | Sortino | PF | DD | P&L | WF+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for item in results:
        if item.get("status") != "OK":
            lines.append(f"| {item['symbol']} | {item.get('status')} | - | - | - | - | - | - | - | - |")
            continue
        m = item["metrics"]
        wf = item["walkforward"]
        lines.append(
            f"| {item['symbol']} | OK | {m.get('total_trades', 0)} | "
            f"{m.get('win_rate', 0):.1f}% | {m.get('sharpe_ratio', 0):.3f} | "
            f"{m.get('sortino_ratio', 0):.3f} | {m.get('profit_factor', 0):.2f} | "
            f"{m.get('max_drawdown_pct', 0):.1f}% | {_money(m.get('net_pnl_usd', 0))} | "
            f"{wf.get('positive_windows', 0)}/{wf.get('total_windows', 0)} |"
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
