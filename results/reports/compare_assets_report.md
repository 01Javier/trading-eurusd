# Reporte comparativo multi-activo

Generado: 2026-07-09 14:12
Cuenta base: $150.00
Riesgo base: 1.50% por operacion

## Conclusión
No hay activo recomendado con criterio conservador completo. El mejor candidato para seguir en demo/paper es **EURUSD**, pero aun no confirma edge por bootstrap.

No se promete una tasa de acierto de 100%. El objetivo es supervivencia, bajo drawdown y validacion estadistica.

## Tabla comparativa

| Activo | Veredicto | Trades | WR | Sharpe | Sortino | PF | DD | P&L | Mes+ | WF+ | Bootstrap IC95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EURUSD | APTO SOLO PAPER | 176 | 48.3% | 0.674 | 1.186 | 1.27 | 11.4% | +$63.13 | 54.0% | 8/10 | [-0.185, 1.503] |
| XAUUSD | NO OPERABLE | - | - | - | - | - | - | - | - | - | - |
| GBPUSD | NO VALIDADO | 182 | 44.5% | 0.253 | 0.422 | 1.07 | 20.3% | +$17.84 | 49.2% | 6/10 | [-0.630, 1.074] |
| USDJPY | NO VALIDADO | 131 | 44.3% | 0.086 | 0.141 | 1.02 | 17.4% | +$2.94 | 38.1% | 4/10 | [-0.803, 0.934] |
| NAS100 | NO VALIDADO | 56 | 33.9% | -1.251 | -1.520 | 0.49 | 22.1% | $-33.19 | 16.7% | 1/10 | [-2.656, -0.250] |
| US30 | NO VALIDADO | 52 | 42.3% | -0.591 | -0.792 | 0.66 | 16.0% | $-21.03 | 12.5% | 0/10 | [-1.635, 0.248] |
| BTCUSD | NO VALIDADO | - | - | - | - | - | - | - | - | - | - |

## Riesgos y gaps
- XAUUSD: no operable con $150; min lot arriesga 10.86% aprox.
- GBPUSD: contrato no confirmado; validar tick value, spread y comision.
- USDJPY: contrato no confirmado; validar tick value, spread y comision.
- NAS100: contrato no confirmado; validar tick value, spread y comision.
- US30: contrato no confirmado; validar tick value, spread y comision.
- BTCUSD: sin datos H4 suficientes; descargar/importar antes de comparar.

## Meta financiera

$100/semana sobre $150 exige 66.7% semanal. $600/mes exige 400% mensual. Ambas metas son no conservadoras salvo que el capital aumente mucho o se acepte un riesgo incompatible con la supervivencia.
