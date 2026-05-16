# Code-Verzeichnis — Reproduktions-Anleitung

## Voraussetzungen

```
Python 3.10+
pip install numpy pandas openpyxl numba statsmodels matplotlib
```

## Quick-Start

```powershell
cd C:\Users\lambr\Downloads\Abgabe_BATH_V7\02_Code

# 1) Hauptbacktest (Kennzahlen + Bootstrap-Inferenz + Slippage-Sensitivitaet)
python backtest_v7.py

# 2) Abbildungen 4, 5, 6, 7 + 4-Panel-Plots aller 5 Titel
python make_plots_v7.py
```

**Erwartete Laufzeit:** je 30-60 Sekunden auf Windows mit Numba (parallel=False).

## Skripte

### `backtest_v7.py`

Erzeugt:
- `baseline_results_v7.csv` (Tab. 3)
- `bootstrap_v7.json` (Tab. 4) — oder uebernimmt vorhandene `bootstrap_v7_B5000.json`
- `slippage_sensitivity_v7.csv` (Tab. 5)

**Quick-Mode:** Wenn `bootstrap_v7_B5000.json` schon im Verzeichnis liegt, wird der
zeitintensive Bootstrap-Schritt uebersprungen und die Werte werden geladen.

Erwartete Hauptzahlen:

| Titel | V7 TR % | V7 MDD % | V7 Sharpe |
|---|---:|---:|---:|
| Logitech | 302.77 | -82.30 | 0.368 |
| Sika | 637.18 | -75.46 | 0.479 |
| Richemont | 898.37 | -67.11 | 0.496 |
| Lonza | 416.10 | -81.82 | 0.412 |
| Partners Group | 598.66 | -61.11 | 0.509 |

**PGHN MDD-Differenz vs B&H: +8.33 pp, p = 0.040 (statistisch signifikant)** — zentraler V7-Befund.

### `make_plots_v7.py`

Erzeugt 7 PNG-Abbildungen fuer die Thesis-Dokumentation:

| Datei | Inhalt | Verwendung in Thesis |
|---|---|---|
| `fig_total_return_v7.png` | Bar-Chart TR aller 5 Titel × 3 Strategien | Abbildung 4 |
| `fig_max_drawdown_v7.png` | Bar-Chart MDD aller 5 Titel × 3 Strategien | Abbildung 5 |
| `fig_pghn_4panel.png` | 4-Panel: Preis+SMA20+Trades, Performance (log), BSADF+Schwellen, Aktienquote | Abbildung 6 |
| `fig_sika_4panel.png` | analog Sika | Abbildung 7 |
| `fig_logitech_4panel.png` | analog Logitech | optional (Anhang) |
| `fig_richemont_4panel.png` | analog Richemont | optional (Anhang) |
| `fig_lonza_4panel.png` | analog Lonza | optional (Anhang) |

### `bsadf_core.py` (Windows-optimiert)

BSADF-Kernimplementierung mit Numba-JIT (`parallel=False`/serial). Auf Windows ohne
initialisierten Threading-Layer 20-30× schneller als die Vorgaengerversion mit `parallel=True`.

## Datei-Uebersicht

| Datei | Zweck |
|---|---|
| `backtest_v7.py` | Hauptbacktest |
| `make_plots_v7.py` | Abbildungen 4-7 + Anhang-Plots |
| `bsadf_core.py` | BSADF-Kern (Numba-JIT, serial) |
| `backtest_v3_referenz.py` | V3-Vorgaengercode |
| `monte_carlo_v2.py` | Evans-Modell-Simulator |
| `Datenreihe.xlsx` | Originaldaten (5 Schweizer Bluechips) |
| `wild_bootstrap_critical_values_v7_B5000.json` | Pre-computed CVs (q94 = 0.101, q99 = 0.995) |
| `bootstrap_v7_B5000.json` | Pre-computed Bootstrap-p-Werte (Quick-Mode-Quelle) |

## Haeufige Probleme

**Bootstrap dauert ewig** → bsadf_core.py muss mit `parallel=False` laufen (Windows-Fix). Kontrollieren mit:
```powershell
findstr parallel= bsadf_core.py
```
Sollte `parallel=False` zeigen.

**Plots werden nicht erzeugt** → `pip install matplotlib`.

**ModuleNotFoundError numba** → `pip install numba`.

**Skript haengt am ersten BSADF-Aufruf** → Numba kompiliert beim ersten Aufruf (cache=True
schreibt `__pycache__/`). Beim zweiten Lauf ist es schneller.
