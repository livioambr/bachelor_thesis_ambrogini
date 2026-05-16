"""
make_plots_v7.py — erzeugt die Abbildungen 4-7 der Thesis aus dem V7-Backtest.

Generiert:
- fig_total_return_v7.png    (Abb. 4: Total Return Vergleich, 5 Titel)
- fig_max_drawdown_v7.png    (Abb. 5: Max Drawdown Vergleich, 5 Titel)
- fig_pghn_4panel.png        (Abb. 6: 4-Panel PGHN)
- fig_sika_4panel.png        (Abb. 7: 4-Panel Sika)
- fig_logitech_4panel.png    (analog Logitech)
- fig_richemont_4panel.png   (analog Richemont)
- fig_lonza_4panel.png       (analog Lonza)

Voraussetzungen: dieselben wie backtest_v7.py
  + matplotlib (pip install matplotlib)

Aufruf: python make_plots_v7.py
Erwartete Laufzeit: ca. 30-60 Sekunden.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

from bsadf_core import calculate_bsadf_fast
from backtest_v7 import (
    load_excel_data, run_buy_and_hold, run_momentum_strategy,
    run_v7_smart_strategy, total_return_pct, max_drawdown_pct,
    AKTIEN, EXCEL_DATEI, MIN_WINDOW, LOOKBACK_WINDOW, SMA_WINDOW, TREND_WINDOW,
)

# Styling-Konstanten
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLOR_BH    = "#7f7f7f"      # grau
COLOR_MOM   = "#ff7f0e"      # orange
COLOR_V7    = "#1f77b4"      # dunkelblau
COLOR_PRICE = "#000000"      # schwarz
COLOR_BSADF = "#9467bd"      # lila
COLOR_LIMIT = "#d62728"      # rot
COLOR_WARN  = "#ff9f1c"      # orange
COLOR_QUOTE = "#2ca02c"      # grün


def load_cvs():
    """Lade Wild-Bootstrap-CVs (B=5000 bevorzugt, sonst B=100, sonst Default)."""
    paths = [
        "wild_bootstrap_critical_values_v7_B5000.json",
        "../outputs/wild_bootstrap_critical_values_v7_B5000.json",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "wild_bootstrap_critical_values_v7_B5000.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "wild_bootstrap_critical_values_v7_B5000.json"),
    ]
    for p in paths:
        if os.path.exists(p):
            return json.load(open(p))
    raise FileNotFoundError("wild_bootstrap_critical_values_v7_B5000.json nicht gefunden.")


def collect_strategy_results(cv):
    """Sammelt fuer alle 5 Titel: TR, MDD je Strategie + V7-Detail-Daten fuer 4-Panel-Plots."""
    q94, q99 = cv["q94"], cv["q99"]
    results = {}
    for asset, sheet in AKTIEN.items():
        print(f">> {asset} ...", flush=True)
        df = load_excel_data(EXCEL_DATEI, sheet)
        prices = df["Price"].values
        scores = calculate_bsadf_fast(df["Price"], MIN_WINDOW, LOOKBACK_WINDOW)

        bh = run_buy_and_hold(prices)
        mom = run_momentum_strategy(prices)
        v7, _, weights, plog, suplog = run_v7_smart_strategy(
            prices, bsadf_scores=scores, q_warn=q94, q_panic=q99,
            use_directional_filter=True, slippage_mode="dynamic",
        )

        sma20 = pd.Series(prices).rolling(SMA_WINDOW).mean().values
        weights_array = np.array(weights, dtype=float)

        results[asset] = {
            "dates": df.index,
            "prices": prices,
            "sma20": sma20,
            "scores": scores,
            "bh": bh, "mom": mom, "v7": v7,
            "weights": weights_array,
            "plog": plog,
            "suplog": suplog,
            "tr_bh": total_return_pct(bh),
            "tr_mom": total_return_pct(mom),
            "tr_v7": total_return_pct(v7),
            "mdd_bh": max_drawdown_pct(bh),
            "mdd_mom": max_drawdown_pct(mom),
            "mdd_v7": max_drawdown_pct(v7),
        }
    return results, q94, q99


def plot_total_return_bars(results, save_path="fig_total_return_v7.png"):
    """Abb. 4: Gruppierter Bar-Chart der Total Returns je Titel und Strategie."""
    assets = list(results.keys())
    bh_vals = [results[a]["tr_bh"] for a in assets]
    mom_vals = [results[a]["tr_mom"] for a in assets]
    v7_vals = [results[a]["tr_v7"] for a in assets]

    x = np.arange(len(assets))
    w = 0.27

    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w, bh_vals, w, label="Buy & Hold", color=COLOR_BH, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x, mom_vals, w, label="Momentum (SMA20)", color=COLOR_MOM, edgecolor="black", linewidth=0.5)
    b3 = ax.bar(x + w, v7_vals, w, label="V7 (dyn. Slippage)", color=COLOR_V7, edgecolor="black", linewidth=0.5)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + (25 if h >= 0 else -50),
                    f"{h:.1f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(assets)
    ax.set_ylabel("Total Return (%)")
    ax.set_title("Abbildung 4: Total Return Vergleich für alle 5 Aktien — V7 (B = 5'000 Wild-Bootstrap-Schwellen)", fontweight="bold")
    ax.legend(loc="upper right")
    ax.axhline(0, color="black", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {save_path}", flush=True)


def plot_max_drawdown_bars(results, save_path="fig_max_drawdown_v7.png"):
    """Abb. 5: Gruppierter Bar-Chart der Max Drawdowns je Titel und Strategie."""
    assets = list(results.keys())
    bh_vals = [results[a]["mdd_bh"] for a in assets]
    mom_vals = [results[a]["mdd_mom"] for a in assets]
    v7_vals = [results[a]["mdd_v7"] for a in assets]

    x = np.arange(len(assets))
    w = 0.27

    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w, bh_vals, w, label="Buy & Hold", color=COLOR_BH, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x, mom_vals, w, label="Momentum (SMA20)", color=COLOR_MOM, edgecolor="black", linewidth=0.5)
    b3 = ax.bar(x + w, v7_vals, w, label="V7 (dyn. Slippage)", color=COLOR_V7, edgecolor="black", linewidth=0.5)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h - 2,
                    f"{h:.1f}", ha="center", va="top", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(assets)
    ax.set_ylabel("Max Drawdown (%)")
    ax.set_title("Abbildung 5: Max Drawdown Vergleich für alle 5 Aktien — V7 (B = 5'000 Wild-Bootstrap-Schwellen)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.axhline(0, color="black", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {save_path}", flush=True)


def plot_4panel(asset, data, q94, q99, save_path):
    """4-Panel-Plot je Titel: (1) Preis + SMA20 + Panik-Trades, (2) Strategien-Performance,
    (3) BSADF-Score + Wild-Bootstrap-Schwellen, (4) V7-Aktienquote."""
    dates = data["dates"]
    prices = data["prices"]
    sma20 = data["sma20"]
    scores = data["scores"]
    bh, mom, v7 = data["bh"], data["mom"], data["v7"]
    weights = data["weights"]
    plog = data["plog"]

    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1.6, 1.0, 0.6]})
    fig.suptitle(f"4-Panel-Analyse der V7-Strategie bei {asset}\n"
                 f"(MIN_WINDOW = 178, score[t-2], Direktionalfilter, "
                 f"q94 = {q94:.3f} / q99 = {q99:.3f})",
                 fontsize=13, fontweight="bold", y=0.995)

    # === Panel 1: Preis + SMA20 + Panik-Trades ===
    ax1 = axes[0]
    ax1.plot(dates, prices, color=COLOR_PRICE, linewidth=1.0, label="Schlusskurs")
    ax1.plot(dates, sma20, color=COLOR_MOM, linewidth=1.0, alpha=0.7, label="SMA20")
    # Panik-Trades markieren
    if plog:
        panic_dates = [dates[t] for t, _, _, _ in plog]
        panic_prices = [p for _, _, _, p in plog]
        ax1.scatter(panic_dates, panic_prices, marker="v", s=80, color=COLOR_LIMIT,
                    edgecolor="black", linewidth=0.5, zorder=10,
                    label=f"Panik-Exit (n = {len(plog)})")
    ax1.set_title("Panel 1: Preis, SMA20 und Panik-Exit-Trades")
    ax1.set_ylabel("Preis (CHF)")
    ax1.legend(loc="upper left")

    # === Panel 2: Performance-Vergleich ===
    ax2 = axes[1]
    ax2.plot(dates, bh, color=COLOR_BH, linewidth=1.2, label=f"Buy & Hold ({data['tr_bh']:.1f} %)")
    ax2.plot(dates, mom, color=COLOR_MOM, linewidth=1.2, alpha=0.85,
             label=f"Momentum SMA20 ({data['tr_mom']:.1f} %)")
    ax2.plot(dates, v7, color=COLOR_V7, linewidth=1.8,
             label=f"V7 (dyn. Slippage) ({data['tr_v7']:.1f} %)")
    ax2.set_title(f"Panel 2: Portfolio-Performance (Start = 100). "
                  f"V7 MDD = {data['mdd_v7']:.1f} %, B&H MDD = {data['mdd_bh']:.1f} %")
    ax2.set_ylabel("Portfolio-Wert")
    ax2.legend(loc="upper left")
    ax2.set_yscale("log")

    # === Panel 3: BSADF-Score + Wild-Bootstrap-Schwellen ===
    ax3 = axes[2]
    ax3.plot(dates, scores, color=COLOR_BSADF, linewidth=0.8, label="BSADF-Score (PSY 2015)")
    ax3.axhline(q94, color=COLOR_WARN, linestyle="--", linewidth=1.0,
                label=f"q94 = {q94:.3f} (Warn)")
    ax3.axhline(q99, color=COLOR_LIMIT, linestyle="--", linewidth=1.2,
                label=f"q99 = {q99:.3f} (Panik)")
    # Trigger-Bereiche schraffieren
    ax3.fill_between(dates, q99, scores, where=(scores > q99),
                     color=COLOR_LIMIT, alpha=0.25, label="Panik-Trigger")
    ax3.fill_between(dates, q94, scores, where=((scores > q94) & (scores <= q99)),
                     color=COLOR_WARN, alpha=0.25, label="Warn-Bereich")
    ax3.set_title("Panel 3: BSADF-Score und Wild-Bootstrap-Schwellen (B = 5'000)")
    ax3.set_ylabel("BSADF (t-Statistik)")
    ax3.legend(loc="upper left", ncol=2, fontsize=8)
    ax3.set_ylim(min(scores.min(), -0.5) - 0.1, max(scores.max(), q99 + 0.3) + 0.2)

    # === Panel 4: Aktienquote ===
    ax4 = axes[3]
    ax4.fill_between(dates, 0, weights * 100, color=COLOR_QUOTE, alpha=0.55, step="post")
    ax4.set_ylabel("Aktienquote (%)")
    ax4.set_ylim(-3, 105)
    ax4.set_yticks([0, 50, 100])
    ax4.set_title("Panel 4: V7-Aktienquote (0 % / 50 % / 100 %)")
    ax4.set_xlabel("Datum")

    # X-Achse formatieren
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {save_path}", flush=True)


def main():
    print("=" * 70, flush=True)
    print("MAKE PLOTS V7 — Abbildungen 4-7 + 4-Panel-Plots aller 5 Titel", flush=True)
    print("=" * 70, flush=True)

    cv = load_cvs()
    q94, q99 = cv["q94"], cv["q99"]
    print(f"Wild-Bootstrap-Schwellen: q94 = {q94:.4f}, q99 = {q99:.4f}\n", flush=True)

    print(">> Sammle Strategie-Resultate fuer alle 5 Titel ...", flush=True)
    results, q94, q99 = collect_strategy_results(cv)

    print("\n>> Erzeuge Abb. 4 (Total Return Vergleich) ...", flush=True)
    plot_total_return_bars(results, "fig_total_return_v7.png")

    print(">> Erzeuge Abb. 5 (Max Drawdown Vergleich) ...", flush=True)
    plot_max_drawdown_bars(results, "fig_max_drawdown_v7.png")

    print(">> Erzeuge 4-Panel-Plots je Titel ...", flush=True)
    fname_map = {
        "Logitech":        "fig_logitech_4panel.png",
        "Sika":            "fig_sika_4panel.png",
        "Richemont":       "fig_richemont_4panel.png",
        "Lonza":           "fig_lonza_4panel.png",
        "Partners Group":  "fig_pghn_4panel.png",
    }
    for asset, fname in fname_map.items():
        plot_4panel(asset, results[asset], q94, q99, fname)

    print("\n" + "=" * 70, flush=True)
    print("FERTIG. 7 Abbildungen erzeugt:", flush=True)
    print("=" * 70, flush=True)
    for f in ["fig_total_return_v7.png", "fig_max_drawdown_v7.png"] + list(fname_map.values()):
        if os.path.exists(f):
            sz = os.path.getsize(f) / 1024
            print(f"  {f}  ({sz:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()