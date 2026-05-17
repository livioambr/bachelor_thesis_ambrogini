"""
make_anhang_outputs.py — Erzeugt die vier Anhang-CSVs der V7-Bachelorarbeit.

Outputs (jeweils im Verzeichnis dieses Skripts):
    anhang_b_bsadf_v2.csv     (Anhang B, Tabelle Anhang 1) — BSADF-max ueber
                              min_window in {30, 60, 120} mit Datum der Maxima.
                              max_lookback = 126 (V3-Vorgaenger-Konfig, siehe
                              Caption Tabelle Anhang 1).
    crash_tag_audit_v7.csv    (Anhang D, Tabelle Anhang 3) — Aggregierte
                              Klassifikation der Panik-Trades je Titel
                              (Bubble / Indifferent / Crash / unterdrueckt).
    pghn_trade_audit_v7.csv   (Anhang E, Tabelle Anhang 4) — Einzelne Panik-
                              Trades der Partners Group inkl. 6-M-Bwd/Fwd.
                              Konvention der Original-Auditierung:
                                  Datum  = idx[t]      (Action-Day)
                                  Preis  = prices[t-2] (Signal-Day-Close)
                                  Bwd 6M = 126 Handelstage rueckwaerts
                                  Fwd 6M = 125 Handelstage vorwaerts
    lag_sensitivity_v7.csv    (Anhang G, Tabelle Anhang 6) — V7-Performance
                              unter score[t-1], score[t-2], score[t-3].

Voraussetzungen:
    Datenreihe.xlsx, bsadf_core.py und backtest_v7.py liegen im selben
    Verzeichnis wie dieses Skript.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from bsadf_core import calculate_bsadf_fast
from backtest_v7 import (
    AKTIEN, EXCEL_DATEI, MIN_WINDOW, LOOKBACK_WINDOW,
    SMA_WINDOW, TREND_WINDOW, RISIKO_FREIER_ZINS,
    TRANSAKTIONS_KOSTEN, TRADING_DAYS,
    SLIP_BASE, SLIP_LAMBDA, SLIP_FLOOR, SLIP_CAP,
    load_excel_data, run_buy_and_hold, run_v7_smart_strategy,
    panic_slippage_dynamic,
    total_return_pct, max_drawdown_pct, sharpe_ratio,
)

# Wild-Bootstrap-Schwellen (B = 5'000, siehe Anhang F)
Q_WARN = 0.1010
Q_PANIC = 0.9949

# 6-Monats-Performance-Fenster (Konvention der Original-Auditierung in der
# Thesis: 126 Tage rueckwaerts, 125 vorwaerts).
WIN_BWD = 126
WIN_FWD = 125

# Anhang B: max_lookback der V3-Vorgaenger-Konfig (Caption Tabelle Anhang 1).
ANHANG_B_LOOKBACK = 126


def classify(bwd_pct):
    """Anhang-D/E-Klassifikation."""
    if bwd_pct > 20.0:
        return "Bubble"
    if bwd_pct < -10.0:
        return "Crash"
    return "Indifferent"


def pct(num, denom):
    return (num / denom - 1.0) * 100.0


# ---------------------------------------------------------------------------
# Anhang B
# ---------------------------------------------------------------------------
def build_anhang_b():
    rows = []
    for asset, sheet in AKTIEN.items():
        df = load_excel_data(EXCEL_DATEI, sheet)
        idx = df.index
        scores_by_mw = {}
        for mw in (30, 60, 120):
            sc = calculate_bsadf_fast(df["Price"], mw, ANHANG_B_LOOKBACK)
            scores_by_mw[mw] = sc
        date_30 = idx[int(np.argmax(scores_by_mw[30]))].strftime("%Y-%m-%d")
        date_60 = idx[int(np.argmax(scores_by_mw[60]))].strftime("%Y-%m-%d")
        rows.append({
            "Asset": asset,
            "max_bsadf_mw30": round(float(scores_by_mw[30].max()), 3),
            "max_bsadf_mw60": round(float(scores_by_mw[60].max()), 3),
            "max_bsadf_mw120": round(float(scores_by_mw[120].max()), 3),
            "datum_max_mw30": date_30,
            "datum_max_mw60": date_60,
        })
    out = pd.DataFrame(rows)
    out.to_csv("anhang_b_bsadf_v2.csv", index=False)
    print("anhang_b_bsadf_v2.csv geschrieben")
    return out


# ---------------------------------------------------------------------------
# Anhang D + E
# ---------------------------------------------------------------------------
def build_anhang_d_and_e():
    agg_rows = []
    pghn_detail_rows = []

    for asset, sheet in AKTIEN.items():
        df = load_excel_data(EXCEL_DATEI, sheet)
        idx = df.index
        prices = df["Price"].values
        scores = calculate_bsadf_fast(df["Price"], MIN_WINDOW, LOOKBACK_WINDOW)

        _, _, _, panic_log, suppressed_log = run_v7_smart_strategy(
            prices, bsadf_scores=scores,
            q_warn=Q_WARN, q_panic=Q_PANIC,
            use_directional_filter=True, slippage_mode="dynamic",
        )

        n_bubble, n_indiff, n_crash = 0, 0, 0
        for (t, vol5, slip, _price_t2_from_log) in panic_log:
            d_action = t
            d_signal = t - 2

            bwd = (pct(prices[d_action], prices[d_action - WIN_BWD])
                   if d_action - WIN_BWD >= 0 else float("nan"))
            fwd = (pct(prices[d_action + WIN_FWD], prices[d_action])
                   if d_action + WIN_FWD < len(prices) else float("nan"))

            cls = classify(bwd) if not np.isnan(bwd) else "Indifferent"
            if cls == "Bubble":
                n_bubble += 1
            elif cls == "Crash":
                n_crash += 1
            else:
                n_indiff += 1

            if asset == "Partners Group":
                pghn_detail_rows.append({
                    "Datum": idx[d_action].strftime("%d.%m.%Y"),
                    "Preis_CHF": round(float(prices[d_signal]), 1),
                    "BSADF": round(float(scores[d_signal]), 3),
                    "vol_5d": round(float(vol5), 4),
                    "Slip_%": round(float(slip) * 100, 2),
                    "Bwd_6M_%": round(bwd, 1),
                    "Fwd_6M_%": (round(fwd, 1) if not np.isnan(fwd) else ""),
                    "Klassifikation": cls,
                })

        agg_rows.append({
            "Asset": asset,
            "V7_Panik_Trades": len(panic_log),
            "Crash_Tag_Bwd_unter_minus10%": n_crash,
            "Bubble_Tag_Bwd_ueber_plus20%": n_bubble,
            "Indifferent": n_indiff,
            "Unterdrueckte_Trigger": len(suppressed_log),
        })

    df_agg = pd.DataFrame(agg_rows)
    df_agg.loc[len(df_agg)] = {
        "Asset": "Summe",
        "V7_Panik_Trades": int(df_agg["V7_Panik_Trades"].sum()),
        "Crash_Tag_Bwd_unter_minus10%": int(df_agg["Crash_Tag_Bwd_unter_minus10%"].sum()),
        "Bubble_Tag_Bwd_ueber_plus20%": int(df_agg["Bubble_Tag_Bwd_ueber_plus20%"].sum()),
        "Indifferent": int(df_agg["Indifferent"].sum()),
        "Unterdrueckte_Trigger": int(df_agg["Unterdrueckte_Trigger"].sum()),
    }
    df_agg.to_csv("crash_tag_audit_v7.csv", index=False)
    print("crash_tag_audit_v7.csv geschrieben")

    df_pghn = pd.DataFrame(pghn_detail_rows)
    df_pghn.to_csv("pghn_trade_audit_v7.csv", index=False)
    print("pghn_trade_audit_v7.csv geschrieben")
    return df_agg, df_pghn


# ---------------------------------------------------------------------------
# Anhang G
# ---------------------------------------------------------------------------
def run_v7_with_lag(prices, scores, lag):
    prices = np.asarray(prices, dtype=float)
    sma = pd.Series(prices).rolling(window=SMA_WINDOW).mean().values
    log_ret = np.diff(np.log(prices))
    vol_5d = np.empty(len(prices))
    vol_5d[:] = np.nan
    for t in range(6, len(prices)):
        vol_5d[t] = np.std(log_ret[t - 5:t], ddof=1)

    portfolio, wealth, prev_w = [100.0], 100.0, 1.0
    n_panic = 0
    for t in range(lag, len(prices)):
        score = scores[t - lag]
        if t < max(SMA_WINDOW, MIN_WINDOW, TREND_WINDOW + lag):
            target = 1.0
        elif prev_w == 0.0:
            re_entry = (score < 0) or (prices[t - lag] > sma[t - lag] and score <= Q_PANIC)
            target = 1.0 if re_entry else 0.0
        else:
            if score > Q_PANIC:
                if t - lag - TREND_WINDOW >= 0:
                    trend_pos = prices[t - lag] > prices[t - lag - TREND_WINDOW]
                else:
                    trend_pos = True
                target = 0.0 if trend_pos else prev_w
            elif score > Q_WARN:
                target = 0.5
            else:
                target = 1.0
        if target != prev_w:
            is_panic = (target == 0.0 and prev_w > 0.0 and score > Q_PANIC)
            if is_panic:
                cost = panic_slippage_dynamic(
                    vol_5d[t - lag], base=SLIP_BASE, lam=SLIP_LAMBDA,
                    floor=SLIP_FLOOR, cap=SLIP_CAP,
                )
                n_panic += 1
            else:
                cost = TRANSAKTIONS_KOSTEN
            wealth -= wealth * abs(target - prev_w) * cost
        prev_w = target
        wealth *= (target * (prices[t] / prices[t - 1])
                   + (1 - target) * (1 + RISIKO_FREIER_ZINS / TRADING_DAYS))
        portfolio.append(wealth)
    while len(portfolio) < len(prices):
        portfolio.insert(0, 100.0)
    return portfolio, n_panic


def build_anhang_g():
    rows = []
    for asset, sheet in AKTIEN.items():
        df = load_excel_data(EXCEL_DATEI, sheet)
        prices = df["Price"].values
        scores = calculate_bsadf_fast(df["Price"], MIN_WINDOW, LOOKBACK_WINDOW)
        bh = run_buy_and_hold(prices)
        bh_tr = round(total_return_pct(bh), 2)
        for lag in (1, 2, 3):
            p, npan = run_v7_with_lag(prices, scores, lag)
            rows.append({
                "Asset": asset,
                "Lag": f"score[t-{lag}]",
                "Total_Return_%": round(total_return_pct(p), 2),
                "Max_Drawdown_%": round(max_drawdown_pct(p), 2),
                "Sharpe": round(sharpe_ratio(p), 3),
                "Panik_Trades": npan,
                "BuyHold_Total_Return_%": bh_tr,
            })
    out = pd.DataFrame(rows)
    out.to_csv("lag_sensitivity_v7.csv", index=False)
    print("lag_sensitivity_v7.csv geschrieben")
    return out


def main():
    os.chdir(SCRIPT_DIR)
    print("== Anhang B (BSADF-Diagnostik, max_lookback = 126) ==")
    print(build_anhang_b().to_string(index=False))
    print()
    print("== Anhang D + E (Crash-Tag-Audit + PGHN-Trade-Audit) ==")
    agg, det = build_anhang_d_and_e()
    print(agg.to_string(index=False))
    print()
    print(det.to_string(index=False))
    print()
    print("== Anhang G (Lag-Sensitivitaet) ==")
    print(build_anhang_g().to_string(index=False))


if __name__ == "__main__":
    main()
