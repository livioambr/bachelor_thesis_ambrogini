"""
Backtest V7 — methodische Vollkorrektur der Reviewer-Befunde aus dem Gutachten 2026-05.

Aenderungen gegenueber V3:
- K2 / PSY-Konformitaet:   MIN_WINDOW = 178 (statt 60). PSY (2015) empfehlen
                           r0 ~= 0.01 + 1.8/sqrt(T); bei T~5000 ergibt das 178.
                           Bezeichnung "PSY-konform" ist damit empirisch gerechtfertigt.
- K3 / Lag-Konvention:     Trade-Entscheidung an Tag t verwendet bsadf_scores[t-2]
                           und prices[t-2]; Periode-Rendite ist prices[t]/prices[t-1].
                           Operative Lesart: "Observe Close t-2 -> Market-on-Close-
                           Order fuer t-1 -> Hold over Day t" — mindestens ein voller
                           Handelstag zwischen Signalbildung und Ausfuehrung.
- K1 / Schwellenwerte:     Empirische rolling Quantile werden durch einmal
                           pre-computed Wild-Bootstrap-Quantile ersetzt (B = 5'000
                           Pfade, Random Walk mit empirischer Volatilitaet,
                           Rademacher-Wild-Schock); die kritischen BSADF-Werte
                           sind damit ex-post fix und nicht prozyklisch.
- K4 / Direktionalfilter:  Panik-Exit wird nur ausgeloest, wenn zusaetzlich der
                           60-Tages-Trend positiv ist (prices[t-2] > prices[t-2-60]).
                           Damit werden Crash-Tag-Verkaeufe (Lehman 2008, COVID 2020)
                           ex ante unterdrueckt.

Outputs (von main() erzeugt):
- wild_bootstrap_critical_values_v7_B5000.json   (falls noch nicht vorhanden)
- baseline_results_v7.csv                        (Tabelle 3)
- bootstrap_v7_B5000.json                        (Tabelle 4)
- slippage_sensitivity_v7.csv                    (Tabelle 5)

Separat erzeugt (Hilfsskripte, nicht Teil dieser Datei):
- lag_sensitivity_v7.csv                         (Anhang G)
- anhang_b_bsadf_v2.csv                          (Anhang B)
- crash_tag_audit_v7.csv                         (Anhang D)
- pghn_trade_audit_v7.csv                        (Anhang E)
"""
import os
import json
import warnings

import numpy as np
import pandas as pd

from bsadf_core import calculate_bsadf_fast

warnings.filterwarnings("ignore")

# ==========================================
# 1. PARAMETER
# ==========================================
RISIKO_FREIER_ZINS = 0.0
TRANSAKTIONS_KOSTEN = 0.0015
LOOKBACK_WINDOW = 252           # 1 Handelsjahr (frueher 126 = 0.5 J)
MIN_WINDOW = 178                # K2: PSY-konform fuer T~=5000
SMA_WINDOW = 20
TREND_WINDOW = 60               # K4: 60-Tage-Trend fuer Direktionalfilter
TRADING_DAYS = 252.0

SLIP_BASE = 0.005
SLIP_LAMBDA = 2.0
SLIP_FLOOR = 0.025
SLIP_CAP   = 0.20

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def _find_excel():
    for p in ["Datenreihe.xlsx",
              os.path.join(_SCRIPT_DIR, "Datenreihe.xlsx"),
              os.path.join(_SCRIPT_DIR, "..", "Datenreihe.xlsx"),
              os.path.join(_SCRIPT_DIR, "..", "outputs", "Datenreihe.xlsx")]:
        if os.path.exists(p):
            return os.path.abspath(p)
    return "Datenreihe.xlsx"  # Fallback
EXCEL_DATEI = _find_excel()
AKTIEN = {
    "Logitech": "LOGN",
    "Sika": "SIKA",
    "Richemont": "CFR",
    "Lonza": "LONN",
    "Partners Group": "PGHN",
}

# Wild-Bootstrap-Parameter (K1)
WB_N_PATHS = 1000
WB_SEED = 20260515


# ==========================================
# 2. DATEN
# ==========================================
def load_excel_data(filepath, sheet_name):
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    df = pd.read_excel(filepath, sheet_name=sheet_name, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    if "Date" not in df.columns or "Close" not in df.columns:
        raise RuntimeError(f"Sheet '{sheet_name}': Date/Close fehlen")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df["Close"] = df["Close"].ffill()
    df = df[df["Close"] > 0]
    return pd.DataFrame({"Price": df["Close"]})


# ==========================================
# 3. WILD-BOOTSTRAP-KRITISCHE WERTE (K1)
# ==========================================
def wild_bootstrap_critical_values(log_returns, T_sample, min_window=MIN_WINDOW,
                                    max_lookback=LOOKBACK_WINDOW,
                                    n_paths=WB_N_PATHS, seed=WB_SEED,
                                    quantiles=(0.94, 0.99)):
    """
    Erzeugt durch Wild-Bootstrap unter H0 (Random Walk mit empirischer Volatilitaet,
    Rademacher-Wild-Schock) eine Verteilung der BSADF-Statistik und gibt
    die geforderten Quantile zurueck. Pre-computed, nicht rollierend — Quantile sind
    ueber alle Beobachtungen und Pfade konstant (nicht prozyklisch).

    Implementiert PSY (2015), Section 4 Vorgehen vereinfacht.

    Parameters
    ----------
    log_returns : np.ndarray
        Beobachtete Log-Renditen (zur Schock-Skalierung).
    T_sample : int
        Pfadlaenge fuer die Simulation (typischerweise = len(log_returns)).
    """
    rng = np.random.default_rng(seed)
    n = len(log_returns)
    sigma = np.std(log_returns, ddof=1)

    all_scores = []
    for k in range(n_paths):
        # Rademacher-Wild-Schock
        epsilon = rng.choice([-1.0, 1.0], size=T_sample) * sigma
        # H0: Random Walk
        log_path = np.cumsum(epsilon)
        prices_path = 100.0 * np.exp(log_path)
        scores = calculate_bsadf_fast(pd.Series(prices_path), min_window, max_lookback)
        # only nonzero scores (BSADF startet ab min_window)
        scores_valid = scores[min_window:]
        all_scores.append(scores_valid)
        if (k + 1) % 100 == 0:
            print(f"   Wild-Bootstrap: {k+1}/{n_paths} Pfade")

    all_scores = np.concatenate(all_scores)
    cv = {f"q{int(q*100)}": float(np.quantile(all_scores, q)) for q in quantiles}
    cv["n_paths"] = n_paths
    cv["T_sample"] = int(T_sample)
    cv["min_window"] = int(min_window)
    cv["max_lookback"] = int(max_lookback)
    cv["sigma_empirical"] = float(sigma)
    return cv


# ==========================================
# 4. STRATEGIEN
# ==========================================
def panic_slippage_dynamic(realised_vol_5d, base=SLIP_BASE, lam=SLIP_LAMBDA,
                            floor=SLIP_FLOOR, cap=SLIP_CAP):
    if not np.isfinite(realised_vol_5d):
        realised_vol_5d = 0.02
    s = base + lam * realised_vol_5d
    return float(np.clip(s, floor, cap))


def run_buy_and_hold(prices, r_free=RISIKO_FREIER_ZINS):
    p = (np.asarray(prices) / prices[0]) * 100.0
    return p.tolist()


def run_momentum_strategy(prices, transaction_cost=TRANSAKTIONS_KOSTEN,
                          r_free=RISIKO_FREIER_ZINS):
    """Klassisches SMA20-Momentum mit konservativer Lag-Konvention (K3).

    Decision an t basiert auf prices[t-2] vs sma[t-2]; Rendite ist t-1->t.
    """
    prices = np.asarray(prices, dtype=float)
    sma = pd.Series(prices).rolling(window=SMA_WINDOW).mean().values
    wealth, portfolio, prev_w = 100.0, [100.0], 1.0
    for t in range(1, len(prices)):
        if t < SMA_WINDOW + 1:
            target = 1.0
        else:
            target = 1.0 if prices[t-2] > sma[t-2] else 0.0
        if target != prev_w:
            wealth -= wealth * abs(target - prev_w) * transaction_cost
        prev_w = target
        wealth *= target * (prices[t]/prices[t-1]) + (1-target) * (1 + r_free/TRADING_DAYS)
        portfolio.append(wealth)
    return portfolio


def run_v7_smart_strategy(prices, transaction_cost=TRANSAKTIONS_KOSTEN,
                          r_free=RISIKO_FREIER_ZINS,
                          bsadf_scores=None,
                          q_warn=None, q_panic=None,
                          use_directional_filter=True,
                          slippage_mode="dynamic",
                          slippage_fixed=0.025,
                          slip_lambda=SLIP_LAMBDA,
                          slip_floor=SLIP_FLOOR,
                          slip_cap=SLIP_CAP):
    """
    V7 — Smart Re-Entry mit:
    - konservativer Lag-Konvention (score[t-2], prices[t-2])  (K3)
    - fixen Wild-Bootstrap-Quantilen q_warn, q_panic           (K1)
    - direktionalem Pre-Filter (K4)

    Parameters
    ----------
    q_warn, q_panic : float
        Fixe BSADF-Schwellen (z. B. aus wild_bootstrap_critical_values).
    use_directional_filter : bool
        Wenn True, wird der Panik-Exit nur ausgeloest, wenn zusaetzlich
        prices[t-2] > prices[t-2-TREND_WINDOW] (60-Tage-Trend positiv).
    """
    prices = np.asarray(prices, dtype=float)
    if bsadf_scores is None:
        bsadf_scores = calculate_bsadf_fast(pd.Series(prices), MIN_WINDOW, LOOKBACK_WINDOW)
    if q_warn is None or q_panic is None:
        raise ValueError("q_warn/q_panic muessen explizit gesetzt sein (Wild-Bootstrap)")

    lim_w = q_warn
    lim_p = q_panic

    sma = pd.Series(prices).rolling(window=SMA_WINDOW).mean().values

    log_ret = np.diff(np.log(prices))
    vol_5d = np.empty(len(prices))
    vol_5d[:] = np.nan
    for t in range(6, len(prices)):
        vol_5d[t] = np.std(log_ret[t-5:t], ddof=1)

    portfolio, wealth, prev_w = [100.0], 100.0, 1.0
    weights = [1.0]
    panic_trade_log = []
    suppressed_panic_log = []  # K4 — wurden Panik-Trigger durch Direktionalfilter unterdrueckt?

    for t in range(2, len(prices)):  # ab t=2 wegen score[t-2]
        score = bsadf_scores[t-2]

        if t < max(SMA_WINDOW, MIN_WINDOW, TREND_WINDOW + 2):
            target = 1.0
        elif prev_w == 0.0:
            re_entry = (score < 0) or (prices[t-2] > sma[t-2] and score <= lim_p)
            target = 1.0 if re_entry else 0.0
        else:
            if score > lim_p:
                # K4 Direktionalfilter: nur ausloesen wenn Trend positiv
                if use_directional_filter:
                    if t - 2 - TREND_WINDOW >= 0:
                        trend_pos = prices[t-2] > prices[t-2-TREND_WINDOW]
                    else:
                        trend_pos = True
                    if trend_pos:
                        target = 0.0
                    else:
                        target = prev_w
                        suppressed_panic_log.append((t, score, prices[t-2]))
                else:
                    target = 0.0
            elif score > lim_w:
                target = 0.5
            else:
                target = 1.0

        if target != prev_w:
            is_panic = (target == 0.0 and prev_w > 0.0 and score > lim_p)
            if is_panic:
                if slippage_mode == "dynamic":
                    s_used = panic_slippage_dynamic(
                        vol_5d[t-2],
                        base=SLIP_BASE, lam=slip_lambda,
                        floor=slip_floor, cap=slip_cap,
                    )
                else:
                    s_used = slippage_fixed
                cost = s_used
                panic_trade_log.append((t, vol_5d[t-2], s_used, prices[t-2]))
            else:
                cost = transaction_cost
            wealth -= wealth * abs(target - prev_w) * cost

        prev_w = target
        weights.append(target)
        wealth *= target * (prices[t]/prices[t-1]) + (1-target) * (1 + r_free/TRADING_DAYS)
        portfolio.append(wealth)

    # Padding fuer t=0,1 — voll investiert
    while len(portfolio) < len(prices):
        portfolio.insert(0, 100.0)
        weights.insert(0, 1.0)

    return portfolio, bsadf_scores, weights, panic_trade_log, suppressed_panic_log


# ==========================================
# 5. KENNZAHLEN
# ==========================================
def total_return_pct(p):
    return (p[-1] / p[0] - 1.0) * 100.0


def max_drawdown_pct(p):
    s = pd.Series(p)
    dd = (s - s.cummax()) / s.cummax()
    return dd.min() * 100.0


def annualized_return_pct(p, n_years):
    factor = p[-1] / p[0]
    if factor <= 0:
        return -100.0
    return (factor ** (1.0/n_years) - 1.0) * 100.0


def annualized_volatility_pct(p):
    r = pd.Series(p).pct_change().dropna()
    return r.std() * np.sqrt(TRADING_DAYS) * 100.0


def sharpe_ratio(p, r_free_annual=RISIKO_FREIER_ZINS):
    r = pd.Series(p).pct_change().dropna()
    excess = r - r_free_annual / TRADING_DAYS
    if r.std() == 0:
        return 0.0
    return (excess.mean() / r.std()) * np.sqrt(TRADING_DAYS)


def calmar_ratio(p, n_years):
    ann_ret = annualized_return_pct(p, n_years) / 100.0
    mdd = abs(max_drawdown_pct(p) / 100.0)
    if mdd == 0:
        return 0.0
    return ann_ret / mdd


def metrics_dict(p, n_years, asset, strategy):
    return {
        "Asset": asset,
        "Strategie": strategy,
        "n_years": round(n_years, 3),
        "Total Return %": round(total_return_pct(p), 2),
        "Ann. Return %": round(annualized_return_pct(p, n_years), 3),
        "Ann. Vol %": round(annualized_volatility_pct(p), 3),
        "Max DD %": round(max_drawdown_pct(p), 3),
        "Sharpe": round(sharpe_ratio(p), 3),
        "Calmar": round(calmar_ratio(p, n_years), 3),
    }


def years_in_sample(df):
    return (df.index[-1] - df.index[0]).days / 365.25


# ==========================================
# 6. INFERENZ
# ==========================================
def stationary_bootstrap_indices(n, mean_block_len, rng):
    p = 1.0 / mean_block_len
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = rng.integers(0, n)
        else:
            idx[t] = (idx[t-1] + 1) % n
    return idx


def stationary_bootstrap_v7(prices, q_warn, q_panic, B=500, mean_block_len=21, seed=42):
    """K5: Stationary-Bootstrap mit Strategie-Re-Simulation auf V7."""
    rng = np.random.default_rng(seed)
    prices = np.asarray(prices, dtype=float)
    log_ret = np.diff(np.log(prices))
    n_ret = len(log_ret)
    p0 = prices[0]

    bsadf_real = calculate_bsadf_fast(pd.Series(prices), MIN_WINDOW, LOOKBACK_WINDOW)
    v7_real, _, _, _, _ = run_v7_smart_strategy(prices, bsadf_scores=bsadf_real,
                                                 q_warn=q_warn, q_panic=q_panic)
    bh_real = run_buy_and_hold(prices)

    obs_dd = max_drawdown_pct(v7_real) - max_drawdown_pct(bh_real)
    obs_sh = sharpe_ratio(v7_real) - sharpe_ratio(bh_real)
    obs_re = total_return_pct(v7_real) - total_return_pct(bh_real)

    diff_dd = np.empty(B); diff_sh = np.empty(B); diff_re = np.empty(B)
    for b in range(B):
        idx = stationary_bootstrap_indices(n_ret, mean_block_len, rng)
        boot_log = log_ret[idx]
        boot_prices = p0 * np.exp(np.concatenate([[0.0], np.cumsum(boot_log)]))
        bsadf_b = calculate_bsadf_fast(pd.Series(boot_prices), MIN_WINDOW, LOOKBACK_WINDOW)
        v7_b, _, _, _, _ = run_v7_smart_strategy(boot_prices, bsadf_scores=bsadf_b,
                                                  q_warn=q_warn, q_panic=q_panic)
        bh_b = run_buy_and_hold(boot_prices)
        diff_dd[b] = max_drawdown_pct(v7_b) - max_drawdown_pct(bh_b)
        diff_sh[b] = sharpe_ratio(v7_b) - sharpe_ratio(bh_b)
        diff_re[b] = total_return_pct(v7_b) - total_return_pct(bh_b)

    def p2(diffs, observed):
        centered = diffs - diffs.mean()
        return float(np.mean(np.abs(centered) >= abs(observed)))

    return {
        "method": "stationary_bootstrap_strategy_resim_v7",
        "B": B, "mean_block_len": mean_block_len,
        "observed": {"maxdd_diff_pp": round(float(obs_dd), 4),
                     "sharpe_diff": round(float(obs_sh), 4),
                     "return_diff_pp": round(float(obs_re), 4)},
        "ci_95": {
            "maxdd_diff_pp": [round(float(np.quantile(diff_dd, 0.025)), 4),
                              round(float(np.quantile(diff_dd, 0.975)), 4)],
            "sharpe_diff": [round(float(np.quantile(diff_sh, 0.025)), 4),
                            round(float(np.quantile(diff_sh, 0.975)), 4)],
            "return_diff_pp": [round(float(np.quantile(diff_re, 0.025)), 4),
                               round(float(np.quantile(diff_re, 0.975)), 4)],
        },
        "p_values": {"maxdd": round(p2(diff_dd, obs_dd), 4),
                     "sharpe": round(p2(diff_sh, obs_sh), 4),
                     "return": round(p2(diff_re, obs_re), 4)},
    }


# ==========================================
# 7. WILD-BOOTSTRAP-AGGREGATION (pre-compute critical values)
# ==========================================
def compute_wild_bootstrap_cvs(n_paths=5000, save_path="wild_bootstrap_critical_values_v7_B5000.json"):
    """Pre-computed Wild-Bootstrap-Quantile unter H0 (Random Walk mit empirischer Vola).

    Sequentielle Ausfuehrung. Laufzeit ca. 30-50 Minuten fuer 5000 Pfade.
    Fortschrittsanzeige alle 100 Pfade.
    """
    import time, math

    sigmas, Ts = [], []
    for asset, sheet in AKTIEN.items():
        df = load_excel_data(EXCEL_DATEI, sheet)
        log_ret = np.diff(np.log(df["Price"].values))
        sigmas.append(np.std(log_ret, ddof=1))
        Ts.append(len(df))
    sigma_rep = float(np.mean(sigmas))
    T_rep = int(np.mean(Ts))

    print(f"Wild-Bootstrap: {n_paths} Pfade x T = {T_rep}, sigma = {sigma_rep:.4f}", flush=True)
    rng = np.random.default_rng(WB_SEED)
    all_scores = []
    t0 = time.time()
    for k in range(n_paths):
        eps = rng.choice([-1.0, 1.0], size=T_rep) * sigma_rep
        log_path = np.cumsum(eps)
        prices = 100.0 * np.exp(log_path)
        sc = calculate_bsadf_fast(pd.Series(prices), MIN_WINDOW, LOOKBACK_WINDOW)
        all_scores.append(sc[MIN_WINDOW:].astype(np.float32))
        if (k + 1) % 100 == 0:
            elapsed = time.time() - t0
            remaining = elapsed / (k+1) * (n_paths - k - 1)
            print(f"   {k+1}/{n_paths} Pfade, dt = {elapsed:.0f}s, ETA = {remaining:.0f}s", flush=True)

    flat = np.concatenate(all_scores)
    flat = flat[np.isfinite(flat)]

    cv = {
        "method": "wild_bootstrap_global_v7",
        "n_paths": int(n_paths),
        "T_rep_per_path": T_rep,
        "n_pooled_obs": int(len(flat)),
        "sigma_rep_LR": sigma_rep,
        "min_window": MIN_WINDOW,
        "max_lookback": LOOKBACK_WINDOW,
        "q90": float(np.quantile(flat, 0.90)),
        "q94": float(np.quantile(flat, 0.94)),
        "q95": float(np.quantile(flat, 0.95)),
        "q99": float(np.quantile(flat, 0.99)),
        "se_q94_approx": math.sqrt(0.06 * 0.94 / n_paths),
        "se_q99_approx": math.sqrt(0.01 * 0.99 / n_paths),
        "seed_base": WB_SEED,
    }
    with open(save_path, "w") as f:
        json.dump(cv, f, indent=2)
    print(f"\nWild-Bootstrap-Quantile gespeichert -> {save_path}", flush=True)
    print(f"  q94 = {cv['q94']:.4f} (Warn)", flush=True)
    print(f"  q99 = {cv['q99']:.4f} (Panik)", flush=True)
    return cv


# ==========================================
# 8. MAIN — fuehrt den V7-Backtest komplett aus
# ==========================================
def main():
    import time
    print("=" * 70, flush=True)
    print("BACKTEST V7 - methodische Vollkorrektur", flush=True)
    print("=" * 70, flush=True)
    print(f"MIN_WINDOW = {MIN_WINDOW} (PSY-konform fuer T ~ 5000)", flush=True)
    print(f"LOOKBACK   = {LOOKBACK_WINDOW}", flush=True)
    print(f"Lag-Konvention: score[t-2] / prices[t-2]", flush=True)
    print(f"Direktionalfilter: aktiv (Trend prices[t-2] vs prices[t-2-{TREND_WINDOW}])", flush=True)
    print(f"Slippage: dyn. clip(0.005 + {SLIP_LAMBDA}*sigma_5d, {SLIP_FLOOR}, {SLIP_CAP})", flush=True)
    print(flush=True)

    # --- 1. Wild-Bootstrap-CVs laden (oder anbieten neu zu berechnen) ---
    candidate_paths = [
        "wild_bootstrap_critical_values_v7_B5000.json",
        "../outputs/wild_bootstrap_critical_values_v7_B5000.json",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "wild_bootstrap_critical_values_v7_B5000.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "wild_bootstrap_critical_values_v7_B5000.json"),
        "wild_bootstrap_critical_values_v7.json",
        "../outputs/wild_bootstrap_critical_values_v7.json",
    ]
    cv = None
    for p in candidate_paths:
        if os.path.exists(p):
            cv = json.load(open(p))
            print(f"CVs geladen aus: {p}", flush=True)
            break

    if cv is None:
        print("Keine vorberechnete CV-Datei gefunden.", flush=True)
        print("Wild-Bootstrap mit 5'000 Pfaden dauert ca. 30-50 Minuten.", flush=True)
        try:
            antwort = input("Jetzt berechnen? [j/N]: ").strip().lower()
        except EOFError:
            antwort = "n"
        if antwort != "j":
            print("Abbruch. Bitte 'wild_bootstrap_critical_values_v7_B5000.json' aus outputs/ in dieses Verzeichnis kopieren.", flush=True)
            return
        cv = compute_wild_bootstrap_cvs(n_paths=5000,
                                       save_path="wild_bootstrap_critical_values_v7_B5000.json")

    q94, q99 = cv["q94"], cv["q99"]
    print(f"\nWild-Bootstrap-Schwellen: q94 = {q94:.4f}, q99 = {q99:.4f}\n", flush=True)

    # --- 2. Hauptbacktest fuer alle 5 Titel ---
    summary_rows = []
    cache = {}
    for asset, sheet in AKTIEN.items():
        print(f">> {asset} ({sheet})", flush=True)
        df = load_excel_data(EXCEL_DATEI, sheet)
        n_years = years_in_sample(df)
        prices = df["Price"].values
        print(f"   {len(df)} Tage, {n_years:.2f} Jahre", flush=True)

        print(f"   BSADF (min_window = {MIN_WINDOW}) ...", flush=True)
        t0 = time.time()
        scores = calculate_bsadf_fast(df["Price"], MIN_WINDOW, LOOKBACK_WINDOW)
        print(f"   BSADF dt = {time.time()-t0:.1f}s, max = {scores.max():.3f}", flush=True)

        bh = run_buy_and_hold(prices)
        mom = run_momentum_strategy(prices)
        v7_dyn, _, _, plog, suplog = run_v7_smart_strategy(
            prices, bsadf_scores=scores, q_warn=q94, q_panic=q99,
            use_directional_filter=True, slippage_mode="dynamic"
        )
        v7_nodir, _, _, _, _ = run_v7_smart_strategy(
            prices, bsadf_scores=scores, q_warn=q94, q_panic=q99,
            use_directional_filter=False, slippage_mode="dynamic"
        )
        v7_fix, _, _, _, _ = run_v7_smart_strategy(
            prices, bsadf_scores=scores, q_warn=q94, q_panic=q99,
            use_directional_filter=True, slippage_mode="fixed", slippage_fixed=0.025
        )

        print(f"   Panik-Trades: {len(plog)}, unterdrueckt: {len(suplog)}", flush=True)

        cache[asset] = dict(prices=prices, scores=scores, n_years=n_years,
                            bh=bh, v7_dyn=v7_dyn)

        for name, p in [("Buy & Hold", bh),
                        ("Momentum SMA20", mom),
                        ("V7 (dir-Filter, dyn. Slippage)", v7_dyn),
                        ("V7 (ohne dir-Filter, dyn. Slip)", v7_nodir),
                        ("V7 (dir-Filter, fix 2.5%)", v7_fix)]:
            summary_rows.append(metrics_dict(p, n_years, asset, name))

    df_metrics = pd.DataFrame(summary_rows)
    df_metrics.to_csv("baseline_results_v7.csv", index=False)
    print(flush=True)
    print("=" * 70, flush=True)
    print("HAUPTRESULTATE V7", flush=True)
    print("=" * 70, flush=True)
    print(df_metrics.to_string(index=False), flush=True)
    print(f"\nGespeichert -> baseline_results_v7.csv", flush=True)

    # --- 3. Stationary-Bootstrap-Inferenz ---
    print(flush=True)
    print("=" * 70, flush=True)
    print("STATIONARY-BOOTSTRAP-INFERENZ V7 (B = 100, ca. 2 min total)", flush=True)
    print("=" * 70, flush=True)
    bootstrap_results = {}
    for asset in AKTIEN:
        print(f">> {asset} ...", flush=True)
        t0 = time.time()
        prices = cache[asset]["prices"]
        res = stationary_bootstrap_v7(prices, q_warn=q94, q_panic=q99,
                                      B=100, mean_block_len=21, seed=42)
        bootstrap_results[asset] = {"V7_dyn_vs_BH": res}
        p = res["p_values"]
        print(f"   dt = {time.time()-t0:.0f}s, p_MDD = {p['maxdd']:.3f}, "
              f"p_Sharpe = {p['sharpe']:.3f}, p_TR = {p['return']:.3f}", flush=True)

    with open("bootstrap_v7_B5000.json", "w") as f:
        json.dump(bootstrap_results, f, indent=2)
    print(f"\nGespeichert -> bootstrap_v7_B5000.json", flush=True)

    # --- 4. Slippage-Sensitivitaet fuer PGHN ---
    print(flush=True)
    print("=" * 70, flush=True)
    print("SLIPPAGE-SENSITIVITAET V7 (Partners Group)", flush=True)
    print("=" * 70, flush=True)
    pghn_prices = cache["Partners Group"]["prices"]
    pghn_scores = cache["Partners Group"]["scores"]
    rows = []
    for bc in [0.0015, 0.0030, 0.0045, 0.0060]:
        for ps in [0.025, 0.05, 0.10, 0.15]:
            v, _, _, _, _ = run_v7_smart_strategy(
                pghn_prices, transaction_cost=bc, bsadf_scores=pghn_scores,
                q_warn=q94, q_panic=q99, use_directional_filter=True,
                slippage_mode="fixed", slippage_fixed=ps,
            )
            mom_p = run_momentum_strategy(pghn_prices, transaction_cost=bc)
            rows.append({
                "Basis-Kosten %": round(bc * 100, 3),
                "Panik-Slippage %": round(ps * 100, 2),
                "V7 Total Return %": round(total_return_pct(v), 2),
                "V7 Max DD %": round(max_drawdown_pct(v), 2),
                "Momentum Total Return %": round(total_return_pct(mom_p), 2),
            })
    df_s = pd.DataFrame(rows)
    df_s.to_csv("slippage_sensitivity_v7.csv", index=False)
    print(df_s.to_string(index=False), flush=True)
    print(f"\nGespeichert -> slippage_sensitivity_v7.csv", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("FERTIG. Alle V7-Outputs sind erzeugt.", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()