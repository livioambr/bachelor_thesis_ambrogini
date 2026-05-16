"""
Backtest V3 - Revision gemaess Reviewer-Korrekturen K2, K5, K6, K7, K10.

Aenderungen gegenueber V2:
- K7: RISIKO_FREIER_ZINS = 0.0 (Cash unverzinst, Konsistenz mit Text).
- K2: MIN_WINDOW = 60 (PSY-konformer; Anhang B zeigt, dass min_window=30
      die BSADF-Suprema bei Lonza und PGHN am 2008-10-10 (Lehman) realisiert,
      d.h. in der Crash- statt Bubble-Phase).
- K6: Volatilitaetsabhaengiges Slippage-Modell (Almgren/Chriss-inspiriert):
        s_t = clip( s_base + lambda * sigma_5d_intraday * 1{Panik},
                    s_floor, s_cap )
      mit lambda=2.0, s_floor=0.025, s_cap=0.20.
      Damit haengt die Panik-Slippage am realisierten Volatilitaetsregime
      des Trigger-Tages und nicht an einer fixen Heuristik.
- K5: Inferenz via Stationary-Bootstrap (Politis/Romano 1994) auf
      Preis-Renditen mit anschliessender Strategie-Re-Simulation
      (V6 und B&H werden auf jedem Bootstrap-Pfad neu durchgespielt).
      Der alte paired-block-Bootstrap auf Strategie-Renditen wird
      zur Transparenz parallel berichtet.
- K10: Kommentar in bsadf_core.py korrigiert (separater Edit).

Outputs:
- baseline_results_v3.csv
- bootstrap_all_v3.json   (mit beiden Methoden)
- slippage_sensitivity_v3.csv
- slippage_lambda_sensitivity_v3.csv  (1D-Sensitivitaet ueber lambda)
"""
import os
import json
import warnings

import numpy as np
import pandas as pd

from bsadf_core import calculate_bsadf_fast

warnings.filterwarnings("ignore")

# ==========================================
# 1. GLOBALE PARAMETER
# ==========================================
RISIKO_FREIER_ZINS = 0.0      # K7: Cash unverzinst
TRANSAKTIONS_KOSTEN = 0.0015  # 15 bp regulaer
LOOKBACK_WINDOW = 126
MIN_WINDOW = 60               # K2: PSY-konformer
SMA_WINDOW = 20
Q_PANIC = 0.99
Q_WARN = 0.94
TRADING_DAYS = 252.0

# K6: dynamische Slippage-Parameter
SLIP_BASE = 0.005             # 50 bp Sockel
SLIP_LAMBDA = 2.0             # Marktimpact-Faktor auf 5-Tage-Vola
SLIP_FLOOR = 0.025            # 2.5% absoluter Floor (Konsistenz mit V2)
SLIP_CAP   = 0.20             # 20% Cap

EXCEL_DATEI = "Datenreihe.xlsx"
AKTIEN = {
    "Logitech": "LOGN",
    "Sika": "SIKA",
    "Richemont": "CFR",
    "Lonza": "LONN",
    "Partners Group": "PGHN",
}


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


def panic_slippage_dynamic(realised_vol_5d, base=SLIP_BASE, lam=SLIP_LAMBDA,
                            floor=SLIP_FLOOR, cap=SLIP_CAP):
    """
    K6: Volatilitaetsabhaengiger Panik-Slippage-Aufschlag.
    Inspiriert von Almgren/Chriss (2000): Marktimpact ist proportional
    zur (realisierten) Volatilitaet.
    """
    if not np.isfinite(realised_vol_5d):
        realised_vol_5d = 0.02
    s = base + lam * realised_vol_5d
    return float(np.clip(s, floor, cap))


# ==========================================
# 3. STRATEGIEN
# ==========================================
def run_buy_and_hold(prices, r_free=RISIKO_FREIER_ZINS):
    p = (prices / prices[0]) * 100.0
    return p.tolist()


def run_momentum_strategy(prices, transaction_cost=TRANSAKTIONS_KOSTEN,
                          r_free=RISIKO_FREIER_ZINS):
    sma = pd.Series(prices).rolling(window=SMA_WINDOW).mean().values
    wealth, portfolio, prev_w = 100.0, [100.0], 1.0
    for t in range(1, len(prices)):
        if t < SMA_WINDOW:
            target = 1.0
        else:
            target = 1.0 if prices[t-1] > sma[t-1] else 0.0
        if target != prev_w:
            wealth -= wealth * abs(target - prev_w) * transaction_cost
        prev_w = target
        wealth *= target * (prices[t]/prices[t-1]) + (1-target) * (1 + r_free/TRADING_DAYS)
        portfolio.append(wealth)
    return portfolio


def run_v6_smart_strategy(prices, transaction_cost=TRANSAKTIONS_KOSTEN,
                          r_free=RISIKO_FREIER_ZINS,
                          bsadf_scores=None,
                          slippage_mode="dynamic",
                          slippage_fixed=0.025,
                          slip_lambda=SLIP_LAMBDA,
                          slip_floor=SLIP_FLOOR,
                          slip_cap=SLIP_CAP):
    """
    V6 - Smart Re-Entry mit echtem BSADF und (per Default) dynamischer
    volatilitaetsabhaengiger Panik-Slippage (K6).

    slippage_mode = 'dynamic' | 'fixed'
    """
    prices = np.asarray(prices, dtype=float)
    if bsadf_scores is None:
        bsadf_scores = calculate_bsadf_fast(pd.Series(prices), MIN_WINDOW, LOOKBACK_WINDOW)

    score_series = pd.Series(bsadf_scores)
    roll_warn = score_series.rolling(LOOKBACK_WINDOW, min_periods=LOOKBACK_WINDOW)\
        .quantile(Q_WARN).values
    roll_panic = score_series.rolling(LOOKBACK_WINDOW, min_periods=LOOKBACK_WINDOW)\
        .quantile(Q_PANIC).values

    sma = pd.Series(prices).rolling(window=SMA_WINDOW).mean().values

    # 5-Tage realisierte Volatilitaet (rolling std der log-returns)
    log_ret = np.diff(np.log(prices))
    vol_5d = np.empty(len(prices))
    vol_5d[:] = np.nan
    for t in range(6, len(prices)):
        vol_5d[t] = np.std(log_ret[t-5:t], ddof=1)

    portfolio, wealth, prev_w = [100.0], 100.0, 1.0
    weights = [1.0]
    panic_trade_log = []  # (t, vol_5d, slippage_used)

    for t in range(1, len(prices)):
        score = bsadf_scores[t-1]
        rw = roll_warn[t-1]
        rp = roll_panic[t-1]
        lim_w = max(rw if not np.isnan(rw) else 0.5, 0.5)
        lim_p = max(rp if not np.isnan(rp) else 1.2, 1.2)

        if t < max(SMA_WINDOW, LOOKBACK_WINDOW):
            target = 1.0
        elif prev_w == 0.0:
            re_entry = (score < 0) or (prices[t-1] > sma[t-1] and score <= lim_p)
            target = 1.0 if re_entry else 0.0
        else:
            if score > lim_p:
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
                        vol_5d[t-1],
                        base=SLIP_BASE,
                        lam=slip_lambda,
                        floor=slip_floor,
                        cap=slip_cap,
                    )
                else:
                    s_used = slippage_fixed
                cost = s_used
                panic_trade_log.append((t, vol_5d[t-1], s_used))
            else:
                cost = transaction_cost
            wealth -= wealth * abs(target - prev_w) * cost

        prev_w = target
        weights.append(target)
        wealth *= target * (prices[t]/prices[t-1]) + (1-target) * (1 + r_free/TRADING_DAYS)
        portfolio.append(wealth)

    return portfolio, bsadf_scores, roll_panic, weights, panic_trade_log


# ==========================================
# 4. KENNZAHLEN
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
# 5. INFERENZ
# ==========================================
def paired_block_bootstrap_strategy_returns(returns_a, returns_b, B=2000, block_len=21,
                                             seed=42):
    """ALT (V2): Paired-Block-Bootstrap auf Strategie-Renditen.
    Wird parallel berichtet (Transparenz). Sieht Strategie-Renditen als
    austauschbar in Bloecken, was bei pfadabhaengigen Allokationen
    diskutabel ist (vgl. Reviewer K5)."""
    rng = np.random.default_rng(seed)
    ra = np.asarray(returns_a)
    rb = np.asarray(returns_b)
    n = len(ra)
    n_blocks = int(np.ceil(n / block_len))

    def maxdd(r):
        w = np.cumprod(1.0 + r) * 100.0
        c = np.maximum.accumulate(w)
        return ((w - c) / c).min() * 100.0

    def sharpe(r):
        sd = r.std()
        if sd == 0:
            return 0.0
        ex = r - RISIKO_FREIER_ZINS / TRADING_DAYS
        return (ex.mean() / sd) * np.sqrt(TRADING_DAYS)

    def tret(r):
        return (np.prod(1.0 + r) - 1.0) * 100.0

    obs_dd = maxdd(ra) - maxdd(rb)
    obs_sh = sharpe(ra) - sharpe(rb)
    obs_re = tret(ra) - tret(rb)

    diff_dd = np.empty(B)
    diff_sh = np.empty(B)
    diff_re = np.empty(B)
    starts_max = n - block_len
    for b in range(B):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
        rab = ra[idx]; rbb = rb[idx]
        diff_dd[b] = maxdd(rab) - maxdd(rbb)
        diff_sh[b] = sharpe(rab) - sharpe(rbb)
        diff_re[b] = tret(rab) - tret(rbb)

    def p2(diffs, observed):
        centered = diffs - diffs.mean()
        return float(np.mean(np.abs(centered) >= abs(observed)))

    return {
        "B": B, "block_length_days": block_len,
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


def stationary_bootstrap_indices(n, mean_block_len, rng):
    """
    Politis/Romano (1994) Stationary-Bootstrap.
    Liefert Index-Sequenz der Laenge n.
    p = 1/mean_block_len ist die Wahrscheinlichkeit fuer Block-Ende.
    """
    p = 1.0 / mean_block_len
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = rng.integers(0, n)
        else:
            idx[t] = (idx[t-1] + 1) % n
    return idx


def stationary_bootstrap_resimulate(prices, B=500, mean_block_len=21, seed=42,
                                     n_years=None, slip_mode="dynamic"):
    """
    K5: Sauberer Bootstrap - blockt PREIS-RENDITEN, baut synthetischen
    Preispfad, rechnet BSADF + V6 + B&H darauf nochmal komplett durch
    und aggregiert die Differenzen. Pfadabhaengigkeit der Strategie
    wird so korrekt unter der Bootstrap-Distribution behandelt.
    """
    rng = np.random.default_rng(seed)
    prices = np.asarray(prices, dtype=float)
    log_ret = np.diff(np.log(prices))
    n_ret = len(log_ret)
    p0 = prices[0]

    # Beobachtete Differenzen unter realer Welt
    bsadf_real = calculate_bsadf_fast(pd.Series(prices), MIN_WINDOW, LOOKBACK_WINDOW)
    v6_real, _, _, _, _ = run_v6_smart_strategy(prices, bsadf_scores=bsadf_real, slippage_mode=slip_mode)
    bh_real = run_buy_and_hold(prices)

    obs_dd = max_drawdown_pct(v6_real) - max_drawdown_pct(bh_real)
    obs_sh = sharpe_ratio(v6_real) - sharpe_ratio(bh_real)
    obs_re = total_return_pct(v6_real) - total_return_pct(bh_real)

    diff_dd = np.empty(B)
    diff_sh = np.empty(B)
    diff_re = np.empty(B)

    for b in range(B):
        idx = stationary_bootstrap_indices(n_ret, mean_block_len, rng)
        boot_log_ret = log_ret[idx]
        boot_prices = p0 * np.exp(np.concatenate([[0.0], np.cumsum(boot_log_ret)]))
        # BSADF auf Bootstrap-Pfad
        bsadf_b = calculate_bsadf_fast(pd.Series(boot_prices), MIN_WINDOW, LOOKBACK_WINDOW)
        v6_b, _, _, _, _ = run_v6_smart_strategy(boot_prices, bsadf_scores=bsadf_b, slippage_mode=slip_mode)
        bh_b = run_buy_and_hold(boot_prices)
        diff_dd[b] = max_drawdown_pct(v6_b) - max_drawdown_pct(bh_b)
        diff_sh[b] = sharpe_ratio(v6_b) - sharpe_ratio(bh_b)
        diff_re[b] = total_return_pct(v6_b) - total_return_pct(bh_b)

    def p2(diffs, observed):
        centered = diffs - diffs.mean()
        return float(np.mean(np.abs(centered) >= abs(observed)))

    return {
        "method": "stationary_bootstrap_with_strategy_resimulation",
        "B": B, "mean_block_len": mean_block_len,
        "slippage_mode": slip_mode,
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
# 6. SLIPPAGE-SENSITIVITAETEN
# ==========================================
def slippage_sensitivity_2d(prices, base_costs, panic_slippages, bsadf_scores):
    rows = []
    for bc in base_costs:
        for ps in panic_slippages:
            v6_p, _, _, _, _ = run_v6_smart_strategy(
                prices, transaction_cost=bc,
                bsadf_scores=bsadf_scores,
                slippage_mode="fixed",
                slippage_fixed=ps,
            )
            mom_p = run_momentum_strategy(prices, transaction_cost=bc)
            rows.append({
                "Basis-Kosten %": round(bc * 100, 3),
                "Panik-Slippage %": round(ps * 100, 2),
                "V6 Total Return %": round(total_return_pct(v6_p), 2),
                "V6 Max DD %": round(max_drawdown_pct(v6_p), 2),
                "Momentum Total Return %": round(total_return_pct(mom_p), 2),
            })
    return pd.DataFrame(rows)


def slippage_sensitivity_lambda(prices, lambdas, bsadf_scores):
    """1D-Sensitivitaet ueber lambda im dynamischen Slippage-Modell."""
    rows = []
    for lam in lambdas:
        v6_p, _, _, _, log = run_v6_smart_strategy(
            prices, bsadf_scores=bsadf_scores,
            slippage_mode="dynamic", slip_lambda=lam,
        )
        avg_slip = (np.mean([s for _, _, s in log]) * 100) if log else None
        rows.append({
            "lambda": lam,
            "Panik-Trades": len(log),
            "Mean Panik-Slippage %": round(avg_slip, 3) if avg_slip is not None else None,
            "V6 Total Return %": round(total_return_pct(v6_p), 2),
            "V6 Max DD %": round(max_drawdown_pct(v6_p), 2),
        })
    return pd.DataFrame(rows)


# ==========================================
# 7. MAIN
# ==========================================
def main():
    print("=" * 70)
    print("BACKTEST V3 - Korrekturen K2, K5, K6, K7, K10")
    print("=" * 70)
    print(f"MIN_WINDOW = {MIN_WINDOW} (K2)")
    print(f"RISIKO_FREIER_ZINS = {RISIKO_FREIER_ZINS} (K7)")
    print(f"Slippage Default = DYNAMISCH (Almgren/Chriss-Stil, K6)")
    print()

    summary_rows = []
    bootstrap_results = {}
    bsadf_cache = {}
    price_cache = {}
    n_years_cache = {}

    for asset, sheet in AKTIEN.items():
        print(f">> {asset} ({sheet})")
        df = load_excel_data(EXCEL_DATEI, sheet)
        n_years = years_in_sample(df)
        n_years_cache[asset] = n_years
        prices = df["Price"].values
        price_cache[asset] = prices
        print(f"   {len(df)} Tage, {n_years:.2f} Jahre")

        print("   -> BSADF (min_window=60) ...")
        scores = calculate_bsadf_fast(df["Price"], MIN_WINDOW, LOOKBACK_WINDOW)
        bsadf_cache[asset] = scores
        nonzero_scores = scores[scores != 0]
        if len(nonzero_scores) > 0:
            print(f"   BSADF: max={scores.max():.2f}, mean(nonzero)={nonzero_scores.mean():.2f}, "
                  f"#>1.5: {(scores > 1.5).sum()}")

        bh_p = run_buy_and_hold(prices)
        mom_p = run_momentum_strategy(prices)
        # Dynamische Slippage als Default
        v6_dyn_p, _, _, _, panic_log = run_v6_smart_strategy(
            prices, bsadf_scores=scores, slippage_mode="dynamic"
        )
        # Auch fixe Variante (2.5%) zum Vergleich mit V2
        v6_fix_p, _, _, _, _ = run_v6_smart_strategy(
            prices, bsadf_scores=scores, slippage_mode="fixed", slippage_fixed=0.025
        )
        if panic_log:
            slips = [s for _,_,s in panic_log]
            print(f"   Panik-Trigger: n={len(panic_log)}, "
                  f"Mean Slippage (dynamisch): {100*np.mean(slips):.2f}%, "
                  f"Max: {100*max(slips):.2f}%")
        else:
            print(f"   Panik-Trigger: keine (Floor wird gefuettert)")

        for name, p in [("Buy & Hold", bh_p), ("Momentum", mom_p),
                        ("V6 Smart (dyn. Slippage)", v6_dyn_p),
                        ("V6 Smart (fix 2.5%)", v6_fix_p)]:
            summary_rows.append(metrics_dict(p, n_years, asset, name))

    df_metrics = pd.DataFrame(summary_rows)
    df_metrics.to_csv("baseline_results_v3.csv", index=False)
    print()
    print("=" * 70)
    print("HAUPTRESULTATE V3 (Tabelle 3 - neu)")
    print("=" * 70)
    print(df_metrics.to_string(index=False))
    print()

    # Inferenz: beide Methoden parallel
    print("=" * 70)
    print("BOOTSTRAP-INFERENZ")
    print("=" * 70)
    for asset in AKTIEN:
        print(f">> {asset}")
        prices = price_cache[asset]
        scores = bsadf_cache[asset]
        bh_p = run_buy_and_hold(prices)
        mom_p = run_momentum_strategy(prices)
        v6_dyn_p, _, _, _, _ = run_v6_smart_strategy(
            prices, bsadf_scores=scores, slippage_mode="dynamic"
        )

        # ALT: paired-block auf Strategie-Renditen
        ret_v6 = pd.Series(v6_dyn_p).pct_change().dropna().values
        ret_bh = pd.Series(bh_p).pct_change().dropna().values
        ret_mom = pd.Series(mom_p).pct_change().dropna().values
        alt_v6 = paired_block_bootstrap_strategy_returns(ret_v6, ret_bh, B=2000, block_len=21)
        alt_mom = paired_block_bootstrap_strategy_returns(ret_mom, ret_bh, B=2000, block_len=21)

        # NEU (K5): stationary bootstrap mit Strategie-Re-Simulation
        print(f"   Stationary-Bootstrap mit Re-Simulation (B=500)...")
        new_v6 = stationary_bootstrap_resimulate(
            prices, B=500, mean_block_len=21, seed=42, slip_mode="dynamic"
        )

        bootstrap_results[asset] = {
            "V6_dyn_vs_BH": {
                "paired_block_strategy_returns_B2000_oldmethod": alt_v6,
                "stationary_bootstrap_strategy_resim_B500_newmethod": new_v6,
            },
            "Momentum_vs_BH": {
                "paired_block_strategy_returns_B2000_oldmethod": alt_mom,
            },
        }
        print(f"   alt p_vals (V6vsBH): mdd={alt_v6['p_values']['maxdd']}, "
              f"sharpe={alt_v6['p_values']['sharpe']}, ret={alt_v6['p_values']['return']}")
        print(f"   new p_vals (V6vsBH): mdd={new_v6['p_values']['maxdd']}, "
              f"sharpe={new_v6['p_values']['sharpe']}, ret={new_v6['p_values']['return']}")

    with open("bootstrap_all_v3.json", "w") as f:
        json.dump(bootstrap_results, f, indent=2)
    print(f"\nBootstrap-Resultate -> bootstrap_all_v3.json")

    # Slippage-Sensitivitaeten fuer Partners Group
    print()
    print("=" * 70)
    print("2D SLIPPAGE-SENSITIVITAET (Partners Group, fixe Slippage)")
    print("=" * 70)
    sens_2d = slippage_sensitivity_2d(
        price_cache["Partners Group"],
        base_costs=[0.0015, 0.0030, 0.0045, 0.0060],
        panic_slippages=[0.025, 0.05, 0.10, 0.15],
        bsadf_scores=bsadf_cache["Partners Group"],
    )
    sens_2d.to_csv("slippage_sensitivity_v3.csv", index=False)
    print(sens_2d.to_string(index=False))

    print()
    print("=" * 70)
    print("1D LAMBDA-SENSITIVITAET (Partners Group, dyn. Slippage)")
    print("=" * 70)
    sens_lam = slippage_sensitivity_lambda(
        price_cache["Partners Group"],
        lambdas=[0.5, 1.0, 2.0, 3.0, 5.0],
        bsadf_scores=bsadf_cache["Partners Group"],
    )
    sens_lam.to_csv("slippage_lambda_sensitivity_v3.csv", index=False)
    print(sens_lam.to_string(index=False))

    print("\nFertig.")


if __name__ == "__main__":
    main()
