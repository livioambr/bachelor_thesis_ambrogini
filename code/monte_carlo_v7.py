"""
Monte Carlo V7 — Evans-Modell mit V7-Strategie und V7-Methodik.

V7-Aenderungen gegenueber monte_carlo_v2.py:
- K2: MIN_WINDOW skaliert PSY-konform mit T (r0 = 0.01 + 1.8/sqrt(T))
       Bei T = 240 ergibt das r0 ~= 0.126 -> MIN_WINDOW = 30
- K3: V7-Smart-Re-Entry mit konservativer Lag-Konvention (score[t-2])
- K1: Pre-computed Wild-Bootstrap-Schwellen unter H0
       (separat fuer T=240, da die Realdaten-CVs auf T~5273 trainiert sind)
- K4: Direktionalfilter im Panik-Trigger (60-Tages-Trend; bei MC: 60 Perioden)

Outputs:
- fig_mc_spaghetti_v7.png          (Abb. 1 oder Mgmt Summary)
- fig_mc_szenario_A_v7.png         (Abb. 2, Normalverlauf)
- fig_mc_szenario_B_v7.png         (mittlere Blasenbildung)
- fig_mc_szenario_C_v7.png         (Abb. 3, Extremfall)
- fig_mc_evolution_v7.png          (V0 vs V3 vs V5 vs V7)
- wild_bootstrap_critical_values_v7_MC.json  (T=240 CVs)
"""
import os, json, warnings, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(101)

from bsadf_core import calculate_bsadf_fast

# ==========================================
# 1. PARAMETER
# ==========================================
EVANS_PERIODEN = 240
EVANS_PFADE = 2500

R_ANNUAL = 0.0075
DRIFT_ANNUAL = 0.0373
VOLA_ANNUAL = 0.1574

EVANS_R = R_ANNUAL
EVANS_DRIFT = DRIFT_ANNUAL
EVANS_VOLA = VOLA_ANNUAL

CHF_MONTHLY_RATE = (1.0 + R_ANNUAL) ** (1.0 / 12.0) - 1.0

EVANS_START_BLASE = 0.5
EVANS_DELTA = 0.5
EVANS_PI = 0.85
EVANS_ALPHA = 1.0

TRANSAKTIONS_KOSTEN = 0.0015
PANIC_SLIPPAGE = 0.025

# V7-MC-Parameter: PSY-konforme Skalierung fuer T = 240
# r0 = 0.01 + 1.8/sqrt(240) ≈ 0.126 -> MIN_WINDOW = 30
T_MC = EVANS_PERIODEN
MIN_WINDOW_MC = max(10, int(round((0.01 + 1.8 / math.sqrt(T_MC)) * T_MC)))   # ≈ 30
LOOKBACK_MC = 60                  # gegenueber 126 bei Realdaten (verhaeltnismaessig kuerzer)
SMA_WINDOW = 20
TREND_WINDOW_MC = 60              # K4: 60-Perioden-Trend fuer Direktionalfilter

WB_N_PATHS_MC = 500               # Wild-Bootstrap-Pfade fuer T=240 H0
WB_SEED = 20260515

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ==========================================
# 2. EVANS-MODELL
# ==========================================
def generiere_evans_pfad():
    """Evans (1991) Periodically Collapsing Bubble. Unveraendert gegenueber V2."""
    dividenden = np.zeros(EVANS_PERIODEN)
    fundamentaldaten = np.zeros(EVANS_PERIODEN)
    blasen = np.zeros(EVANS_PERIODEN)
    preise = np.zeros(EVANS_PERIODEN)

    dividenden[0] = 1.0
    blasen[0] = EVANS_START_BLASE
    fundamentaldaten[0] = (EVANS_DRIFT * (1 + EVANS_R) / EVANS_R**2) + (dividenden[0] / EVANS_R)
    preise[0] = fundamentaldaten[0] + blasen[0]

    for t in range(1, EVANS_PERIODEN):
        schock = np.random.normal(0, EVANS_VOLA)
        dividenden[t] = max(0.01, EVANS_DRIFT + dividenden[t-1] + schock)
        fundamentaldaten[t] = (EVANS_DRIFT * (1 + EVANS_R) / EVANS_R**2) + (dividenden[t] / EVANS_R)
        u_t = np.random.lognormal(mean=0, sigma=0.01)

        if blasen[t-1] <= EVANS_ALPHA:
            blasen[t] = (1 + EVANS_R) * blasen[t-1] * u_t
        else:
            if np.random.rand() <= EVANS_PI:
                wachstum = (1/EVANS_PI) * (1 + EVANS_R) * (blasen[t-1] - (EVANS_DELTA / (1 + EVANS_R)))
                blasen[t] = (EVANS_DELTA + wachstum) * u_t
            else:
                blasen[t] = EVANS_DELTA * u_t
        preise[t] = fundamentaldaten[t] + blasen[t]

    return pd.DataFrame({"Fundamental": fundamentaldaten,
                         "Bubble": blasen,
                         "Price": preise})


# ==========================================
# 3. WILD-BOOTSTRAP-CVs FUER MC (T = 240)
# ==========================================
def compute_mc_cvs(n_paths=WB_N_PATHS_MC, sigma=EVANS_VOLA,
                   save_path=None):
    """Pre-computed Wild-Bootstrap-Quantile unter H0 fuer T = 240.

    Im Gegensatz zu den Realdaten-CVs (T = 5273, sigma ~ 0.02 p.d.) sind die
    MC-Pfade T = 240 Perioden lang mit hoeherer Per-Periode-Vola (~0.16).
    Daher braucht es separate CVs.
    """
    if save_path is None:
        save_path = os.path.join(_SCRIPT_DIR, "wild_bootstrap_critical_values_v7_MC.json")

    if os.path.exists(save_path):
        print(f"MC-CVs geladen aus: {save_path}", flush=True)
        return json.load(open(save_path))

    print(f"Berechne Wild-Bootstrap-CVs fuer MC (T = {T_MC}, n_paths = {n_paths}) ...", flush=True)
    rng = np.random.default_rng(WB_SEED)
    all_scores = []
    t0 = time.time()
    for k in range(n_paths):
        eps = rng.choice([-1.0, 1.0], size=T_MC) * sigma
        log_path = np.cumsum(eps)
        prices = 100.0 * np.exp(log_path)
        sc = calculate_bsadf_fast(pd.Series(prices), MIN_WINDOW_MC, LOOKBACK_MC)
        all_scores.append(sc[MIN_WINDOW_MC:].astype(np.float32))
        if (k + 1) % 100 == 0:
            print(f"   {k+1}/{n_paths}, dt={time.time()-t0:.0f}s", flush=True)

    flat = np.concatenate(all_scores)
    flat = flat[np.isfinite(flat)]
    cv = {
        "method": "wild_bootstrap_global_v7_MC",
        "n_paths": int(n_paths),
        "T_per_path": T_MC,
        "n_pooled_obs": int(len(flat)),
        "sigma": float(sigma),
        "min_window": MIN_WINDOW_MC,
        "max_lookback": LOOKBACK_MC,
        "q90": float(np.quantile(flat, 0.90)),
        "q94": float(np.quantile(flat, 0.94)),
        "q95": float(np.quantile(flat, 0.95)),
        "q99": float(np.quantile(flat, 0.99)),
        "se_q94": math.sqrt(0.06*0.94/n_paths),
        "se_q99": math.sqrt(0.01*0.99/n_paths),
        "seed_base": WB_SEED,
    }
    with open(save_path, "w") as f:
        json.dump(cv, f, indent=2)
    print(f"MC-CVs gespeichert -> {save_path}", flush=True)
    print(f"  q94 = {cv['q94']:.4f}, q99 = {cv['q99']:.4f}", flush=True)
    return cv


# ==========================================
# 4. STRATEGIEN
# ==========================================
def run_momentum(df):
    prices = df["Price"].values
    sma = df["Price"].rolling(window=SMA_WINDOW).mean().values
    wealth, portfolio, prev_w = 100.0, [100.0], 1.0
    for t in range(1, len(prices)):
        if t < SMA_WINDOW + 1:
            target = 1.0
        else:
            target = 1.0 if prices[t-2] > sma[t-2] else 0.0  # K3: t-2
        if target != prev_w:
            wealth -= wealth * abs(target - prev_w) * TRANSAKTIONS_KOSTEN
        prev_w = target
        wealth *= target * (prices[t]/prices[t-1]) + (1-target) * (1 + CHF_MONTHLY_RATE)
        portfolio.append(wealth)
    return portfolio


def run_v7_mc(df, q_warn, q_panic, scores=None, use_directional_filter=True):
    """V7-Smart-Re-Entry auf MC-Pfaden mit Wild-Bootstrap-Schwellen + Direktionalfilter."""
    prices = df["Price"].values
    if scores is None:
        scores = calculate_bsadf_fast(df["Price"], MIN_WINDOW_MC, LOOKBACK_MC)

    sma = df["Price"].rolling(SMA_WINDOW).mean().values
    portfolio, wealth, prev_w = [100.0], 100.0, 1.0
    weights = [1.0]

    for t in range(2, len(prices)):  # K3: ab t=2
        score = scores[t-2]
        if t < max(SMA_WINDOW, MIN_WINDOW_MC, TREND_WINDOW_MC + 2):
            target = 1.0
        elif prev_w == 0.0:
            re_entry = (score < 0) or (prices[t-2] > sma[t-2] and score <= q_panic)
            target = 1.0 if re_entry else 0.0
        else:
            if score > q_panic:
                if use_directional_filter and (t - 2 - TREND_WINDOW_MC >= 0):
                    trend_pos = prices[t-2] > prices[t-2-TREND_WINDOW_MC]
                    target = 0.0 if trend_pos else prev_w
                else:
                    target = 0.0
            elif score > q_warn:
                target = 0.5
            else:
                target = 1.0

        if target != prev_w:
            cost = PANIC_SLIPPAGE if (target == 0.0 and prev_w > 0.0 and score > q_panic) else TRANSAKTIONS_KOSTEN
            wealth -= wealth * abs(target - prev_w) * cost
        prev_w = target
        weights.append(target)
        wealth *= target * (prices[t]/prices[t-1]) + (1-target) * (1 + CHF_MONTHLY_RATE)
        portfolio.append(wealth)

    # Padding fuer t=0,1
    while len(portfolio) < len(prices):
        portfolio.insert(0, 100.0)
        weights.insert(0, 1.0)
    return portfolio, scores, weights


def run_v3_static(df, scores, threshold=1.8):
    """Vorgaengerstrategie V3 (statischer Exit) zum Vergleich."""
    prices = df["Price"].values
    p, w, out = [100.0], 100.0, False
    for t in range(1, len(prices)):
        if scores[t-1] > threshold:
            out = True
        if out:
            w *= 1 + CHF_MONTHLY_RATE
        else:
            w *= prices[t] / prices[t-1]
        p.append(w)
    return p


# ==========================================
# 5. PLOTTING
# ==========================================
def plot_4panel_szenario(df, title, fname, q_warn, q_panic):
    """4-Panel-Plot fuer einen Evans-Pfad: Anatomie, Performance, BSADF, Aktienquote."""
    p_v7, scores, weights = run_v7_mc(df, q_warn, q_panic)
    bh = (df["Price"] / df["Price"].iloc[0] * 100).values
    mom = run_momentum(df)
    weights_arr = np.array(weights, dtype=float)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(13, 14), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.5, 1.0, 0.7]})
    fig.suptitle(title + f"\n(MIN_WINDOW = {MIN_WINDOW_MC}, score[t-2], Direktionalfilter, q94 = {q_warn:.3f}/q99 = {q_panic:.3f})",
                 fontsize=13, fontweight="bold", y=0.995)

    ax1.plot(df.index, df["Price"], color="black", linewidth=1.0, label="Marktpreis")
    ax1.plot(df.index, df["Fundamental"], color="blue", linestyle="--", linewidth=1.0, label="Fundament")
    ax1.fill_between(df.index, df["Fundamental"], df["Price"],
                     where=(df["Price"] >= df["Fundamental"]),
                     color="red", alpha=0.18, label="Blasenkomponente")
    ax1.set_title("Anatomie: Preis vs. Fundament (Evans-Modell)")
    ax1.legend(loc="upper left")

    ax2.plot(df.index, bh, color="#7f7f7f", linewidth=1.2, label="Buy & Hold")
    ax2.plot(df.index, mom, color="#ff7f0e", linewidth=1.2, label=f"Momentum SMA{SMA_WINDOW}")
    ax2.plot(df.index, p_v7, color="#1f77b4", linewidth=1.8, label="V7 Smart")
    ax2.set_title("Performance der Strategien")
    ax2.legend(loc="upper left")

    ax3.plot(df.index, scores, color="purple", linewidth=0.9, label="BSADF Score (PSY 2015)")
    ax3.axhline(q_warn, color="#ff9f1c", linestyle="--", linewidth=1.0, label=f"q94 = {q_warn:.3f} (Warn)")
    ax3.axhline(q_panic, color="#d62728", linestyle="--", linewidth=1.2, label=f"q99 = {q_panic:.3f} (Panik)")
    ax3.fill_between(df.index, q_panic, scores, where=(scores > q_panic),
                     color="red", alpha=0.30, label="Panik-Trigger")
    ax3.set_title("Signalgenerierung BSADF + Wild-Bootstrap-Schwellen (V7)")
    ax3.legend(loc="upper left", ncol=2, fontsize=8)

    ax4.fill_between(df.index, 0, weights_arr * 100, color="green", alpha=0.5, step="post")
    ax4.set_ylim(-3, 105)
    ax4.set_title("V7-Aktienquote (0 % / 50 % / 100 %)")
    ax4.set_xlabel("Periode (Ticks)")

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {fname}", flush=True)


def plot_spaghetti(simulationen, df_normal, df_medium, df_extrem, fname):
    plt.figure(figsize=(13, 7))
    for df in simulationen:
        plt.plot(df.index, df["Price"], alpha=0.03, color="gray", linewidth=1)
    plt.plot(simulationen[0].index, simulationen[0]["Fundamental"],
             color="black", linestyle="--", linewidth=2, label="Fundament")
    for df, color, label in [(df_normal, "green", "Szenario A (Normal)"),
                              (df_medium, "orange", "Szenario B (Medium)"),
                              (df_extrem, "red", "Szenario C (Extrem)")]:
        if df is not None:
            plt.plot(df.index, df["Price"], color=color, linewidth=2.5, label=label)
    plt.yscale("log")
    plt.title(f"Evans-Modell: {len(simulationen)} Pfade (log-Skala) — V7-Konfiguration",
              fontsize=14, fontweight="bold")
    plt.xlabel("Periode (Ticks)")
    plt.legend(loc="upper left")
    plt.grid(True, which="both", ls="-", alpha=0.1)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  -> {fname}", flush=True)


def plot_evolution(all_sims_subset, q_warn, q_panic, fname):
    """V0 vs V3 vs V7 — Erwartungswert ueber alle Pfade."""
    v0, v3, v7 = [], [], []
    for df in all_sims_subset:
        scores = calculate_bsadf_fast(df["Price"], MIN_WINDOW_MC, LOOKBACK_MC)
        v0.append((df["Price"] / df["Price"].iloc[0]) * 100)
        v3.append(run_v3_static(df, scores, threshold=1.8))
        p_v7, _, _ = run_v7_mc(df, q_warn, q_panic, scores=scores)
        v7.append(p_v7)

    plt.figure(figsize=(13, 7))
    plt.plot(np.mean(v0, axis=0), color="gray", linestyle="--", linewidth=2,
             label="V0: Buy & Hold")
    plt.plot(np.mean(v3, axis=0), color="orange", linewidth=2.5,
             label="V3: Statischer Exit (BSADF > 1.8)")
    plt.plot(np.mean(v7, axis=0), color="#1f77b4", linewidth=3,
             label="V7: Smart Re-Entry + Direktionalfilter (Final)")
    plt.title("Strategieentwicklung mit V7 BSADF + Direktionalfilter (Erwartungswert ueber 300 Pfade)",
              fontsize=14, fontweight="bold")
    plt.xlabel("Periode (Ticks)")
    plt.ylabel("Durchschnittlicher Portfolio-Wert (Start = 100)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  -> {fname}", flush=True)


# ==========================================
# 6. MAIN
# ==========================================
def main():
    print("=" * 70, flush=True)
    print("MONTE CARLO V7 — Evans-Modell mit V7-Strategie", flush=True)
    print("=" * 70, flush=True)
    print(f"T = {T_MC}, MIN_WINDOW_MC = {MIN_WINDOW_MC} (PSY-konform), LOOKBACK_MC = {LOOKBACK_MC}", flush=True)
    print(f"Lag: score[t-2], Direktionalfilter: aktiv (Trend = {TREND_WINDOW_MC} Perioden)", flush=True)
    print()

    cv = compute_mc_cvs(n_paths=WB_N_PATHS_MC)
    q_warn, q_panic = cv["q94"], cv["q99"]
    print(f"\nMC-Wild-Bootstrap-Schwellen: q94 = {q_warn:.4f}, q99 = {q_panic:.4f}\n", flush=True)

    print(f"Generiere {EVANS_PFADE} Evans-Pfade ...", flush=True)
    np.random.seed(101)
    alle_sims, bubbles = [], []
    t0 = time.time()
    for i in range(EVANS_PFADE):
        df = generiere_evans_pfad()
        alle_sims.append(df)
        bubbles.append(df["Bubble"].max())
        if (i+1) % 500 == 0:
            print(f"   {i+1} Pfade, dt={time.time()-t0:.0f}s", flush=True)

    # Szenarien
    crash_pfade = [df for df in alle_sims
                   if df["Bubble"].max() > 50.0
                   and df["Bubble"].idxmax() < 225
                   and df["Bubble"].iloc[-1] < df["Bubble"].max() * 0.1]
    df_extrem = (crash_pfade[np.argmax([df["Bubble"].max() for df in crash_pfade])]
                 if crash_pfade else alle_sims[np.argmax(bubbles)])
    max_b = df_extrem["Bubble"].max()
    ziel_med = (4.0 + max_b) / 4
    df_normal = alle_sims[np.argmin([abs(b - 4.0) for b in bubbles])]
    df_medium = alle_sims[np.argmin([abs(b - ziel_med) for b in bubbles])]

    print("\nGeneriere Plots ...", flush=True)
    plot_spaghetti(alle_sims, df_normal, df_medium, df_extrem, "fig_mc_spaghetti_v7.png")
    plot_4panel_szenario(df_normal, "Szenario A (Normalverlauf) — V7 Strategie",
                         "fig_mc_szenario_A_v7.png", q_warn, q_panic)
    plot_4panel_szenario(df_medium, "Szenario B (Mittlere Blasenbildung) — V7 Strategie",
                         "fig_mc_szenario_B_v7.png", q_warn, q_panic)
    plot_4panel_szenario(df_extrem, "Szenario C (Extremereignis / Kollaps) — V7 Strategie",
                         "fig_mc_szenario_C_v7.png", q_warn, q_panic)

    subset = alle_sims[:300]
    plot_evolution(subset, q_warn, q_panic, "fig_mc_evolution_v7.png")

    print("\n" + "=" * 70, flush=True)
    print("FERTIG. MC-V7-Plots erzeugt.", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
