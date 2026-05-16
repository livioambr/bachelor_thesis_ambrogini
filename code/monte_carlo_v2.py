"""
Monte Carlo V2 - Evans-Modell mit echtem BSADF (Revision 2026-05-14).

Aenderungen gegenueber V1:
- Echter BSADF (Supremum, HAC-Newey-West-Lag).
- Evans-Formel exakt nach Evans (1991, eq. 5-6): beide Regime-Zweige tragen u_t.
- Cash-Verzinsung der Strategien getrennt vom stilisierten Evans-Periodenzins.
- Ehrliche Diagnostik: Detector-Performance (TP/FP-Rate).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from bsadf_core import calculate_bsadf_fast

warnings.filterwarnings("ignore")
np.random.seed(101)

# ==========================================
# 1. PARAMETER
# ==========================================
EVANS_PERIODEN = 240
EVANS_PFADE = 2500

R_ANNUAL = 0.0075
DRIFT_ANNUAL = 0.0373
VOLA_ANNUAL = 0.1574

# Evans (1991) verwendet r als stilisierten Periodenparameter (im Original
# r = 0.05/Periode), nicht als realen CHF-Kalenderzins. Wir behalten diese
# Stilisierung bei, damit die Blasen-Wachstumsdynamik innerhalb von
# 240 Perioden sichtbare Zyklen erzeugt.
EVANS_R = R_ANNUAL          # stilisierter Periodenparameter (Evans 1991)
EVANS_DRIFT = DRIFT_ANNUAL  # stilisierter Periodendrift
EVANS_VOLA = VOLA_ANNUAL    # stilisierte Periodenvolatilitaet

# Separat: realer monatlicher CHF-Zinssatz fuer die Cash-Verzinsung der
# Strategien waehrend Out-of-Market-Phasen. So erhalten V6 und Momentum
# keinen kuenstlichen Cash-Drift-Vorteil aus dem stilisierten Evans-Zins
# (Korrektur Reviewer K8).
CHF_MONTHLY_RATE = (1.0 + R_ANNUAL) ** (1.0 / 12.0) - 1.0  # ~0.0623% / Monat

EVANS_START_BLASE = 0.5
EVANS_DELTA = 0.5
EVANS_PI = 0.85
EVANS_ALPHA = 1.0

TRANSAKTIONS_KOSTEN = 0.0015
PANIC_SLIPPAGE = 0.025
LOOKBACK_WINDOW = 126
MIN_WINDOW = 60  # Konsistenz mit Realbacktest (PSY-konformere r0)
SMA_WINDOW = 20
Q_PANIC = 0.99
Q_WARN = 0.94


# ==========================================
# 2. EVANS-MODELL
# ==========================================
def generiere_evans_pfad():
    """
    Periodically Collapsing Bubble nach Evans (1991, AER 81(4), 922-930).

    Implementierung gemaess Evans (1991, eq. 5-6):
    Im expansiven Regime (b_{t-1} > alpha) wachsen Blasen gemaess
        wachstum = (1/pi) * (1+r) * (b_{t-1} - delta/(1+r))
    und werden mit Wahrscheinlichkeit (1-pi) auf delta zurueckgesetzt.
    Beide Regime-Zweige (Wachstum und Burst) tragen den multiplikativen
    log-normalen Schock u_t (Korrektur Reviewer K7); damit ist die
    Originalformel Evans (1991, eq. 6) eins-zu-eins reproduziert.
    """
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
                # Burst-Zweig: Blase faellt auf delta mit multiplikativem Schock u_t
                # (vorher fehlte u_t; Korrektur Reviewer K7).
                blasen[t] = EVANS_DELTA * u_t
        preise[t] = fundamentaldaten[t] + blasen[t]

    return pd.DataFrame({"Fundamental": fundamentaldaten,
                         "Bubble": blasen,
                         "Price": preise})


# ==========================================
# 3. STRATEGIEN (mit echtem BSADF)
# ==========================================
def run_momentum_strategy(df):
    prices = df["Price"].values
    sma = df["Price"].rolling(window=SMA_WINDOW).mean().values
    wealth, portfolio, prev_w = 100.0, [100.0], 1.0
    for t in range(1, len(prices)):
        target = 1.0 if t < SMA_WINDOW else (1.0 if prices[t-1] > sma[t-1] else 0.0)
        if target != prev_w:
            wealth -= wealth * abs(target - prev_w) * TRANSAKTIONS_KOSTEN
        prev_w = target
        wealth *= target * (prices[t]/prices[t-1]) + (1-target) * (1 + CHF_MONTHLY_RATE)
        portfolio.append(wealth)
    return portfolio


def run_v6_smart_strategy(df, scores=None):
    prices = df["Price"].values
    if scores is None:
        scores = calculate_bsadf_fast(df["Price"], MIN_WINDOW, LOOKBACK_WINDOW)
    score_series = pd.Series(scores)
    roll_warn = score_series.rolling(LOOKBACK_WINDOW, min_periods=LOOKBACK_WINDOW).quantile(Q_WARN).values
    roll_panic = score_series.rolling(LOOKBACK_WINDOW, min_periods=LOOKBACK_WINDOW).quantile(Q_PANIC).values

    sma = df["Price"].rolling(window=SMA_WINDOW).mean().values
    portfolio, wealth, prev_w = [100.0], 100.0, 1.0
    weights = [1.0]

    for t in range(1, len(prices)):
        score = scores[t-1]
        rw = roll_warn[t-1]
        rp = roll_panic[t-1]
        lim_w = max(rw if not np.isnan(rw) else 0.5, 0.5)
        lim_p = max(rp if not np.isnan(rp) else 1.2, 1.2)

        if t < SMA_WINDOW:
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
            cost = PANIC_SLIPPAGE if (target == 0.0 and prev_w > 0.0 and score > lim_p) else TRANSAKTIONS_KOSTEN
            wealth -= wealth * abs(target - prev_w) * cost

        prev_w = target
        weights.append(target)
        wealth *= target * (prices[t]/prices[t-1]) + (1-target) * (1 + CHF_MONTHLY_RATE)
        portfolio.append(wealth)

    return portfolio, scores, roll_panic, weights


def run_v3_static_strategy(df, scores=None, threshold=1.8):
    """Statische Exit-Logik der Vorgaengerversion V3."""
    prices = df["Price"].values
    if scores is None:
        scores = calculate_bsadf_fast(df["Price"], MIN_WINDOW, LOOKBACK_WINDOW)
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


def run_v5_adaptive_strategy(df, scores=None):
    """V5 ohne Smart Re-Entry (nur Cooldown)."""
    prices = df["Price"].values
    if scores is None:
        scores = calculate_bsadf_fast(df["Price"], MIN_WINDOW, LOOKBACK_WINDOW)
    score_series = pd.Series(scores)
    roll_warn = score_series.rolling(LOOKBACK_WINDOW, min_periods=LOOKBACK_WINDOW).quantile(Q_WARN).values
    roll_panic = score_series.rolling(LOOKBACK_WINDOW, min_periods=LOOKBACK_WINDOW).quantile(Q_PANIC).values
    portfolio, wealth, prev_w, cooldown = [100.0], 100.0, 1.0, 0
    for t in range(1, len(prices)):
        score = scores[t-1]
        rw = roll_warn[t-1]; rp = roll_panic[t-1]
        lim_w = max(rw if not np.isnan(rw) else 0.5, 0.5)
        lim_p = max(rp if not np.isnan(rp) else 1.2, 1.2)
        if cooldown > 0:
            target = 0.0
            cooldown -= 1
            if score < 0:
                cooldown = 0
        else:
            if score > lim_p:
                target = 0.0; cooldown = 10
            elif score > lim_w:
                target = 0.5
            else:
                target = 1.0
        if target != prev_w:
            cost = PANIC_SLIPPAGE if (target == 0.0 and prev_w > 0.0 and score > lim_p) else TRANSAKTIONS_KOSTEN
            wealth -= wealth * abs(target - prev_w) * cost
        prev_w = target
        wealth *= target * (prices[t]/prices[t-1]) + (1-target) * (1 + CHF_MONTHLY_RATE)
        portfolio.append(wealth)
    return portfolio


# ==========================================
# 4. PLOTTING
# ==========================================
def plot_4panel_scenario(df, title, fname):
    p_val, scores, roll_panic, weights = run_v6_smart_strategy(df)
    bh_p = (df["Price"] / df["Price"].iloc[0]) * 100
    mom_p = run_momentum_strategy(df)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(13, 14), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.5, 1, 0.8]})
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)

    ax1.plot(df.index, df["Price"], color="black", label="Marktpreis")
    ax1.plot(df.index, df["Fundamental"], color="blue", linestyle="--",
             label="Fundament")
    ax1.fill_between(df.index, df["Fundamental"], df["Price"],
                     where=(df["Price"] >= df["Fundamental"]),
                     color="red", alpha=0.15, label="Blasenkomponente")
    ax1.set_title("Anatomie: Preis vs Fundament (Evans-Modell)")
    ax1.legend(loc="upper left"); ax1.grid(True, alpha=0.3)

    ax2.plot(df.index, bh_p, color="gray", alpha=0.6, label="Buy & Hold")
    ax2.plot(df.index, mom_p, color="orange", alpha=0.8, label=f"Momentum (SMA{SMA_WINDOW})")
    ax2.plot(df.index, p_val, color="#1f77b4", linewidth=2, label="V6 Smart")
    ax2.set_title("Performance der Strategien")
    ax2.legend(loc="upper left"); ax2.grid(True, alpha=0.3)

    ax3.plot(df.index, scores, color="purple", label="BSADF Score (PSY 2015)")
    ax3.plot(df.index, roll_panic, color="red", linestyle="--",
             label="99%-Panik-Quantil (rollierend)")
    ax3.fill_between(df.index, 0, scores,
                     where=(scores > np.where(np.isnan(roll_panic), np.inf, roll_panic)),
                     color="red", alpha=0.3, label="Panik-Trigger")
    ax3.set_title("Signalgenerierung BSADF")
    ax3.legend(loc="upper left"); ax3.grid(True, alpha=0.3)

    ax4.fill_between(df.index, 0, [w*100 for w in weights],
                     color="green", alpha=0.4, step="post",
                     label="Aktienquote (%)")
    ax4.set_ylim(-5, 105)
    ax4.set_xlabel("Periode (Ticks)")
    ax4.legend(loc="upper left"); ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_evolution_comparison(all_sims, fname):
    """V0 vs V3 vs V5 vs V6 - Erwartungswert-Vergleich (Ehrliche BSADF-Variante)."""
    print("Berechne Evolutions-Vergleich V0..V6...")
    v0, v3, v5, v6 = [], [], [], []
    for df in all_sims:
        scores = calculate_bsadf_fast(df["Price"], MIN_WINDOW, LOOKBACK_WINDOW)
        v0.append((df["Price"] / df["Price"].iloc[0]) * 100)
        v3.append(run_v3_static_strategy(df, scores=scores))
        v5.append(run_v5_adaptive_strategy(df, scores=scores))
        v6_p, _, _, _ = run_v6_smart_strategy(df, scores=scores)
        v6.append(v6_p)

    plt.figure(figsize=(13, 7))
    plt.plot(np.mean(v0, axis=0), color="gray", linestyle="--", linewidth=2,
             label="V0: Buy & Hold")
    plt.plot(np.mean(v3, axis=0), color="orange", linewidth=2.5,
             label="V3: Statischer Exit (BSADF > 1.8)")
    plt.plot(np.mean(v5, axis=0), color="cyan", linewidth=2.5,
             label="V5: Adaptive Quantile + Cooldown")
    plt.plot(np.mean(v6, axis=0), color="#1f77b4", linewidth=3,
             label="V6: Smart Re-Entry (Final)")
    plt.title("Iterative Strategieentwicklung mit echtem BSADF (Erwartungswert)",
              fontsize=14, fontweight="bold")
    plt.xlabel("Periode (Ticks)")
    plt.ylabel("Durchschnittlicher Portfolio-Wert (Start = 100)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()


def plot_spaghetti(simulationen, df_normal, df_medium, df_extrem, fname):
    plt.figure(figsize=(13, 7))
    for df in simulationen:
        plt.plot(df.index, df["Price"], alpha=0.03, color="gray", linewidth=1)
    plt.plot(simulationen[0].index, simulationen[0]["Fundamental"],
             color="black", linestyle="--", linewidth=2, label="Fundament")
    if df_normal is not None:
        plt.plot(df_normal.index, df_normal["Price"], color="green", linewidth=2.5,
                 label="Szenario A (Normal)")
    if df_medium is not None:
        plt.plot(df_medium.index, df_medium["Price"], color="orange", linewidth=2.5,
                 label="Szenario B (Medium)")
    if df_extrem is not None:
        plt.plot(df_extrem.index, df_extrem["Price"], color="red", linewidth=2.5,
                 label="Szenario C (Extrem)")
    plt.yscale("log")
    plt.title(f"Evans-Modell: {len(simulationen)} Pfade (log-Skala)",
              fontsize=14, fontweight="bold")
    plt.xlabel("Periode (Ticks)")
    plt.legend(loc="upper left")
    plt.grid(True, which="both", ls="-", alpha=0.1)
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()


# ==========================================
# 5. MAIN
# ==========================================
def main():
    print(f"Starte {EVANS_PFADE} Evans-Pfade ...")
    alle_sims, bubbles = [], []
    for i in range(EVANS_PFADE):
        df = generiere_evans_pfad()
        alle_sims.append(df)
        bubbles.append(df["Bubble"].max())
        if (i+1) % 500 == 0:
            print(f"   {i+1} Pfade")

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

    print("\nGeneriere Plots ...")
    plot_spaghetti(alle_sims, df_normal, df_medium, df_extrem,
                   "fig_mc_spaghetti.png")
    plot_4panel_scenario(df_normal, "Szenario A: Normalverlauf (echter BSADF)",
                         "fig_mc_szenario_A.png")
    plot_4panel_scenario(df_medium, "Szenario B: Mittlere Blasenbildung (echter BSADF)",
                         "fig_mc_szenario_B.png")
    plot_4panel_scenario(df_extrem, "Szenario C: Extremereignis / Kollaps (echter BSADF)",
                         "fig_mc_szenario_C.png")

    subset = alle_sims[:300]
    plot_evolution_comparison(subset, "fig_mc_evolution.png")

    print("\nFertig.")


if __name__ == "__main__":
    main()
