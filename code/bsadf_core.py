"""
BSADF (Backward Sup ADF) Implementierung gemaess Phillips, Shi & Yu (2015).

Vektorisiertes NumPy-Implementation mit Newey-West HAC-korrigierten
Standardfehlern. Die Anzahl der HAC-Lags wird automatisch nach der
Newey-West-Empfehlung q = floor(4 * (T/100)^(2/9)) bestimmt.

Fuer jeden Endpunkt t:
    Fenster [r1, t] mit r1 von max(0, t - max_lookback) bis t - min_window
    Berechne ADF-Koeffizienten-t-Statistik fuer jedes Fenster
    BSADF[t] = max ueber alle r1
"""
import numpy as np


def newey_west_lag(T):
    """Empfohlene HAC-Lags nach Newey-West (1987, 1994)."""
    return max(1, int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0))))


def adf_tstat_hac(y_lag, dy):
    """
    Berechnet den t-Wert des Slope-Koeffizienten beta in der Regression
        dy_t = alpha + beta * y_{lag,t} + eps_t
    mit Newey-West HAC-Standardfehlern.

    Parameters
    ----------
    y_lag : 1D array
        Lagged level series (length n).
    dy : 1D array
        First differences (length n).

    Returns
    -------
    t_stat : float
        t-Wert des beta-Koeffizienten. NaN bei Singularitaet.
    """
    n = len(y_lag)
    if n < 10:
        return np.nan

    # Design-Matrix mit Konstante
    X = np.empty((n, 2))
    X[:, 0] = 1.0
    X[:, 1] = y_lag

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return np.nan

    beta = XtX_inv @ (X.T @ dy)
    resid = dy - X @ beta

    # HAC: Newey-West mit empfohlener Lag-Anzahl
    q = newey_west_lag(n)

    # Score-Matrix: u_t = X_t * resid_t  (shape n x 2)
    u = X * resid[:, None]

    # Gamma_0
    S = u.T @ u

    # Gamma_j fuer j=1..q mit Bartlett-Gewichten
    for j in range(1, q + 1):
        weight = 1.0 - j / (q + 1.0)
        Gamma_j = u[j:].T @ u[:-j]
        S = S + weight * (Gamma_j + Gamma_j.T)

    # HAC-Kovarianzmatrix: V = (X'X)^{-1} S (X'X)^{-1}
    V = XtX_inv @ S @ XtX_inv

    var_beta = V[1, 1]
    if var_beta <= 0 or not np.isfinite(var_beta):
        return np.nan

    se_beta = np.sqrt(var_beta)
    return beta[1] / se_beta


def calculate_bsadf(series, min_window=30, max_lookback=126):
    """
    Backward Sup ADF (BSADF) gemaess PSY (2015).

    Fuer jeden Endpunkt t wird das Supremum der ADF-t-Statistiken ueber
    Fenster mit variablem Startpunkt gebildet:
        BSADF_t = sup_{r1 in [t - max_lookback, t - min_window]} ADF_t(r1, t)

    Parameters
    ----------
    series : pd.Series oder np.ndarray
        Preisreihe.
    min_window : int
        Kleinstes Sub-Fenster, ueber das eine ADF-Regression laeuft.
    max_lookback : int
        Maximales Backward-Fenster (=  groesster moeglicher r1-Abstand).

    Returns
    -------
    bsadf : np.ndarray
        BSADF-Statistik (Laenge = Laenge der Serie). Werte vor Index
        min_window sind 0.
    """
    if hasattr(series, "values"):
        y = series.values.astype(float)
    else:
        y = np.asarray(series, dtype=float)

    # Normierung, damit numerische Skala unabhaengig vom Preisniveau
    y = y / y[0]

    n = len(y)
    bsadf = np.zeros(n)

    # Vorberechnung: dy[t] = y[t+1] - y[t]; y_lag[t] = y[t]
    dy_full = np.diff(y)
    y_lag_full = y[:-1]

    for t in range(min_window, n):
        # Endpunkt t (inklusive im Sinne der Beobachtung y[t]):
        # y_lag_full[r1:t] liefert y[r1..t-1], dy_full[r1:t] = y[r1+1..t] - y[r1..t-1].
        # Die Ex-ante-Eigenschaft fuer eine Handelsentscheidung an Tag t
        # entsteht erst durch den Aufrufer, der score = bsadf_scores[t-1]
        # verwendet (siehe backtest_v2.run_v6_smart_strategy).
        end = t  # exklusiv in Slicing
        earliest_start = max(0, t - max_lookback)
        latest_start = t - min_window

        if latest_start <= earliest_start:
            continue

        best = -np.inf
        for r1 in range(earliest_start, latest_start + 1):
            yl = y_lag_full[r1:end]
            dy_w = dy_full[r1:end]
            if len(yl) < min_window:
                continue
            ts = adf_tstat_hac(yl, dy_w)
            if np.isfinite(ts) and ts > best:
                best = ts

        if np.isfinite(best):
            bsadf[t] = best

    return bsadf


# -----------------------------------------------------------------------------
# JIT-beschleunigte Variante (falls numba verfuegbar; sonst Fallback)
# -----------------------------------------------------------------------------
try:
    from numba import njit, prange

    @njit(cache=True, fastmath=True)
    def _adf_tstat_hac_nb(y_lag, dy):
        n = y_lag.shape[0]
        if n < 10:
            return np.nan

        # X = [1, y_lag]
        # XtX
        s_y = 0.0
        s_yy = 0.0
        for i in range(n):
            s_y += y_lag[i]
            s_yy += y_lag[i] * y_lag[i]

        det = n * s_yy - s_y * s_y
        if det == 0.0:
            return np.nan

        inv00 = s_yy / det
        inv01 = -s_y / det
        inv11 = n / det

        # XtY
        sd = 0.0
        syd = 0.0
        for i in range(n):
            sd += dy[i]
            syd += y_lag[i] * dy[i]

        beta0 = inv00 * sd + inv01 * syd
        beta1 = inv01 * sd + inv11 * syd

        # Residuen
        resid = np.empty(n)
        for i in range(n):
            resid[i] = dy[i] - beta0 - beta1 * y_lag[i]

        # HAC-Lag
        q = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))

        # Score-Matrix u_t = (resid_t, y_lag_t * resid_t)
        u0 = resid.copy()
        u1 = y_lag * resid

        # Gamma_0
        S00 = 0.0
        S01 = 0.0
        S11 = 0.0
        for i in range(n):
            S00 += u0[i] * u0[i]
            S01 += u0[i] * u1[i]
            S11 += u1[i] * u1[i]

        # Gamma_j mit Bartlett-Gewichten
        for j in range(1, q + 1):
            w = 1.0 - j / (q + 1.0)
            G00 = 0.0
            G01 = 0.0
            G10 = 0.0
            G11 = 0.0
            for i in range(j, n):
                G00 += u0[i] * u0[i - j]
                G01 += u0[i] * u1[i - j]
                G10 += u1[i] * u0[i - j]
                G11 += u1[i] * u1[i - j]
            S00 += w * (G00 + G00)
            S11 += w * (G11 + G11)
            S01 += w * (G01 + G10)

        # V = (XtX)^-1 S (XtX)^-1
        # XtX^-1 = [[inv00, inv01],[inv01, inv11]]
        a = inv01 * S00 + inv11 * S01
        b = inv01 * S01 + inv11 * S11
        var_beta1 = inv01 * a + inv11 * b

        if var_beta1 <= 0.0 or not np.isfinite(var_beta1):
            return np.nan

        return beta1 / np.sqrt(var_beta1)

    @njit(cache=True, parallel=True, fastmath=True)
    def _bsadf_nb(y, min_window, max_lookback):
        n = y.shape[0]
        bsadf = np.zeros(n)
        dy_full = np.empty(n - 1)
        for i in range(n - 1):
            dy_full[i] = y[i + 1] - y[i]
        y_lag_full = y[:-1]

        for t in prange(min_window, n):
            earliest = t - max_lookback
            if earliest < 0:
                earliest = 0
            latest = t - min_window
            if latest <= earliest:
                continue
            best = -1e18
            found = False
            for r1 in range(earliest, latest + 1):
                yl = y_lag_full[r1:t]
                dy_w = dy_full[r1:t]
                if yl.shape[0] < min_window:
                    continue
                ts = _adf_tstat_hac_nb(yl, dy_w)
                if np.isfinite(ts) and ts > best:
                    best = ts
                    found = True
            if found:
                bsadf[t] = best
        return bsadf

    def calculate_bsadf_fast(series, min_window=30, max_lookback=126):
        if hasattr(series, "values"):
            y = series.values.astype(np.float64)
        else:
            y = np.asarray(series, dtype=np.float64)
        y = y / y[0]
        return _bsadf_nb(y, min_window, max_lookback)

    HAS_NUMBA = True

except ImportError:
    HAS_NUMBA = False

    def calculate_bsadf_fast(series, min_window=30, max_lookback=126):
        return calculate_bsadf(series, min_window, max_lookback)
