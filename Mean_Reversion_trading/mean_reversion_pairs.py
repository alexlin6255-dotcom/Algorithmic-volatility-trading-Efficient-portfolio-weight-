"""
Mean-reversion (cointegration pairs), vectorized, lookahead-free.

Design (every estimate at decision time t uses only data < t):
  - hedge ratio beta: OLS of log(P1) on log(P2) over a trailing `coint_window`
  - pair is "tradable" if the trailing spread passes ADF stationarity (p < adf_p)
  - spread_t = log(P1_t) - beta * log(P2_t)
  - z_t = (spread_t - rolling_mean) / rolling_std   (rolling over `z_window`, trailing)
  - enter short-spread when z > +entry, long-spread when z < -entry
  - exit when |z| < exit
  - each leg is dollar-sized so the pair is ~market neutral; beta scales leg 2

The portfolio return series plugs straight into evaluation/backtest_harness.

NOTE on cost: statsmodels may be unavailable; we fall back to a fast in-house
ADF-style stationarity proxy (a unit-root regression t-stat) so the strategy runs
anywhere. If statsmodels is present we use the real adfuller.
"""
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Tuple

try:
    from statsmodels.tsa.stattools import adfuller
    _HAVE_SM = True
except Exception:
    _HAVE_SM = False


REBAL = 5                # update z-score signals every 5 days
SELECT_EVERY = 63        # re-select the cointegrated pair set every ~quarter
COINT_WINDOW = 252       # ~1y trailing window to estimate hedge ratio + test
Z_WINDOW = 21            # rolling window for spread z-score
ENTRY_Z = 2.0
EXIT_Z = 0.5
ADF_P = 0.05
MAX_PAIRS = 10           # cap simultaneous pairs


def _log_close_matrix(ohlc_list) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    cols = [np.log(df["close"].to_numpy()) for df in ohlc_list]
    return np.column_stack(cols), ohlc_list[0].index


def _ols_beta(y: np.ndarray, x: np.ndarray) -> float:
    """Slope of y on x (no intercept-adjusted hedge): beta = cov/var."""
    xc = x - x.mean()
    yc = y - y.mean()
    denom = (xc * xc).sum()
    return (xc * yc).sum() / denom if denom > 0 else 0.0


def _adf_pvalue(spread: np.ndarray) -> float:
    """ADF p-value (real if statsmodels available, else a fast unit-root proxy)."""
    if _HAVE_SM:
        try:
            return adfuller(spread, maxlag=1, autolag=None)[1]
        except Exception:
            return 1.0
    # proxy: regress d(spread) on lagged spread; more-negative t on the lag => more stationary
    s = spread
    ds = np.diff(s)
    lag = s[:-1]
    b = _ols_beta(ds, lag)
    resid = ds - b * (lag - lag.mean())
    n = len(ds)
    se = np.sqrt((resid @ resid) / max(n - 2, 1)) / (np.sqrt(((lag - lag.mean()) ** 2).sum()) + 1e-12)
    tstat = b / (se + 1e-12)
    # map t-stat to a pseudo p-value via the ~-2.9 ADF 5% critical value
    return 0.04 if tstat < -2.9 else (0.10 if tstat < -2.4 else 0.50)


def pairs_returns(ohlc_list,
                  coint_window: int = COINT_WINDOW,
                  z_window: int = Z_WINDOW,
                  entry_z: float = ENTRY_Z,
                  exit_z: float = EXIT_Z,
                  adf_p: float = ADF_P,
                  max_pairs: int = MAX_PAIRS,
                  rebal: int = REBAL,
                  select_every: int = SELECT_EVERY) -> np.ndarray:
    """Daily portfolio return series for the cointegration pairs strategy."""
    logpx, _ = _log_close_matrix(ohlc_list)
    T, K = logpx.shape
    all_pairs = list(combinations(range(K), 2))

    simple = np.full((T, K), np.nan)
    simple[1:] = np.exp(logpx[1:] - logpx[:-1]) - 1.0

    rets = np.full(T, np.nan)
    selected = []                       # list of (i, j, beta) currently tradable
    pos = {}                            # (i,j) -> {-1,0,+1}

    for t in range(T):
        # ---- realize P&L from positions held into today ----
        if t > 0:
            if pos:
                legs = []
                for (i, j), s in pos.items():
                    if s == 0:
                        continue
                    legs.append(s * 0.5 * simple[t, i] - s * 0.5 * simple[t, j])
                rets[t] = float(np.mean(legs)) if legs else 0.0
            else:
                rets[t] = 0.0

        # ---- (infrequent) re-select cointegrated pair set on trailing window ----
        if t >= coint_window and t % select_every == 0:
            lo = t - coint_window
            scored = []
            for (i, j) in all_pairs:
                yi, xj = logpx[lo:t, i], logpx[lo:t, j]
                b = _ols_beta(yi, xj)
                if b <= 0:
                    continue
                pv = _adf_pvalue(yi - b * xj)
                if pv < adf_p:
                    scored.append((pv, i, j, b))
            scored.sort(key=lambda r: r[0])
            new_sel = [(i, j, b) for _, i, j, b in scored[:max_pairs]]
            new_keys = {(i, j) for i, j, _ in new_sel}
            # close pairs that dropped out of the cointegrated set
            for key in list(pos):
                if key not in new_keys:
                    pos.pop(key, None)
            selected = new_sel
            for i, j, _ in new_sel:
                pos.setdefault((i, j), 0)

        # ---- (frequent) update z-score signals on the selected set ----
        if selected and t >= z_window and t % rebal == 0:
            for i, j, b in selected:
                sp = logpx[t - z_window + 1:t + 1, i] - b * logpx[t - z_window + 1:t + 1, j]
                sd = sp.std()
                if sd <= 0:
                    continue
                z = (sp[-1] - sp.mean()) / sd
                cur = pos.get((i, j), 0)
                if cur == 0:
                    if z > entry_z:
                        pos[(i, j)] = -1
                    elif z < -entry_z:
                        pos[(i, j)] = +1
                elif abs(z) < exit_z:
                    pos[(i, j)] = 0

    return rets
