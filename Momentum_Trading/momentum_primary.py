"""
Momentum primary model (vectorized) + meta-labeling event extraction.

Logic mirrors the team's momentum_module.Model:
  - top-2 assets by trailing `lookback` return
  - equal weight, long-only, fully invested
  - rebalance every 21 trading days
Integer-share rounding and cash drag from the OO version are dropped: they don't
affect whether the *signal* has an edge, which is what we're testing.

CAUSALITY: every feature at decision day t uses only data up to and including t.
Labels look forward (that's allowed — they're the target, not an input).
"""
import numpy as np
import pandas as pd
from typing import Tuple

REBAL = 21
TOP_N = 2          # default; override per-call for larger universes


def _close_matrix(ohlc_list) -> Tuple[np.ndarray, pd.DatetimeIndex, list]:
    """Stack per-ticker close columns into a T x K matrix."""
    cols = [df["close"].to_numpy() for df in ohlc_list]
    mat = np.column_stack(cols)
    return mat, ohlc_list[0].index, [f"mkt{i}" for i in range(len(ohlc_list))]


def momentum_returns(ohlc_list, lookback: int, top_n: int = TOP_N) -> np.ndarray:
    """Run the momentum strategy, return the daily portfolio return series."""
    px, _, _ = _close_matrix(ohlc_list)
    T, K = px.shape
    alloc = np.zeros(K)          # dollar value held per asset
    cash = 1.0
    equity = 1.0
    rets = np.full(T, np.nan)

    for t in range(T):
        if t > 0:
            growth = px[t] / px[t - 1]
            alloc = alloc * growth
            new_equity = alloc.sum() + cash
            rets[t] = new_equity / equity - 1.0
            equity = new_equity
        if t >= lookback and t % REBAL == 0:
            mom = px[t] / px[t - lookback] - 1.0
            top = np.argsort(mom)[-top_n:]
            target = equity / top_n
            alloc = np.zeros(K)
            alloc[top] = target
            cash = equity - alloc.sum()
    return rets


def benchmark_equal_weight(ohlc_list) -> np.ndarray:
    """Buy-and-hold equal weight basket — the 'just ride the drift' baseline."""
    px, _, _ = _close_matrix(ohlc_list)
    T, K = px.shape
    daily = px[1:] / px[:-1] - 1.0          # (T-1) x K
    port = daily.mean(axis=1)
    return np.concatenate([[np.nan], port])


# ----------------------------------------------------------------------------
# Meta-labeling dataset: one row per (rebalance day, picked asset).
# ----------------------------------------------------------------------------
def build_meta_dataset(ohlc_list, lookback: int, top_n: int = TOP_N) -> pd.DataFrame:
    px, index, _ = _close_matrix(ohlc_list)
    T, K = px.shape
    logpx = np.log(px)
    ret = np.full_like(px, np.nan)
    ret[1:] = px[1:] / px[:-1] - 1.0

    # rolling helpers (causal: stat at t uses window ending at t)
    def roll_std(arr, w):
        out = np.full_like(arr, np.nan)
        for k in range(K):
            s = pd.Series(arr[:, k]).rolling(w).std().to_numpy()
            out[:, k] = s
        return out

    def roll_mean(arr, w):
        out = np.full_like(arr, np.nan)
        for k in range(K):
            s = pd.Series(arr[:, k]).rolling(w).mean().to_numpy()
            out[:, k] = s
        return out

    vol20 = roll_std(ret, 20)
    ma50 = roll_mean(px, 50)
    sd50 = roll_std(px, 50)

    rows = []
    rebal_days = [t for t in range(T) if t >= lookback and t % REBAL == 0]
    for di, t in enumerate(rebal_days):
        mom = px[t] / px[t - lookback] - 1.0
        order = np.argsort(mom)              # ascending
        ranks = np.empty(K, dtype=int)
        ranks[order] = np.arange(K)          # 0=worst .. K-1=best
        top = np.argsort(mom)[-top_n:]
        disp = np.nanstd(mom)                # cross-sectional dispersion (regime proxy)

        # forward window = until next rebalance (non-overlapping by construction)
        t_next = rebal_days[di + 1] if di + 1 < len(rebal_days) else T - 1
        if t_next <= t:
            continue

        for k in top:
            fwd = px[t_next, k] / px[t, k] - 1.0     # forward return of this pick
            short5 = px[t, k] / px[t - 5, k] - 1.0 if t >= 5 else np.nan
            ma_z = (px[t, k] - ma50[t, k]) / sd50[t, k] if sd50[t, k] and not np.isnan(sd50[t, k]) else np.nan
            rows.append({
                "t": t, "t_next": t_next, "date": index[t], "asset": k,
                # ---- features (all known at t) ----
                "f_mom": mom[k],
                "f_rank": ranks[k],
                "f_vol20": vol20[t, k],
                "f_ma_z": ma_z,
                "f_short5": short5,
                "f_dispersion": disp,
                "f_mom_vol": mom[k] / vol20[t, k] if vol20[t, k] and not np.isnan(vol20[t, k]) else np.nan,
                # ---- label & bookkeeping ----
                "fwd_ret": fwd,
                "label": int(fwd > 0),
            })

    df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    return df


FEATURES = ["f_mom", "f_rank", "f_vol20", "f_ma_z", "f_short5", "f_dispersion", "f_mom_vol"]
