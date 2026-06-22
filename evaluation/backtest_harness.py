"""
Shared evaluation harness.

One entry point for every strategy: strategy -> daily return series -> metrics -> MCPT.
Keeping all strategies behind this single door is what guarantees the metric is
computed identically on real and permuted data (the asymmetry that quietly biases
a permutation p-value is exactly what this prevents).
"""
import numpy as np
import pandas as pd
from typing import List, Union, Callable, Dict


TRADING_DAYS = 252


# ----------------------------------------------------------------------------
# Metrics  (all computed from a daily *return* series, decimal form)
# ----------------------------------------------------------------------------
def metrics_from_returns(daily_ret: np.ndarray) -> Dict[str, float]:
    daily_ret = np.asarray(daily_ret, dtype=float)
    daily_ret = daily_ret[~np.isnan(daily_ret)]
    if daily_ret.size < 2:
        return {"sharpe": np.nan, "cagr": np.nan, "total_return": np.nan,
                "max_dd": np.nan, "profit_factor": np.nan, "vol": np.nan}

    equity = np.cumprod(1.0 + daily_ret)
    total_return = equity[-1] - 1.0
    years = daily_ret.size / TRADING_DAYS
    cagr = equity[-1] ** (1.0 / years) - 1.0 if years > 0 else np.nan

    vol = daily_ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
    mean_ann = daily_ret.mean() * TRADING_DAYS
    sharpe = mean_ann / vol if vol > 0 else np.nan

    running_max = np.maximum.accumulate(equity)
    max_dd = (equity / running_max - 1.0).min()

    gains = daily_ret[daily_ret > 0].sum()
    losses = -daily_ret[daily_ret < 0].sum()
    profit_factor = gains / losses if losses > 0 else np.inf

    return {"sharpe": sharpe, "cagr": cagr, "total_return": total_return,
            "max_dd": max_dd, "profit_factor": profit_factor, "vol": vol}


# ----------------------------------------------------------------------------
# Bar permutation  (multi-market: ONE permutation applied across all tickers,
# so contemporaneous cross-sectional correlation is preserved while time-order
# / momentum persistence is destroyed). This is the version studied earlier.
# ----------------------------------------------------------------------------
def get_permutation(ohlc: Union[pd.DataFrame, List[pd.DataFrame]],
                    start_index: int = 0, seed=None):
    assert start_index >= 0
    rng = np.random.default_rng(seed)

    single = not isinstance(ohlc, list)
    if single:
        ohlc = [ohlc]
    time_index = ohlc[0].index
    for mkt in ohlc:
        assert np.all(time_index == mkt.index), "Indexes do not match"
    n_markets = len(ohlc)
    n_bars = len(ohlc[0])

    perm_index = start_index + 1
    perm_n = n_bars - perm_index

    start_bar = np.empty((n_markets, 4))
    relative_open = np.empty((n_markets, perm_n))
    relative_high = np.empty((n_markets, perm_n))
    relative_low = np.empty((n_markets, perm_n))
    relative_close = np.empty((n_markets, perm_n))

    for mkt_i, reg_bars in enumerate(ohlc):
        log_bars = np.log(reg_bars[["open", "high", "low", "close"]])
        start_bar[mkt_i] = log_bars.iloc[start_index].to_numpy()
        r_o = (log_bars["open"] - log_bars["close"].shift()).to_numpy()
        r_h = (log_bars["high"] - log_bars["open"]).to_numpy()
        r_l = (log_bars["low"] - log_bars["open"]).to_numpy()
        r_c = (log_bars["close"] - log_bars["open"]).to_numpy()
        relative_open[mkt_i] = r_o[perm_index:]
        relative_high[mkt_i] = r_h[perm_index:]
        relative_low[mkt_i] = r_l[perm_index:]
        relative_close[mkt_i] = r_c[perm_index:]

    idx = np.arange(perm_n)
    perm1 = rng.permutation(idx)          # intrabar trio, locked together
    relative_high = relative_high[:, perm1]
    relative_low = relative_low[:, perm1]
    relative_close = relative_close[:, perm1]
    perm2 = rng.permutation(idx)          # gaps, shuffled independently
    relative_open = relative_open[:, perm2]

    perm_ohlc = []
    for mkt_i, reg_bars in enumerate(ohlc):
        log_bars = np.log(reg_bars[["open", "high", "low", "close"]]).to_numpy().copy()
        perm_bars = np.zeros((n_bars, 4))
        perm_bars[:start_index] = log_bars[:start_index]
        perm_bars[start_index] = start_bar[mkt_i]
        for i in range(perm_index, n_bars):
            k = i - perm_index
            perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[mkt_i][k]
            perm_bars[i, 1] = perm_bars[i, 0] + relative_high[mkt_i][k]
            perm_bars[i, 2] = perm_bars[i, 0] + relative_low[mkt_i][k]
            perm_bars[i, 3] = perm_bars[i, 0] + relative_close[mkt_i][k]
        perm_bars = np.exp(perm_bars)
        perm_bars = pd.DataFrame(perm_bars, index=time_index,
                                 columns=["open", "high", "low", "close"])
        perm_ohlc.append(perm_bars)

    return perm_ohlc if n_markets > 1 else perm_ohlc[0]


# ----------------------------------------------------------------------------
# In-sample MCPT.
#   optimize_fn(ohlc_list) -> (best_metric, best_param)   re-runs the WHOLE
#   parameter search and returns the best achievable metric on that dataset.
# Comparing the real best against the distribution of best-on-noise is what
# makes this catch optimisation/selection bias, not just luck.
# ----------------------------------------------------------------------------
def in_sample_mcpt(ohlc_list: List[pd.DataFrame],
                   optimize_fn: Callable,
                   n_perms: int = 200,
                   start_index: int = 0,
                   seed: int = 0):
    real_metric, real_param = optimize_fn(ohlc_list)

    null = np.empty(n_perms)
    for p in range(n_perms):
        perm = get_permutation(ohlc_list, start_index=start_index, seed=seed + p + 1)
        if not isinstance(perm, list):
            perm = [perm]
        m, _ = optimize_fn(perm)
        null[p] = m

    beat = int(np.sum(null >= real_metric))
    p_value = (beat + 1) / (n_perms + 1)
    return {
        "real_metric": real_metric,
        "real_param": real_param,
        "null": null,
        "p_value": p_value,
        "null_mean": float(np.nanmean(null)),
        "null_p95": float(np.nanpercentile(null, 95)),
    }
