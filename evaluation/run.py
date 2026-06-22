"""
Run the whole pipeline:
  1. load data, build OHLC list, sanity-check
  2. momentum primary: optimize lookback, report vs equal-weight benchmark
  3. in-sample MCPT on momentum (re-optimize per permutation)
  4. meta-labeling on the chosen-lookback momentum: purged OOF, filtered vs primary
"""

from Momentum_Trading.momentum_primary import (
    momentum_returns, benchmark_equal_weight, build_meta_dataset, FEATURES
)
from evaluation.meta_model import purged_oof_predictions, filtered_vs_primary
from evaluation.backtest_harness import metrics_from_returns, in_sample_mcpt
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

LOOKBACK_GRID = [20, 40, 60, 90, 120]
MCPT_METRIC = "sharpe"
N_PERMS = 200
TOP_N = 2          # set via --top_n


def load_ohlc(path="SPY.csv"):
    data = pd.read_csv(path, index_col=0, header=[0, 1], parse_dates=True)
    tickers = list(data["Close"].columns)
    ohlc_list = []
    for tk in tickers:
        df = pd.DataFrame({
            "open": data["Open"][tk], "high": data["High"][tk],
            "low": data["Low"][tk], "close": data["Close"][tk],
        }).dropna()
        ohlc_list.append(df)
    # align on common index
    idx = ohlc_list[0].index
    for df in ohlc_list[1:]:
        idx = idx.intersection(df.index)
    ohlc_list = [df.loc[idx] for df in ohlc_list]
    return ohlc_list, tickers


def sanity_check(ohlc_list, tickers):
    print("\n--- data sanity ---")
    flagged = []
    for tk, df in zip(tickers, ohlc_list):
        lr = np.diff(np.log(df["close"].to_numpy()))
        mx = np.abs(lr).max()
        if mx >= 0.5:
            flagged.append((tk, mx))
    print(f"{len(tickers)} tickers, {len(ohlc_list[0])} common bars "
          f"({ohlc_list[0].index.min().date()} -> {ohlc_list[0].index.max().date()})")
    if flagged:
        print("  POSSIBLE SPLIT ARTIFACTS:", flagged)
    else:
        print("  no daily |log ret| >= 0.5 — split/div adjustment looks clean")


def optimize_lookback(ohlc_list):
    """Return (best metric, best lookback) over the grid. Used for real and each perm."""
    best_m, best_p = -np.inf, None
    for lb in LOOKBACK_GRID:
        rets = momentum_returns(ohlc_list, lb, top_n=TOP_N)
        m = metrics_from_returns(rets)[MCPT_METRIC]
        if np.isnan(m):
            continue
        if m > best_m:
            best_m, best_p = m, lb
    return best_m, best_p


def main():
    global TOP_N
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="SPY.csv")
    ap.add_argument("--top_n", type=int, default=2)
    ap.add_argument("--n_perms", type=int, default=N_PERMS)
    ap.add_argument("--out", default="pipeline_results.png")
    args = ap.parse_args()
    TOP_N = args.top_n

    ohlc_list, tickers = load_ohlc(args.data)
    print(f"loaded {len(tickers)} tickers from {args.data}")
    print(f"tickers: {tickers}")
    sanity_check(ohlc_list, tickers)

    # ---- 2. primary momentum, optimized ----
    print(
        f"\n========== MOMENTUM (primary), lookback scan, top-{TOP_N} ==========")
    for lb in LOOKBACK_GRID:
        m = metrics_from_returns(momentum_returns(ohlc_list, lb, top_n=TOP_N))
        print(f"  lookback={lb:3d}  Sharpe={m['sharpe']:.3f}  CAGR={m['cagr']:.2%}"
              f"  maxDD={m['max_dd']:.2%}  PF={m['profit_factor']:.2f}")
    best_sharpe, best_lb = optimize_lookback(ohlc_list)
    print(f"  -> best lookback = {best_lb} (Sharpe {best_sharpe:.3f})")

    bench = metrics_from_returns(benchmark_equal_weight(ohlc_list))
    print(f"  equal-weight buy&hold benchmark: Sharpe={bench['sharpe']:.3f}"
          f"  CAGR={bench['cagr']:.2%}  maxDD={bench['max_dd']:.2%}")
    print(f"  edge over basket (Sharpe): {best_sharpe - bench['sharpe']:+.3f}")

    # ---- 3. in-sample MCPT ----
    print(
        f"\n========== IN-SAMPLE MCPT ({args.n_perms} perms, re-optimizing lookback each) ==========")
    res = in_sample_mcpt(ohlc_list, optimize_lookback,
                         n_perms=args.n_perms, seed=0)
    print(
        f"  real best Sharpe      : {res['real_metric']:.3f}  (lookback {res['real_param']})")
    print(f"  noise best Sharpe mean: {res['null_mean']:.3f}")
    print(f"  noise best Sharpe p95 : {res['null_p95']:.3f}")
    print(f"  p-value               : {res['p_value']:.3f}")
    verdict = ("edge survives selection bias" if res["p_value"] < 0.05
               else "NOT distinguishable from optimized noise")
    print(f"  verdict               : {verdict}")

    # ---- 4. meta-labeling ----
    print(
        f"\n========== META-LABELING on lookback={best_lb} momentum, top-{TOP_N} ==========")
    ds = build_meta_dataset(ohlc_list, best_lb, top_n=TOP_N)
    print(
        f"  events: {len(ds)}  (label base rate = {ds['label'].mean():.2%} profitable picks)")

    best_oof = None
    for kind in ("logit", "gb"):
        oof, auc, aucs = purged_oof_predictions(
            ds, n_splits=5, embargo=2, model_kind=kind)
        cov = np.mean(~np.isnan(oof))
        print(f"  [{kind}] purged OOF AUC = {auc:.3f}   (fold AUCs: "
              f"{', '.join(f'{a:.2f}' for a in aucs)};  OOF coverage {cov:.0%})")
        if kind == "gb":
            ds_oof, best_oof = ds, oof

    thr = np.nanmedian(best_oof)
    prim, filt, prim_p, filt_p = filtered_vs_primary(
        ds_oof, best_oof, threshold=thr)
    print(f"\n  filter threshold P>= {thr:.3f}")
    print(f"  {'metric':<16}{'primary':>12}{'meta-filtered':>16}")
    for k in ("total_return", "sharpe_ann", "hit_rate", "max_dd", "n_periods"):
        pv, fv = prim[k], filt[k]
        fmt = (lambda x: f"{x:.2%}") if k in ("total_return", "hit_rate", "max_dd") else \
              (lambda x: f"{x:.3f}") if k == "sharpe_ann" else (
                  lambda x: f"{int(x)}")
        print(f"  {k:<16}{fmt(pv):>12}{fmt(fv):>16}")

    # ---- plot ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].hist(res["null"], bins=30, color="#7A899E",
               alpha=0.8, label="noise (best-of-grid)")
    ax[0].axvline(res["real_metric"], color="#F2C14E", lw=2.2, label="real")
    ax[0].set_title(f"In-sample MCPT  (p = {res['p_value']:.3f})")
    ax[0].set_xlabel("best Sharpe over lookback grid")
    ax[0].legend()

    ax[1].plot(np.cumprod(1 + prim_p), color="#7A899E",
               label="primary momentum")
    ax[1].plot(np.cumprod(1 + filt_p), color="#3FB8A0", label="meta-filtered")
    ax[1].set_title("Primary vs meta-filtered (OOF, per-rebalance equity)")
    ax[1].set_xlabel("rebalance period")
    ax[1].set_ylabel("growth of $1")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
