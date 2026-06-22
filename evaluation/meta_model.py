"""
Meta-model: learn 'should I trust this momentum pick?' on top of the primary.

Validation is purged + embargoed and walk-forward in time. Even though the
forward windows here tile rather than overlap (holding == rebalance gap), we
still embargo a few events at each train/test seam so no test event's window
touches a training event's window. OOF predictions are used for the filtered
backtest so the comparison itself carries no lookahead.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from Momentum_Trading.momentum_primary import FEATURES


def purged_oof_predictions(df: pd.DataFrame, n_splits: int = 5,
                           embargo: int = 2, model_kind: str = "gb"):
    """Walk-forward expanding-window CV with an embargo. Returns OOF P(label=1)."""
    n = len(df)
    oof = np.full(n, np.nan)
    fold_size = n // (n_splits + 1)
    aucs = []

    for s in range(1, n_splits + 1):
        train_end = fold_size * s
        test_start = train_end + embargo          # embargo gap
        test_end = min(train_end + fold_size, n)
        if test_start >= test_end:
            continue
        tr = df.iloc[:train_end]
        te = df.iloc[test_start:test_end]

        Xtr, ytr = tr[FEATURES].to_numpy(), tr["label"].to_numpy()
        Xte, yte = te[FEATURES].to_numpy(), te["label"].to_numpy()
        if len(np.unique(ytr)) < 2:
            continue

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        if model_kind == "logit":
            clf = LogisticRegression(max_iter=1000, C=0.5)
        else:
            clf = GradientBoostingClassifier(
                n_estimators=120, max_depth=2, learning_rate=0.05,
                subsample=0.8, random_state=0)
        clf.fit(Xtr_s, ytr)
        prob = clf.predict_proba(Xte_s)[:, 1]
        oof[test_start:test_end] = prob
        if len(np.unique(yte)) == 2:
            aucs.append(roc_auc_score(yte, prob))

    return oof, (float(np.mean(aucs)) if aucs else np.nan), aucs


def filtered_vs_primary(df: pd.DataFrame, oof: np.ndarray, threshold: float):
    """
    Build a per-event 'taken/skipped' comparison using OOF probabilities.
    Aggregates the two picks per rebalance into a portfolio leg return so the
    filtered and primary curves are comparable on the same events.
    """
    d = df.copy()
    d["meta_p"] = oof
    d = d.dropna(subset=["meta_p"])

    # group by rebalance day; each day has up to TOP_N picks
    primary_period, filtered_period = [], []
    for t, g in d.groupby("t"):
        # primary: equal weight across the picks (what momentum does)
        primary_period.append(g["fwd_ret"].mean())
        # filtered: keep only picks the meta-model trusts; else that capital is cash (0)
        keep = g[g["meta_p"] >= threshold]
        if len(keep) == 0:
            filtered_period.append(0.0)
        else:
            # capital splits equally across the ORIGINAL picks; skipped legs -> 0
            filtered_period.append(keep["fwd_ret"].sum() / len(g))

    primary_period = np.array(primary_period)
    filtered_period = np.array(filtered_period)

    def summarize(period_rets):
        eq = np.cumprod(1.0 + period_rets)
        total = eq[-1] - 1.0
        # ~12 periods/yr (21d); annualize period Sharpe
        mu, sd = period_rets.mean(), period_rets.std(ddof=1)
        sharpe = (mu / sd) * np.sqrt(12) if sd > 0 else np.nan
        wins = (period_rets > 0).mean()
        rm = np.maximum.accumulate(eq)
        mdd = (eq / rm - 1).min()
        return {"total_return": total, "sharpe_ann": sharpe,
                "hit_rate": wins, "max_dd": mdd, "n_periods": len(period_rets)}

    return summarize(primary_period), summarize(filtered_period), primary_period, filtered_period
