"""Held-out TRAIN probe for Track C2. DEV is never read.

Fits a few representations on 2,500 queries and scores a disjoint 500-query
slice under clean data and under the corruptions that could be the five
hidden versions. Selection uses this file only; the shipped fit is separate.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time

import numpy as np

from harness.cv import grouped_kfold
from harness.metrics import mean_ndcg_from_scores
from scripts.data_cache import load_xyq
from scripts.lambdamart_utils import default_params, fit
from submission.ranking import within_query_ranks

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def query_pct(X, q):
    out = np.empty(X.shape, dtype=np.float32)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    for a, b in zip(starts, ends):
        out[a:b] = within_query_ranks(np.asarray(X[a:b], dtype=np.float64))
    return out


def group_aggs(X):
    g = np.asarray(X[:, :125], dtype=np.float32).reshape(X.shape[0], 25, 5)
    return np.concatenate([g.max(2), g.mean(2), g.min(2)], axis=1)


def featurize(X, q, mode):
    pct = query_pct(X, q)
    if mode == "pct":
        return pct
    if mode == "both":
        return np.hstack([X, pct]).astype(np.float32)
    aggs = group_aggs(X)
    return np.hstack([X, pct, aggs, query_pct(aggs, q)]).astype(np.float32)


def field_copies(X, y, q, n_clean):
    xs, ys, qs = [], [], []
    q = np.asarray(q, dtype=object)
    for i in range(n_clean):
        xs.append(X)
        ys.append(y)
        qs.append(np.array([f"{v}#c{i}" for v in q], dtype=object))
    for f in range(5):
        Z = np.zeros_like(X)
        Z[:, f:125:5] = X[:, f:125:5]
        Z[:, 125:] = X[:, 125:]
        xs.append(Z)
        ys.append(y)
        qs.append(np.array([f"{v}#f{f}" for v in q], dtype=object))
    return np.vstack(xs), np.concatenate(ys), np.concatenate(qs)


def corruptions(X):
    rng = np.random.RandomState(0)
    out = {"clean": X, "log1p": np.sign(X) * np.log1p(np.abs(X)), "times10": X * 10}
    std = np.clip(X.std(0), 1e-6, None).astype(np.float32)
    out["zglob"] = ((X - X.mean(0)) / std).astype(np.float32)
    out["add0.1"] = (X + 0.1 * std * rng.randn(*X.shape).astype(np.float32)).astype(np.float32)
    scale = np.exp(rng.normal(0, 1, size=X.shape[1])).astype(np.float32)
    out["lognorm"] = (X * scale).astype(np.float32)
    perm = X.copy()
    for g in range(25):
        perm[:, 5 * g:5 * g + 5] = X[:, 5 * g + rng.permutation(5)]
    out["permute"] = perm
    only, drop = [], []
    for f in range(5):
        Z = np.zeros_like(X)
        Z[:, f:125:5] = X[:, f:125:5]
        Z[:, 125:] = X[:, 125:]
        only.append(Z)
        D = X.copy()
        D[:, f:125:5] = 0
        drop.append(D)
    return out, only, drop


def score(model, X, y, q, mode, rounds):
    F = featurize(X, q, mode)
    return {r: mean_ndcg_from_scores(model.predict(F, num_threads=4, num_iteration=r), y, q, 10) for r in rounds}


def main():
    X, y, q = load_xyq(os.path.join(ROOT, "data", "train", "train.txt"))
    tr_idx, va_idx = grouped_kfold(q, k=5, seed=0)[0]
    # Disjoint queries, capped so four fits finish in one pass.
    tr_q = list(dict.fromkeys(q[tr_idx].tolist()))[:2500]
    va_q = list(dict.fromkeys(q[va_idx].tolist()))[:500]
    tr_keep = np.isin(q, tr_q)
    va_keep = np.isin(q, va_q)
    Xtr, ytr, qtr = X[tr_keep], y[tr_keep], q[tr_keep]
    Xva, yva, qva = X[va_keep], y[va_keep], q[va_keep]
    print(f"train queries {len(tr_q)} rows {len(ytr)} | val queries {len(va_q)} rows {len(yva)}", flush=True)

    rounds = (80, 160, 240)
    specs = [
        ("base_both", "both", False, dict(num_leaves=32, learning_rate=0.05, min_data_in_leaf=101,
                                           lambda_l2=6.0, feature_fraction=0.7, max_bin=64, extra_trees=True)),
        ("strong_both", "both", False, dict(num_leaves=127, learning_rate=0.05, min_data_in_leaf=40,
                                             lambda_l2=1.0, feature_fraction=0.8, max_bin=255)),
        ("strong_pct", "pct", False, dict(num_leaves=127, learning_rate=0.05, min_data_in_leaf=40,
                                           lambda_l2=1.0, feature_fraction=0.9, max_bin=255)),
        ("strong_aggs", "aggs", False, dict(num_leaves=127, learning_rate=0.05, min_data_in_leaf=40,
                                             lambda_l2=1.0, feature_fraction=0.8, max_bin=255)),
        ("field_both", "both", True, dict(num_leaves=63, learning_rate=0.05, min_data_in_leaf=50,
                                           lambda_l2=1.0, feature_fraction=0.8, max_bin=255)),
        ("field_aggs", "aggs", True, dict(num_leaves=63, learning_rate=0.05, min_data_in_leaf=50,
                                           lambda_l2=1.0, feature_fraction=0.8, max_bin=255)),
    ]
    cor, only, drop = corruptions(Xva)
    models = {}
    for name, mode, copies, kw in specs:
        t0 = time.perf_counter()
        Xt, yt, qt = (field_copies(Xtr, ytr, qtr, n_clean=2) if copies else (Xtr, ytr, qtr))
        Ft = featurize(Xt, qt, mode)
        params = default_params(seed=1, num_threads=10, deterministic=False, **kw)
        model = fit(Ft, yt, qt, params, max(rounds))
        models[name] = (model, mode)
        print(f"\n== {name} features {Ft.shape[1]} copies {copies} {time.perf_counter() - t0:.0f}s ==", flush=True)
        rows = {k: score(model, v, yva, qva, mode, rounds) for k, v in cor.items()}
        only_s = [score(model, v, yva, qva, mode, rounds) for v in only]
        drop_s = [score(model, v, yva, qva, mode, rounds) for v in drop]
        for r in rounds:
            only_m = float(np.mean([s[r] for s in only_s]))
            drop_m = float(np.mean([s[r] for s in drop_s]))
            hard = np.mean([rows["log1p"][r], rows["times10"][r], rows["zglob"][r], rows["lognorm"][r]])
            print(f"  trees {r:3d}  clean {rows['clean'][r]:.4f}  only {only_m:.4f}  drop {drop_m:.4f}  "
                  f"perm {rows['permute'][r]:.4f}  add {rows['add0.1'][r]:.4f}  "
                  f"scale-mean {hard:.4f}", flush=True)


if __name__ == "__main__":
    main()
