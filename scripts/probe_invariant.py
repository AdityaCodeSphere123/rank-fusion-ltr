"""Compare invariant feature layouts on a left-out slice of TRAIN.

DEV is not read. The fit is 1,200 queries and the score is 400 other queries.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from harness.cv import grouped_kfold
from harness.metrics import mean_ndcg_from_scores
from scripts.data_cache import load_xyq
from scripts.lambdamart_utils import default_params, fit
from scripts.train_model import _expand, _run_codes

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _take(X, y, q, idx, n, seed):
    q = np.asarray(q)
    sub = q[idx]
    starts = np.flatnonzero(np.r_[True, sub[1:] != sub[:-1]])
    ends = np.r_[starts[1:], len(sub)]
    order = np.random.RandomState(seed).permutation(len(starts))[:n]
    order.sort()
    rows = np.concatenate([idx[starts[i]:ends[i]] for i in order])
    return X[rows], y[rows], np.asarray(q[rows])


def _corrupt(X):
    rng = np.random.RandomState(0)
    X = np.asarray(X, dtype=np.float32)
    std = np.clip(X.std(0), 1e-6, None).astype(np.float32)
    out = {
        "clean": X,
        "log1p": (np.sign(X) * np.log1p(np.abs(X))).astype(np.float32),
        "zscore": ((X - X.mean(0)) / std).astype(np.float32),
        "x10": np.asarray(X * 10, dtype=np.float32),
        "add0.05": (X + 0.05 * std * rng.randn(*X.shape).astype(np.float32)).astype(np.float32),
        "add0.20": (X + 0.20 * std * rng.randn(*X.shape).astype(np.float32)).astype(np.float32),
        "mul0.20": (X * (1.0 + 0.20 * rng.randn(*X.shape).astype(np.float32))).astype(np.float32),
    }
    cell = np.array(X, copy=True)
    cell[rng.rand(*X.shape) < 0.10] = 0
    out["cell10"] = cell
    perm = np.array(X, copy=True)
    for g in range(25):
        perm[:, 5 * g:5 * g + 5] = X[:, 5 * g + rng.permutation(5)]
    out["permute"] = perm
    return out


def _layout(X, q, mode):
    codes = _run_codes(np.asarray(q))
    both = _expand(X, codes, False, with_z=True)
    d = X.shape[1]
    raw, pct, z = both[:, :d], both[:, d:2 * d], both[:, 2 * d:]
    if mode == "raw_pct_z":
        return both
    if mode == "pct_z":
        return np.hstack([pct, z])
    if mode == "pct":
        return pct
    if mode == "pct_zzlog":
        log = (np.sign(X) * np.log1p(np.abs(X))).astype(np.float32)
        zlog = _expand(log, codes, False, with_z=True)[:, 2 * d:]
        return np.hstack([pct, z, zlog])
    if mode == "raw_pct_z_zlog":
        log = (np.sign(X) * np.log1p(np.abs(X))).astype(np.float32)
        zlog = _expand(log, codes, False, with_z=True)[:, 2 * d:]
        return np.hstack([raw, pct, z, zlog])
    raise ValueError(mode)


def main():
    X, y, q = load_xyq(os.path.join(ROOT, "data", "train", "train.txt"))
    tr, va = grouped_kfold(q, k=5, seed=1)[0]
    Xtr, ytr, qtr = _take(X, y, q, tr, 1200, 1)
    Xva, yva, qva = _take(X, y, q, va, 400, 2)
    corrupted = _corrupt(Xva)
    params = default_params(63, 0.05, 20, lambda_l2=0.1, seed=1, feature_fraction=0.8,
                            bagging_fraction=0.9, bagging_freq=1, num_threads=6)
    keys = list(corrupted)
    for mode in ("raw_pct_z_zlog",):
        print(f"== {mode} ==", flush=True)
        packed = {name: _layout(raw, qva, mode) for name, raw in corrupted.items()}
        model = fit(_layout(Xtr, qtr, mode), ytr, _run_codes(qtr), params, 200)
        scores = {}
        for key in keys:
            pred = model.predict(packed[key], num_threads=4)
            scores[key] = mean_ndcg_from_scores(pred, yva, qva, 10)
        corrupt = float(np.mean([scores[k] for k in keys if k != "clean"]))
        detail = " ".join(f"{k} {scores[k]:.4f}" for k in keys)
        print(f"{detail}  corrupt {corrupt:.4f}", flush=True)


if __name__ == "__main__":
    main()
