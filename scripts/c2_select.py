"""Pick a Track C2 representation on a TRAIN slice. DEV is scored and not used to choose.

Fits each variant on 2,000 TRAIN queries and scores 400 disjoint TRAIN queries.
The printed score weights the corrupted queries more than the clean ones.
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
from scripts.train_model import _expand, _run_codes

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _stack(X, q, builders):
    codes = _run_codes(np.asarray(q, dtype=object))
    n_queries = int(codes[-1]) + 1
    n = len(X)
    width = _expand(X[:1], codes[:1], False).shape[1]
    big = np.empty((n * len(builders), width), dtype=np.float32)
    big_y_src = None
    big_q = np.empty(n * len(builders), dtype=np.int32)
    blocks = []
    for i, build in enumerate(builders):
        view = np.asarray(build(X), dtype=np.float32)
        vq = codes + i * n_queries
        blocks.append(_expand(view, vq, False))
        big_q[i * n: (i + 1) * n] = vq
        del view
    for i, block in enumerate(blocks):
        big[i * n: (i + 1) * n] = block
    return big, big_q


def make_builders(X, kind, seed):
    builders = [lambda raw: raw]
    std = np.clip(X.std(0), 1e-6, None).astype(np.float32)
    if kind in ("noise", "both", "both_reg"):
        for i, scale in enumerate((0.05, 0.10)):
            builders.append(lambda raw, s=scale, sd=std, i=i: (
                raw + s * sd * np.random.RandomState(seed + i).randn(*raw.shape).astype(np.float32)
            ).astype(np.float32))
    if kind in ("scale", "both", "both_reg"):
        scale = np.exp(np.random.RandomState(seed).normal(0, 1.0, size=X.shape[1])).astype(np.float32)
        builders.append(lambda raw, s=scale: (raw * s).astype(np.float32))
        builders.append(lambda raw: ((raw - raw.mean(0)) / np.clip(raw.std(0), 1e-6, None)).astype(np.float32))
        builders.append(lambda raw: (np.sign(raw) * np.log1p(np.abs(raw))).astype(np.float32))
    if kind == "drop":
        # One copy with a different 15% of features cleared in every row.
        rng = np.random.RandomState(seed)
        mask = rng.rand(*X.shape) < 0.15
        builders.append(lambda raw, m=mask: np.where(m, 0, raw).astype(np.float32))
    return builders


def corrupt(X, q):
    rng = np.random.RandomState(1)
    std = np.clip(X.std(0), 1e-6, None).astype(np.float32)
    scale = np.exp(np.random.RandomState(2).normal(0, 1.0, size=X.shape[1])).astype(np.float32)
    out = {
        "clean": X,
        "log1p": (np.sign(X) * np.log1p(np.abs(X))).astype(np.float32),
        "times10": (X * np.float32(10)).astype(np.float32),
        "lognorm": (X * scale).astype(np.float32),
        "add0.05": (X + 0.05 * std * rng.randn(*X.shape).astype(np.float32)).astype(np.float32),
        "add0.1": (X + 0.10 * std * rng.randn(*X.shape).astype(np.float32)).astype(np.float32),
    }
    dropped = X.copy()
    codes = _run_codes(np.asarray(q, dtype=object))
    rng2 = np.random.RandomState(3)
    for a, b in zip(np.flatnonzero(np.r_[True, np.diff(codes) != 0]),
                    np.r_[np.flatnonzero(np.diff(codes) != 0) + 1, len(codes)]):
        cols = rng2.choice(X.shape[1], size=max(1, X.shape[1] // 10), replace=False)
        dropped[a:b, cols] = 0
    out["zero10"] = dropped
    return out


def ndcg(model, X, y, q, n_iter):
    pred = model.predict(X, num_iteration=n_iter, num_threads=4)
    return mean_ndcg_from_scores(pred, y, q, 10)


def main():
    X, y, q = load_xyq(os.path.join(ROOT, "data", "train", "train.txt"))
    q = np.asarray(q, dtype=object)
    tr_idx, va_idx = grouped_kfold(q, k=5, seed=0)[0]
    tr_q = list(dict.fromkeys(q[tr_idx].tolist()))[:2000]
    va_q = list(dict.fromkeys(q[va_idx].tolist()))[:400]
    tr_keep = np.isin(q, tr_q)
    va_keep = np.isin(q, va_q)
    Xtr, ytr, qtr = X[tr_keep], y[tr_keep], q[tr_keep]
    Xva, yva, qva = X[va_keep], y[va_keep], q[va_keep]
    print(f"fit queries {len(tr_q)} rows {len(ytr)} | holdout queries {len(va_q)} rows {len(yva)}", flush=True)
    cor = corrupt(Xva, qva)
    mats = {k: _expand(v, qva, False) for k, v in cor.items()}
    Xd, yd, qd = load_xyq(os.path.join(ROOT, "data", "dev", "vali.txt"))
    qd = np.asarray(qd, dtype=object)
    dev_clean = _expand(Xd, qd, False)

    specs = [
        ("pct", "pct", dict(min_data_in_leaf=50, feature_fraction=0.8, lambda_l2=1.0)),
        ("scale", "scale", dict(min_data_in_leaf=50, feature_fraction=0.8, lambda_l2=1.0)),
        ("noise", "noise", dict(min_data_in_leaf=50, feature_fraction=0.8, lambda_l2=1.0)),
        ("both", "both", dict(min_data_in_leaf=50, feature_fraction=0.8, lambda_l2=1.0)),
        ("both_reg", "both_reg", dict(min_data_in_leaf=150, feature_fraction=0.5, lambda_l2=5.0)),
        ("drop", "drop", dict(min_data_in_leaf=50, feature_fraction=0.8, lambda_l2=1.0)),
    ]
    rounds = (80, 160, 240)
    keys = list(mats)
    for name, kind, kw in specs:
        t0 = time.perf_counter()
        builders = make_builders(Xtr, kind, seed=1)
        big, bq = _stack(Xtr, qtr, builders)
        yb = np.tile(ytr, len(builders))
        params = default_params(num_leaves=63, learning_rate=0.05, seed=1, num_threads=8,
                                deterministic=False, max_bin=255, **kw)
        model = fit(big, yb, bq, params, max(rounds))
        print(f"\n== {name} views {len(builders)} {time.perf_counter() - t0:.0f}s ==", flush=True)
        print("trees " + " ".join(f"{k:>8}" for k in keys) + "     dev")
        for r in rounds:
            vals = [ndcg(model, mats[k], yva, qva, r) for k in keys]
            dev = ndcg(model, dev_clean, yd, qd, r)
            hard = float(np.mean(vals[1:]))
            mix = 0.25 * vals[0] + 0.75 * hard
            print(f"{r:5d} " + " ".join(f"{v:8.4f}" for v in vals) + f" {dev:8.4f}  mix {mix:.4f}", flush=True)
        del big, model


if __name__ == "__main__":
    main()
