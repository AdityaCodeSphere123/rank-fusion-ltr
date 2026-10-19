"""Pick a tree count for raw + percentile features, with and without within-query z-scores.

The fit uses four fifths of TRAIN plus DEV. The remaining fifth of TRAIN is scored
clean and under corruptions. DEV is reported too; those rows were fit.

    .venv/bin/python scripts/select_qz_views.py
"""
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse

import numpy as np

from harness.cv import grouped_kfold
from harness.metrics import mean_ndcg_from_scores
from scripts.data_cache import load_xyq
from scripts.lambdamart_utils import default_params, fit
from scripts.train_model import _expand, _run_codes

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
KEYS = ["clean", "log1p", "zscore", "x10", "add0.05", "add0.20", "mul0.20", "cell10", "permute"]


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


def _pack(raw_by_name, q, with_z):
    codes = _run_codes(np.asarray(q))
    return {name: _expand(raw, codes, False, with_z=with_z) for name, raw in raw_by_name.items()}


def _view_list(X, q, seed):
    X = np.asarray(X, dtype=np.float32)
    q = np.asarray(q)
    std = X.std(axis=0).astype(np.float32)
    noise = (X + 0.05 * std * np.random.RandomState(seed).randn(*X.shape).astype(np.float32)).astype(np.float32)
    log = (np.sign(X) * np.log1p(np.abs(X))).astype(np.float32)
    z = ((X - X.mean(0)) / np.clip(std, 1e-6, None)).astype(np.float32)

    def qscale(raw=X, q=q, seed=seed):
        out = np.empty_like(raw)
        rng = np.random.RandomState(seed + 17)
        starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
        ends = np.r_[starts[1:], len(q)]
        drawn = np.exp(rng.normal(0.0, 1.0, size=(len(starts), raw.shape[1]))).astype(np.float32)
        for i, (a, b) in enumerate(zip(starts, ends)):
            out[a:b] = raw[a:b] * drawn[i]
        return out

    return [X, noise, log, z, qscale()]


def _stack_views(X, y, q, with_z, seed):
    views = _view_list(X, q, seed)
    base = _run_codes(np.asarray(q))
    n_queries = int(base[-1]) + 1
    width = _expand(views[0][:1], base[:1], False, with_z=with_z).shape[1]
    n = len(y)
    big = np.empty((n * len(views), width), dtype=np.float32)
    big_y = np.empty(n * len(views), dtype=y.dtype)
    big_c = np.empty(n * len(views), dtype=np.int32)
    for i, view in enumerate(views):
        a = i * n
        codes = base + i * n_queries
        big[a:a + n] = _expand(view, codes, False, with_z=with_z)
        big_y[a:a + n] = y
        big_c[a:a + n] = codes
        del view
    del views
    return big, big_y, big_c


def _report(model, packed, y, q, dev_pack, yd, qd, trees):
    scores = {}
    for key in KEYS:
        pred = model.predict(packed[key], num_iteration=trees, num_threads=4)
        scores[key] = mean_ndcg_from_scores(pred, y, q, 10)
    dev = mean_ndcg_from_scores(model.predict(dev_pack, num_iteration=trees, num_threads=4), yd, qd, 10)
    corrupt = float(np.mean([scores[k] for k in KEYS if k != "clean"]))
    detail = " ".join(f"{k} {scores[k]:.4f}" for k in KEYS)
    print(f"trees {trees:4d}  dev {dev:.4f}  holdout {detail}  corrupt {corrupt:.4f}", flush=True)
    return dev, corrupt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=800)
    ap.add_argument("--every", type=int, default=200)
    ap.add_argument("--with-z", action="store_true")
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--num-threads", type=int, default=8)
    args = ap.parse_args()

    X, y, q = load_xyq(os.path.join(ROOT, "data", "train", "train.txt"))
    Xd, yd, qd = load_xyq(os.path.join(ROOT, "data", "dev", "vali.txt"))
    q = np.asarray(q)
    qd = np.asarray(qd)
    tr_idx, va_idx = grouped_kfold(q, k=5, seed=0)[0]
    print(f"with_z {args.with_z} leaves {args.num_leaves} fit queries holdout-excluded", flush=True)
    print("pack holdout", flush=True)
    packed = _pack(_corrupt(X[va_idx]), q[va_idx], args.with_z)
    yva, qva = y[va_idx], q[va_idx]
    print(f"holdout rows {len(yva)} width {packed['clean'].shape[1]}", flush=True)
    dev_pack = _expand(Xd, _run_codes(qd), False, with_z=args.with_z)
    print("stack fit", flush=True)
    fit_x = np.vstack([X[tr_idx], Xd])
    fit_y = np.concatenate([y[tr_idx], yd])
    fit_q = np.concatenate([q[tr_idx], qd])
    del X, Xd
    gc.collect()
    big_x, big_y, big_c = _stack_views(fit_x, fit_y, fit_q, args.with_z, seed=1)
    del fit_x
    gc.collect()
    print(f"fit rows {len(big_y)} cols {big_x.shape[1]}", flush=True)
    params = default_params(
        args.num_leaves, 0.05, 20, lambda_l2=0.1, seed=1,
        feature_fraction=0.5, bagging_fraction=0.9, bagging_freq=1,
        num_threads=args.num_threads,
    )

    def _checkpoint(env, every=args.every, packed=packed, yva=yva, qva=qva, dev_pack=dev_pack, yd=yd, qd=qd):
        n = env.iteration + 1
        if n % every == 0 or n == args.trees:
            _report(env.model, packed, yva, qva, dev_pack, yd, qd, n)

    fit(big_x, big_y, big_c, params, args.trees, callbacks=[_checkpoint])


if __name__ == "__main__":
    main()
