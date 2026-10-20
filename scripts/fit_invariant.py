"""Fit the percentile + z-score + log z-score model.

A left-out fifth of TRAIN is scored clean and under corruptions. Pass
--refit-trees to fit every TRAIN and DEV query and write the artifact.
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
from scripts.train_model import _run_codes
from submission.ranking import within_query_ranks, within_query_z

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
KEYS = ["clean", "log1p", "zscore", "x10", "add0.05", "add0.20", "mul0.20", "cell10", "permute"]


def _blocks(X, q):
    q = np.asarray(q)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    return list(zip(starts, ends))


def _matrix(X, q):
    """Same columns rerank_with_ltr builds for a model of width 3D."""
    X = np.asarray(X, dtype=np.float64)
    out = np.empty((len(X), X.shape[1] * 3), dtype=np.float32)
    for a, b in _blocks(X, q):
        block = X[a:b]
        logged = np.where(np.isfinite(block), np.sign(block) * np.log1p(np.abs(block)), np.nan)
        out[a:b] = np.hstack([
            within_query_ranks(block),
            within_query_z(block),
            within_query_z(logged),
        ]).astype(np.float32)
    return out


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


def _noise(X, seed):
    X = np.asarray(X, dtype=np.float32)
    std = X.std(axis=0).astype(np.float32)
    return (X + 0.05 * std * np.random.RandomState(seed).randn(*X.shape).astype(np.float32)).astype(np.float32)


def _stack(X, y, q, seed):
    views = [np.asarray(X, dtype=np.float32), _noise(X, seed)]
    base = _run_codes(np.asarray(q))
    n_queries = int(base[-1]) + 1
    n = len(y)
    width = X.shape[1] * 3
    big = np.empty((n * len(views), width), dtype=np.float32)
    big_y = np.empty(n * len(views), dtype=y.dtype)
    big_c = np.empty(n * len(views), dtype=np.int32)
    for i, view in enumerate(views):
        a = i * n
        codes = base + i * n_queries
        # Rebuild a q-like vector so _matrix groups rows the same way as `codes`.
        big[a:a + n] = _matrix(view, codes)
        big_y[a:a + n] = y
        big_c[a:a + n] = codes
    return big, big_y, big_c


def _report(model, packed, y, q, dev_x, yd, qd, trees):
    scores = {}
    for key in KEYS:
        pred = model.predict(packed[key], num_iteration=trees, num_threads=4)
        scores[key] = mean_ndcg_from_scores(pred, y, q, 10)
    dev = mean_ndcg_from_scores(model.predict(dev_x, num_iteration=trees, num_threads=4), yd, qd, 10)
    corrupt = float(np.mean([scores[k] for k in KEYS if k != "clean"]))
    detail = " ".join(f"{k} {scores[k]:.4f}" for k in KEYS)
    print(f"trees {trees:4d}  dev {dev:.4f}  left-out {detail}  corrupt {corrupt:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=600)
    ap.add_argument("--every", type=int, default=200)
    ap.add_argument("--refit-trees", type=int, default=0)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "submission", "model.txt"))
    args = ap.parse_args()
    params = default_params(127, 0.05, 20, lambda_l2=0.1, seed=1, feature_fraction=0.5,
                            bagging_fraction=0.9, bagging_freq=1, num_threads=args.num_threads)
    X, y, q = load_xyq(os.path.join(ROOT, "data", "train", "train.txt"))
    Xd, yd, qd = load_xyq(os.path.join(ROOT, "data", "dev", "vali.txt"))
    q = np.asarray(q)
    qd = np.asarray(qd)

    if args.refit_trees:
        print(f"refit all train+dev, {args.refit_trees} trees", flush=True)
        fit_x = np.vstack([X, Xd])
        fit_y = np.concatenate([y, yd])
        fit_q = np.concatenate([q, qd])
        dev_x = _matrix(Xd, qd)
        del X, Xd
        gc.collect()
        big_x, big_y, big_c = _stack(fit_x, fit_y, fit_q, seed=1)
        del fit_x
        gc.collect()
        model = fit(big_x, big_y, big_c, params, args.refit_trees)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        model.save_model(args.out)
        dev = mean_ndcg_from_scores(model.predict(dev_x, num_threads=4), yd, qd, 10)
        print(f"wrote {args.out} features {model.num_feature()} trees {model.num_trees()}")
        print(f"dev nDCG@10 {dev:.4f} (in-training)", flush=True)
        return

    tr, va = grouped_kfold(q, k=5, seed=0)[0]
    print("pack left-out", flush=True)
    packed = {name: _matrix(raw, q[va]) for name, raw in _corrupt(X[va]).items()}
    yva, qva = y[va], q[va]
    dev_x = _matrix(Xd, qd)
    fit_x = np.vstack([X[tr], Xd])
    fit_y = np.concatenate([y[tr], yd])
    fit_q = np.concatenate([q[tr], qd])
    del X, Xd
    gc.collect()
    print("stack fit", flush=True)
    big_x, big_y, big_c = _stack(fit_x, fit_y, fit_q, seed=1)
    del fit_x
    gc.collect()
    print(f"fit rows {len(big_y)} cols {big_x.shape[1]}", flush=True)

    def _checkpoint(env):
        n = env.iteration + 1
        if n % args.every == 0 or n == args.trees:
            _report(env.model, packed, yva, qva, dev_x, yd, qd, n)

    fit(big_x, big_y, big_c, params, args.trees, callbacks=[_checkpoint])


if __name__ == "__main__":
    main()
