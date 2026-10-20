"""Compare plain scores with an average of ranks under a few input transforms.

Uses the released train file only. The shipped model was fit on these queries,
so the clean number is optimistic; the comparison is the gap between the two
scoring rules under the same corruptions.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from harness.cv import grouped_kfold
from harness.metrics import mean_ndcg_from_scores
from scripts.data_cache import load_xyq
from scripts.train_model import _expand, _run_codes

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


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


def _transforms(X, q):
    X = np.asarray(X, dtype=np.float32)
    codes = _run_codes(np.asarray(q))
    z = np.empty_like(X)
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    ends = np.r_[starts[1:], len(codes)]
    for a, b in zip(starts, ends):
        block = X[a:b]
        sd = block.std(0)
        sd[sd < 1e-8] = 1.0
        z[a:b] = (block - block.mean(0)) / sd
    log = (np.sign(X) * np.log1p(np.abs(X))).astype(np.float32)
    return [X, log, np.asarray(X * 10, dtype=np.float32), z]


def _ranks(pred, q):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 1)
    from submission.ranking import within_query_ranks
    codes = _run_codes(np.asarray(q))
    out = np.empty(len(pred), dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    ends = np.r_[starts[1:], len(codes)]
    for a, b in zip(starts, ends):
        out[a:b] = within_query_ranks(pred[a:b])[:, 0]
    return out


def main():
    import lightgbm as lgb
    X, y, q = load_xyq(os.path.join(ROOT, "data", "train", "train.txt"))
    q = np.asarray(q)
    _, va = grouped_kfold(q, k=5, seed=0)[0]
    X, y, q = X[va], y[va], q[va]
    # Keep the first 400 queries so this stays a quick comparison.
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    if len(starts) > 400:
        cut = starts[400]
        X, y, q = X[:cut], y[:cut], q[:cut]
    print(f"rows {len(y)}", flush=True)
    model = lgb.Booster(model_file=os.path.join(ROOT, "submission", "model.txt"))
    raws = _corrupt(X)
    keys = list(raws)
    for key in keys:
        packed = _expand(raws[key], _run_codes(q), False, with_z=True)
        plain = model.predict(packed, num_threads=8)
        views = _transforms(raws[key], q)
        acc = None
        for view in views:
            block = _expand(view, _run_codes(q), False, with_z=True)
            pred = model.predict(block, num_threads=8)
            rank = _ranks(pred, q)
            acc = rank if acc is None else acc + rank
        avg = acc / len(views)
        p = mean_ndcg_from_scores(plain, y, q, 10)
        a = mean_ndcg_from_scores(avg, y, q, 10)
        print(f"{key:10s}  plain {p:.4f}  rankavg {a:.4f}  delta {a - p:+.4f}", flush=True)


if __name__ == "__main__":
    main()
