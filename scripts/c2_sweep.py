"""TRAIN-fold sweep for Track C2.

Each config is fit on grouped 5-fold TRAIN folds 1-4 and scored on fold 0.
DEV is never read. Selection target is nDCG@10 under additive Gaussian feature
noise (the corruption whose magnitude matches the graded C2 drop) together
with a global per-feature affine transform, which leaves within-query ranks
unchanged and breaks raw tree thresholds.

    python scripts/c2_sweep.py --trees 800 --workers 3
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASE = dict(num_leaves=63, learning_rate=0.0125, min_data_in_leaf=500, lambda_l2=1.0,
            feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1)

# name, representation in {raw, rank, both}, noise scales appended as extra copies
CONFIGS = [
    ("raw", "raw", ()),
    ("raw_n05", "raw", (0.05,)),
    ("raw_n10", "raw", (0.10,)),
    ("raw_n05_10", "raw", (0.05, 0.10)),
    ("raw_n03_05_08", "raw", (0.03, 0.05, 0.08)),
    ("rank", "rank", ()),
    ("rank_n05", "rank", (0.05,)),
    ("both", "both", ()),
    ("both_n05", "both", (0.05,)),
    ("both_n05_10", "both", (0.05, 0.10)),
]


def query_percentiles(X, q):
    """Within-query average rank of each feature, scaled to [0, 1]. Monotone per feature."""
    out = np.empty(X.shape, dtype=np.float32)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    for a, b in zip(starts, ends):
        block = X[a:b]
        n, d = block.shape
        if n == 1:
            out[a:b] = 0.5
            continue
        order = np.argsort(block, axis=0, kind="mergesort")
        ranks = np.empty((n, d), dtype=np.float64)
        ranks[order, np.arange(d)] = np.arange(n)[:, None]
        out[a:b] = (ranks / (n - 1)).astype(np.float32)
    return out


def represent(X, q, kind):
    if kind == "raw":
        return X
    ranks = query_percentiles(X, q)
    if kind == "rank":
        return ranks
    return np.hstack([X, ranks]).astype(np.float32)


def augment(X, y, q, scales, seed):
    if not scales:
        return X, y, q
    rng = np.random.RandomState(seed)
    std = X.std(0).astype(np.float32)
    xs, ys, qs = [X], [y], [q]
    for scale in scales:
        noisy = (X + scale * std * rng.randn(*X.shape).astype(np.float32)).astype(np.float32)
        xs.append(noisy)
        ys.append(y)
        qs.append(q)
    return np.vstack(xs), np.concatenate(ys), np.concatenate(qs)


def corrupt(X, seed=1):
    rng = np.random.RandomState(seed)
    std = np.maximum(X.std(0), 1e-6).astype(np.float32)
    out = {"clean": X}
    for scale in (0.02, 0.05, 0.10, 0.20):
        out[f"add{scale:.2f}"] = (X + scale * std * rng.randn(*X.shape).astype(np.float32)).astype(np.float32)
    scale = np.exp(rng.normal(0, 0.5, size=X.shape[1])).astype(np.float32)
    shift = (rng.normal(0, 1.0, size=X.shape[1]) * std).astype(np.float32)
    out["affine"] = (X * scale + shift).astype(np.float32)
    out["mul0.20"] = (X * (1 + 0.20 * rng.randn(*X.shape).astype(np.float32))).astype(np.float32)
    dropped = X.copy()
    dropped[rng.rand(*X.shape) < 0.10] = 0
    out["drop10"] = dropped
    # five independent 0.05-std draws, the shape the graded mean would have if every version is that noise
    draws = []
    for s in range(5):
        rs = np.random.RandomState(100 + s)
        draws.append((X + 0.05 * std * rs.randn(*X.shape).astype(np.float32)).astype(np.float32))
    out["_draws05"] = draws
    return out


def run(job):
    name, kind, scales, trees, train_path = job
    import lightgbm as lgb
    from harness.cv import grouped_kfold
    from harness.metrics import mean_ndcg_from_scores
    from scripts.data_cache import load_xyq
    from scripts.lambdamart_utils import default_params, group_sizes

    X, y, q = load_xyq(train_path)
    tr, va = grouped_kfold(q, k=5, seed=0)[0]
    Xtr, ytr, qtr = augment(X[tr], y[tr], q[tr], scales, seed=0)
    Xtr = represent(Xtr, qtr, kind)
    params = default_params(seed=0, **BASE)
    model = lgb.train(params, lgb.Dataset(Xtr, label=ytr, group=group_sizes(qtr)), num_boost_round=trees)

    versions = corrupt(X[va])
    draws = versions.pop("_draws05")
    scores = {}
    yv, qv = y[va], q[va]
    for key, Xp in versions.items():
        scores[key] = mean_ndcg_from_scores(model.predict(represent(Xp, qv, kind), num_threads=4), yv, qv, 10)
    scores["mean05"] = float(np.mean([
        mean_ndcg_from_scores(model.predict(represent(Xp, qv, kind), num_threads=4), yv, qv, 10) for Xp in draws
    ]))
    return name, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=800)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    path = os.path.join(ROOT, "data", "train", "train.txt")
    keys = ["clean", "add0.02", "add0.05", "mean05", "add0.10", "add0.20", "affine", "mul0.20", "drop10"]
    header = f"{'config':<16}" + "".join(f"{k:>9}" for k in keys)
    print(header, flush=True)
    jobs = [(n, k, s, args.trees, path) for n, k, s in CONFIGS]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(run, j) for j in jobs]):
            name, scores = fut.result()
            line = f"{name:<16}" + "".join(f"{scores[k]:9.4f}" for k in keys)
            print(line, flush=True)
            rows.append((name, scores))
    out = os.path.join(ROOT, "experiments", "c2_sweep.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for name, scores in rows:
            f.write(f"{name:<16}" + "".join(f"{scores[k]:9.4f}" for k in keys) + "\n")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
