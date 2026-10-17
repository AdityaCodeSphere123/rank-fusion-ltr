"""One-fold probes for LambdaMART knobs that the CV sweep never touched.

Trains on TRAIN folds != 0 (grouped 5-fold, seed 0) and scores TRAIN fold 0. Never touches DEV.
Each config is the shipped setting plus one change. Also prints a few feature perturbations so a
gain that only exists on clean data is visible.

    .venv/bin/python scripts/probe_ltr.py --workers 2
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

CONFIGS = [
    ("base", {}),
    ("trunc10", {"lambdarank_truncation_level": 10}),
    ("trunc5", {"lambdarank_truncation_level": 5}),
    ("trunc20", {"lambdarank_truncation_level": 20}),
    ("smooth10", {"path_smooth": 10.0}),
    ("trunc10_smooth10", {"lambdarank_truncation_level": 10, "path_smooth": 10.0}),
]


def _perturb(X, q, rng):
    std = X.std(0)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    Xm = X.copy()
    for a, b in zip(starts, ends):
        Xm[a:b, rng.rand(X.shape[1]) < 0.10] = 0.0
    return {
        "clean": X,
        "mult10": (X * (1 + 0.10 * rng.randn(*X.shape))).astype(np.float32),
        "add0.1std": (X + (0.1 * std * rng.randn(*X.shape))).astype(np.float32),
        "mask10": Xm,
    }


def run(job):
    name, extra, trees, train_path = job
    import lightgbm as lgb
    from harness.cv import grouped_kfold
    from harness.metrics import mean_ndcg_from_scores
    from scripts.data_cache import load_xyq
    from scripts.lambdamart_utils import default_params, group_sizes

    X, y, q = load_xyq(train_path)
    tr, va = grouped_kfold(q, k=5, seed=0)[0]
    params = default_params(seed=0, **BASE, **extra)
    model = lgb.train(params, lgb.Dataset(X[tr], label=y[tr], group=group_sizes(q[tr])), num_boost_round=trees)
    rng = np.random.RandomState(0)
    out = {"name": name, "trees": trees}
    for pname, Xp in _perturb(X[va], q[va], rng).items():
        out[pname] = round(mean_ndcg_from_scores(model.predict(Xp), y[va], q[va], 10), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--train", default=os.path.join(ROOT, "data", "train", "train.txt"))
    args = ap.parse_args()
    jobs = [(n, e, args.trees, args.train) for n, e in CONFIGS]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            clean = r["clean"]
            print(f"{r['name']:<18} clean {clean:.4f}  " + "  ".join(
                f"{k} {r[k]:.4f} ({r[k] - clean:+.4f})" for k in ("mult10", "add0.1std", "mask10")), flush=True)


if __name__ == "__main__":
    main()
