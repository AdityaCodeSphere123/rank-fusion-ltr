"""Train LambdaMART variants on TRAIN folds != 0 and score TRAIN fold 0 under additive noise.

The leaderboard C2 number (0.3533 against a clean 0.4845) matches this failure mode. DEV is not used.
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

# name, extra lightgbm params, noise scale mixed into training (0 = clean data only), copies
CONFIGS = [
    ("bin63", {"max_bin": 63}, 0.0),
    ("bin15", {"max_bin": 15}, 0.0),
    ("noise0.1", {}, 0.1),
    ("noise0.2", {}, 0.2),
    ("bin31_noise0.1", {"max_bin": 31}, 0.1),
    ("bin63_noise0.15", {"max_bin": 63}, 0.15),
]


def run(job):
    name, extra, noise, trees, train_path = job
    import lightgbm as lgb
    from harness.cv import grouped_kfold
    from harness.metrics import mean_ndcg_from_scores
    from scripts.data_cache import load_xyq
    from scripts.lambdamart_utils import default_params, group_sizes

    X, y, q = load_xyq(train_path)
    tr, va = grouped_kfold(q, k=5, seed=0)[0]
    Xtr, ytr, qtr = X[tr], y[tr], q[tr]
    groups = group_sizes(qtr)
    if noise > 0:
        rng = np.random.RandomState(0)
        std = Xtr.std(0).astype(np.float32)
        noisy = (Xtr + noise * std * rng.randn(*Xtr.shape).astype(np.float32)).astype(np.float32)
        Xtr = np.vstack([Xtr, noisy])
        ytr = np.concatenate([ytr, ytr])
        groups = groups + groups
    params = default_params(seed=0, **BASE, **extra)
    model = lgb.train(params, lgb.Dataset(Xtr, label=ytr, group=groups), num_boost_round=trees)

    rng = np.random.RandomState(1)
    std = X[va].std(0).astype(np.float32)
    tests = {
        "clean": X[va],
        "add0.1": (X[va] + 0.1 * std * rng.randn(*X[va].shape).astype(np.float32)).astype(np.float32),
        "add0.2": (X[va] + 0.2 * std * rng.randn(*X[va].shape).astype(np.float32)).astype(np.float32),
    }
    out = {"name": name}
    for k, Xp in tests.items():
        out[k] = round(mean_ndcg_from_scores(model.predict(Xp, num_threads=4), y[va], q[va], 10), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=800)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    path = os.path.join(ROOT, "data", "train", "train.txt")
    jobs = [(n, e, z, args.trees, path) for n, e, z in CONFIGS]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(run, j) for j in jobs]):
            r = fut.result()
            print(f"{r['name']:<18} clean {r['clean']:.4f}  add0.1 {r['add0.1']:.4f}  add0.2 {r['add0.2']:.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
