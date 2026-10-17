"""Tie-breaker for LambdaMART configs with near-equal CV accuracy: how much does each lose under generic
perturbations of unseen queries? (Track C2 scores the submitted model on undisclosed "hidden versions" of the
held-out split; we do not know how they are built, so this only probes general sensitivity.)

Each config is trained on TRAIN folds != 0 (grouped 5-fold, seed 0) and scored on TRAIN fold 0 -- never DEV --
clean and under: multiplicative feature noise, additive noise scaled by each feature's std, random feature
masking (a random 10% of columns set to 0 per query), and candidate-pool subsampling (50% of each query's docs).
Perturbations use fixed seeds, so every config sees the same perturbed data.

    python scripts/robustness_eval.py --configs experiments/robust_configs.json --workers 3
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "experiments", "robustness.jsonl")


def perturbations(X, y, q, seed=0):
    rng = np.random.RandomState(seed)
    std = X.std(0)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    out = {"clean": (X, y, q)}
    out["mult_noise_10%"] = (X * (1 + 0.10 * rng.randn(*X.shape)).astype(np.float32), y, q)
    out["add_noise_0.1std"] = (X + (0.1 * std * rng.randn(*X.shape)).astype(np.float32), y, q)
    Xm = X.copy()
    for a, b in zip(starts, ends):
        Xm[a:b, rng.rand(X.shape[1]) < 0.10] = 0.0
    out["mask_10%_features"] = (Xm, y, q)
    keep = np.concatenate([np.sort(rng.choice(np.arange(a, b), max(1, (b - a) // 2), replace=False))
                           for a, b in zip(starts, ends)])
    out["subsample_50%_docs"] = (X[keep], y[keep], q[keep])
    return out


def run(args):
    cfg, train_path = args
    import lightgbm as lgb
    from harness.cv import grouped_kfold
    from harness.metrics import mean_ndcg_from_scores
    from scripts.data_cache import load_xyq
    from scripts.lambdamart_utils import default_params, group_sizes

    X, y, q = load_xyq(train_path)
    tr, va = grouped_kfold(q, k=5, seed=0)[0]
    params = default_params(seed=0, **{k: v for k, v in cfg.items() if not k.startswith("_")})
    model = lgb.train(params, lgb.Dataset(X[tr], label=y[tr], group=group_sizes(q[tr])),
                      num_boost_round=cfg["_trees"])
    res = {"config": cfg, "fold0": {}}
    for name, (Xp, yp, qp) in perturbations(X[va], y[va], q[va]).items():
        res["fold0"][name] = mean_ndcg_from_scores(model.predict(Xp), yp, qp, 10)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--train", default=os.path.join(ROOT, "data", "train", "train.txt"))
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    cfgs = json.load(open(args.configs, encoding="utf-8"))
    with ProcessPoolExecutor(max_workers=min(args.workers, 4)) as ex:
        for fut in as_completed([ex.submit(run, (c, args.train)) for c in cfgs]):
            r = fut.result()
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            clean = r["fold0"]["clean"]
            print(json.dumps({k: v for k, v in r["config"].items()}), "clean %.4f" % clean,
                  " ".join(f"{k} {v:.4f} ({v - clean:+.4f})" for k, v in r["fold0"].items() if k != "clean"),
                  flush=True)


if __name__ == "__main__":
    main()
