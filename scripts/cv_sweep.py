"""Grouped 5-fold CV sweep of LambdaMART settings on TRAIN only. This makes the CV table in
docs/SUBMISSION_NOTES.md.

For each (config, fold) one model is trained up to the largest tree count. Train-fold and
validation-fold nDCG@10 are then read at every fixed tree count with predict(num_iteration=n).
The first n trees are exactly the model num_boost_round=n would give, so this is a sweep over
fixed tree counts and not early stopping on the reported fold. Folds are split by query.

    python scripts/cv_sweep.py --grid leaves     # results go to experiments/cv_results.jsonl
    python scripts/cv_sweep.py --grid lr
    python scripts/cv_sweep.py --report          # table of everything run so far

Each job uses 4 threads. Finished (config, fold) jobs are skipped when the sweep is run again.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "experiments", "cv_results.jsonl")

BASE = {"num_leaves": 63, "learning_rate": 0.05, "min_data_in_leaf": 100, "lambda_l2": 1.0,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1}

TREES = [100, 200, 300, 500, 750, 1000, 1500, 2000]

GRIDS = {
    "leaves": [dict(BASE, num_leaves=nl, _trees=TREES) for nl in (31, 63, 127, 255)],
    "lr": [dict(BASE, min_data_in_leaf=500, feature_fraction=0.5, learning_rate=lr, _trees=TREES)
           for lr in (0.05, 0.025, 0.0125)],
}


def config_name(cfg):
    keys = sorted(k for k in cfg if not k.startswith("_"))
    return ",".join(f"{k}={cfg[k]}" for k in keys)


def _load(train_path):
    from scripts.data_cache import load_xyq
    return load_xyq(train_path)


def run_job(args):
    """Train one config on one fold's training part; score train + val parts at each tree count."""
    cfg, fold, k, seed, train_path = args
    import lightgbm as lgb
    from harness.cv import grouped_kfold
    from harness.metrics import mean_ndcg_from_scores
    from scripts.lambdamart_utils import default_params, group_sizes

    X, y, q = _load(train_path)
    tr_idx, va_idx = grouped_kfold(q, k=k, seed=seed)[fold]
    trees = sorted(cfg["_trees"])
    params = default_params(seed=seed, **{kk: v for kk, v in cfg.items() if not kk.startswith("_")})
    t0 = time.perf_counter()
    model = lgb.train(params, lgb.Dataset(X[tr_idx], label=y[tr_idx], group=group_sizes(q[tr_idx])),
                      num_boost_round=trees[-1])
    fit_s = time.perf_counter() - t0
    res = {"config": config_name(cfg), "params": {kk: v for kk, v in cfg.items() if not kk.startswith("_")},
           "fold": fold, "k": k, "seed": seed, "fit_seconds": round(fit_s, 1), "trees": {}}
    for n in trees:
        res["trees"][str(n)] = {
            "train": mean_ndcg_from_scores(model.predict(X[tr_idx], num_iteration=n), y[tr_idx], q[tr_idx], 10),
            "val": mean_ndcg_from_scores(model.predict(X[va_idx], num_iteration=n), y[va_idx], q[va_idx], 10)}
    return res


def done_keys():
    keys = set()
    if os.path.isfile(OUT):
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                keys.add((r["config"], r["fold"], r["k"], r["seed"], tuple(sorted(r["trees"], key=int))))
    return keys


def report(path=OUT):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for n, m in r["trees"].items():
                rows.setdefault((r["config"], int(n)), {}).setdefault(r["fold"], (m["train"], m["val"], r["fit_seconds"]))
    print(f"{'config':<120} {'trees':>5} {'folds':>5} {'train':>7} {'val':>7} {'val_std':>7} {'val/train':>9} {'fit_s':>6}")
    for (cfg, n), folds in sorted(rows.items()):
        tr = np.array([v[0] for v in folds.values()])
        va = np.array([v[1] for v in folds.values()])
        fs = np.mean([v[2] for v in folds.values()])
        print(f"{cfg:<120} {n:>5} {len(folds):>5} {tr.mean():>7.4f} {va.mean():>7.4f} {va.std():>7.4f} "
              f"{va.mean() / tr.mean():>9.4f} {fs:>6.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=sorted(GRIDS), default="leaves")
    ap.add_argument("--train", default=os.path.join(ROOT, "data", "train", "train.txt"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="parallel jobs, each with num_threads=4 (max 4)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return
    if args.workers > 4:
        raise SystemExit("at most 4 worker processes")
    cfgs = GRIDS[args.grid]
    _load(args.train)  # fill the cache once before the worker processes start
    have = done_keys()
    jobs = [(c, f, args.folds, args.seed, args.train) for c in cfgs for f in range(args.folds)
            if (config_name(c), f, args.folds, args.seed, tuple(sorted(map(str, c["_trees"]), key=int))) not in have]
    print(f"{len(jobs)} jobs to run ({len(cfgs)} configs x {args.folds} folds, minus finished)", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(run_job, j) for j in jobs]):
            res = fut.result()
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
            best = max(res["trees"].items(), key=lambda kv: kv[1]["val"])
            print(f"[{time.perf_counter() - t0:7.0f}s] fold {res['fold']} fit {res['fit_seconds']:.0f}s "
                  f"best val {best[1]['val']:.4f} @ {best[0]} trees  | {res['config']}", flush=True)


if __name__ == "__main__":
    main()
