"""Benchmark Track C1 recipe variants on benchmark-like tasks carved out of MSLR TRAIN (never DEV).

The hidden benchmarks are undisclosed, so the recipe's rules are checked on tasks that differ from full MSLR
in the ways the spec says they may: number of training queries, feature dimension (random feature subsets),
label levels (0-4 collapsed to 0-2), and list length (documents per query subsampled). Every task draws
disjoint query sets for its train / vali / test parts from TRAIN with its own seed; the recipe sees train + vali
only (exactly like recipe.train) and is scored by nDCG@10 on the task's test queries.

    python scripts/recipe_sim.py --variants default starter --workers 3
    python scripts/recipe_sim.py --report

Results are appended to experiments/recipe_sim.jsonl.
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
OUT = os.path.join(ROOT, "experiments", "recipe_sim.jsonl")

# name -> (n_train_q, n_vali_q, n_test_q, n_features or None, max_label, max_docs_per_query or None)
TASKS = {
    "q300_d136_l4":        (300, 100, 1500, None, 4, None),
    "q1000_d46_l2":        (1000, 300, 1500, 46, 2, None),
    "q1000_d136_l4_docs20": (1000, 300, 1500, None, 4, 20),
    "q2500_d90_l4":        (2500, 800, 1500, 90, 4, None),
    "q4000_d136_l2":       (4000, 500, 1500, None, 2, None),
}

# variant name -> kwargs for recipe.fit_arrays (overrides of the rule-based parameters) or "starter"
VARIANTS = {
    "default": {},
    "no_refit": {"refit": False},
    "starter": "starter",
    "leaves_x2": "leaves_x2",
    "leaves_half": "leaves_half",
    "ff_1.0": {"feature_fraction": 1.0},
    "ff_0.5": {"feature_fraction": 0.5},
    "mdl_x4": "mdl_x4",
    "lr_0.1": {"learning_rate": 0.1},
    "lr_0.025": {"learning_rate": 0.025},
    "trunc10": {"lambdarank_truncation_level": 10},
    "qz": {},
    "qr": {},
    "qz_trunc10": {"lambdarank_truncation_level": 10},
}


def make_task(name, seed):
    from scripts.data_cache import load_xyq
    n_tr, n_va, n_te, n_feat, max_label, max_docs = TASKS[name]
    X, y, q = load_xyq(os.path.join(ROOT, "data", "train", "train.txt"))
    rng = np.random.RandomState(1000 + seed)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    order = rng.permutation(len(starts))
    cols = np.sort(rng.choice(X.shape[1], n_feat, replace=False)) if n_feat else np.arange(X.shape[1])

    def part(qs):
        rows = []
        for qi in qs:
            r = np.arange(starts[qi], ends[qi])
            if max_docs and len(r) > max_docs:
                r = np.sort(rng.choice(r, max_docs, replace=False))
            rows.append(r)
        sizes = [len(r) for r in rows]
        rows = np.concatenate(rows)
        yy = y[rows] if max_label == 4 else np.minimum(y[rows], 2)
        return X[rows][:, cols], yy, sizes

    a, b = n_tr, n_tr + n_va
    return part(order[:a]), part(order[a:b]), part(order[b:b + n_te])


def query_relative(X, sizes, kind):
    """Append a within-query transform of every column. Trees are not invariant to this: the new columns
    are invariant to a per-feature shift and scale, which raw values are not. kind 'z' is a z-score,
    kind 'r' is the average rank in [0, 1]."""
    extra = np.empty_like(X)
    i = 0
    for n in sizes:
        block = X[i:i + n]
        if kind == "z":
            mu = block.mean(axis=0)
            sd = block.std(axis=0)
            sd = np.where(sd < 1e-6, 1.0, sd)
            extra[i:i + n] = (block - mu) / sd
        else:
            # average rank, 0 = worst. argsort twice; ties share the mean rank of the tie group.
            order = np.argsort(block, axis=0, kind="stable")
            ranks = np.empty_like(order)
            rows = np.arange(n)[:, None]
            ranks[order, np.arange(block.shape[1])] = rows
            extra[i:i + n] = ranks / max(n - 1, 1)
        i += n
    return np.hstack([X, extra]).astype(np.float32, copy=False)


def run(job):
    task, seed, variant = job
    import lightgbm as lgb
    from harness.metrics import mean_ndcg_from_scores
    from submission import recipe

    (X, y, g), (Xv, yv, gv), (Xt, yt, gt) = make_task(task, seed)
    rel = "z" if variant in ("qz", "qz_trunc10") else ("r" if variant == "qr" else None)
    if rel is not None:
        X, Xv, Xt = query_relative(X, g, rel), query_relative(Xv, gv, rel), query_relative(Xt, gt, rel)
    qt = np.repeat(np.arange(len(gt)), gt)
    t0 = time.perf_counter()
    spec = VARIANTS[variant]
    if spec == "starter":                                   # the shipped starter recipe: fixed 100 trees
        params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [10], "num_leaves": 15,
                  "learning_rate": 0.05, "min_data_in_leaf": 50, "lambda_l2": 1.0, "feature_fraction": 0.8,
                  "bagging_fraction": 0.8, "bagging_freq": 1, "seed": seed, "deterministic": True,
                  "force_row_wise": True, "verbose": -1, "num_threads": 4}
        booster = lgb.train(params, lgb.Dataset(X, label=y, group=g), num_boost_round=100)
        n_iter, info = 100, {}
    else:
        kw = dict(spec) if isinstance(spec, dict) else {}
        if spec in ("leaves_x2", "leaves_half", "mdl_x4"):
            base = recipe.rule_params(X.shape[0], len(g), X.shape[1], 5, seed)
            if spec == "leaves_x2":
                kw["num_leaves"] = 2 * base["num_leaves"] + 1
            elif spec == "leaves_half":
                kw["num_leaves"] = max(3, base["num_leaves"] // 2)
            else:
                kw["min_data_in_leaf"] = 4 * base["min_data_in_leaf"]
        booster, n_iter, info = recipe.fit_arrays(X, y, g, Xv, yv, gv, seed=seed, **kw)
    secs = time.perf_counter() - t0
    nd = mean_ndcg_from_scores(booster.predict(Xt, num_iteration=n_iter), yt, qt, 10)
    return {"task": task, "seed": seed, "variant": variant, "test_ndcg@10": nd, "seconds": round(secs, 1),
            "n_train_rows": int(X.shape[0]), "D": int(X.shape[1]), "info": info}


def report():
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    tasks = sorted({r["task"] for r in rows})
    variants = sorted({r["variant"] for r in rows})
    tab = {}
    for r in rows:
        tab.setdefault((r["variant"], r["task"]), []).append(r["test_ndcg@10"])
    print(f"{'variant':<12}" + "".join(f"{t[:20]:>22}" for t in tasks) + f"{'mean':>8}")
    for v in variants:
        vals = [np.mean(tab[(v, t)]) if (v, t) in tab else np.nan for t in tasks]
        print(f"{v:<12}" + "".join(f"{x:>22.4f}" for x in vals) + f"{np.nanmean(vals):>8.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["default"])
    ap.add_argument("--tasks", nargs="+", default=sorted(TASKS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return
    if args.workers > 4:
        raise SystemExit("at most 4 worker processes")
    from scripts.data_cache import load_xyq
    load_xyq(os.path.join(ROOT, "data", "train", "train.txt"))
    jobs = [(t, s, v) for t in args.tasks for s in args.seeds for v in args.variants]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(run, j) for j in jobs]):
            r = fut.result()
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            print(f"{r['task']:<24} seed {r['seed']} {r['variant']:<12} test nDCG@10 {r['test_ndcg@10']:.4f} "
                  f"({r['seconds']:.0f}s) {r['info']}", flush=True)


if __name__ == "__main__":
    main()
