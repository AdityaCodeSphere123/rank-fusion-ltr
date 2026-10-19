"""Pick a tree count for the within-query z-score model, then refit.

    .venv/bin/python scripts/fit_qz.py --trees 800
    .venv/bin/python scripts/fit_qz.py --refit-trees 600
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


def _codes(q):
    return _run_codes(np.asarray(q))


def _corrupt(X, q):
    rng = np.random.RandomState(0)
    X = np.asarray(X, dtype=np.float32)
    std = np.clip(X.std(0), 1e-6, None).astype(np.float32)
    out = {
        "clean": X,
        "log1p": (np.sign(X) * np.log1p(np.abs(X))).astype(np.float32),
        "zscore": ((X - X.mean(0)) / std).astype(np.float32),
        "x10": np.asarray(X * 10, dtype=np.float32),
        "add0.05": (X + 0.05 * std * rng.randn(*X.shape).astype(np.float32)).astype(np.float32),
    }
    cell = np.array(X, copy=True)
    cell[rng.rand(*X.shape) < 0.10] = 0
    out["cell10"] = cell
    perm = np.array(X, copy=True)
    for g in range(25):
        perm[:, 5 * g:5 * g + 5] = X[:, 5 * g + rng.permutation(5)]
    out["permute"] = perm
    return out


def _pack(raw_by_name, q):
    codes = _codes(q)
    return {name: _expand(raw, codes, False, with_z=True) for name, raw in raw_by_name.items()}


def _report(model, packed, y, q, trees):
    keys = ["clean", "log1p", "zscore", "x10", "add0.05", "cell10", "permute"]
    scores = {}
    for key in keys:
        pred = model.predict(packed[key], num_iteration=trees, num_threads=4)
        scores[key] = mean_ndcg_from_scores(pred, y, q, 10)
    corrupt = [scores[k] for k in keys if k != "clean"]
    scores["mean"] = float(np.mean(corrupt))
    detail = " ".join(f"{k} {scores[k]:.4f}" for k in keys)
    print(f"trees {trees:4d}  {detail}  mean {scores['mean']:.4f}", flush=True)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=800)
    ap.add_argument("--every", type=int, default=200)
    ap.add_argument("--refit-trees", type=int, default=0, help="skip the curve and refit all of TRAIN+DEV")
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "submission", "model.txt"))
    args = ap.parse_args()

    params = default_params(63, 0.05, 80, lambda_l2=1.0, seed=0, feature_fraction=0.6,
                            bagging_fraction=0.8, bagging_freq=1, num_threads=args.num_threads)
    X, y, q = load_xyq(os.path.join(ROOT, "data", "train", "train.txt"))
    Xd, yd, qd = load_xyq(os.path.join(ROOT, "data", "dev", "vali.txt"))
    q = np.asarray(q)
    qd = np.asarray(qd)

    if args.refit_trees:
        print(f"refit on all TRAIN+DEV, {args.refit_trees} trees", flush=True)
        dev_x = _expand(Xd, _codes(qd), False, with_z=True)
        big_x = np.vstack([
            _expand(X, _codes(q), False, with_z=True),
            dev_x,
        ])
        big_y = np.concatenate([y, yd])
        big_c = np.concatenate([_codes(q), _codes(qd) + int(_codes(q)[-1]) + 1])
        del X, Xd
        gc.collect()
        model = fit(big_x, big_y, big_c, params, args.refit_trees)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        model.save_model(args.out)
        dev_nd = mean_ndcg_from_scores(model.predict(dev_x, num_threads=4), yd, qd, 10)
        print(f"wrote {args.out} features {model.num_feature()} trees {model.num_trees()}")
        print(f"dev nDCG@10 {dev_nd:.4f} (in-training, dev rows were fit)", flush=True)
        return

    tr_idx, va_idx = grouped_kfold(q, k=5, seed=0)[0]
    print("featurize holdout", flush=True)
    packed = _pack(_corrupt(X[va_idx], q[va_idx]), q[va_idx])
    yva, qva = y[va_idx], q[va_idx]
    print(f"holdout rows {len(yva)} width {packed['clean'].shape[1]}", flush=True)
    dev_codes = _codes(qd) + 10_000_000
    tr_codes = _codes(q[tr_idx])
    print("featurize fit", flush=True)
    big_x = np.vstack([
        _expand(X[tr_idx], tr_codes, False, with_z=True),
        _expand(Xd, dev_codes, False, with_z=True),
    ])
    big_y = np.concatenate([y[tr_idx], yd])
    big_c = np.concatenate([tr_codes, dev_codes])
    del X, Xd
    gc.collect()
    print(f"fit rows {len(big_y)}", flush=True)

    def _checkpoint(env, every=args.every, packed=packed, yva=yva, qva=qva):
        n = env.iteration + 1
        if n % every == 0 or n == args.trees:
            _report(env.model, packed, yva, qva, n)

    fit(big_x, big_y, big_c, params, args.trees, callbacks=[_checkpoint])


if __name__ == "__main__":
    main()
