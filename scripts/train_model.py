"""Train the LambdaMART artifact that submission/ranking.py loads (offline; grading never retrains).

    python scripts/train_model.py --train data/train --out submission/model.txt --trees 300

--train takes a split file or a directory holding train.txt. Pass --use-dev to ALSO train on the dev split;
that raises your dev nDCG (Tracks A and B) without helping on queries the model has not seen, so your public
leaderboard position will overstate where you stand (assignment Section 7). Whatever you do, declare it in your
submission notes (Section 10).

The submitted submission/model.txt was produced with TRAIN only (no --use-dev); the exact command, with the
hyperparameters chosen by grouped 5-fold CV on TRAIN (scripts/cv_sweep.py), is in Decision.md section 4.
LightGBM runs with num_threads=4, deterministic=True, force_row_wise=True and a fixed seed, so the artifact is
reproducible. DEV is only scored after training, as a reported number.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import time

import numpy as np

from harness.feature_io import read_svmlight_ltr, stack
from scripts.lambdamart_utils import default_params, fit, ndcg10


def _resolve(path, name):
    return os.path.join(path, name) if os.path.isdir(path) else path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/toy")
    ap.add_argument("--dev", default=None, help="dev file/dir; used for reporting only unless --use-dev")
    ap.add_argument("--out", default="submission/model.txt")
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--num-leaves", type=int, default=15)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=50)
    ap.add_argument("--lambda-l2", type=float, default=1.0)
    ap.add_argument("--feature-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-freq", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feature-noise", type=float, default=0.0,
                    help="append a second copy of the training rows with Gaussian feature noise of this many "
                         "feature-stds (0 = off). Labels are unchanged. Chosen on a TRAIN fold for Track C2.")
    ap.add_argument("--use-dev", action="store_true", help="also train on dev (lowers Track C; declare it)")
    args = ap.parse_args()

    train = read_svmlight_ltr(_resolve(args.train, "train.txt"))
    dev_path = _resolve(args.dev, "vali.txt") if args.dev else _resolve(args.train, "vali.txt")
    dev = read_svmlight_ltr(dev_path) if os.path.isfile(dev_path) else None
    if args.use_dev:
        if dev is None:
            raise SystemExit("--use-dev needs a dev file")
        merged = dict(train); merged.update(dev); train_used = merged
    else:
        train_used = train
    X, y, q = stack(train_used)
    n_clean = len(y)
    if args.feature_noise > 0:
        rng = np.random.RandomState(args.seed)
        std = X.std(axis=0).astype(np.float32)
        noisy = X + args.feature_noise * std * rng.randn(*X.shape).astype(np.float32)
        X = np.vstack([X, noisy.astype(np.float32)])
        y = np.concatenate([y, y])
        q = np.concatenate([q, q])
    params = default_params(args.num_leaves, args.learning_rate, args.min_data_in_leaf, lambda_l2=args.lambda_l2,
                            seed=args.seed, feature_fraction=args.feature_fraction,
                            bagging_fraction=args.bagging_fraction, bagging_freq=args.bagging_freq)
    t0 = time.perf_counter()
    model = fit(X, y, q, params, args.trees)
    fit_s = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save_model(args.out)
    print(f"trained on {len(train_used)} queries ({'train+dev' if args.use_dev else 'train only'}) -> {args.out} "
          f"({args.trees} trees, feature_noise={args.feature_noise}, {fit_s:.0f}s)")
    print(f"params: {params}")
    print(f"train nDCG@10 = {ndcg10(model, X[:n_clean], y[:n_clean], q[:n_clean]):.4f}")
    if dev is not None and not args.use_dev:
        Xd, yd, qd = stack(dev)
        print(f"dev   nDCG@10 = {ndcg10(model, Xd, yd, qd):.4f}")


if __name__ == "__main__":
    main()
