"""Train the LambdaMART artifact that submission/ranking.py loads (offline; grading never retrains).

    python scripts/train_model.py --train data/train --out submission/model.txt --trees 300

--train takes a split file or a directory holding train.txt. Pass --use-dev to ALSO train on the dev split;
that raises your dev nDCG (Tracks A and B) without helping on queries the model has not seen, so your public
leaderboard position will overstate where you stand (assignment Section 7). Whatever you do, declare it in your
submission notes (Section 10).

"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import time

import numpy as np

from harness.feature_io import read_svmlight_ltr, stack
from scripts.lambdamart_utils import default_params, fit, ndcg10
from submission.ranking import within_query_ranks


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
    ap.add_argument("--max-bin", type=int, default=255)
    ap.add_argument("--extra-trees", action="store_true",
                    help="draw each split threshold at random between the bin edges")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feature-noise", default="0",
                    help="comma-separated Gaussian noise scales, in units of each feature's TRAIN std. "
                         "Each scale appends one extra copy of the rows (0 = off). Labels are unchanged.")
    ap.add_argument("--query-ranks", action="store_true",
                    help="append within-query feature ranks, the same transform rerank_with_ltr applies "
                         "when the model has twice as many features as the input")
    ap.add_argument("--single-field-copies", action="store_true",
                    help="append five copies of the rows, each keeping one MSLR field "
                         "(body, anchor, title, url, whole document) plus features 126-136")
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
    scales = [float(s) for s in str(args.feature_noise).split(",") if float(s) > 0]
    if scales:
        rng = np.random.RandomState(args.seed)
        std = X.std(axis=0).astype(np.float32)
        copies_x, copies_y, copies_q = [X], [y], [q]
        for scale in scales:
            noisy = (X + scale * std * rng.randn(*X.shape).astype(np.float32)).astype(np.float32)
            copies_x.append(noisy)
            copies_y.append(y)
            copies_q.append(q)
        X = np.vstack(copies_x)
        y = np.concatenate(copies_y)
        q = np.concatenate(copies_q)
    if args.single_field_copies:
        copies_x, copies_y, copies_q = [X], [y], [q]
        for field in range(5):
            kept = np.zeros_like(X)
            kept[:, field:125:5] = X[:, field:125:5]
            kept[:, 125:] = X[:, 125:]
            copies_x.append(kept)
            copies_y.append(y)
            copies_q.append(q)
        X = np.vstack(copies_x)
        y = np.concatenate(copies_y)
        q = np.concatenate(copies_q)
    if args.query_ranks:
        # Ranks are computed inside each copy, so a noisy copy does not inherit the clean order.
        ranks = np.empty_like(X)
        starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
        ends = np.r_[starts[1:], len(q)]
        for a, b in zip(starts, ends):
            ranks[a:b] = within_query_ranks(X[a:b])
        X = np.hstack([X, ranks]).astype(np.float32)
    params = default_params(args.num_leaves, args.learning_rate, args.min_data_in_leaf, lambda_l2=args.lambda_l2,
                            seed=args.seed, feature_fraction=args.feature_fraction,
                            bagging_fraction=args.bagging_fraction, bagging_freq=args.bagging_freq,
                            max_bin=args.max_bin, extra_trees=args.extra_trees)
    t0 = time.perf_counter()
    model = fit(X, y, q, params, args.trees)
    fit_s = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save_model(args.out)
    print(f"trained on {len(train_used)} queries ({'train+dev' if args.use_dev else 'train only'}) -> {args.out} "
          f"({args.trees} trees, feature_noise={args.feature_noise}, "
          f"single_field_copies={args.single_field_copies}, {fit_s:.0f}s)")
    print(f"params: {params}")
    print(f"train nDCG@10 = {ndcg10(model, X[:n_clean], y[:n_clean], q[:n_clean]):.4f}")
    if dev is not None:
        Xd, yd, qd = stack(dev)
        if args.query_ranks:
            ranks = np.empty_like(Xd)
            starts = np.flatnonzero(np.r_[True, qd[1:] != qd[:-1]])
            ends = np.r_[starts[1:], len(qd)]
            for a, b in zip(starts, ends):
                ranks[a:b] = within_query_ranks(Xd[a:b])
            Xd = np.hstack([Xd, ranks]).astype(np.float32)
        label = "dev nDCG@10 (in-training, dev rows were fit)" if args.use_dev else "dev   nDCG@10"
        print(f"{label} = {ndcg10(model, Xd, yd, qd):.4f}")


if __name__ == "__main__":
    main()
