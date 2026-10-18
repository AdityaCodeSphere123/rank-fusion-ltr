"""Grouped 5-fold cross-validation sweep over the number of trees (assignment Section 4.3).

    python scripts/cv_lambdamart.py --train data/train --trees 25 50 100 200 400

For each tree count we train on 4 folds and score the 5th. NO early stopping is used on the reported fold.
The gap between the train-fold and validation-fold nDCG@10 is your local estimate of how much a model loses on
queries it has not seen: a model whose gap widens with more trees is overfitting. Pick the tree count where validation stops improving.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse

import numpy as np

from harness.cv import grouped_kfold
from harness.feature_io import read_svmlight_ltr, stack
from scripts.lambdamart_utils import default_params, fit, ndcg10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/toy")
    ap.add_argument("--trees", type=int, nargs="+", default=[10, 30, 100, 300])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--num-leaves", type=int, default=15)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    path = os.path.join(args.train, "train.txt") if os.path.isdir(args.train) else args.train
    X, y, q = stack(read_svmlight_ltr(path))
    folds = grouped_kfold(q, k=args.folds, seed=args.seed)
    params = default_params(args.num_leaves, args.learning_rate, args.min_data_in_leaf, seed=args.seed)
    print(f"{'trees':>6} {'train':>8} {'val':>8} {'val/train':>10}")
    for n in args.trees:
        tr, va = [], []
        for tr_idx, va_idx in folds:
            m = fit(X[tr_idx], y[tr_idx], q[tr_idx], params, n)
            tr.append(ndcg10(m, X[tr_idx], y[tr_idx], q[tr_idx]))
            va.append(ndcg10(m, X[va_idx], y[va_idx], q[va_idx]))
        print(f"{n:>6} {np.mean(tr):>8.4f} {np.mean(va):>8.4f} {min(1.0, np.mean(va) / np.mean(tr)):>10.4f}")


if __name__ == "__main__":
    main()
