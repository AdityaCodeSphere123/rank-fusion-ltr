"""Vectorised fusion probes on TRAIN (and a reported DEV number). Does not change the submission."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from harness.metrics import mean_ndcg_from_scores
from scripts.data_cache import load_xyq

# 0-based columns for features 110, 108, 120, 125
COLS = np.array([109, 107, 119, 124])


def fused_scores(X, q, weights, power=1.0):
    scores = np.empty(len(q), np.float64)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    w = np.asarray(weights, dtype=np.float64)
    for a, b in zip(starts, ends):
        block = X[a:b][:, COLS].astype(np.float64)
        lo = block.min(axis=0)
        hi = block.max(axis=0)
        span = hi - lo
        norm = np.divide(block - lo, span, out=np.zeros_like(block), where=span > 0)
        if power != 1.0:
            norm = np.power(norm, power)
        scores[a:b] = norm @ w
    return scores


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    specs = [
        ("3ch", (1, 1, 1, 0), 1.0),
        ("4ch", (1, 1, 1, 1), 1.0),
        ("3ch^0.5", (1, 1, 1, 0), 0.5),
        ("3ch^2", (1, 1, 1, 0), 2.0),
        ("w 1,1,1.5,0", (1, 1, 1.5, 0), 1.0),
        ("w 1,1.5,1,0", (1, 1.5, 1, 0), 1.0),
        ("w 1.5,1,1,0", (1.5, 1, 1, 0), 1.0),
        ("w 1,0.5,1,0", (1, 0.5, 1, 0), 1.0),
        ("w 1,1,2,0", (1, 1, 2, 0), 1.0),
        ("w 2,1,1,0", (2, 1, 1, 0), 1.0),
        ("w 1,1,1,0.25", (1, 1, 1, 0.25), 1.0),
    ]
    for split in ("data/train/train.txt", "data/dev/vali.txt"):
        X, y, q = load_xyq(os.path.join(root, split))
        print(f"== {split} ==")
        for name, w, p in specs:
            nd = mean_ndcg_from_scores(fused_scores(X, q, w, p), y, q, 10)
            print(f"  {name:<16} nDCG@10 {nd:.4f}", flush=True)


if __name__ == "__main__":
    main()
