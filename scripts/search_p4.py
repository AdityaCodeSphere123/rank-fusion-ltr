"""Two last pairwise checks against the shipped file.

A longer, slower schedule is compared on a disjoint tenth of TRAIN. If it does
not win, the shipped schedule is refit on all of TRAIN for a fixed epoch budget
and kept only when DEV rises.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

from harness.cv import grouped_kfold
from scripts.data_cache import load_xyq
from submission.pointwise_pairwise import dev_ndcg, make_scorer, save_scorer, train_model

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BAR_SECOND = 0.4339
BAR_DEV = 0.4531

SLOW = dict(
    kind="pairwise", seed=0, epochs=60, patience=12, lr=5e-4, lr_decay=0.97,
    pairs_per_query=2000, queries_per_batch=64, hidden=[256, 256, 256],
    pair_weighting="delta_ndcg", scale="winsor", verbose=False,
)


def _indices(q):
    folds = grouped_kfold(q, k=10, seed=0)
    monitor = folds[0][1]
    second = folds[1][1]
    return monitor, second


def _subset(X, y, q, drop):
    keep = np.array([i for i in range(len(y)) if i not in drop])
    return X[keep], y[keep], q[keep]


def _fit(kw, train, monitor):
    net, std, hist = train_model(kw["kind"], train, monitor, **{k: v for k, v in kw.items() if k != "kind"})
    return net, std, hist


def main():
    torch.set_num_threads(4)
    X, y, q = load_xyq(os.path.join(ROOT, "data", "train", "train.txt"))
    dev = load_xyq(os.path.join(ROOT, "data", "dev", "vali.txt"))
    monitor, second = _indices(q)
    drop = set(monitor.tolist()) | set(second.tolist())
    print("start slower", flush=True)
    net, std, hist = _fit(SLOW, _subset(X, y, q, drop), (X[monitor], y[monitor], q[monitor]))
    scorer = make_scorer(net, std)
    best = max(hist, key=lambda h: h["dev_ndcg@10"])
    second_s = dev_ndcg(scorer, X[second], y[second], q[second])
    dev_s = dev_ndcg(scorer, *dev)
    print(f"slower epochs {len(hist)} best {best['epoch']} monitor {best['dev_ndcg@10']:.4f} "
          f"second {second_s:.4f} dev {dev_s:.4f}", flush=True)
    path = os.path.join(ROOT, "submission", "pairwise_model.pt")
    if second_s > BAR_SECOND:
        kw = dict(SLOW)
        print("refit slower on nine tenths", flush=True)
        net, std, hist = _fit(kw, _subset(X, y, q, set(monitor.tolist())), (X[monitor], y[monitor], q[monitor]))
        scorer = make_scorer(net, std)
        dev_s = dev_ndcg(scorer, *dev)
        best = max(hist, key=lambda h: h["dev_ndcg@10"])
        print(f"refit monitor {best['dev_ndcg@10']:.4f} dev {dev_s:.4f}", flush=True)
        if dev_s > BAR_DEV:
            save_scorer(path, net, std, kw["hidden"])
            print(f"wrote {path}", flush=True)
            return
    kw = dict(SLOW)
    kw.update(epochs=40, patience=40, lr=5e-4, lr_decay=0.95)
    print("start all-train refit of the shipped schedule", flush=True)
    net, std, hist = _fit(kw, (X, y, q), (X[monitor], y[monitor], q[monitor]))
    scorer = make_scorer(net, std)
    dev_s = dev_ndcg(scorer, *dev)
    best = max(hist, key=lambda h: h["dev_ndcg@10"])
    print(f"all-train epochs {len(hist)} best {best['epoch']} in-sample {best['dev_ndcg@10']:.4f} "
          f"dev {dev_s:.4f}", flush=True)
    if dev_s > BAR_DEV:
        save_scorer(path, net, std, kw["hidden"])
        print(f"wrote {path}", flush=True)
    else:
        print("kept the saved pairwise file", flush=True)


if __name__ == "__main__":
    main()
