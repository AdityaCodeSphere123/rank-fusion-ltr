"""Compare pairwise training settings on a disjoint tenth of TRAIN.

DEV is scored after the choice and is not used to pick a setting. This script
does not write pairwise_model.pt.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

from harness.cv import grouped_kfold
from scripts.data_cache import load_xyq
from submission.pointwise_pairwise import dev_ndcg, make_scorer, train_model

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

BASE = dict(
    kind="pairwise", seed=0, epochs=25, patience=6, lr=1e-3, lr_decay=0.9,
    pairs_per_query=2000, queries_per_batch=64, hidden=[256, 256, 256],
    pair_weighting="delta_ndcg", verbose=False,
)

CONFIGS = [
    ("base", {}),
    ("lambda", {"pair_sampling": "lambda"}),
    ("winsor", {"scale": "winsor"}),
    ("noise", {"input_noise": 0.1}),
    ("lambda_slow", {"pair_sampling": "lambda", "lr": 5e-4, "lr_decay": 0.95, "epochs": 30, "patience": 8}),
    ("winsor_lambda", {"scale": "winsor", "pair_sampling": "lambda"}),
]


def _split(X, y, q, keep_second=False):
    folds = grouped_kfold(q, k=10, seed=0)
    monitor = folds[0][1]
    second = folds[1][1]
    drop = set(monitor.tolist())
    if not keep_second:
        drop |= set(second.tolist())
    keep = np.array([i for i in range(len(y)) if i not in drop])
    return ((X[keep], y[keep], q[keep]),
            (X[monitor], y[monitor], q[monitor]),
            (X[second], y[second], q[second]))


def _run(name, extra, train, monitor, second, dev):
    kw = dict(BASE)
    kw.update(extra)
    net, std, hist = train_model(kw["kind"], train, monitor, **{k: v for k, v in kw.items() if k != "kind"})
    scorer = make_scorer(net, std)
    best = max(hist, key=lambda h: h["dev_ndcg@10"])
    second_s = dev_ndcg(scorer, *second)
    dev_s = dev_ndcg(scorer, *dev)
    print(f"{name} epochs {len(hist)} best {best['epoch']} monitor {best['dev_ndcg@10']:.4f} "
          f"second {second_s:.4f} dev {dev_s:.4f}", flush=True)
    return second_s, dev_s, net, std, kw


def main():
    torch.set_num_threads(4)
    X, y, q = load_xyq(os.path.join(ROOT, "data", "train", "train.txt"))
    dev = load_xyq(os.path.join(ROOT, "data", "dev", "vali.txt"))
    train, monitor, second = _split(X, y, q, keep_second=False)
    best = None
    for name, extra in CONFIGS:
        print(f"start {name}", flush=True)
        second_s, dev_s, net, std, kw = _run(name, extra, train, monitor, second, dev)
        if best is None or second_s > best[0]:
            best = (second_s, dev_s, name, extra, kw)
    print(f"winner {best[2]} second {best[0]:.4f} dev {best[1]:.4f}", flush=True)
    print("comparison only; pairwise_model.pt is left unchanged", flush=True)


if __name__ == "__main__":
    main()
