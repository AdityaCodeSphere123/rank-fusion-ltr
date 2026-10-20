"""Screen more pairwise settings against the shipped clipped-scale schedule.

A setting replaces pairwise_model.pt only when a disjoint tenth of TRAIN beats
the shipped comparison score and the nine-tenth refit also scores higher on DEV.
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

BASE = dict(
    kind="pairwise", seed=0, epochs=40, patience=10, lr=5e-4, lr_decay=0.95,
    pairs_per_query=2000, queries_per_batch=64, hidden=[256, 256, 256],
    pair_weighting="delta_ndcg", scale="winsor", verbose=False,
)

CONFIGS = [
    ("approx8", {"pair_weighting": "approx_ndcg", "approx_alpha": 8.0, "queries_per_batch": 16}),
    ("approx4", {"pair_weighting": "approx_ndcg", "approx_alpha": 4.0, "queries_per_batch": 16}),
    ("listnet_t8", {"pair_weighting": "listnet", "listnet_temp": 8.0, "queries_per_batch": 32}),
    ("listnet_label", {"pair_weighting": "listnet", "listnet_on": "label", "listnet_temp": 1.0, "queries_per_batch": 32}),
    ("swa", {"swa_start": 15}),
    ("pairs4k", {"pairs_per_query": 4000, "queries_per_batch": 32}),
    ("wd", {"weight_decay": 1e-5}),
    ("medwinsor", {"scale": "medwinsor"}),
    ("winsor2", {"scale": "winsor2"}),
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
    return second_s, extra


def main():
    torch.set_num_threads(4)
    X, y, q = load_xyq(os.path.join(ROOT, "data", "train", "train.txt"))
    dev = load_xyq(os.path.join(ROOT, "data", "dev", "vali.txt"))
    train, monitor, second = _split(X, y, q, keep_second=False)
    best = None
    for name, extra in CONFIGS:
        print(f"start {name}", flush=True)
        second_s, extra = _run(name, extra, train, monitor, second, dev)
        if best is None or second_s > best[0]:
            best = (second_s, name, extra)
    print(f"best {best[1]} second {best[0]:.4f}", flush=True)
    if best[0] <= BAR_SECOND:
        print("kept the saved pairwise file", flush=True)
        return
    train, monitor, second = _split(X, y, q, keep_second=True)
    kw = dict(BASE)
    kw.update(best[2])
    net, std, hist = train_model(kw["kind"], train, monitor, **{k: v for k, v in kw.items() if k != "kind"})
    scorer = make_scorer(net, std)
    best_ep = max(hist, key=lambda h: h["dev_ndcg@10"])
    dev_s = dev_ndcg(scorer, *dev)
    print(f"refit {best[1]} monitor {best_ep['dev_ndcg@10']:.4f} dev {dev_s:.4f}", flush=True)
    if dev_s <= BAR_DEV:
        print("refit DEV did not clear the saved file; pairwise file not replaced", flush=True)
        return
    path = os.path.join(ROOT, "submission", "pairwise_model.pt")
    save_scorer(path, net, std, kw["hidden"])
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
