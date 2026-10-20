"""Two more Track P settings. Replaces pairwise_model.pt only on a clear win."""
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


def _split(X, y, q, keep_second=False):
    folds = grouped_kfold(q, k=10, seed=0)
    monitor, second = folds[0][1], folds[1][1]
    drop = set(monitor.tolist())
    if not keep_second:
        drop |= set(second.tolist())
    keep = np.array([i for i in range(len(y)) if i not in drop])
    return ((X[keep], y[keep], q[keep]),
            (X[monitor], y[monitor], q[monitor]),
            (X[second], y[second], q[second]))


def _oversample(X, y, q):
    """Repeat queries that contain a grade of 2 or more, so rare relevant docs are seen twice."""
    change = np.r_[True, q[1:] != q[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(q)]
    blocks = []
    for a, b in zip(starts, ends):
        blocks.append((X[a:b], y[a:b], q[a:b]))
        if float(y[a:b].max()) >= 2:
            blocks.append((X[a:b], y[a:b], q[a:b]))
    return (np.vstack([b[0] for b in blocks]),
            np.concatenate([b[1] for b in blocks]),
            np.concatenate([b[2] for b in blocks]))


def _fit(name, extra, train, monitor, second, dev, oversample=False):
    kw = dict(BASE)
    kw.update(extra)
    if oversample:
        train = _oversample(*train)
    net, std, hist = train_model(kw["kind"], train, monitor, **{k: v for k, v in kw.items() if k != "kind"})
    scorer = make_scorer(net, std)
    best = max(hist, key=lambda h: h["dev_ndcg@10"])
    second_s = dev_ndcg(scorer, *second)
    dev_s = dev_ndcg(scorer, *dev)
    print(f"{name} epochs {len(hist)} best {best['epoch']} monitor {best['dev_ndcg@10']:.4f} "
          f"second {second_s:.4f} dev {dev_s:.4f}", flush=True)
    return second_s, kw


def main():
    torch.set_num_threads(4)
    X, y, q = load_xyq(os.path.join(ROOT, "data", "train", "train.txt"))
    dev = load_xyq(os.path.join(ROOT, "data", "dev", "vali.txt"))
    train, monitor, second = _split(X, y, q, keep_second=False)
    runs = []
    print("start mixed", flush=True)
    runs.append(("mixed",) + _fit("mixed", {"scale": "mixed"}, train, monitor, second, dev))
    print("start oversample", flush=True)
    runs.append(("oversample",) + _fit("oversample", {}, train, monitor, second, dev, oversample=True))
    runs.sort(key=lambda r: -r[1])
    name, second_s, kw = runs[0]
    print(f"best {name} second {second_s:.4f}", flush=True)
    if second_s <= BAR_SECOND:
        print("kept the saved pairwise file", flush=True)
        return
    train, monitor, _second = _split(X, y, q, keep_second=True)
    if name == "oversample":
        train = _oversample(*train)
    net, std, hist = train_model(kw["kind"], train, monitor, **{k: v for k, v in kw.items() if k != "kind"})
    dev_s = dev_ndcg(make_scorer(net, std), *dev)
    print(f"refit {name} dev {dev_s:.4f}", flush=True)
    if dev_s <= BAR_DEV:
        print("refit DEV did not clear the saved file", flush=True)
        return
    path = os.path.join(ROOT, "submission", "pairwise_model.pt")
    save_scorer(path, net, std, kw["hidden"])
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
