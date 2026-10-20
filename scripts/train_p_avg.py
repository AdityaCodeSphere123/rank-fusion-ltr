"""Train several pairwise nets on the same TRAIN split and average their weights.

The split is fixed (seed 0). Only the network seed changes. DEV is scored
once at the end and is not used to pick an epoch.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import copy

import numpy as np
import torch

from harness.cv import grouped_kfold
from scripts.data_cache import load_xyq
from submission.pointwise_pairwise import dev_ndcg, make_scorer, save_scorer, train_model

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _split(X, y, q):
    folds = grouped_kfold(q, k=10, seed=0)
    monitor = folds[0][1]
    holdout = folds[1][1]
    drop = set(monitor.tolist()) | set(holdout.tolist())
    keep = np.array([i for i in range(len(y)) if i not in drop])
    return (X[keep], y[keep], q[keep]), (X[monitor], y[monitor], q[monitor]), (X[holdout], y[holdout], q[holdout])


def _average(states):
    acc = None
    for state in states:
        if acc is None:
            acc = {k: v.detach().float().clone() for k, v in state.items()}
        else:
            for k, v in state.items():
                acc[k] += v.detach().float()
    n = float(len(states))
    for k in acc:
        acc[k] /= n
    return acc


def _fit(train, monitor, seed, gain_weight):
    net, std, hist = train_model(
        "pairwise", train, monitor, seed=seed, epochs=25, patience=5, lr=1e-3, lr_decay=0.9,
        pairs_per_query=2000, queries_per_batch=64, hidden=[256, 256, 256],
        pair_weighting="delta_ndcg", gain_weight=gain_weight, verbose=True,
    )
    best = max(hist, key=lambda h: h["dev_ndcg@10"])
    print(f"seed {seed} gain {gain_weight} best epoch {best['epoch']} monitor {best['dev_ndcg@10']:.4f}", flush=True)
    return net, std


def main():
    X, y, q = load_xyq(os.path.join(ROOT, "data", "train", "train.txt"))
    dev = load_xyq(os.path.join(ROOT, "data", "dev", "vali.txt"))
    train, monitor, holdout = _split(X, y, q)
    torch.set_num_threads(4)
    nets = []
    std = None
    for seed in (0, 1, 2):
        net, std, = _fit(train, monitor, seed, 0.0)
        hold = dev_ndcg(make_scorer(net, std), *holdout)
        print(f"seed {seed} left-out {hold:.4f}", flush=True)
        nets.append(copy.deepcopy(net))
    net, std = _fit(train, monitor, 0, 0.1)
    hold = dev_ndcg(make_scorer(net, std), *holdout)
    print(f"gain 0.1 left-out {hold:.4f} dev {dev_ndcg(make_scorer(net, std), *dev):.4f}", flush=True)
    avg_net = nets[0]
    avg_net.load_state_dict(_average([n.state_dict() for n in nets]))
    scorer = make_scorer(avg_net, std)
    print(f"avg3 left-out {dev_ndcg(scorer, *holdout):.4f} dev {dev_ndcg(scorer, *dev):.4f}", flush=True)
    path = os.path.join(ROOT, "submission", "pairwise_model_avg.pt")
    save_scorer(path, avg_net, std, [256, 256, 256])
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
