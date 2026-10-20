"""Continue the shipped pairwise net for a few epochs at a small learning rate.

The monitor is a disjoint tenth of TRAIN. Another tenth is only scored.
The artifact is replaced only when that left-out score rises.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import copy

import numpy as np
import torch

from harness.cv import grouped_kfold
from harness.neural import ScoringNetwork
from harness.preprocessing import Standardizer
from scripts.data_cache import load_xyq
from submission.pointwise_pairwise import (
    _query_rows, dev_ndcg, lambdarank_query_step, make_scorer, save_scorer, set_seed,
)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    torch.set_num_threads(4)
    X, y, q = load_xyq(os.path.join(ROOT, "data", "train", "train.txt"))
    dev = load_xyq(os.path.join(ROOT, "data", "dev", "vali.txt"))
    folds = grouped_kfold(q, k=10, seed=0)
    monitor_i, hold_i = folds[0][1], folds[1][1]
    drop = set(monitor_i.tolist()) | set(hold_i.tolist())
    keep = np.array([i for i in range(len(y)) if i not in drop])
    train = (X[keep], y[keep], q[keep])
    monitor = (X[monitor_i], y[monitor_i], q[monitor_i])
    holdout = (X[hold_i], y[hold_i], q[hold_i])

    path = os.path.join(ROOT, "submission", "pairwise_model.pt")
    blob = torch.load(path, weights_only=True)
    std = Standardizer.from_dict(blob["standardizer"])
    net = ScoringNetwork(int(blob["config"]["n_features"]), blob["config"]["hidden"])
    net.load_state_dict(blob["state_dict"])
    base_hold = dev_ndcg(make_scorer(net, std), *holdout)
    base_dev = dev_ndcg(make_scorer(net, std), *dev)
    print(f"start left-out {base_hold:.4f} dev {base_dev:.4f}", flush=True)

    set_seed(0)
    Xt = torch.from_numpy(std.transform(train[0]))
    yt = torch.from_numpy(np.asarray(train[1], dtype=np.float32))
    rows = _query_rows(train[2])
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    rng = np.random.RandomState(0)
    best_hold, best_state = base_hold, copy.deepcopy(net.state_dict())
    for epoch in range(1, 9):
        net.train()
        order = rng.permutation(len(rows))
        for b in range(0, len(order), 64):
            batch = [rows[i] for i in order[b:b + 64]]
            lambdarank_query_step(net, opt, Xt, yt, batch, 2000, rng)
        for group in opt.param_groups:
            group["lr"] *= 0.9
        hold = dev_ndcg(make_scorer(net, std), *holdout)
        mon = dev_ndcg(make_scorer(net, std), *monitor)
        print(f"epoch {epoch} monitor {mon:.4f} left-out {hold:.4f}", flush=True)
        if hold > best_hold:
            best_hold, best_state = hold, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    hold = dev_ndcg(make_scorer(net, std), *holdout)
    dev_s = dev_ndcg(make_scorer(net, std), *dev)
    print(f"best left-out {hold:.4f} dev {dev_s:.4f}", flush=True)
    if hold > base_hold + 0.002:
        save_scorer(path, net, std, blob["config"]["hidden"])
        print(f"replaced {path}", flush=True)
    else:
        print("kept the shipped pairwise net", flush=True)


if __name__ == "__main__":
    main()
