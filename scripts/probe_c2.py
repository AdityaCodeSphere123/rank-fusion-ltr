"""Score the shipped LambdaMART model under candidate Track-C2 corruptions.

Selection evidence is the TRAIN fold-0 number. DEV is printed only to compare the drop with the
leaderboard (Track B 0.4845 -> Track C2 0.3533). Nothing is trained.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from harness.cv import grouped_kfold
from harness.metrics import mean_ndcg_from_scores
from scripts.data_cache import load_xyq

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIELDS = ["body", "anchor", "title", "url", "whole_document"]


def ndcg(model, X, y, q):
    Xp = X
    if model.num_feature() == X.shape[1] * 2:
        from submission.ranking import within_query_ranks
        ranks = np.empty((len(X), X.shape[1]), dtype=np.float64)
        starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
        ends = np.r_[starts[1:], len(q)]
        for a, b in zip(starts, ends):
            ranks[a:b] = within_query_ranks(X[a:b])
        Xp = np.hstack([X, ranks])
    return mean_ndcg_from_scores(model.predict(Xp, num_threads=2), y, q, 10)


def versions(X):
    rng = np.random.RandomState(0)
    std = X.std(0).astype(np.float32)
    out = {"clean": X}
    for scale in (0.05, 0.1, 0.2, 0.5, 1.0):
        out[f"add_{scale}std"] = (X + scale * std * rng.randn(*X.shape).astype(np.float32)).astype(np.float32)
        out[f"mult_{scale}"] = (X * (1 + scale * rng.randn(*X.shape).astype(np.float32))).astype(np.float32)
    # one version per MSLR field, that field zeroed (features 1..125 are 25 groups x 5 fields)
    for f, name in enumerate(FIELDS):
        Z = X.copy()
        Z[:, f:125:5] = 0
        out[f"drop_{name}"] = Z
    for f, name in enumerate(FIELDS):
        Z = np.zeros_like(X)
        Z[:, f:125:5] = X[:, f:125:5]
        Z[:, 125:] = X[:, 125:]
        out[f"only_{name}"] = Z
    # random projections back to 136 dims (a dense mixture of the original coordinates)
    for k, rank in enumerate((136, 64)):
        R = rng.normal(scale=1 / np.sqrt(rank), size=(X.shape[1], rank)).astype(np.float32)
        proj = X @ R
        if rank < X.shape[1]:
            padded = np.zeros_like(X)
            padded[:, :rank] = proj
            proj = padded
        out[f"proj_rank{rank}_{k}"] = proj.astype(np.float32)
    # permute features inside each group of 5 fields
    perm = X.copy()
    for g in range(25):
        perm[:, 5 * g:5 * g + 5] = X[:, 5 * g + rng.permutation(5)]
    out["permute_fields"] = perm
    # scale each feature by a log-normal factor (a different normalisation per version)
    scale = np.exp(rng.normal(0, 1, size=X.shape[1])).astype(np.float32)
    out["lognormal_scale"] = X * scale
    return out


def report(tag, model, X, y, q):
    print(f"== {tag} ==")
    scores = {}
    for name, Xp in versions(X).items():
        scores[name] = ndcg(model, Xp, y, q)
        print(f"  {name:<22} {scores[name]:.4f}", flush=True)
    drops = [scores[f"drop_{n}"] for n in FIELDS]
    print(f"  mean drop-one-field   {np.mean(drops):.4f}")
    print(f"  mean add_0.1/0.2/0.5  {np.mean([scores['add_0.1std'], scores['add_0.2std'], scores['add_0.5std']]):.4f}")


def main():
    import lightgbm as lgb
    model = lgb.Booster(model_file=os.path.join(ROOT, "submission", "model.txt"))
    X, y, q = load_xyq(os.path.join(ROOT, "data", "train", "train.txt"))
    _, va = grouped_kfold(q, k=5, seed=0)[0]
    report("TRAIN fold 0", model, X[va], y[va], q[va])
    Xd, yd, qd = load_xyq(os.path.join(ROOT, "data", "dev", "vali.txt"))
    report("DEV (report only)", model, Xd, yd, qd)


if __name__ == "__main__":
    main()
