"""Compare the C1 recipe with and without within-query percentiles.

Uses a fixed slice of MSLR TRAIN. One task keeps the original features. The
other multiplies every column by a log-normal constant, which is the arbitrary
scale a hidden benchmark is allowed to have. Scores are nDCG@10 on queries
that were not in that task's train or vali.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from harness.metrics import mean_ndcg_from_scores
from scripts.data_cache import load_xyq
from scripts.lambdamart_utils import group_sizes
from submission.ranking import within_query_ranks
from submission.recipe import fit_arrays


def _append_ranks(X, sizes):
    ranks = np.empty(X.shape, dtype=np.float32)
    row = 0
    for n in sizes:
        ranks[row:row + n] = within_query_ranks(X[row:row + n])
        row += n
    return np.hstack([np.asarray(X, dtype=np.float32), ranks])


def _split(X, y, q, seed):
    rng = np.random.RandomState(seed)
    queries = np.array(list(dict.fromkeys(q.tolist())))
    rng.shuffle(queries)
    train_q, vali_q, test_q = queries[:800], queries[800:1000], queries[1000:1200]
    def take(wanted):
        m = np.isin(q, wanted)
        return X[m], y[m], group_sizes(q[m])
    return take(train_q), take(vali_q), take(test_q), q


def _score(booster, n_iter, X, y, sizes):
    # One score per row; rebuild qids from the group sizes.
    q = np.repeat(np.arange(len(sizes)), sizes)
    pred = booster.predict(X, num_iteration=n_iter, num_threads=2)
    return mean_ndcg_from_scores(pred, y, q, 10)


def run(name, X, y, q):
    tr, va, te, _ = _split(X, y, q, seed=0)
    raw_b, raw_n, raw_info = fit_arrays(*tr, *va, seed=0, refit=True, num_threads=2)
    ranked_b, ranked_n, ranked_info = fit_arrays(
        _append_ranks(tr[0], tr[2]), tr[1], tr[2],
        _append_ranks(va[0], va[2]), va[1], va[2], seed=0, refit=True, num_threads=2)
    raw = _score(raw_b, raw_n, te[0], te[1], te[2])
    ranked = _score(ranked_b, ranked_n, _append_ranks(te[0], te[2]), te[1], te[2])
    print(f"{name:<16} raw {raw:.4f}  ranks {ranked:.4f}  "
          f"rounds {raw_info.get('best_iteration')}/{ranked_info.get('best_iteration')}  "
          f"refit {raw_info.get('refit_rounds')}/{ranked_info.get('refit_rounds')}", flush=True)


def main():
    X, y, q = load_xyq(os.path.join(os.path.dirname(__file__), "..", "data", "train", "train.txt"))
    q = np.asarray(q)
    keep = np.isin(q, list(dict.fromkeys(q.tolist()))[:1200])
    X, y, q = X[keep], y[keep], q[keep]
    run("original", X, y, q)
    # One positive scale per query per feature. Rank order inside a query is
    # unchanged, so percentiles stay put while absolute tree thresholds move.
    rng = np.random.RandomState(1)
    scaled = X.copy()
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    for a, b in zip(starts, ends):
        scaled[a:b] *= np.exp(rng.normal(0, 1.5, size=X.shape[1])).astype(np.float32)
    run("per_query_scale", scaled, y, q)


if __name__ == "__main__":
    main()
