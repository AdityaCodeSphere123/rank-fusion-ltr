"""
harness/cv.py -- grouped (by query) k-fold cross-validation splits.

Why grouped: all documents of a query must land in the same fold. Splitting
by row puts near-duplicate documents of one query in both the training and
validation sides, which inflates the validation score -- exactly the
overfitting cross-validation is supposed to measure (assignment Section 4.3).
"""
from typing import List, Sequence, Tuple

import numpy as np


def grouped_kfold(qids: Sequence, k: int = 5, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """qids: one query id per ROW. Returns [(train_row_idx, val_row_idx), ...]
    of length k. Queries are shuffled with `seed`, then dealt round-robin into
    k folds, so fold sizes differ by at most one query. Deterministic."""
    qids = np.asarray(qids, dtype=object)
    unique = list(dict.fromkeys(qids.tolist()))  # preserves first-seen order
    if k < 2:
        raise ValueError("k must be >= 2")
    if len(unique) < k:
        raise ValueError(f"need at least k={k} queries, got {len(unique)}")
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(unique))
    fold_of_query = {}
    for rank, pos in enumerate(order):
        fold_of_query[unique[pos]] = rank % k
    row_fold = np.array([fold_of_query[q] for q in qids.tolist()])
    splits = []
    for f in range(k):
        val_idx = np.where(row_fold == f)[0]
        train_idx = np.where(row_fold != f)[0]
        splits.append((train_idx, val_idx))
    return splits
