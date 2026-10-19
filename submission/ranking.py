"""Competition entry.

fuse_candidates: unweighted CombSUM of BM25 (document and title) and LMIR.DIR.
LMIR.JM is dropped; on TRAIN every channel-weight grid set it to 0.

rerank_with_ltr: the LightGBM model loaded by prepare(). Rows are padded or
truncated to the model's width. Two MSLR layouts are recognised. Twice the
candidate width is raw features plus within-query percentiles. Width
2*D + 150 (D >= 125) also appends, for each of the 25 field-groups, the max,
mean and min across the five fields, and the percentiles of those aggregates.
Every other width, including recipe.py models, is scored on the raw vectors.

prepare() only loads the artifact and raises if it is missing. Both rankers
return at most k (doc_id, score) pairs, best first, ids drawn only from the
candidate list, no duplicates. Ties keep candidate order.
"""
import os
from typing import Dict, List, Tuple

import numpy as np

from submission.fusion import combsum_scores

FUSION_CHANNELS = ("bm25_whole_document", "bm25_title", "lmir_dir_whole_document")

# MSLR features 1..125 are 25 groups x 5 fields. The aggregates below are
# invariant to the order of the five fields, and the max survives when the
# other four fields are cleared.
_N_GROUPS = 25
_N_FIELDS = 5
_N_AGGS = _N_GROUPS * 3

_MODEL = None
_N_FEATURES = None


def prepare(model_path: str) -> None:
    global _MODEL, _N_FEATURES
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"trained model artifact not found at {model_path!r}; run scripts/train_model.py and "
            f"include the artifact in your submission (grading never retrains)"
        )
    import lightgbm as lgb
    model = lgb.Booster(model_file=model_path)
    _MODEL, _N_FEATURES = model, model.num_feature()


def _unique(ids: List[str]) -> List[str]:
    return list(dict.fromkeys(ids))


def fuse_candidates(qid: str, candidate_doc_ids: List[str], base_scores: Dict[str, Dict[str, float]],
                    k: int) -> List[Tuple[str, float]]:
    ids = _unique(candidate_doc_ids)
    if not ids:
        return []
    fused = combsum_scores({d: base_scores[d] for d in ids}, channels=FUSION_CHANNELS)
    ranked = sorted(ids, key=lambda d: -fused[d])
    return [(d, fused[d]) for d in ranked[:k]]


def sanitize(X: np.ndarray) -> np.ndarray:
    """Replace non-finite values and values above 1e30 with NaN.

    LightGBM treats NaN as missing. Finite values are left unchanged.
    """
    X = np.asarray(X, dtype=np.float64)
    bad = ~np.isfinite(X) | (np.abs(X) > 1e30)
    if not bad.any():
        return X
    X = X.copy()
    X[bad] = np.nan
    return X


def within_query_ranks(X: np.ndarray) -> np.ndarray:
    """Within-query average-tie percentile of each feature, in [0, 1].

    One document, and any feature that is constant in the query, is 0. Tied
    values share a percentile, so document order does not matter. A monotone
    map of one feature leaves its percentile unchanged. Non-finite entries
    tie at the bottom of that feature.
    """
    X = np.asarray(X, dtype=np.float64)
    bad = ~np.isfinite(X)
    if bad.any():
        X = X.copy()
        finite = X[~bad]
        floor = (float(finite.min()) - 1.0) if finite.size else 0.0
        X[bad] = floor
    n, d = X.shape
    out = np.zeros((n, d), dtype=np.float64)
    if n <= 1:
        return out
    order = np.argsort(X, axis=0, kind="mergesort")
    sorted_x = np.take_along_axis(X, order, axis=0)
    denom = float(n - 1)
    for j in range(d):
        col = sorted_x[:, j]
        if col[0] == col[-1]:
            continue
        cuts = np.flatnonzero(col[1:] != col[:-1]) + 1
        starts = np.r_[0, cuts]
        ends = np.r_[cuts, n]
        avg = (starts + ends - 1) / 2.0 / denom
        out[order[np.repeat(np.arange(len(starts)), ends - starts), j], j] = np.repeat(avg, ends - starts)
    return out


def group_aggregates(X: np.ndarray) -> np.ndarray:
    """Max, mean and min across the five fields of each of the 25 MSLR groups."""
    g = np.asarray(X[:, :_N_GROUPS * _N_FIELDS], dtype=np.float64).reshape(X.shape[0], _N_GROUPS, _N_FIELDS)
    return np.concatenate([g.max(axis=2), g.mean(axis=2), g.min(axis=2)], axis=1)


def _feature_matrix(ids: List[str], features: Dict[str, List[float]], width: int) -> np.ndarray:
    rows = [np.asarray(features[d], dtype=np.float64).ravel() for d in ids]
    observed = max((r.shape[0] for r in rows), default=0)
    X = np.zeros((len(rows), observed), dtype=np.float64)
    for i, r in enumerate(rows):
        n = min(observed, r.shape[0])
        X[i, :n] = r[:n]
    X = sanitize(X)
    # 2D+150: raw, percentiles, field max/mean/min, percentiles of those.
    if observed >= _N_GROUPS * _N_FIELDS and width == 2 * observed + 2 * _N_AGGS:
        aggs = group_aggregates(X)
        X = np.hstack([X, within_query_ranks(X), aggs, within_query_ranks(aggs)])
    elif observed > 0 and width == 2 * observed:
        X = np.hstack([X, within_query_ranks(X)])
    elif width != observed:
        if width > observed:
            padded = np.zeros((len(rows), width), dtype=np.float64)
            padded[:, :observed] = X
            X = padded
        else:
            X = X[:, :width]
    return X


def rerank_with_ltr(qid: str, candidate_doc_ids: List[str], features: Dict[str, List[float]],
                    k: int) -> List[Tuple[str, float]]:
    if _MODEL is None:
        raise RuntimeError("call prepare(model_path) first")
    ids = _unique(candidate_doc_ids)
    if not ids:
        return []
    X = _feature_matrix(ids, features, _N_FEATURES)
    preds = _MODEL.predict(X, num_threads=1)
    order = np.argsort(-preds, kind="stable")
    return [(ids[i], float(preds[i])) for i in order[:k]]
