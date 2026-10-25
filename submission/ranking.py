"""Competition entry.

fuse_candidates adds up the min-max scores of BM25 on the whole document,
BM25 on the title and LMIR.DIR. LMIR.JM is left out because every weight search
on TRAIN gave it a weight of 0.

rerank_with_ltr scores the candidates with the LightGBM model loaded by
prepare(). Our models see each raw feature plus its percentile and z-score
inside the query, so we rebuild those columns here before scoring.
"""
import os
from typing import Dict, List, Tuple

import numpy as np

from submission.fusion import combsum_scores

FUSION_CHANNELS = ("bm25_whole_document", "bm25_title", "lmir_dir_whole_document")

_MODEL = None
_N_FEATURES = None
_RAW_WIDTH = None


def prepare(model_path: str) -> None:
    global _MODEL, _N_FEATURES, _RAW_WIDTH
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"trained model artifact not found at {model_path!r}; run scripts/train_model.py and "
            f"include the artifact in your submission (grading never retrains)"
        )
    import lightgbm as lgb
    model = lgb.Booster(model_file=model_path)
    _MODEL, _N_FEATURES, _RAW_WIDTH = model, model.num_feature(), _named_raw_width(model.feature_name())


def _named_raw_width(names: List[str]):
    """Raw feature count of a recipe.py model, read from its column names. None for other models."""
    if not names or len(names) % 3:
        return None
    d = len(names) // 3
    expected = [f"{p}_{i}" for p in ("raw", "pct", "wqz") for i in range(d)]
    return d if list(names) == expected else None


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
    """Turn inf, nan and huge values above 1e30 into NaN. LightGBM treats NaN as missing."""
    X = np.asarray(X, dtype=np.float64)
    bad = ~np.isfinite(X) | (np.abs(X) > 1e30)
    if not bad.any():
        return X
    X = X.copy()
    X[bad] = np.nan
    return X


def within_query_ranks(X: np.ndarray) -> np.ndarray:
    """Percentile of each feature inside one query, from 0 to 1.

    Equal values share the same percentile so the order of documents does not
    matter. A constant feature or a one document query gives 0. Missing values
    count as the smallest value.
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


def within_query_z(X: np.ndarray) -> np.ndarray:
    """Z-score of each feature inside one query.

    It does not change if a feature is shifted or rescaled. A constant feature or
    a one document query gives 0 and missing values are ignored.
    """
    X = np.asarray(X, dtype=np.float64)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd >= 1e-8), sd, 1.0)
    out = (X - mu) / sd
    out[~np.isfinite(out)] = 0.0
    return out


def _feature_matrix(ids: List[str], features: Dict[str, List[float]]) -> np.ndarray:
    rows = [np.asarray(features[d], dtype=np.float64).ravel() for d in ids]
    d = _RAW_WIDTH
    if d is None:
        # The MSLR model has unnamed columns. It has three columns per input feature and
        # a plain model has one
        observed = max((r.shape[0] for r in rows), default=0)
        d = observed if _N_FEATURES == 3 * observed else _N_FEATURES
    # A sparse file may never mention its last features, so pad or cut every row to d
    X = np.zeros((len(rows), d), dtype=np.float64)
    for i, r in enumerate(rows):
        n = min(d, r.shape[0])
        X[i, :n] = r[:n]
    X = sanitize(X)
    if _N_FEATURES == 3 * d:
        X = np.hstack([X, within_query_ranks(X), within_query_z(X)])
    return X


def rerank_with_ltr(qid: str, candidate_doc_ids: List[str], features: Dict[str, List[float]],
                    k: int) -> List[Tuple[str, float]]:
    if _MODEL is None:
        raise RuntimeError("call prepare(model_path) first")
    ids = _unique(candidate_doc_ids)
    if not ids:
        return []
    X = _feature_matrix(ids, features)
    # every row is scored on its own so more threads only make big queries faster
    preds = _MODEL.predict(X, num_threads=4 if len(ids) > 200 else 1)
    order = np.argsort(-preds, kind="stable")
    return [(ids[i], float(preds[i])) for i in order[:k]]
