"""Competition entry.

fuse_candidates: unweighted CombSUM of BM25 (document and title) and LMIR.DIR.
LMIR.JM is dropped; on TRAIN every channel-weight grid set it to 0.

rerank_with_ltr: the LightGBM model loaded by prepare(). Rows are padded or
truncated to the model's width. A model with twice the candidate width was
trained on raw features plus within-query ranks (within_query_ranks); every
other width, including recipe.py models, is scored on the raw vectors.

prepare() only loads the artifact and raises if it is missing. Both rankers
return at most k (doc_id, score) pairs, best first, ids drawn only from the
candidate list, no duplicates. Ties keep candidate order.
"""
import os
from typing import Dict, List, Tuple

import numpy as np

from submission.fusion import combsum_scores

FUSION_CHANNELS = ("bm25_whole_document", "bm25_title", "lmir_dir_whole_document")

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


def within_query_ranks(X: np.ndarray) -> np.ndarray:
    """Stable within-query rank of each feature, scaled to [0, 1]. One document is 0.5.

    A per-feature monotone map leaves this unchanged, so splits on it survive a
    hidden version that only rescales features.
    """
    n, d = X.shape
    if n <= 1:
        return np.full((n, d), 0.5, dtype=np.float64)
    order = np.argsort(X, axis=0, kind="mergesort")
    ranks = np.empty((n, d), dtype=np.float64)
    ranks[order, np.arange(d)] = np.arange(n, dtype=np.float64)[:, None]
    return ranks / (n - 1)


def _feature_matrix(ids: List[str], features: Dict[str, List[float]], width: int) -> np.ndarray:
    rows = [np.asarray(features[d], dtype=np.float64).ravel() for d in ids]
    observed = max((r.shape[0] for r in rows), default=0)
    X = np.zeros((len(rows), observed), dtype=np.float64)
    for i, r in enumerate(rows):
        n = min(observed, r.shape[0])
        X[i, :n] = r[:n]
    # 2x width: raw features concatenated with within-query ranks (the MSLR model).
    if observed > 0 and width == 2 * observed:
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
