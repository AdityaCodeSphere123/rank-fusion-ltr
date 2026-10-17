"""
submission/ranking.py -- your competition entry (assignment Sections 4.2, 5).

    prepare(model_path)
    fuse_candidates(qid, candidate_doc_ids, base_scores, k)   -> Track A
    rerank_with_ltr(qid, candidate_doc_ids, features, k)      -> Tracks B and C

fuse_candidates: CombSUM (min-max per channel over the candidate pool, then summed) over three of the four
channels -- BM25 whole document, BM25 title, LMIR.DIR whole document -- with equal weights. LMIR.JM is left
out: on TRAIN, every grid of channel weights put it at 0, and dropping it beat plain 4-channel CombSUM, CombMNZ
and RRF for every c (scripts/fusion_eval.py, Decision.md). Selected on TRAIN only.

rerank_with_ltr: the LightGBM LambdaMART model loaded by prepare() (submission/model.txt, trained on TRAIN only
by scripts/train_model.py; for Track C1 whatever submission/recipe.py wrote). It accepts any feature width:
rows are zero-padded / truncated to the model's feature count, the SVMLight convention for absent indices.

The harness never trains anything: prepare() only LOADS the artifact, and must raise if it is missing (no silent
fallback to training or to an untrained model). Both ranking functions return at most k (doc_id, score) pairs,
best first, using ONLY doc_ids from `candidate_doc_ids`, with no duplicates; ties keep candidate order.
`base_scores`: {doc_id: {channel: score}} (the 4 named channels).
`features`:    {doc_id: [full feature vector]}.
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
    ranked = sorted(ids, key=lambda d: -fused[d])  # stable: ties keep candidate order
    return [(d, fused[d]) for d in ranked[:k]]


def _feature_matrix(ids: List[str], features: Dict[str, List[float]], width: int) -> np.ndarray:
    rows = [np.asarray(features[d], dtype=np.float64).ravel() for d in ids]
    X = np.zeros((len(rows), width), dtype=np.float64)
    for i, r in enumerate(rows):
        n = min(width, r.shape[0])
        X[i, :n] = r[:n]
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
