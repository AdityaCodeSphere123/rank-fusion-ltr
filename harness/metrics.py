"""
harness/metrics.py -- the exact metrics your leaderboard score is computed
from: nDCG@10 (primary) and MAP@10, plus MRR and P@k for your own analysis.
Shared, readable and unit-tested (tests/test_metrics.py) so there is no
ambiguity about how scoring works; the same code path scores the public dev
set and, at grading time, the withheld held-out set.

Conventions (same as Assignments 1 and 2):
  - nDCG uses graded relevance (MSLR labels 0-4) and the (2^rel - 1) gain.
  - MAP, MRR and P@k binarise relevance: a document is relevant if its label
    is > 0.
  - "MAP@10" is Average Precision over the top 10 returned, normalised by the
    TRUE number of relevant documents in the query's candidate pool (not just
    those found), so a query with more than 10 relevant documents can never
    reach AP = 1.0. This matches trec_eval on a truncated run.
  - A ranking must never contain a doc_id twice. evaluate_run() keeps only
    each doc_id's first occurrence before scoring; run_harness.py separately
    rejects duplicate-returning submissions outright. Without this the
    unbounded-above metrics could be inflated by repeating a relevant doc.
  - Documents of a query with no judgment entry count as non-relevant.
"""
import math
from typing import Dict, List, Tuple

RunType = Dict[str, List[Tuple[str, float]]]  # qid -> [(doc_id, score), ...], best first
QrelsType = Dict[str, Dict[str, int]]  # qid -> {doc_id: relevance}


def _relevance(qrels_for_q: Dict[str, int], doc_id: str) -> int:
    return qrels_for_q.get(doc_id, 0)


def _dedup_keep_first(doc_ids: List[str]) -> List[str]:
    """Keep each doc_id's first (best-ranked) occurrence only, dropping
    any later repeats. See the module docstring's "note on duplicate
    doc_ids" for why this exists — every metric below sums a per-rank
    contribution with no upper cap, so letting the same relevant document
    count more than once lets nDCG@10/MAP@10 blow past 1.0."""
    seen = set()
    out = []
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def dcg_at_k(gains: List[int], k: int) -> float:
    """gains: relevance values in ranked order (already truncated or not;
    only the first k are used). DCG = sum_i (2^rel_i - 1) / log2(i + 1),
    1-indexed."""
    dcg = 0.0
    for i, rel in enumerate(gains[:k], start=1):
        dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_doc_ids: List[str], qrels_for_q: Dict[str, int], k: int = 10) -> float:
    if not qrels_for_q:
        return 0.0
    gains = [_relevance(qrels_for_q, doc_id) for doc_id in ranked_doc_ids[:k]]
    dcg = dcg_at_k(gains, k)
    ideal_gains = sorted(qrels_for_q.values(), reverse=True)
    idcg = dcg_at_k(ideal_gains, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision(ranked_doc_ids: List[str], qrels_for_q: Dict[str, int]) -> float:
    """Average Precision over `ranked_doc_ids`, exactly as given — this
    function does NOT truncate internally. The caller (evaluate_run,
    below) passes an already rank-10-truncated list to produce "MAP@10",
    matching nDCG@10's cutoff; pass a longer or shorter list yourself if
    you want AP at a different depth. Normalises by the *true* total
    relevant-document count from qrels_for_q, not by however many of
    those were found in `ranked_doc_ids` — so AP is capped below 1.0 for
    any query with more relevant documents than the list you passed in.
    """
    n_relevant = sum(1 for rel in qrels_for_q.values() if rel > 0)
    if n_relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, doc_id in enumerate(ranked_doc_ids, start=1):
        if _relevance(qrels_for_q, doc_id) > 0:
            hits += 1
            precision_sum += hits / i
    return precision_sum / n_relevant


def reciprocal_rank(ranked_doc_ids: List[str], qrels_for_q: Dict[str, int]) -> float:
    for i, doc_id in enumerate(ranked_doc_ids, start=1):
        if _relevance(qrels_for_q, doc_id) > 0:
            return 1.0 / i
    return 0.0


def precision_at_k(ranked_doc_ids: List[str], qrels_for_q: Dict[str, int], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = ranked_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if _relevance(qrels_for_q, doc_id) > 0)
    return hits / k


def evaluate_run(run: RunType, qrels: QrelsType, k: int = 10) -> Dict:
    """Evaluate `run` against `qrels`. Returns a dict with per-query
    metrics and corpus-level (macro) averages. Queries present in `run`
    but absent from `qrels` are skipped with a warning key so they don't
    silently zero out your average; queries present in `qrels` but
    missing from `run` are scored as 0 across all metrics (you didn't
    answer them)."""
    per_query = {}
    skipped_no_qrels = []

    all_qids = set(qrels.keys()) | set(run.keys())
    for qid in sorted(all_qids):
        if qid not in qrels:
            skipped_no_qrels.append(qid)
            continue
        qrels_for_q = qrels[qid]
        ranked_doc_ids = [doc_id for doc_id, _score in run.get(qid, [])]
        ranked_doc_ids = _dedup_keep_first(ranked_doc_ids)
        per_query[qid] = {
            "ndcg@10": ndcg_at_k(ranked_doc_ids, qrels_for_q, k=10),
            # Fixed at 10 (not the `k` argument to this function) to match
            # nDCG@10's cutoff and keep "MAP@10" an honest label regardless
            # of what --k the harness happened to be invoked with — see
            # average_precision()'s docstring.
            "map@10": average_precision(ranked_doc_ids[:10], qrels_for_q),
            "mrr": reciprocal_rank(ranked_doc_ids, qrels_for_q),
            f"p@{k}": precision_at_k(ranked_doc_ids, qrels_for_q, k),
            "num_ranked": len(ranked_doc_ids),
        }

    n = len(per_query)
    if n == 0:
        aggregate = {"ndcg@10": 0.0, "map@10": 0.0, "mrr": 0.0, f"p@{k}": 0.0}
    else:
        aggregate = {
            "ndcg@10": sum(m["ndcg@10"] for m in per_query.values()) / n,
            "map@10": sum(m["map@10"] for m in per_query.values()) / n,
            "mrr": sum(m["mrr"] for m in per_query.values()) / n,
            f"p@{k}": sum(m[f"p@{k}"] for m in per_query.values()) / n,
        }

    return {
        "per_query": per_query,
        "aggregate": aggregate,
        "num_queries_scored": n,
        "skipped_no_qrels": skipped_no_qrels,
    }


def mean_ndcg_from_scores(scores, labels, qids, k: int = 10) -> float:
    """Mean nDCG@k over queries, computed straight from parallel per-ROW
    arrays (model scores, graded labels, query ids) -- the convenient form for
    training-time evaluation and cross-validation. Same gain/discount as
    ndcg_at_k(); ties in `scores` keep row order (stable sort); queries whose
    labels are all zero contribute 0, exactly like ndcg_at_k(). Rows of one
    query need not be contiguous."""
    import numpy as np

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    by_query: Dict[str, List[int]] = {}
    for i, q in enumerate(qids):
        by_query.setdefault(q, []).append(i)
    total = 0.0
    for rows in by_query.values():
        rows = np.asarray(rows)
        order = rows[np.argsort(-scores[rows], kind="stable")]
        dcg = dcg_at_k([int(labels[i]) for i in order], k)
        idcg = dcg_at_k(sorted((int(labels[i]) for i in rows), reverse=True), k)
        total += dcg / idcg if idcg > 0 else 0.0
    return total / len(by_query) if by_query else 0.0
