"""
harness/run_harness.py -- runs your submission end-to-end on a labelled LTR
file (e.g. data/toy/vali.txt or the real dev split) and reports dev metrics,
exactly the way the grading harness does (minus the withheld held-out set).

Single in-process script (assignment Section 4.2): prepare(), fuse_candidates()
and rerank_with_ltr() are called directly. Per query, the candidate pool is
simply all rows of that query's block in the file.

Usage:
    python -m harness.run_harness \\
        --features data/toy/vali.txt \\
        --model submission/model.txt \\
        --report-out runs/dev_report.json
"""
import argparse
import inspect
import json
import os
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

from harness import feature_names, leaderboard
from harness.feature_io import read_svmlight_ltr, to_qrels
from harness.metrics import evaluate_run

EXPECTED_SIGNATURES = [
    ("prepare", ["model_path"]),
    ("fuse_candidates", ["qid", "candidate_doc_ids", "base_scores", "k"]),
    ("rerank_with_ltr", ["qid", "candidate_doc_ids", "features", "k"]),
]


def check_conformance(submission_module, model_path: Optional[str] = None) -> List[str]:
    """Human-readable problems (empty list = conformant). Checks signatures
    and, if model_path is given, that prepare() refuses a missing artifact
    instead of silently training or falling back to an untrained model (assignment Section 5)."""
    problems = []
    for name, expected in EXPECTED_SIGNATURES:
        fn = getattr(submission_module, name, None)
        if fn is None or not callable(fn):
            problems.append(f"submission.ranking.{name} is missing or not callable")
            continue
        params = list(inspect.signature(fn).parameters.keys())
        if params != expected:
            problems.append(f"submission.ranking.{name} has parameters {params}, expected {expected}")
    if problems:
        return problems
    missing = "/nonexistent/path/to/model.artifact"
    try:
        submission_module.prepare(missing)
    except Exception:  # noqa: BLE001 -- any exception is the correct behaviour
        pass
    else:
        problems.append(
            "prepare() accepted a nonexistent model path without raising -- it must fail loudly when "
            "the trained artifact is missing, never fall back to training or an untrained model"
        )
    return problems


def _validate_and_sort_results(
    results: List[Tuple[str, float]], qid: str, k: int, valid_doc_ids: Optional[Set[str]] = None
) -> List[Tuple[str, float]]:
    """Anti-gaming check on every ranking call: <= k results, no duplicate
    doc_ids, and (when given) only doc_ids from the candidate pool the
    function was handed. Returns the results re-sorted by score descending."""
    if len(results) > k:
        raise ValueError(f"qid={qid}: returned {len(results)} results, more than k={k}")
    doc_ids = [d for d, _s in results]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError(f"qid={qid}: returned a duplicate doc_id in {doc_ids!r}")
    if valid_doc_ids is not None:
        invalid = [d for d in doc_ids if d not in valid_doc_ids]
        if invalid:
            raise ValueError(
                f"qid={qid}: returned doc_id(s) {invalid!r} not present in the candidate pool it was "
                f"given -- a reranker may only return documents from its input."
            )
    return sorted(results, key=lambda pair: pair[1], reverse=True)


def rank_dataset(
    dataset, fuse_fn: Optional[Callable], ltr_fn: Optional[Callable], k: int = 10,
    channels: Optional[Dict[str, int]] = None,
) -> Tuple[dict, dict]:
    """Run fusion and/or LTR over every query of `dataset`; returns
    (fusion_run, ltr_run) as {qid: [(doc_id, score), ...]} (empty if the
    corresponding callable is None). channels=None uses MSLR's
    BASE_RANKER_CHANNELS; pass {} for datasets without pinned channels."""
    channels = feature_names.BASE_RANKER_CHANNELS if channels is None else channels
    fusion_run, ltr_run = {}, {}
    for qid, q in dataset.items():
        pool = list(q.doc_ids)
        valid = set(pool)
        if fuse_fn is not None:
            base_scores = {
                d: {name: float(row[idx - 1]) for name, idx in channels.items()}
                for d, row in zip(pool, q.features)
            }
            fusion_run[qid] = _validate_and_sort_results(fuse_fn(qid, pool, base_scores, k), qid, k, valid)
        if ltr_fn is not None:
            feats = {d: [float(v) for v in row] for d, row in zip(pool, q.features)}
            ltr_run[qid] = _validate_and_sort_results(ltr_fn(qid, pool, feats, k), qid, k, valid)
    return fusion_run, ltr_run


def run(features_path: str, model_path: str, report_out: str = None, n_features: Optional[int] = None) -> dict:
    import submission.ranking as submission_module

    problems = check_conformance(submission_module, model_path)
    if problems:
        raise SystemExit("Conformance problems:\n  " + "\n  ".join(problems))

    dataset = read_svmlight_ltr(features_path, n_features)
    qrels = to_qrels(dataset)

    t0 = time.perf_counter()
    submission_module.prepare(model_path)
    prepare_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    fusion_run, ltr_run = rank_dataset(dataset, submission_module.fuse_candidates, submission_module.rerank_with_ltr)
    rank_seconds = time.perf_counter() - t0

    summary = leaderboard.summarize(
        evaluate_run(fusion_run, qrels)["aggregate"], evaluate_run(ltr_run, qrels)["aggregate"]
    )
    summary["timing"] = {"prepare_seconds": prepare_seconds, "ranking_seconds": rank_seconds}
    summary["num_queries"] = len(dataset)

    if report_out:
        os.makedirs(os.path.dirname(report_out) or ".", exist_ok=True)
        with open(report_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True, help="labelled LTR file, e.g. data/toy/vali.txt")
    parser.add_argument("--model", default="submission/model.txt")
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--n-features", type=int, default=None)
    args = parser.parse_args()

    summary = run(args.features, args.model, args.report_out, args.n_features)
    print("== Local sanity-check only -- NOT your real leaderboard score ==")
    print(f"queries scored: {summary['num_queries']}")
    print(f"fuse_candidates()  nDCG@10: {summary['fusion_dev_ndcg@10']:.4f}   MAP@10: {summary['fusion_dev_map@10']:.4f}")
    print(f"rerank_with_ltr()  nDCG@10: {summary['ltr_dev_ndcg@10']:.4f}   MAP@10: {summary['ltr_dev_map@10']:.4f}")
    print()
    print(summary["note"])


if __name__ == "__main__":
    main()
