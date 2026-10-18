"""CI gate: the ranking interface works end to end on toy data. (Does NOT test your TODOs.)"""
import os

import pytest

import submission.ranking as ranking
from harness import run_harness
from harness.feature_io import read_svmlight_ltr

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL = os.path.join(ROOT, "submission", "model.txt")
VALI = os.path.join(ROOT, "data", "toy", "vali.txt")


def test_signatures_and_missing_artifact_fails_loudly():
    assert run_harness.check_conformance(ranking, MODEL) == []


def test_end_to_end_toy():
    summary = run_harness.run(VALI, MODEL)
    assert summary["num_queries"] > 0
    assert 0.0 <= summary["fusion_dev_ndcg@10"] <= 1.0
    assert summary["ltr_dev_ndcg@10"] > 0.5


def test_results_obey_contract():
    ds = read_svmlight_ltr(VALI)
    ranking.prepare(MODEL)
    fusion_run, ltr_run = run_harness.rank_dataset(ds, ranking.fuse_candidates, ranking.rerank_with_ltr, k=10)
    for run in (fusion_run, ltr_run):
        for qid, res in run.items():
            ids = [d for d, _ in res]
            assert len(ids) <= 10 and len(ids) == len(set(ids))
            assert set(ids) <= set(ds[qid].doc_ids)


def test_validator_rejects_foreign_and_duplicate_ids():
    with pytest.raises(ValueError):
        run_harness._validate_and_sort_results([("x", 1.0)], "q", 10, {"a"})
    with pytest.raises(ValueError):
        run_harness._validate_and_sort_results([("a", 1.0), ("a", 0.5)], "q", 10, {"a"})
    with pytest.raises(ValueError):
        run_harness._validate_and_sort_results([("a", 1.0), ("b", 0.5)], "q", 1, {"a", "b"})


def test_prepare_raises_without_artifact():
    with pytest.raises(Exception):
        ranking.prepare("/nonexistent/model.txt")
