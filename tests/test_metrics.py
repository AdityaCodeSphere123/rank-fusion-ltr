import math

import numpy as np

from harness.metrics import average_precision, mean_ndcg_from_scores, ndcg_at_k


def test_ndcg_perfect_and_worst():
    qrels = {"a": 3, "b": 1, "c": 0}
    assert ndcg_at_k(["a", "b", "c"], qrels, 10) == 1.0
    assert ndcg_at_k(["c"], qrels, 10) == 0.0


def test_ndcg_graded_gain_formula():
    # gains (2^l - 1): ideal = 7/1 + 1/log2(3); ranking b,a => 1/1 + 7/log2(3)
    qrels = {"a": 3, "b": 1}
    ideal = 7 + 1 / math.log2(3)
    got = 1 + 7 / math.log2(3)
    assert abs(ndcg_at_k(["b", "a"], qrels, 10) - got / ideal) < 1e-9


def test_map_binarized_at_positive_label():
    qrels = {"a": 2, "b": 0, "c": 1}
    # relevant = {a, c}; ranking a,b,c -> (1/1 + 2/3)/2
    assert abs(average_precision(["a", "b", "c"], qrels) - (1 + 2 / 3) / 2) < 1e-9


def test_mean_ndcg_from_scores_matches_manual():
    scores = np.array([3.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    labels = np.array([2, 0, 1, 0, 1, 0])
    qids = np.array(["a", "a", "a", "b", "b", "b"])
    v = mean_ndcg_from_scores(scores, labels, qids, k=10)
    assert 0.0 <= v <= 1.0
    assert abs(mean_ndcg_from_scores(labels.astype(float), labels, qids, k=10) - 1.0) < 1e-12
