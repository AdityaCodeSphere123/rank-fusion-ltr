import pytest

from harness import leaderboard as lb


def test_weights_sum_to_one():
    total = (lb.TRACK_A_NDCG_WEIGHT + lb.TRACK_A_MAP_WEIGHT + lb.TRACK_B_WEIGHT + lb.TRACK_P_WEIGHT
             + lb.TRACK_C1_WEIGHT + lb.TRACK_C2_WEIGHT)
    assert abs(total - 1.0) < 1e-12  # A 10 + B 10 + P 10 + C1 35 + C2 35, no bonus (assignment Section 7)


def test_percentile_ties_average():
    p = lb._percentile_ranks([1.0, 1.0, 3.0])
    assert p[0] == p[1] and p[2] == max(p)


def test_generalization_ratio_clamped():
    assert lb.generalization_ratio(0.5, 0.6) == 1.0
    assert abs(lb.generalization_ratio(0.8, 0.4) - 0.5) < 1e-12
    assert lb.generalization_ratio(0.0, 0.3) == 0.0


def test_generalization_ratio_clamped():
    assert lb.generalization_ratio(0.5, 0.6) == 1.0
    assert abs(lb.generalization_ratio(0.8, 0.4) - 0.5) < 1e-12
    assert lb.generalization_ratio(0.0, 0.3) == 0.0


def test_track_c2_is_percentile_of_projected_ndcg():
    c2 = lb.track_c2_component({"a": 0.50, "b": 0.45, "c": 0.0})
    assert c2["a"] == lb.TRACK_C2_WEIGHT and c2["c"] == 0.0 and 0 < c2["b"] < c2["a"]


def test_track_c1_averages_percentiles_over_hidden_benchmarks_and_failures_rank_last():
    c1 = lb.track_c1_component({"a": [0.60, 0.40], "b": [0.55, 0.45], "failed": [0.0, 0.0]})
    assert c1["failed"] == 0.0
    assert abs(c1["a"] - c1["b"]) < 1e-12 and abs(c1["a"] - 0.75 * lb.TRACK_C1_WEIGHT) < 1e-12  # each wins one set, is 2nd on the other
    one_set = lb.track_c1_component({"a": [0.6], "b": [0.5]})
    assert abs(one_set["a"] - lb.TRACK_C1_WEIGHT) < 1e-12 and one_set["b"] == 0.0


def test_tie_band_merges_near_ties_in_a_and_b_only():
    b = lb.track_b_component({"x": 0.4801, "y": 0.4823, "z": 0.50})
    assert b["x"] == b["y"] < b["z"]          # 0.4801 and 0.4823 are in the same 0.005 band
    assert lb.band(0.4801) == lb.band(0.4823) != lb.band(0.4860)
    a = lb.track_a_component({"x": 0.5000, "y": 0.5004}, {"x": 0.3, "y": 0.3})
    assert a["x"] == a["y"]
    c2 = lb.track_c2_component({"x": 0.4801, "y": 0.4823})
    assert c2["x"] < c2["y"]                  # C2 uses exact percentiles


def test_track_p_exact_percentile_and_failed_artifact_last():
    p = lb.track_p_component({"a": 0.4801, "b": 0.4823, "broken": 0.0})
    assert p["broken"] == 0.0 and p["a"] < p["b"] == lb.TRACK_P_WEIGHT
