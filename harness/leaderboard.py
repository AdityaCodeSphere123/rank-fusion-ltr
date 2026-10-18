"""
harness/leaderboard.py -- combine raw metrics into the weighted leaderboard
score defined in the assignment (Section 7). Weights sum to 100%, no bonus:

    Track A  (fusion, MSLR dev)                      10%  (nDCG@10 80% / MAP@10 20%)
  + Track B  (LTR dev accuracy, MSLR)                10%
  + Track P  (pairwise model, MSLR held-out)         10%
  + Track C1 (transfer to hidden benchmarks)       35%
  + Track C2 (hidden robustness test)               35%

Every track is percentile-ranked against the whole class before weighting
(0 for the lowest team, 1 for the highest, ties share the average percentile),
so a single team running this locally cannot compute its real score; course
staff assemble the class leaderboard.

Tracks A and B (dev, where teams cluster) first round every metric to the nearest
TIE_BAND = 0.005, so scores within one band share the average percentile and
differences smaller than the evaluation noise are not rewarded. P, C1 and C2
use exact percentiles.

Track P: staff load the team's submission/pairwise_model.pt with their own copy
of harness/neural.py and score it on the MSLR held-out split; the nDCG@10 is
percentile-ranked (exact). An artifact that does not load scores 0. Dev scores
are shown on the live leaderboard for feedback only.

Track C1: staff run the team's submission/recipe.py once on each of two
undisclosed benchmarks and evaluate rerank_with_ltr() on each one's held-out
split. Each benchmark's nDCG@10 is percentile-ranked separately and the
percentiles are averaged. A recipe that fails or times out on a benchmark
scores 0 there (ranked last).

Track C2: a hidden robustness test. Staff run the team's submitted model (the
same rerank_with_ltr() as Track B, nothing retrained) on hidden versions of the
MSLR held-out data; the mean nDCG@10 over those versions is percentile-ranked.
"""
import math
from typing import Dict, List, Optional, Sequence

TRACK_A_NDCG_WEIGHT = 0.08
TRACK_A_MAP_WEIGHT = 0.02
TRACK_A_WEIGHT = TRACK_A_NDCG_WEIGHT + TRACK_A_MAP_WEIGHT  # 0.10
TRACK_B_WEIGHT = 0.10
TRACK_P_WEIGHT = 0.10
TRACK_C1_WEIGHT = 0.35
TRACK_C2_WEIGHT = 0.35
TIE_BAND = 0.005  # Tracks A and B: metrics are rounded to this before ranking


def _percentile_ranks(values: List[float]) -> List[float]:
    """Percentile rank in [0, 1] per value: 0 for the single lowest, 1 for the
    single highest, linear between; ties share the averaged-rank percentile; a
    class of one maps to 1.0. Identical to Assignments 1 and 2."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    order = sorted(range(n), key=lambda i: values[i])
    percentiles = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        pct = ((i + j) / 2.0) / (n - 1)
        for k in range(i, j + 1):
            percentiles[order[k]] = pct
        i = j + 1
    return percentiles


def band(x: float, width: float = TIE_BAND) -> float:
    """Round to the nearest `width` (half up), so near-ties land in the same band and share a percentile."""
    return math.floor(x / width + 0.5 + 1e-9) * width


def track_a_component(team_ndcg: Dict[str, float], team_map: Dict[str, float]) -> Dict[str, float]:
    teams = list(team_ndcg.keys())
    ndcg_pct = _percentile_ranks([band(team_ndcg[t]) for t in teams])
    map_pct = _percentile_ranks([band(team_map[t]) for t in teams])
    return {teams[i]: TRACK_A_NDCG_WEIGHT * ndcg_pct[i] + TRACK_A_MAP_WEIGHT * map_pct[i]
            for i in range(len(teams))}


def track_b_component(team_dev_ndcg: Dict[str, float]) -> Dict[str, float]:
    teams = list(team_dev_ndcg.keys())
    pct = _percentile_ranks([band(team_dev_ndcg[t]) for t in teams])
    return {teams[i]: TRACK_B_WEIGHT * pct[i] for i in range(len(teams))}


def generalization_ratio(dev_ndcg: float, heldout_ndcg: float) -> float:
    """held-out nDCG@10 / dev nDCG@10, clamped to [0, 1] (0.0 if dev <= 0). Not part of any track's score."""
    if dev_ndcg <= 0:
        return 0.0
    return max(0.0, min(1.0, heldout_ndcg / dev_ndcg))


def track_p_component(team_pairwise_heldout_ndcg: Dict[str, float]) -> Dict[str, float]:
    """{team: held-out nDCG@10 of pairwise_model.pt}; artifacts that fail to load must be passed as 0.0."""
    teams = list(team_pairwise_heldout_ndcg.keys())
    pct = _percentile_ranks([team_pairwise_heldout_ndcg[t] for t in teams])
    return {teams[i]: TRACK_P_WEIGHT * pct[i] for i in range(len(teams))}


def track_c1_component(team_scores: Dict[str, Sequence[float]]) -> Dict[str, float]:
    """team_scores: {team: [nDCG@10 on hidden benchmark 1, 2, ...]} (same order and length for every team).
    Failed or timed-out benchmarks must be passed as 0.0, never omitted. Each benchmark is percentile-ranked
    across teams; the team's score is TRACK_C1_WEIGHT x the mean percentile."""
    teams = list(team_scores.keys())
    if not teams:
        return {}
    n_sets = len(next(iter(team_scores.values())))
    mean_pct = [0.0] * len(teams)
    for k in range(n_sets):
        pct = _percentile_ranks([team_scores[t][k] for t in teams])
        mean_pct = [m + p / n_sets for m, p in zip(mean_pct, pct)]
    return {teams[i]: TRACK_C1_WEIGHT * mean_pct[i] for i in range(len(teams))}


def track_c2_component(team_projected_ndcg: Dict[str, float]) -> Dict[str, float]:
    """team_projected_ndcg: {team: mean nDCG@10 over the hidden held-out versions}; teams whose
    model could not be run must be passed as 0.0, never omitted."""
    teams = list(team_projected_ndcg.keys())
    pct = _percentile_ranks([team_projected_ndcg[t] for t in teams])
    return {teams[i]: TRACK_C2_WEIGHT * pct[i] for i in range(len(teams))}


def summarize(
    fusion_dev_metrics: Dict[str, float],
    ltr_dev_metrics: Dict[str, float],
    baseline_ltr_dev_ndcg: Optional[float] = None,
) -> Dict:
    """Local single-team sanity summary -- NOT the real leaderboard score."""
    result = {
        "fusion_dev_ndcg@10": fusion_dev_metrics["ndcg@10"],
        "fusion_dev_map@10": fusion_dev_metrics["map@10"],
        "ltr_dev_ndcg@10": ltr_dev_metrics["ndcg@10"],
        "ltr_dev_map@10": ltr_dev_metrics["map@10"],
        "note": (
            "These are RAW numbers. The real Tracks A, B, C1, C2 percentile-rank each "
            "against the whole class (weights A10/B10/P10/C1 35/C2 35). C1 needs the hidden benchmarks "
            "and C2 hidden held-out data, so neither can be computed locally: use "
            "scripts/run_recipe.py (C1 dry run on synthetic transfer sets) and scripts/cv_lambdamart.py "
            "(your local train/validation gap within TRAIN)."
        ),
    }
    if baseline_ltr_dev_ndcg is not None:
        result["beats_baseline_ltr_dev_ndcg@10"] = ltr_dev_metrics["ndcg@10"] > baseline_ltr_dev_ndcg
    return result
