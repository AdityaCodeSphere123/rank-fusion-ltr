"""
harness/feature_names.py -- the named structure of MSLR-WEB10K's 136 features
and the small set of "base ranker" channels that fuse_candidates() receives
(assignment Section 6).

The index table below follows the official MSLR-WEB10K feature list (Microsoft
Research, "Introducing LETOR 4.0 Datasets" / the feature-definition table that
ships with the download): 25 feature groups x 5 document fields, then 11
single features.

All indices in this file are 1-BASED, exactly as they appear in the
SVMLight-format files ("1:0.5 2:3.0 ..."). Column c of a feature matrix
built by harness.feature_io is index c + 1.
"""
from typing import Dict, List

N_FEATURES_MSLR = 136

FIELDS = ["body", "anchor", "title", "url", "whole_document"]

# 25 groups x 5 fields = features 1..125, in this order.
GROUPS = [
    "covered_query_term_number", "covered_query_term_ratio", "stream_length", "idf",
    "sum_tf", "min_tf", "max_tf", "mean_tf", "variance_tf",
    "sum_normalized_tf", "min_normalized_tf", "max_normalized_tf", "mean_normalized_tf",
    "variance_normalized_tf",
    "sum_tfidf", "min_tfidf", "max_tfidf", "mean_tfidf", "variance_tfidf",
    "boolean_model", "vector_space_model", "bm25", "lmir_abs", "lmir_dir", "lmir_jm",
]

# Features 126..136, one each.
SINGLES = [
    "number_of_slash_in_url", "length_of_url", "inlink_number", "outlink_number",
    "pagerank", "siterank", "quality_score", "quality_score_2",
    "query_url_click_count", "url_click_count", "url_dwell_time",
]


def feature_index(group: str, field: str = "whole_document") -> int:
    """1-based MSLR feature index for (group, field), or for a single feature
    when `group` is one of SINGLES (field ignored)."""
    if group in SINGLES:
        return 126 + SINGLES.index(group)
    return 5 * GROUPS.index(group) + 1 + FIELDS.index(field)


def feature_name_list() -> List[str]:
    names = [f"{g}:{f}" for g in GROUPS for f in FIELDS] + list(SINGLES)
    assert len(names) == N_FEATURES_MSLR
    return names


# The channels fuse_candidates() receives, name -> 1-based feature index.
# Chosen because they are independently well-understood scoring functions
# that genuinely disagree with one another (different fields / model families).
BASE_RANKER_CHANNELS: Dict[str, int] = {
    "bm25_whole_document": feature_index("bm25", "whole_document"),        # 110
    "bm25_title": feature_index("bm25", "title"),                          # 108
    "lmir_dir_whole_document": feature_index("lmir_dir", "whole_document"),  # 120
    "lmir_jm_whole_document": feature_index("lmir_jm", "whole_document"),    # 125
}


def base_scores_from_row(row) -> Dict[str, float]:
    """Extract the named base-ranker scores from one feature row (a sequence
    of length >= max channel index). Higher is better for every channel."""
    return {name: float(row[idx - 1]) for name, idx in BASE_RANKER_CHANNELS.items()}
