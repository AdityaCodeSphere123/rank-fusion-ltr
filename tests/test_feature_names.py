from harness import feature_names as fn


def test_layout():
    assert len(fn.feature_name_list()) == 136
    assert fn.feature_index("covered_query_term_number", "body") == 1
    assert fn.feature_index("bm25", "whole_document") == 110
    assert fn.feature_index("lmir_jm", "whole_document") == 125
    assert fn.feature_index("pagerank") == 130
    assert fn.feature_index("url_dwell_time") == 136
    assert set(fn.BASE_RANKER_CHANNELS.values()) == {110, 108, 120, 125}
