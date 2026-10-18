import numpy as np

from harness.feature_io import read_svmlight_ltr, stack, to_qrels, write_svmlight_ltr


def test_roundtrip_and_synth_ids(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("2 qid:7 1:0.5 3:1.5\n0 qid:7 2:2.0\n1 qid:9 1:1.0 136:4.0\n")
    ds = read_svmlight_ltr(str(p))
    assert list(ds) == ["7", "9"]
    assert ds["7"].doc_ids == ["7:0", "7:1"]
    assert ds["7"].features.shape == (2, 136)
    assert ds["9"].features[0, 135] == 4.0
    q = tmp_path / "u.txt"
    write_svmlight_ltr(str(q), ds)
    ds2 = read_svmlight_ltr(str(q))
    assert np.allclose(ds["9"].features, ds2["9"].features)
    X, y, qids = stack(ds)
    assert X.shape == (3, 136) and list(y) == [2, 0, 1] and list(qids) == ["7", "7", "9"]
    assert to_qrels(ds)["7"] == {"7:0": 2, "7:1": 0}
