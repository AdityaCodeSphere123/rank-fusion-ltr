import numpy as np

from harness.cv import grouped_kfold


def test_queries_never_split_and_folds_partition():
    qids = np.repeat(np.arange(23), 4)
    folds = grouped_kfold(qids, k=5, seed=0)
    seen = []
    for tr, va in folds:
        assert not set(qids[tr]) & set(qids[va])
        assert len(tr) + len(va) == len(qids)
        seen.extend(va.tolist())
    assert sorted(seen) == list(range(len(qids)))
    sizes = [len(set(qids[va])) for _, va in folds]
    assert max(sizes) - min(sizes) <= 1


def test_deterministic_and_seeded():
    qids = np.repeat(np.arange(20), 3)
    a, b, c = grouped_kfold(qids, 5, 0), grouped_kfold(qids, 5, 0), grouped_kfold(qids, 5, 1)
    assert all((x[1] == y[1]).all() for x, y in zip(a, b))
    assert any((x[1] != y[1]).any() for x, y in zip(a, c))
