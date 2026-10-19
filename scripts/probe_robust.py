"""Train-fold probe for Track C2. DEV is never read.

Fits on 1,500 TRAIN queries and scores a disjoint 400-query slice under
clean data and under corruptions. The selection number is the mean of the
corruptions. Nothing here is the course held-out split.
"""
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from harness.cv import grouped_kfold
from harness.metrics import mean_ndcg_from_scores
from scripts.data_cache import load_xyq
from scripts.lambdamart_utils import default_params, fit
from scripts.train_model import _expand, _run_codes

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# (name, feature mode, training views, extra LightGBM params)
# Views are applied to the raw matrix before within-query features, so a
# percentile column still describes the view the tree actually sees.
CONFIGS = [
    ("both", "both", ("clean",), {}),
    ("both_trunc10", "both", ("clean",), {"lambdarank_truncation_level": 10}),
    ("both_views", "both", ("clean", "log", "z", "qscale"), {}),
    ("both_perm", "both", ("clean", "permute"), {}),
    ("both_cell", "both", ("clean", "cell"), {}),
    ("both_noise", "both", ("clean", "noise"), {}),
    ("both_views_perm", "both", ("clean", "log", "z", "permute"), {}),
    ("qz", "qz", ("clean",), {}),
    ("aggs", "aggs", ("clean",), {}),
    ("pct", "pct", ("clean",), {}),
    ("aggs_views", "aggs", ("clean", "log", "permute"), {}),
]


def _subsample(X, y, q, idx, n_queries, seed):
    q = np.asarray(q)
    sub_q = q[idx]
    starts_all = np.flatnonzero(np.r_[True, sub_q[1:] != sub_q[:-1]])
    ends_all = np.r_[starts_all[1:], len(sub_q)]
    order = np.random.RandomState(seed).permutation(len(starts_all))[:n_queries]
    order.sort()
    take = np.concatenate([idx[starts_all[i]:ends_all[i]] for i in order])
    return X[take], y[take], np.asarray(q[take])


def _qz(X, q):
    out = np.empty(np.shape(X), dtype=np.float32)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    for a, b in zip(starts, ends):
        block = np.asarray(X[a:b], dtype=np.float64)
        sd = block.std(axis=0)
        sd[sd < 1e-8] = 1.0
        out[a:b] = ((block - block.mean(axis=0)) / sd).astype(np.float32)
    return out


def _featurize(X, q, mode):
    codes = _run_codes(np.asarray(q))
    if mode == "pct":
        both = _expand(X, codes, False)
        return both[:, X.shape[1]:]
    if mode == "both":
        return _expand(X, codes, False)
    if mode == "qz":
        return np.hstack([_expand(X, codes, False), _qz(X, codes)])
    if mode == "aggs":
        return _expand(X, codes, True)
    raise ValueError(mode)


def _view(X, q, kind, seed):
    X = np.asarray(X, dtype=np.float32)
    if kind == "clean":
        return X
    if kind == "log":
        return (np.sign(X) * np.log1p(np.abs(X))).astype(np.float32)
    if kind == "z":
        sd = np.clip(X.std(0), 1e-6, None).astype(np.float32)
        return ((X - X.mean(0)) / sd).astype(np.float32)
    rng = np.random.RandomState(seed)
    if kind == "noise":
        return (X + 0.1 * X.std(0).astype(np.float32) * rng.randn(*X.shape).astype(np.float32)).astype(np.float32)
    if kind == "cell":
        out = np.array(X, copy=True)
        out[rng.rand(*X.shape) < 0.10] = 0
        return out
    if kind == "permute":
        out = np.array(X, copy=True)
        for g in range(25):
            out[:, 5 * g:5 * g + 5] = X[:, 5 * g + rng.permutation(5)]
        return out
    if kind == "qscale":
        out = np.empty_like(X)
        starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
        ends = np.r_[starts[1:], len(q)]
        scales = np.exp(rng.normal(0.0, 1.0, size=(len(starts), X.shape[1]))).astype(np.float32)
        for i, (a, b) in enumerate(zip(starts, ends)):
            out[a:b] = X[a:b] * scales[i]
        return out
    raise ValueError(kind)


def _corruptions(X, q):
    rng = np.random.RandomState(0)
    X = np.asarray(X, dtype=np.float32)
    std = X.std(0).astype(np.float32)
    return {
        "log1p": _view(X, q, "log", 0),
        "zscore": _view(X, q, "z", 0),
        "x10": np.asarray(X * 10, dtype=np.float32),
        "add0.05": (X + 0.05 * std * rng.randn(*X.shape).astype(np.float32)).astype(np.float32),
        "add0.10": (X + 0.10 * std * rng.randn(*X.shape).astype(np.float32)).astype(np.float32),
        "cell10": _view(X, q, "cell", 2),
        "permute": _view(X, q, "permute", 3),
        "qscale": _view(X, q, "qscale", 4),
    }


def main():
    X, y, q = load_xyq(os.path.join(ROOT, "data", "train", "train.txt"))
    tr_idx, va_idx = grouped_kfold(q, k=5, seed=0)[0]
    Xtr, ytr, qtr = _subsample(X, y, q, tr_idx, 1500, seed=1)
    Xva, yva, qva = _subsample(X, y, q, va_idx, 400, seed=2)
    del X
    gc.collect()
    print(f"train {len(ytr)} rows, val {len(yva)} rows", flush=True)
    cor = _corruptions(Xva, qva)
    # Featurize each val matrix once per mode.
    modes = sorted({m for _, m, _, _ in CONFIGS})
    val_feat = {m: {"clean": _featurize(Xva, qva, m)} for m in modes}
    for m in modes:
        for name, raw in cor.items():
            val_feat[m][name] = _featurize(raw, qva, m)
        print(f"val features {m} width {val_feat[m]['clean'].shape[1]}", flush=True)
    del cor
    gc.collect()

    names = ["clean", "log1p", "zscore", "x10", "add0.05", "add0.10", "cell10", "permute", "qscale"]
    print(f"{'config':<18}" + "".join(f"{n:>9}" for n in names) + f"{'mean':>9}", flush=True)
    for name, mode, views, extra in CONFIGS:
        blocks_x, blocks_y, blocks_c = [], [], []
        offset = 0
        for v_i, view in enumerate(views):
            raw = _view(Xtr, qtr, view, 10 + v_i)
            feat = _featurize(raw, qtr, mode)
            codes = _run_codes(qtr) + offset
            offset = int(codes[-1]) + 1
            blocks_x.append(feat)
            blocks_y.append(ytr)
            blocks_c.append(codes)
            del raw, feat
        big_x = np.vstack(blocks_x)
        big_y = np.concatenate(blocks_y)
        big_c = np.concatenate(blocks_c)
        del blocks_x
        params = default_params(63, 0.05, 80, lambda_l2=1.0, seed=0, feature_fraction=0.6,
                                bagging_fraction=0.8, bagging_freq=1, num_threads=4)
        params.update(extra)
        model = fit(big_x, big_y, big_c, params, 250)
        del big_x, big_y, big_c
        gc.collect()
        scores = []
        row = f"{name:<18}"
        for key in names:
            nd = mean_ndcg_from_scores(model.predict(val_feat[mode][key], num_threads=4), yva, qva, 10)
            scores.append(nd)
            row += f"{nd:9.4f}"
        mean = float(np.mean(scores[1:]))
        print(row + f"{mean:9.4f}", flush=True)
        del model
        gc.collect()


if __name__ == "__main__":
    main()
