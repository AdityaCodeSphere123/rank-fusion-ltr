"""How much a rank-feature blend moves the shipped model under corruption.

DEV was inside the shipped fit, so clean DEV is an in-training number. The
corruptions are still informative: they break raw-value splits. A blend is
kept only when the clean drop stays inside the Track B rounding band (0.0025)
and the mean over corruptions rises. DEV labels are not a hidden-test peek.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from harness.metrics import mean_ndcg_from_scores
from scripts.data_cache import load_xyq
from scripts.train_model import _expand

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# 0-based columns of BM25 title, BM25 whole document, LMIR.DIR whole document.
_CHANNELS = (107, 109, 119)


def _variants(X):
    rng = np.random.RandomState(0)
    X = np.asarray(X, dtype=np.float32)
    std = X.std(0).astype(np.float32)
    out = {
        "clean": X,
        "log1p": np.sign(X) * np.log1p(np.abs(X)),
        "zscore": (X - X.mean(0)) / np.clip(std, 1e-6, None),
        "x10": X * 10,
        "add0.05": X + 0.05 * std * rng.randn(*X.shape).astype(np.float32),
        "add0.10": X + 0.10 * std * rng.randn(*X.shape).astype(np.float32),
    }
    cell = np.array(X, copy=True)
    cell[rng.rand(*X.shape) < 0.10] = 0
    out["cell10"] = cell
    perm = np.array(X, copy=True)
    for g in range(25):
        perm[:, 5 * g:5 * g + 5] = X[:, 5 * g + rng.permutation(5)]
    out["permute"] = perm
    starts = None
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}, starts


def _query_scale(X, q):
    rng = np.random.RandomState(1)
    out = np.empty_like(X, dtype=np.float32)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    scales = np.exp(rng.normal(0.0, 0.5, size=(len(starts), X.shape[1]))).astype(np.float32)
    X = np.asarray(X, dtype=np.float32)
    for i, (a, b) in enumerate(zip(starts, ends)):
        out[a:b] = X[a:b] * scales[i]
    return out


def _pct_rank(scores, q):
    """Within-query average-tie percentile of a 1-d score, in [0, 1]."""
    out = np.zeros(len(scores), dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    for a, b in zip(starts, ends):
        n = b - a
        if n <= 1:
            continue
        order = np.argsort(scores[a:b], kind="mergesort")
        # average ties
        vals = scores[a:b][order]
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i + 1
            while j < n and vals[j] == vals[i]:
                j += 1
            ranks[i:j] = 0.5 * (i + j - 1) / (n - 1)
            i = j
        out[a + order] = ranks
    return out


def main():
    import lightgbm as lgb

    X, y, q = load_xyq(os.path.join(ROOT, "data", "dev", "vali.txt"))
    q = np.asarray(q)
    # 600 queries is enough to rank blend weights; full DEV is 2,000.
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    keep = np.concatenate([np.arange(starts[i], ends[i]) for i in range(min(600, len(starts)))])
    X, y, q = X[keep], y[keep], q[keep]
    # Contiguous query codes so _expand's change-point logic holds.
    code = np.cumsum(np.r_[True, q[1:] != q[:-1]]).astype(np.int32) - 1
    model = lgb.Booster(model_file=os.path.join(ROOT, "submission", "model.txt"))
    variants, _ = _variants(X)
    variants["qscale"] = _query_scale(X, q)

    packed = {}
    for name, raw in variants.items():
        print(f"featurize {name}", flush=True)
        packed[name] = _expand(raw, code, False)

    d = packed["clean"].shape[1] // 2
    channel_pct = packed["clean"][:, [d + c for c in _CHANNELS]]
    # Channel percentiles are invariant to a per-feature monotone map, so they
    # are taken from the clean matrix only when the corruption is monotone.
    # For noise and permutation they must be recomputed, which packed already did.
    weights = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
    print(f"rows {len(y)} features {packed['clean'].shape[1]}", flush=True)
    header = f"{'variant':<10}" + "".join(f"{w:>8.2f}" for w in weights)
    print(header)
    means = {w: [] for w in weights}
    for name, block in packed.items():
        pred = model.predict(block, num_threads=4)
        pred_pct = _pct_rank(pred, q)
        robust = block[:, [d + c for c in _CHANNELS]].mean(axis=1)
        # robust is already in [0, 1]
        row = f"{name:<10}"
        for w in weights:
            blended = (1.0 - w) * pred_pct + w * robust
            nd = mean_ndcg_from_scores(blended, y, q, 10)
            row += f"{nd:8.4f}"
            if name != "clean":
                means[w].append(nd)
        print(row, flush=True)
    print(f"{'MEAN':<10}" + "".join(f"{np.mean(means[w]):8.4f}" for w in weights))
    # Also the unblended raw model score, for reference.
    print("raw model (no blend)")
    raw_vals = []
    for name, block in packed.items():
        nd = mean_ndcg_from_scores(model.predict(block, num_threads=4), y, q, 10)
        raw_vals.append((name, nd))
        print(f"  {name:<10} {nd:.4f}")
    _ = channel_pct


if __name__ == "__main__":
    main()
