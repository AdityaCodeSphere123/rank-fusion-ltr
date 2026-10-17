"""Compare fusion methods on TRAIN (selection) and DEV (reported only) -- assignment Section 4.1, Track A.

Methods: each single channel, RRF for a grid of c, CombSUM, CombMNZ, and channel-weighted RRF / CombSUM (weights
on a small grid). Everything is a vectorised re-implementation of submission/fusion.py (same stable tie-breaking
by candidate order) so the whole grid runs in seconds; the chosen method is then checked end to end through
submission.ranking.fuse_candidates with harness.run_harness.

Selection is done on TRAIN only. "cv" columns: for each of 5 query folds of TRAIN, pick the best method/params
on the other 4 folds and score it on the held-out fold -- an honest estimate of the selection procedure. DEV
numbers are printed for reporting only and are never used to choose.

    python scripts/fusion_eval.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import itertools
import json

import numpy as np

from harness.cv import grouped_kfold
from harness.feature_names import BASE_RANKER_CHANNELS
from scripts.data_cache import load_xyq

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CH = list(BASE_RANKER_CHANNELS)                    # bm25_whole, bm25_title, lmir_dir, lmir_jm
COLS = [BASE_RANKER_CHANNELS[c] - 1 for c in CH]
C_GRID = [1, 2, 5, 10, 20, 30, 45, 60, 80, 100, 150, 250, 500]
W_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]


def query_blocks(q):
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    return list(zip(starts, np.r_[starts[1:], len(q)]))


def per_query_metrics(S, y):
    """S: (n_docs, n_methods) fused scores of one query; returns (ndcg@10, map@10) arrays over methods.
    Ties are broken by candidate (row) order, like the stable sorts in fusion.py / ranking.py."""
    order = np.argsort(-S, axis=0, kind="stable")[:10]          # (<=10, M)
    lab = y[order]
    disc = 1.0 / np.log2(np.arange(2, lab.shape[0] + 2))[:, None]
    dcg = ((2.0 ** lab - 1) * disc).sum(0)
    ideal = np.sort(y)[::-1][:10]
    idcg = ((2.0 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2))).sum()
    nd = dcg / idcg if idcg > 0 else np.zeros(S.shape[1])
    rel = (lab > 0).astype(float)
    nrel = (y > 0).sum()
    prec = np.cumsum(rel, 0) / np.arange(1, lab.shape[0] + 1)[:, None]
    ap = (prec * rel).sum(0) / nrel if nrel > 0 else np.zeros(S.shape[1])
    return nd, ap


def method_table():
    """[(name, kind, params)] -- kind in rrf / comb."""
    m = [(f"single:{c}", "single", {"ch": i}) for i, c in enumerate(CH)]
    m += [(f"rrf c={c}", "rrf", {"c": c, "w": (1, 1, 1, 1)}) for c in C_GRID]
    m += [("combsum", "combsum", {"w": (1, 1, 1, 1)}), ("combmnz", "combmnz", {})]
    for w in itertools.product([1.0], W_GRID, W_GRID, W_GRID):
        if w == (1, 1, 1, 1):
            continue
        for c in (10, 20, 30, 60, 100):
            m.append((f"wrrf c={c} w={w}", "rrf", {"c": c, "w": w}))
        m.append((f"wcombsum w={w}", "combsum", {"w": w}))
        m.append((f"wzsum w={w}", "zsum", {"w": w}))
    m.append(("zsum", "zsum", {"w": (1, 1, 1, 1)}))
    return m


def fused_matrix(B, methods):
    """B: (n, 4) channel scores of one query -> (n, n_methods) fused scores."""
    n = B.shape[0]
    ranks = np.empty_like(B, dtype=np.float64)
    for j in range(B.shape[1]):
        o = np.argsort(-B[:, j], kind="stable")
        ranks[o, j] = np.arange(1, n + 1)
    lo, hi = B.min(0), B.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    norm = np.where(hi > lo, (B - lo) / span, 0.0)
    sd = B.std(0)
    z = np.where(sd > 0, (B - B.mean(0)) / np.where(sd > 0, sd, 1.0), 0.0)
    out = np.empty((n, len(methods)))
    for k, (_, kind, p) in enumerate(methods):
        if kind == "single":
            out[:, k] = B[:, p["ch"]]
        elif kind == "rrf":
            out[:, k] = (np.asarray(p["w"]) / (p["c"] + ranks)).sum(1)
        elif kind == "combsum":
            out[:, k] = norm @ np.asarray(p["w"])
        elif kind == "zsum":
            out[:, k] = z @ np.asarray(p["w"])
        else:
            out[:, k] = norm.sum(1) * (norm > 0).sum(1)
    return out


def evaluate(path, methods):
    X, y, q = load_xyq(path)
    B = X[:, COLS].astype(np.float64)
    blocks = query_blocks(q)
    nd = np.zeros((len(blocks), len(methods)))
    ap = np.zeros_like(nd)
    for i, (a, b) in enumerate(blocks):
        nd[i], ap[i] = per_query_metrics(fused_matrix(B[a:b], methods), y[a:b])
    return nd, ap, np.array([q[a] for a, _ in blocks])


def main():
    methods = method_table()
    names = [m[0] for m in methods]
    tr_nd, tr_ap, tr_q = evaluate(os.path.join(ROOT, "data", "train", "train.txt"), methods)
    dv_nd, dv_ap, _ = evaluate(os.path.join(ROOT, "data", "dev", "vali.txt"), methods)

    # 5-fold selection CV on TRAIN: choose by nDCG@10 on 4 folds, score the 5th
    folds = grouped_kfold(tr_q, k=5, seed=0)
    picks, cv_nd, cv_ap = [], [], []
    for tr_idx, va_idx in folds:
        best = int(np.argmax(tr_nd[tr_idx].mean(0)))
        picks.append(names[best])
        cv_nd.append(tr_nd[va_idx, best].mean())
        cv_ap.append(tr_ap[va_idx, best].mean())

    # restricted selection CV: only the unweighted methods (RRF c-grid, CombSUM, CombMNZ)
    plain = [i for i, n in enumerate(names) if not n.startswith(("wrrf", "wcombsum", "wzsum", "single", "zsum"))]
    cv_plain = []
    for tr_idx, va_idx in folds:
        best = plain[int(np.argmax(tr_nd[tr_idx][:, plain].mean(0)))]
        cv_plain.append((names[best], tr_nd[va_idx, best].mean(), tr_ap[va_idx, best].mean()))

    show = [i for i, n in enumerate(names) if not n.startswith(("wrrf", "wcombsum", "wzsum"))]
    order_w = np.argsort(-tr_nd.mean(0))
    print(f"{'method':<52} {'TRAIN nDCG':>10} {'TRAIN MAP':>9} {'DEV nDCG':>9} {'DEV MAP':>8}")
    for i in show + [i for i in order_w if i not in show][:15]:
        print(f"{names[i]:<52} {tr_nd[:, i].mean():>10.4f} {tr_ap[:, i].mean():>9.4f} "
              f"{dv_nd[:, i].mean():>9.4f} {dv_ap[:, i].mean():>8.4f}")
    print("\nselection CV (all methods, by TRAIN nDCG@10):", picks)
    print(f"  held-out-fold nDCG@10 {np.mean(cv_nd):.4f} +- {np.std(cv_nd):.4f}   MAP@10 {np.mean(cv_ap):.4f}")
    print("selection CV (unweighted only):", [p[0] for p in cv_plain])
    print(f"  held-out-fold nDCG@10 {np.mean([p[1] for p in cv_plain]):.4f}   MAP@10 {np.mean([p[2] for p in cv_plain]):.4f}")
    best = int(order_w[0])
    print(f"\nbest on full TRAIN: {names[best]}  TRAIN {tr_nd[:, best].mean():.4f}  DEV {dv_nd[:, best].mean():.4f} "
          f"/ MAP {dv_ap[:, best].mean():.4f}")
    os.makedirs(os.path.join(ROOT, "experiments"), exist_ok=True)
    with open(os.path.join(ROOT, "experiments", "fusion_results.json"), "w", encoding="utf-8") as f:
        json.dump({n: {"train_ndcg": tr_nd[:, i].mean(), "train_map": tr_ap[:, i].mean(),
                       "dev_ndcg": dv_nd[:, i].mean(), "dev_map": dv_ap[:, i].mean()}
                   for i, n in enumerate(names)}, f, indent=1)


if __name__ == "__main__":
    main()
