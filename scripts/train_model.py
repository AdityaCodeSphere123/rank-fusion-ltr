"""Train the LambdaMART model that submission/ranking.py loads. Grading never retrains it.

    python scripts/train_model.py --train data/train --out submission/model.txt --trees 300

--train takes a file or a folder with train.txt in it. --use-dev also trains on the dev split.
The extra flags add changed copies of the training rows (noise, missing cells, shuffled fields
and so on) so the model holds up on the hidden C2 versions. The exact commands for the shipped
model are in docs/SUBMISSION_NOTES.md.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import time

import numpy as np

from scripts.data_cache import load_xyq, resolve
from scripts.lambdamart_utils import default_params, fit, ndcg10
from submission.ranking import within_query_ranks, within_query_z


def _query_spans(q):
    """(start, end) row ranges of each contiguous query."""
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    return list(zip(starts, np.r_[starts[1:], len(q)]))


def _query_ranks(X, q):
    ranks = np.empty(np.shape(X), dtype=np.float64)
    for a, b in _query_spans(q):
        ranks[a:b] = within_query_ranks(np.asarray(X[a:b], dtype=np.float64))
    return ranks.astype(np.float32)


def _query_z(X, q):
    out = np.empty(np.shape(X), dtype=np.float32)
    for a, b in _query_spans(q):
        out[a:b] = within_query_z(np.asarray(X[a:b], dtype=np.float64))
    return out


def _expand(X, q):
    """Raw features, then their percentile and z-score inside each query. Same layout as rerank_with_ltr."""
    return np.hstack([np.asarray(X, dtype=np.float32), _query_ranks(X, q), _query_z(X, q)])


def _run_codes(q):
    """One integer per row, constant on a contiguous query and unique across queries."""
    change = np.r_[True, q[1:] != q[:-1]]
    return np.cumsum(change).astype(np.int32) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/toy")
    ap.add_argument("--dev", default=None, help="dev file/dir; used for reporting only unless --use-dev")
    ap.add_argument("--out", default="submission/model.txt")
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--num-leaves", type=int, default=15)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=50)
    ap.add_argument("--lambda-l2", type=float, default=1.0)
    ap.add_argument("--feature-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-freq", type=int, default=1)
    ap.add_argument("--max-bin", type=int, default=255)
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feature-noise", default="0",
                    help="comma-separated Gaussian noise scales, in units of each feature's TRAIN std. "
                         "Each scale appends one extra copy of the rows (0 = off). Labels are unchanged.")
    ap.add_argument("--query-z", action="store_true",
                    help="add each feature's percentile and z-score inside its query (model width 3D)")
    ap.add_argument("--log-view", action="store_true",
                    help="append a log1p copy of the rows")
    ap.add_argument("--z-view", action="store_true",
                    help="append a per-feature z-scored copy of the rows")
    ap.add_argument("--query-scale", action="store_true",
                    help="append a copy where every query is rescaled by its own per-feature log-normal draw")
    ap.add_argument("--permute-per-query", action="store_true",
                    help="append a copy where the five fields inside each feature group are shuffled per query")
    ap.add_argument("--cell-view", action="store_true",
                    help="append a copy with --cell-rate of feature cells set to 0")
    ap.add_argument("--cell-rate", type=float, default=0.10,
                    help="fraction of cells zeroed in the cell view")
    ap.add_argument("--cell-missing", action="store_true",
                    help="append a copy with --cell-rate of cells set to missing")
    ap.add_argument("--drop-field-view", action="store_true",
                    help="append a copy that zeroes one of the five fields, chosen per query")
    ap.add_argument("--drop-field-count", type=int, default=1,
                    help="how many of the five fields to zero in the drop-field view")
    ap.add_argument("--mul-view", type=float, default=0.0,
                    help="if positive, append a copy scaled by 1 + this times unit noise")
    ap.add_argument("--clean-copies", type=int, default=1,
                    help="how many uncorrupted copies to keep when stacking row copies")
    ap.add_argument("--use-dev", action="store_true", help="also train on dev")
    ap.add_argument("--init-model", default=None, help="keep boosting from this LightGBM model file")
    args = ap.parse_args()

    X, y, q = load_xyq(resolve(args.train, "train.txt"))
    q = np.asarray(q, dtype=object)
    dev_path = resolve(args.dev, "vali.txt") if args.dev else resolve(args.train, "vali.txt")
    dev = None
    if os.path.isfile(dev_path):
        Xd, yd, qd = load_xyq(dev_path)
        dev = (Xd, yd, np.asarray(qd, dtype=object))
    if args.use_dev:
        if dev is None:
            raise SystemExit("--use-dev needs a dev file")
        X = np.vstack([X, dev[0]])
        y = np.concatenate([y, dev[1]])
        q = np.concatenate([q, dev[2]])
    n_queries_fit = len(set(q.tolist()))
    n_clean = len(y)
    scales = [float(s) for s in str(args.feature_noise).split(",") if float(s) > 0]
    base_codes = _run_codes(np.asarray(q))
    n_queries = int(base_codes[-1]) + 1
    # Each copy is one function of the raw rows. Copies get new query ids so a percentile never
    # mixes two copies of one query. Every copy has its own fixed seed.
    builders = []
    if args.clean_copies > 0:
        builders.append(lambda raw: raw)
    if scales:
        std = X.std(axis=0).astype(np.float32)
        for i, scale in enumerate(scales):
            builders.append(lambda raw, s=scale, i=i: (
                raw + s * std * np.random.RandomState(args.seed + i).randn(*raw.shape).astype(np.float32)))
    for _ in range(1, args.clean_copies):
        builders.append(lambda raw: raw)
    if args.log_view:
        builders.append(lambda raw: np.sign(raw) * np.log1p(np.abs(raw)))
    if args.z_view:
        builders.append(lambda raw: (raw - raw.mean(0)) / np.clip(raw.std(0), 1e-6, None))
    if args.query_scale:
        def _qscale(raw):
            rng = np.random.RandomState(args.seed + 17)
            spans = _query_spans(q)
            scale = np.exp(rng.normal(0.0, 1.0, size=(len(spans), raw.shape[1]))).astype(np.float32)
            out = np.empty_like(raw)
            for i, (a, b) in enumerate(spans):
                out[a:b] = raw[a:b] * scale[i]
            return out
        builders.append(_qscale)
    if args.permute_per_query:
        def _permute(raw):
            rng = np.random.RandomState(args.seed + 23)
            out = raw.copy()
            for a, b in _query_spans(q):
                for g in range(25):
                    out[a:b, 5 * g:5 * g + 5] = raw[a:b, 5 * g + rng.permutation(5)]
            return out
        builders.append(_permute)
    if args.cell_view:
        def _cell(raw):
            out = raw.copy()
            out[np.random.RandomState(args.seed + 29).rand(*out.shape) < args.cell_rate] = 0
            return out
        builders.append(_cell)
    if args.cell_missing:
        def _missing(raw):
            out = raw.copy()
            out[np.random.RandomState(args.seed + 37).rand(*out.shape) < args.cell_rate] = np.nan
            return out
        builders.append(_missing)
    if args.drop_field_view:
        n_drop = max(1, min(4, args.drop_field_count))

        def _drop_field(raw):
            rng = np.random.RandomState(args.seed + 41)
            out = raw.copy()
            for a, b in _query_spans(q):
                for field in rng.choice(5, size=n_drop, replace=False):
                    out[a:b, int(field):125:5] = 0
            return out
        builders.append(_drop_field)
    if args.mul_view > 0:
        builders.append(lambda raw: raw * (
            1.0 + args.mul_view * np.random.RandomState(args.seed + 31).randn(*raw.shape).astype(np.float32)))
    want_ranks = args.query_z
    if len(builders) > 1 or want_ranks:
        # fill one big array copy by copy so all copies are never in memory twice
        width = X.shape[1] * (3 if want_ranks else 1)
        gib = n_clean * len(builders) * width * 4 / 1024**3
        print(f"stacking {len(builders)} views, {width} columns, {gib:.1f} GiB", flush=True)
        big = np.empty((n_clean * len(builders), width), dtype=np.float32)
        big_y = np.empty(n_clean * len(builders), dtype=y.dtype)
        big_q = np.empty(n_clean * len(builders), dtype=np.int32)
        for i, build in enumerate(builders):
            view = np.asarray(build(X), dtype=np.float32)
            vq = base_codes + i * n_queries
            block = _expand(view, vq) if want_ranks else view
            a = i * n_clean
            big[a:a + n_clean] = block
            big_y[a:a + n_clean] = y
            big_q[a:a + n_clean] = vq
            del view, block
        del X
        X, y, q = big, big_y, big_q
    params = default_params(args.num_leaves, args.learning_rate, args.min_data_in_leaf, lambda_l2=args.lambda_l2,
                            seed=args.seed, feature_fraction=args.feature_fraction,
                            bagging_fraction=args.bagging_fraction, bagging_freq=args.bagging_freq,
                            max_bin=args.max_bin, num_threads=args.num_threads)
    t0 = time.perf_counter()
    model = fit(X, y, q, params, args.trees, init_model=args.init_model)
    fit_s = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save_model(args.out)
    where = "train+dev" if args.use_dev else "train only"
    print(f"trained on {n_queries_fit} queries ({where}) -> {args.out} "
          f"({args.trees} trees, features={X.shape[1]}, clean_copies={args.clean_copies}, {fit_s:.0f}s)")
    print(f"params: {params}")
    print(f"train nDCG@10 = {ndcg10(model, X[:n_clean], y[:n_clean], q[:n_clean]):.4f}")
    if dev is not None:
        Xd, yd, qd = dev
        if want_ranks:
            Xd = _expand(Xd, np.asarray(qd))
        label = "dev nDCG@10 (in-training, dev rows were fit)" if args.use_dev else "dev   nDCG@10"
        print(f"{label} = {ndcg10(model, Xd, yd, qd):.4f}")


if __name__ == "__main__":
    main()
