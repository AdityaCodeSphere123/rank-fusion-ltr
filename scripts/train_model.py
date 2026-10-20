"""Train the LambdaMART artifact that submission/ranking.py loads (offline; grading never retrains).

    python scripts/train_model.py --train data/train --out submission/model.txt --trees 300

--train takes a split file or a directory holding train.txt. Pass --use-dev to ALSO train on the dev split;
that raises your dev nDCG (Tracks A and B) without helping on queries the model has not seen, so your public
leaderboard position will overstate where you stand (assignment Section 7). Whatever you do, declare it in your
submission notes (Section 10).

"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import time

import numpy as np

from scripts.data_cache import load_xyq
from scripts.lambdamart_utils import default_params, fit, ndcg10
from submission.ranking import within_query_ranks, within_query_z


def _resolve(path, name):
    return os.path.join(path, name) if os.path.isdir(path) else path


def _query_ranks(X, q):
    ranks = np.empty(np.shape(X), dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    for a, b in zip(starts, ends):
        ranks[a:b] = within_query_ranks(np.asarray(X[a:b], dtype=np.float64))
    return ranks.astype(np.float32)


def _query_z(X, q):
    out = np.empty(np.shape(X), dtype=np.float32)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    for a, b in zip(starts, ends):
        out[a:b] = within_query_z(np.asarray(X[a:b], dtype=np.float64))
    return out


def _expand(X, q, with_z=False):
    """Raw features plus within-query percentiles. with_z appends within-query
    z-scores (width 3D). Same layout as rerank_with_ltr."""
    parts = [np.asarray(X, dtype=np.float32), _query_ranks(X, q)]
    if with_z:
        parts.append(_query_z(X, q))
    return np.hstack(parts)


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
    ap.add_argument("--query-ranks", action="store_true",
                    help="append within-query feature ranks, the same transform rerank_with_ltr applies "
                         "when the model has twice as many features as the input")
    ap.add_argument("--query-z", action="store_true",
                    help="also append within-query z-scores (model width 3D). Implies --query-ranks")
    ap.add_argument("--log-view", action="store_true",
                    help="append a log1p copy of the rows")
    ap.add_argument("--z-view", action="store_true",
                    help="append a per-feature z-scored copy of the rows")
    ap.add_argument("--query-scale", action="store_true",
                    help="append a copy where every query is rescaled by its own per-feature log-normal draw")
    ap.add_argument("--permute-view", action="store_true",
                    help="append a copy with the five fields inside each group shuffled")
    ap.add_argument("--permute-per-query", action="store_true",
                    help="draw a fresh field shuffle for every query (implies --permute-view)")
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

    X, y, q = load_xyq(_resolve(args.train, "train.txt"))
    q = np.asarray(q, dtype=object)
    dev_path = _resolve(args.dev, "vali.txt") if args.dev else _resolve(args.train, "vali.txt")
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
    # Views are built one at a time. Copy ids are shifted so a percentile never
    # mixes two copies of the same query.
    builders = []
    if args.clean_copies > 0:
        builders.append(("clean", lambda raw, _f=None: raw))
    if scales:
        std = X.std(axis=0).astype(np.float32)
        for i, scale in enumerate(scales):
            builders.append(("noise", lambda raw, s=scale, sd=std, i=i: (
                raw + s * sd * np.random.RandomState(args.seed + i).randn(*raw.shape).astype(np.float32)
            ).astype(np.float32)))
    for i in range(1, max(0, args.clean_copies)):
        builders.append(("clean", lambda raw: raw))
    if args.log_view:
        builders.append(("log", lambda raw: (np.sign(raw) * np.log1p(np.abs(raw))).astype(np.float32)))
    if args.z_view:
        builders.append(("z", lambda raw: ((raw - raw.mean(0)) / np.clip(raw.std(0), 1e-6, None)).astype(np.float32)))
    if args.query_scale:
        q_scale = q

        def _qscale(raw, q=q_scale, seed=args.seed):
            raw = np.asarray(raw, dtype=np.float32)
            out = np.empty_like(raw)
            rng = np.random.RandomState(seed + 17)
            starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
            ends = np.r_[starts[1:], len(q)]
            scales = np.exp(rng.normal(0.0, 1.0, size=(len(starts), raw.shape[1]))).astype(np.float32)
            for i, (a, b) in enumerate(zip(starts, ends)):
                out[a:b] = raw[a:b] * scales[i]
            return out

        builders.append(("qscale", _qscale))
    if args.permute_per_query:
        args.permute_view = True
    if args.permute_view:
        per_query = args.permute_per_query
        q_perm = q

        def _permute(raw, seed=args.seed, per_query=per_query, q=q_perm):
            raw = np.asarray(raw, dtype=np.float32)
            out = np.array(raw, copy=True)
            rng = np.random.RandomState(seed + 23)
            if not per_query:
                for g in range(25):
                    out[:, 5 * g:5 * g + 5] = raw[:, 5 * g + rng.permutation(5)]
                return out
            starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
            ends = np.r_[starts[1:], len(q)]
            for a, b in zip(starts, ends):
                block = out[a:b]
                src = raw[a:b]
                for g in range(25):
                    block[:, 5 * g:5 * g + 5] = src[:, 5 * g + rng.permutation(5)]
            return out

        builders.append(("permute", _permute))
    if args.cell_view:
        rate = float(args.cell_rate)

        def _cell(raw, seed=args.seed, rate=rate):
            raw = np.asarray(raw, dtype=np.float32)
            out = np.array(raw, copy=True)
            rng = np.random.RandomState(seed + 29)
            out[rng.rand(*out.shape) < rate] = 0
            return out

        builders.append(("cell", _cell))
    if args.cell_missing:
        rate = float(args.cell_rate)

        def _missing(raw, seed=args.seed, rate=rate):
            raw = np.asarray(raw, dtype=np.float32)
            out = np.array(raw, dtype=np.float32, copy=True)
            rng = np.random.RandomState(seed + 37)
            out[rng.rand(*out.shape) < rate] = np.nan
            return out

        builders.append(("missing", _missing))
    if args.drop_field_view:
        q_drop = q

        n_drop = max(1, min(4, int(args.drop_field_count)))

        def _drop_field(raw, seed=args.seed, q=q_drop, n_drop=n_drop):
            raw = np.asarray(raw, dtype=np.float32)
            out = np.array(raw, copy=True)
            rng = np.random.RandomState(seed + 41)
            starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
            ends = np.r_[starts[1:], len(q)]
            for a, b in zip(starts, ends):
                for field in rng.choice(5, size=n_drop, replace=False):
                    out[a:b, int(field):125:5] = 0
            return out

        builders.append(("dropfield", _drop_field))
    if args.mul_view > 0:
        mul_scale = float(args.mul_view)

        def _mul(raw, s=mul_scale, seed=args.seed):
            raw = np.asarray(raw, dtype=np.float32)
            rng = np.random.RandomState(seed + 31)
            return (raw * (1.0 + s * rng.randn(*raw.shape).astype(np.float32))).astype(np.float32)

        builders.append(("mul", _mul))
    if args.query_z:
        args.query_ranks = True
    want_ranks = (args.query_ranks or args.log_view or args.z_view or args.query_scale or args.permute_view
                  or args.cell_view or args.cell_missing or args.drop_field_view or args.mul_view > 0)
    if len(builders) > 1 or want_ranks:
        # Fill one preallocated block so the copies are not all live at once.
        probe_q = base_codes
        probe = _expand(X[:1], probe_q[:1], with_z=args.query_z) if want_ranks else np.asarray(X[:1], dtype=np.float32)
        width = probe.shape[1]
        del probe
        gib = n_clean * len(builders) * width * 4 / 1024**3
        print(f"stacking {len(builders)} views, {width} columns, {gib:.1f} GiB", flush=True)
        big = np.empty((n_clean * len(builders), width), dtype=np.float32)
        big_y = np.empty(n_clean * len(builders), dtype=y.dtype)
        big_q = np.empty(n_clean * len(builders), dtype=np.int32)
        for i, (_, build) in enumerate(builders):
            view = np.asarray(build(X), dtype=np.float32)
            vq = base_codes + i * n_queries
            block = _expand(view, vq, with_z=args.query_z) if want_ranks else view
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
            Xd = _expand(Xd, np.asarray(qd), with_z=args.query_z)
        label = "dev nDCG@10 (in-training, dev rows were fit)" if args.use_dev else "dev   nDCG@10"
        print(f"{label} = {ndcg10(model, Xd, yd, qd):.4f}")


if __name__ == "__main__":
    main()
