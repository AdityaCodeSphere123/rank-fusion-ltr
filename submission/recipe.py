"""Track C1 recipe. Staff call train(train_path, vali_path, out_model_path, seed=0).

Writes one LightGBM LambdaMART model file that averages up to five members.
Each query is fit on its raw features, their within-query percentiles, and
their within-query z-scores. The columns are named raw_i, pct_i and wqz_i,
and rerank_with_ltr rebuilds them from those names. Values above 1e30 are
stored as missing.

Leaf count, min_data_in_leaf, column sampling and L2 follow the training-set
shape, not MSLR constants. The members differ in seed and in tree shape
(the size rule, deeper, shallower, fewer columns, slower learning rate), so
the average hedges the shape choice on a dataset we cannot tune on. Each
member early-stops on vali nDCG@10 and is refit on train+vali, its round
count scaled by the row ratio. How many members run, and their round caps,
come from a size-based cost model, so the model depends only on the data
and the seed. A wall-clock guard is a safety net only.
"""
import time

import numpy as np

from submission.ranking import sanitize, within_query_ranks, within_query_z

TIME_BUDGET_S = 15 * 60
NUM_THREADS = 4
MAX_MEMBERS = 5


def _dense_block(lines, width_hint: int):
    width = max([width_hint] + [int(idx.max()) for idx, _ in lines if idx.size])
    B = np.zeros((len(lines), width), dtype=np.float32)
    for r, (idx, val) in enumerate(lines):
        if idx.size:
            B[r, idx - 1] = val
    return B


def _read(path: str, block_rows: int = 8192):
    """SVMLight file to dense float32 blocks. Absent indices are 0; '# ...' is ignored."""
    labels, qids, blocks, pending = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            hash_pos = line.find("#")
            if hash_pos >= 0:
                line = line[:hash_pos]
            parts = line.split(None, 2)
            if not parts:
                continue
            if len(parts) < 2 or not parts[1].startswith("qid:"):
                raise ValueError(f"{path}: malformed LTR line {line[:80]!r}")
            labels.append(int(float(parts[0])))
            qids.append(parts[1][4:])
            if len(parts) > 2 and parts[2].strip():
                flat = np.array(parts[2].replace(":", " ").split(), dtype=np.float64)
                val = flat[1::2]
                val[~np.isfinite(val) | (np.abs(val) > 1e30)] = np.nan
                pending.append((flat[0::2].astype(np.int64), val))
            else:
                pending.append((np.zeros(0, np.int64), np.zeros(0)))
            if len(pending) == block_rows:
                blocks.append(_dense_block(pending, 0))
                pending = []
    if pending:
        blocks.append(_dense_block(pending, 0))
    return blocks, np.asarray(labels, dtype=np.int64), qids


def _to_matrix(blocks, labels, qids, width: int):
    """Dense X with rows grouped by qid (first appearance, file order within a query)."""
    X = np.zeros((len(qids), width), dtype=np.float32)
    r = 0
    for B in blocks:
        X[r:r + B.shape[0], :B.shape[1]] = B
        r += B.shape[0]
    first = {}
    for q in qids:
        first.setdefault(q, len(first))
    qnum = np.fromiter((first[q] for q in qids), dtype=np.int64, count=len(qids))
    if len(qnum) and np.any(np.diff(qnum) < 0):
        order = np.argsort(qnum, kind="stable")
        X, labels, qnum = X[order], labels[order], qnum[order]
    sizes = np.bincount(qnum, minlength=len(first)).tolist()
    return X, labels, sizes


def view_names(width: int):
    """Column names saved in the model. ranking.prepare() reads the raw width back from them."""
    return ([f"raw_{i}" for i in range(width)] + [f"pct_{i}" for i in range(width)]
            + [f"wqz_{i}" for i in range(width)])


def _fill_views(out, X, sizes):
    """Write raw features, within-query percentiles and within-query z-scores into out.

    Same columns as ranking._feature_matrix builds at scoring time. One query
    at a time, so the statistics never cross queries.
    """
    d = X.shape[1]
    start = 0
    for n in sizes:
        block = sanitize(X[start:start + n])
        out[start:start + n, :d] = block
        out[start:start + n, d:2 * d] = within_query_ranks(block)
        out[start:start + n, 2 * d:] = within_query_z(block)
        start += n


def load_views(train_path: str, vali_path: str):
    """(X, y, train group sizes, vali group sizes, raw width). X holds train rows, then vali rows."""
    tr = _read(train_path)
    va = _read(vali_path)
    width = max([1] + [B.shape[1] for B in tr[0] + va[0]])
    Xt, yt, gt = _to_matrix(*tr, width)
    del tr
    Xv, yv, gv = _to_matrix(*va, width)
    del va
    n = Xt.shape[0]
    X = np.empty((n + Xv.shape[0], 3 * width), dtype=np.float32)
    _fill_views(X[:n], Xt, gt)
    del Xt
    _fill_views(X[n:], Xv, gv)
    return X, np.concatenate([yt, yv]), gt, gv, width


# Seconds per (row x sampled column) per round at 31 leaves on 4 threads (column-wise
# histograms), measured on a 500k-row, 300-feature case, times a slower-grader factor.
# Only data size and round counts enter the plan, so the model is deterministic given the seed.
SEC_PER_UNIT = 8.5e-10
SLOWDOWN = 3.0
BOOST_BUDGET_S = 0.7 * TIME_BUDGET_S
MIN_ROUNDS, MAX_ROUNDS = 100, 3000


def _round_cost_s(n_rows: int, n_features: int, params) -> float:
    leaves_factor = max(0.5, (params["num_leaves"] / 31.0) ** 0.45)
    return SEC_PER_UNIT * SLOWDOWN * n_rows * max(1.0, n_features * params["feature_fraction"]) * leaves_factor


def rule_params(n_rows: int, n_features: int, seed: int, **overrides):
    """Leaves grow with sqrt(rows); min_data_in_leaf with rows; fewer columns as D grows."""
    num_leaves = int(np.clip(2 ** round(np.log2(max(n_rows, 1) ** 0.5 / 16.0)) - 1, 7, 63))
    min_data_in_leaf = int(np.clip(n_rows / 4000.0, 20, 200))
    feature_fraction = float(np.clip(1.0 - 0.15 * np.log2(max(n_features, 1) / 32.0), 0.5, 1.0))
    params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
        "num_leaves": num_leaves, "min_data_in_leaf": min_data_in_leaf, "learning_rate": 0.05,
        "feature_fraction": feature_fraction, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 10.0 if n_rows < 50_000 else 1.0, "max_bin": 255, "feature_pre_filter": False,
        "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed, "data_random_seed": seed,
        "deterministic": True, "force_col_wise": True, "num_threads": NUM_THREADS, "verbose": -1,
    }
    params.update(overrides)
    return params


def member_params(base, m: int):
    """Member m: 0 is the size rule, 1 deeper, 2 shallower, 3 fewer columns, 4 slower learning rate."""
    p = dict(base)
    nl, mdl = base["num_leaves"], base["min_data_in_leaf"]
    if m == 1:
        p.update(num_leaves=min(255, 2 * nl + 1), min_data_in_leaf=2 * mdl)
    elif m == 2:
        p.update(num_leaves=max(7, nl // 2), min_data_in_leaf=max(10, mdl // 2))
    elif m == 3:
        p.update(feature_fraction=max(0.3, 0.6 * base["feature_fraction"]))
    elif m == 4:
        p.update(learning_rate=0.6 * base["learning_rate"])
    s = base["seed"] + 7919 * m
    p.update(seed=s, bagging_seed=s, feature_fraction_seed=s)
    return p


class _Deadline:
    """Stop boosting once wall-clock passes the deadline. Safety net only."""
    order = 30

    def __init__(self, deadline: float):
        self.deadline = deadline
        self.fired = False

    def __call__(self, env):
        if time.perf_counter() > self.deadline:
            self.fired = True
            import lightgbm as lgb
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list or [])


def merge_boosters(members):
    """One LightGBM model string whose score is the mean of the members' scores.

    members: (booster, n_iter) pairs on the same columns. Trees are concatenated
    and every leaf value is divided by the member count. tree_sizes is dropped,
    so LightGBM parses the trees in order.
    """
    k = len(members)
    texts = [b.model_to_string(num_iteration=n) for b, n in members]
    head = texts[0].split("\nTree=0\n", 1)[0]
    head = "\n".join(line for line in head.split("\n") if not line.startswith("tree_sizes="))
    tail = texts[0].split("\nend of trees\n", 1)[1]
    trees, t = [], 0
    for text in texts:
        body = text.split("\nTree=0\n", 1)[1].split("\nend of trees\n", 1)[0]
        for block in ("Tree=0\n" + body).split("\n\n\n"):
            block = block.strip("\n")
            if not block:
                continue
            lines = block.split("\n")
            lines[0] = f"Tree={t}"
            for i, line in enumerate(lines):
                key, _, vals = line.partition("=")
                if key in ("leaf_value", "internal_value") and vals:
                    lines[i] = key + "=" + " ".join(repr(float(v) / k) for v in vals.split(" "))
            trees.append("\n".join(lines))
            t += 1
    return head + "\n\n" + "\n\n\n".join(trees) + "\n\n\nend of trees\n" + tail


def fit_ensemble(full, n_train: int, n_vali: int, D: int, seed: int, deadline: float):
    """Members on full.subset(train), early-stopped on full.subset(vali), refit on full.

    Returns ([(booster, n_iter)], info). Members are added while the cost model
    says the next one, fit plus refit, still fits in BOOST_BUDGET_S.
    """
    import lightgbm as lgb

    base = rule_params(n_train, D, seed)
    have_vali = n_vali > 0
    dtrain = full.subset(np.arange(n_train), params=base) if have_vali else full
    dvalid = full.subset(np.arange(n_train, n_train + n_vali), params=base) if have_vali else None
    n_all = n_train + n_vali
    spent, members, info = 0.0, [], {"base": {k: base[k] for k in ("num_leaves", "min_data_in_leaf",
                                                                    "learning_rate", "feature_fraction")},
                                     "members": []}
    first_rounds = None
    for m in range(MAX_MEMBERS):
        params = member_params(base, m)
        fit_cost = _round_cost_s(n_train, D, params)
        refit_cost = _round_cost_s(n_all, D, params)
        lr_scale = base["learning_rate"] / params["learning_rate"]
        if m == 0:
            # the first member always runs; cap its rounds so it alone fits the budget
            cap = int(np.clip(BOOST_BUDGET_S / (fit_cost + refit_cost * n_all / n_train), MIN_ROUNDS, MAX_ROUNDS))
        else:
            expect = first_rounds * lr_scale
            if spent + expect * (fit_cost + refit_cost * n_all / n_train) > BOOST_BUDGET_S:
                break
            cap = int(np.clip(2 * expect, MIN_ROUNDS, MAX_ROUNDS))
        if time.perf_counter() > deadline - 60:
            break
        patience = int(np.clip(cap // 10, 50, 200))
        guard = _Deadline(deadline)
        if have_vali:
            b = lgb.train(params, dtrain, num_boost_round=cap, valid_sets=[dvalid],
                          callbacks=[guard, lgb.early_stopping(patience, first_metric_only=True, verbose=False)])
            best = int(b.best_iteration) if b.best_iteration > 0 else b.current_iteration()
            ran = min(cap, best + patience)
        else:
            b = lgb.train(params, dtrain, num_boost_round=cap, callbacks=[guard])
            best = ran = b.current_iteration()
        spent += ran * fit_cost
        if guard.fired:
            if not members:
                members.append((b, best))
            break
        if first_rounds is None:
            first_rounds = best
        rec = {"m": m, "num_leaves": params["num_leaves"], "best": best}
        if have_vali:
            rounds = max(1, int(round(best * n_all / n_train)))
            refit_guard = _Deadline(deadline)
            b = lgb.train(params, full, num_boost_round=rounds, callbacks=[refit_guard])
            spent += rounds * refit_cost
            if refit_guard.fired and members:
                break
            rec["refit"] = rounds
            best = b.current_iteration()
        members.append((b, best))
        info["members"].append(rec)
    info["predicted_s"] = round(spent, 1)
    return members, info


def _hold_out_train(X, y, g):
    """Vali is empty or has no relevant document: early-stop on every fifth train query instead.

    Returns rows reordered as kept queries then held-out queries, and both group lists.
    """
    starts = np.r_[0, np.cumsum(g)[:-1]].astype(np.int64)
    hold = np.zeros(len(g), dtype=bool)
    hold[4::5] = True
    if not hold.any() or hold.all() or y[np.concatenate(
            [np.arange(s, s + n) for s, n, h in zip(starts, g, hold) if h])].max() <= 0:
        return X, y, list(g), []
    rows = np.concatenate([np.arange(s, s + n) for s, n, h in zip(starts, g, hold) if not h]
                          + [np.arange(s, s + n) for s, n, h in zip(starts, g, hold) if h])
    return (X[rows], y[rows], [n for n, h in zip(g, hold) if not h], [n for n, h in zip(g, hold) if h])


def train(train_path: str, vali_path: str, out_model_path: str, seed: int = 0) -> None:
    import lightgbm as lgb

    t0 = time.perf_counter()
    X, y, g, gv, D = load_views(train_path, vali_path)
    n_train = int(sum(g))
    n_vali = X.shape[0] - n_train
    if n_vali == 0 or y[n_train:].max() <= 0:
        X, y, g, gv = _hold_out_train(X[:n_train], y[:n_train], g)
        n_train = int(sum(g))
        n_vali = X.shape[0] - n_train
    ds_params = {"max_bin": 255, "feature_pre_filter": False, "data_random_seed": seed,
                 "num_threads": NUM_THREADS, "verbose": -1}
    full = lgb.Dataset(X, label=y, group=list(g) + list(gv), feature_name=view_names(D),
                       params=ds_params, free_raw_data=True).construct()
    del X
    members, _info = fit_ensemble(full, n_train, n_vali, 3 * D, seed, deadline=t0 + TIME_BUDGET_S)
    if len(members) == 1:
        b, n = members[0]
        b.save_model(out_model_path, num_iteration=n)
        return
    merged = merge_boosters(members)
    lgb.Booster(model_str=merged)  # raises if the merged text does not parse
    with open(out_model_path, "w", encoding="utf-8") as f:
        f.write(merged)
