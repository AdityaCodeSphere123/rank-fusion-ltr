"""Track C1 training recipe. Staff call train(train_path, vali_path, out_model_path, seed=0).

The recipe trains up to six LambdaMART models and saves their average as one
LightGBM model file. Half of them see the raw features and half also see two
extra copies of every feature that are computed inside each query: its
percentile and its z-score. Neither kind won on every dataset we tried so we
use both.

Tree size, leaf size and column sampling are set from the size of the data and
not copied from MSLR. Each model stops early on the vali file and is then
trained again on train plus vali. How many models we train is decided from the
data size alone so the result only depends on the data and the seed.
"""
import time

import numpy as np

from submission.ranking import sanitize, within_query_ranks, within_query_z

NUM_THREADS = 4
# Staff stop us at 20 minutes. We stop adding trees at 15 to leave room for saving
TIME_BUDGET_S = 15 * 60

# Rough cost of one boosting round per row per used column, measured on a Mac with 4 threads.
# SLOWDOWN assumes the grading machine is slower. On full MSLR TRAIN the plan used 669
# predicted seconds while boosting really took 262 so there is plenty of room
SEC_PER_UNIT = 8.5e-10
SLOWDOWN = 2.5
BOOST_BUDGET_S = 0.7 * TIME_BUDGET_S
MIN_ROUNDS, MAX_ROUNDS = 100, 3000
# Each rerank call must finish in 2 seconds so the whole ensemble is kept below this many trees
MAX_TOTAL_TREES = 8000

# The models in the order we add them. The first part says which features the model
# sees and the second part says how its trees are shaped
SCHEDULE = [("views", "rule"), ("raw", "rule"), ("views", "deep"), ("raw", "deep"),
            ("views", "shallow"), ("raw", "narrow")]


def _dense_block(lines):
    width = max([0] + [int(idx.max()) for idx, _ in lines if idx.size])
    B = np.zeros((len(lines), width), dtype=np.float32)
    for r, (idx, val) in enumerate(lines):
        if idx.size:
            B[r, idx - 1] = val
    return B


def _read(path: str, block_rows: int = 8192):
    """Read an SVMLight file into dense blocks. Missing features are 0 and comments are skipped."""
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
                # some datasets use huge numbers like 1.8e308 to mean missing
                val[~np.isfinite(val) | (np.abs(val) > 1e30)] = np.nan
                pending.append((flat[0::2].astype(np.int64), val))
            else:
                pending.append((np.zeros(0, np.int64), np.zeros(0)))
            if len(pending) == block_rows:
                blocks.append(_dense_block(pending))
                pending = []
    if pending:
        blocks.append(_dense_block(pending))
    return blocks, np.asarray(labels, dtype=np.int64), qids


def _place(blocks, labels, qids, out):
    """Copy the parsed rows into out with all rows of a query next to each other.

    Each block is freed as soon as it is copied so we never hold two full copies
    of the data. The grading machine has only 8 GB of memory.
    """
    first = {}
    for q in qids:
        first.setdefault(q, len(first))
    qnum = np.fromiter((first[q] for q in qids), dtype=np.int64, count=len(qids))
    order = np.argsort(qnum, kind="stable")
    dest = np.empty(len(qids), dtype=np.int64)
    dest[order] = np.arange(len(qids))
    out[:] = 0.0
    r = 0
    while blocks:
        B = blocks.pop(0)
        out[dest[r:r + B.shape[0]], :B.shape[1]] = B
        r += B.shape[0]
    return labels[order], np.bincount(qnum, minlength=len(first)).tolist()


def view_names(width: int):
    """Column names stored in the model. prepare() reads the raw width back from them."""
    return ([f"raw_{i}" for i in range(width)] + [f"pct_{i}" for i in range(width)]
            + [f"wqz_{i}" for i in range(width)])


def _fill_views(X, sizes, d: int):
    """Fill the percentile and z-score columns one query at a time, the same way rerank_with_ltr does."""
    start = 0
    for n in sizes:
        block = sanitize(X[start:start + n, :d])
        X[start:start + n, :d] = block
        X[start:start + n, d:2 * d] = within_query_ranks(block)
        X[start:start + n, 2 * d:] = within_query_z(block)
        start += n


def load_views(train_path: str, vali_path: str):
    """Return X, labels, train query sizes, vali query sizes and the raw width.

    X holds the train rows first and then the vali rows, with 3 * width columns.
    """
    tr = _read(train_path)
    va = _read(vali_path)
    width = max([1] + [B.shape[1] for B in tr[0] + va[0]])
    n = len(tr[2])
    X = np.empty((n + len(va[2]), 3 * width), dtype=np.float32)
    yt, gt = _place(*tr, X[:n, :width])
    del tr
    yv, gv = _place(*va, X[n:, :width])
    del va
    _fill_views(X[:n], gt, width)
    _fill_views(X[n:], gv, width)
    return X, np.concatenate([yt, yv]), gt, gv, width


def _round_cost_s(n_rows: int, n_features: int, params) -> float:
    leaves_factor = max(0.5, (params["num_leaves"] / 31.0) ** 0.45)
    return SEC_PER_UNIT * SLOWDOWN * n_rows * max(1.0, n_features * params["feature_fraction"]) * leaves_factor


def rule_params(n_rows: int, n_features: int, seed: int):
    """LightGBM settings that grow with the data.

    More rows give more leaves and bigger leaves. More columns give a smaller
    column sample. Small data gets stronger L2 so it does not overfit.
    """
    num_leaves = int(np.clip(2 ** round(np.log2(max(n_rows, 1) ** 0.5 / 16.0)) - 1, 7, 63))
    min_data_in_leaf = int(np.clip(n_rows / 4000.0, 20, 200))
    feature_fraction = float(np.clip(1.0 - 0.15 * np.log2(max(n_features, 1) / 32.0), 0.5, 1.0))
    return {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
        "num_leaves": num_leaves, "min_data_in_leaf": min_data_in_leaf, "learning_rate": 0.05,
        "feature_fraction": feature_fraction, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 10.0 if n_rows < 50_000 else 1.0, "max_bin": 255, "feature_pre_filter": False,
        "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed, "data_random_seed": seed,
        "deterministic": True, "force_col_wise": True, "num_threads": NUM_THREADS, "verbose": -1,
    }


def member_params(base, m: int):
    """Settings for model m. Deep doubles the leaves, shallow halves them and narrow samples fewer columns."""
    p = dict(base)
    nl, mdl = base["num_leaves"], base["min_data_in_leaf"]
    shape = SCHEDULE[m][1]
    if shape == "deep":
        p.update(num_leaves=min(255, 2 * nl + 1), min_data_in_leaf=2 * mdl)
    elif shape == "shallow":
        p.update(num_leaves=max(7, nl // 2), min_data_in_leaf=max(10, mdl // 2))
    elif shape == "narrow":
        p.update(feature_fraction=max(0.3, 0.6 * base["feature_fraction"]))
    s = base["seed"] + 7919 * m
    p.update(seed=s, bagging_seed=s, feature_fraction_seed=s)
    return p


class _Deadline:
    """Stops boosting if the clock runs out. This is only a safety net for a very slow machine."""
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
    """Join several boosters into one LightGBM model whose score is their average.

    We put all trees one after another and divide every leaf value by the number
    of models. The tree_sizes line is dropped so LightGBM reads the trees in order.
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


def fit_ensemble(datasets, n_train: int, n_vali: int, D: int, seed: int, deadline: float):
    """Train the models in SCHEDULE and return a list of (booster, number of trees).

    Each model stops early on vali and is then trained again on train plus vali
    for the same number of rounds scaled up by the extra rows. We only add a model
    when the cost estimate says it still fits in the time budget.
    """
    import lightgbm as lgb

    have_vali = n_vali > 0
    n_all = n_train + n_vali
    cols = {"views": 3 * D, "raw": D}
    bases = {rep: rule_params(n_train, cols[rep], seed) for rep in cols}
    parts = {}
    for rep, full in datasets.items():
        if have_vali:
            parts[rep] = (full.subset(np.arange(n_train), params=bases[rep]),
                          full.subset(np.arange(n_train, n_all), params=bases[rep]))
        else:
            parts[rep] = (full, None)
    spent, total_trees, members, first_rounds = 0.0, 0, [], {}
    for m, (rep, _shape) in enumerate(SCHEDULE):
        params = member_params(bases[rep], m)
        fit_cost = _round_cost_s(n_train, cols[rep], params)
        refit_cost = _round_cost_s(n_all, cols[rep], params)
        member_cost = fit_cost + refit_cost * n_all / n_train
        if m == 0:
            # the first model always runs and its rounds are capped so it fits the budget alone
            cap = int(np.clip(BOOST_BUDGET_S / member_cost, MIN_ROUNDS, MAX_ROUNDS))
        else:
            # guess this model's rounds from the first model of the same kind
            expect = first_rounds.get(rep, first_rounds["views"])
            if spent + expect * member_cost > BOOST_BUDGET_S:
                continue  # a cheaper model later in the list may still fit
            if total_trees + expect * n_all / n_train > MAX_TOTAL_TREES:
                continue
            cap = int(np.clip(2 * expect, MIN_ROUNDS, MAX_ROUNDS))
        if time.perf_counter() > deadline - 60:
            break
        patience = int(np.clip(cap // 10, 50, 200))
        guard = _Deadline(deadline)
        dtrain, dvalid = parts[rep]
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
        first_rounds.setdefault(rep, best)
        if have_vali:
            rounds = max(1, int(round(best * n_all / n_train)))
            refit_guard = _Deadline(deadline)
            b = lgb.train(params, datasets[rep], num_boost_round=rounds, callbacks=[refit_guard])
            spent += rounds * refit_cost
            if refit_guard.fired and members:
                break
            best = b.current_iteration()
        members.append((b, best))
        total_trees += best
    return members


def _hold_out_train(X, y, g):
    """Used when vali is empty or has no relevant document. Every fifth train query becomes vali.

    The rows come back as kept queries first and held out queries last.
    """
    starts = np.r_[0, np.cumsum(g)[:-1]].astype(np.int64)
    hold = np.zeros(len(g), dtype=bool)
    hold[4::5] = True
    if not hold.any() or y[np.concatenate(
            [np.arange(s, s + n) for s, n, h in zip(starts, g, hold) if h])].max() <= 0:
        return X, y, list(g), []
    rows = np.concatenate([np.arange(s, s + n) for s, n, h in zip(starts, g, hold) if not h]
                          + [np.arange(s, s + n) for s, n, h in zip(starts, g, hold) if h])
    return X[rows], y[rows], [n for n, h in zip(g, hold) if not h], [n for n, h in zip(g, hold) if h]


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
    # LightGBM picks bin edges from a sample of rows kept as doubles. A smaller sample on wide
    # data keeps that copy near 0.6 GB
    sample_rows = int(np.clip(0.6e9 / (8 * 3 * D), 50_000, 200_000))
    ds_params = {"max_bin": 255, "feature_pre_filter": False, "data_random_seed": seed,
                 "bin_construct_sample_cnt": sample_rows, "num_threads": NUM_THREADS, "verbose": -1}
    group = list(g) + list(gv)
    # The raw columns are the first D columns of X. Trees trained on them therefore point at
    # the right columns inside the merged model too
    datasets = {"raw": lgb.Dataset(np.ascontiguousarray(X[:, :D]), label=y, group=group,
                                   feature_name=view_names(D)[:D], params=ds_params).construct(),
                "views": lgb.Dataset(X, label=y, group=group, feature_name=view_names(D),
                                     params=ds_params).construct()}
    del X
    members = fit_ensemble(datasets, n_train, n_vali, D, seed, deadline=t0 + TIME_BUDGET_S)
    if len(members) == 1:
        b, n = members[0]
        b.save_model(out_model_path, num_iteration=n)
        return
    merged = merge_boosters(members)
    lgb.Booster(model_str=merged)  # fails loudly here if the merged text is broken
    with open(out_model_path, "w", encoding="utf-8") as f:
        f.write(merged)
