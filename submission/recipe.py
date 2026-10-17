"""
submission/recipe.py -- your training recipe (assignment Section 4.1, Track C1).

Staff call train() ONCE per hidden benchmark (two undisclosed flat-feature LTR datasets) on that
benchmark's train and vali files, in an isolated directory, then evaluate the model it writes on the benchmark's
held-out split through YOUR prepare() / rerank_with_ltr(). Contract:

  * signature: train(train_path, vali_path, out_model_path, seed=0) -> None
  * the files are flat-feature SVMLight (label qid:N 1:v ... D:v), D unknown (at most ~300), labels integers
    0..4 (possibly fewer levels), at most ~500k training rows; feature meanings are NOT given
  * write a LightGBM text model (or whatever your prepare() loads) to out_model_path
  * 20 minutes wall-clock PER BENCHMARK on 4 CPU threads; no network; read only the two input files; deterministic given seed
  * a recipe that raises, times out, or writes a model prepare() cannot load scores 0 on that benchmark

What this recipe does (rules, not MSLR constants; derivation and measurements in Decision.md):
  1. Parse both files (one C-level float conversion per line, folded into dense float32 blocks; same semantics as
     harness.feature_io: absent indices = 0, trailing '# ...' comments ignored, rows grouped by qid in order of
     first appearance). Width D = the largest feature index in EITHER file, so the model accepts every held-out row
     (rerank_with_ltr also zero-pads / truncates rows to the model's width).
  2. Hyperparameters from the data shape (rule_params / plan): leaves and min_data_in_leaf grow with the number of
     training rows, feature_fraction shrinks as D grows, stronger L2 for small data, learning rate 0.05 (0.1 only
     if the size-based cost model allows fewer than 800 rounds), max rounds from that cost model.
  3. Fit on train, monitoring nDCG@10 on the vali file for early stopping (the ONE use of vali for model
     selection) -> best round count B.
  4. Refit on train + vali with round(B * (n_train + n_vali) / n_train) rounds -- more queries for the final model,
     tree count scaled to the extra data -- if the size-based cost estimate fits the budget; otherwise keep the
     early-stopped train-only model from step 3.
  5. A wall-clock guard stops boosting early if the run ever approaches the cap (a safety net that only fires on a
     machine far slower than the cost model assumes; all regular decisions depend on data size only, so the
     output is deterministic given the seed).

Test it locally on the synthetic transfer sets:  python scripts/run_recipe.py
"""
import time

import numpy as np

TIME_BUDGET_S = 15 * 60          # stay well inside the 20-minute cap (parse + fit + refit + save)
NUM_THREADS = 4


# ---- parsing ---------------------------------------------------------------------------------------------------

def _dense_block(lines, width_hint: int):
    """[(idx int array, val array)] -> dense float32 block (len(lines), max index in block)."""
    width = max([width_hint] + [int(idx.max()) for idx, _ in lines if idx.size])
    B = np.zeros((len(lines), width), dtype=np.float32)
    for r, (idx, val) in enumerate(lines):
        if idx.size:
            B[r, idx - 1] = val
    return B


def _read(path: str, block_rows: int = 8192):
    """SVMLight LTR file -> (list of dense float32 row blocks, labels int64 [n], qids list[str] [n]).
    Same semantics as harness.feature_io.read_svmlight_ltr: absent indices are 0, '# ...' comments ignored.
    Each line is converted with one C-level float parse; lines are folded into dense blocks every
    `block_rows` rows so memory stays ~ the size of the final float32 matrix."""
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
                pending.append((flat[0::2].astype(np.int64), flat[1::2]))
            else:
                pending.append((np.zeros(0, np.int64), np.zeros(0)))
            if len(pending) == block_rows:
                blocks.append(_dense_block(pending, 0))
                pending = []
    if pending:
        blocks.append(_dense_block(pending, 0))
    return blocks, np.asarray(labels, dtype=np.int64), qids


def _to_matrix(blocks, labels, qids, width: int):
    """Dense float32 X (n, width) with rows grouped by qid (first-appearance order, file order within a
    query, exactly like harness.feature_io), labels, and the per-query group sizes LightGBM needs."""
    X = np.zeros((len(qids), width), dtype=np.float32)
    r = 0
    for B in blocks:
        X[r:r + B.shape[0], :B.shape[1]] = B
        r += B.shape[0]
    first = {}
    for q in qids:
        first.setdefault(q, len(first))
    qnum = np.fromiter((first[q] for q in qids), dtype=np.int64, count=len(qids))
    if len(qnum) and np.any(np.diff(qnum) < 0):           # some query is split: regroup (stable)
        order = np.argsort(qnum, kind="stable")
        X, labels, qnum = X[order], labels[order], qnum[order]
    sizes = np.bincount(qnum, minlength=len(first)).tolist()
    return X, labels, sizes


def load_pair(train_path: str, vali_path: str):
    tr, va = _read(train_path), _read(vali_path)
    width = max([1] + [B.shape[1] for B in tr[0] + va[0]])
    return _to_matrix(*tr, width), _to_matrix(*va, width)


# ---- rules -----------------------------------------------------------------------------------------------------

# Cost model: seconds per (row x used feature) per boosting round with 4 threads, measured on the development
# machine (Decision.md), times a safety factor for a slower grader. Only data SIZE enters the plan, so the
# plan -- and therefore the model -- is deterministic given the seed.
SEC_PER_UNIT = 2.0e-9
SLOWDOWN = 3.0
FIT_BUDGET_S = 0.55 * TIME_BUDGET_S       # first fit (with early stopping) may use at most this, predicted
REFIT_TOTAL_BUDGET_S = 0.8 * TIME_BUDGET_S  # predicted fit + refit must fit in this, else keep the first model


def _round_cost_s(n_rows: int, n_features: int, params) -> float:
    leaves_factor = max(0.5, (params["num_leaves"] / 31.0) ** 0.45)   # measured: 127 leaves ~1.84x 31 leaves
    return SEC_PER_UNIT * SLOWDOWN * n_rows * max(1.0, n_features * params["feature_fraction"]) * leaves_factor


def rule_params(n_rows: int, n_queries: int, n_features: int, n_labels: int, seed: int, **overrides):
    """LightGBM LambdaMART parameters as functions of the data shape (see Decision.md, 'recipe rules')."""
    # capacity grows with the amount of data: leaves ~ sqrt(rows) / 16, clipped to [7, 63]
    num_leaves = int(np.clip(2 ** round(np.log2(max(n_rows, 1) ** 0.5 / 16.0)) - 1, 7, 63))
    # a leaf must be supported by enough rows: ~ rows / 4000, clipped to [20, 200]
    min_data_in_leaf = int(np.clip(n_rows / 4000.0, 20, 200))
    # wide inputs: sample fewer columns per tree (decorrelates trees, cheaper); ~1.0 for very narrow inputs
    feature_fraction = float(np.clip(1.0 - 0.15 * np.log2(max(n_features, 1) / 32.0), 0.5, 1.0))
    params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
        "num_leaves": num_leaves, "min_data_in_leaf": min_data_in_leaf, "learning_rate": 0.05,
        "feature_fraction": feature_fraction, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 10.0 if n_rows < 50_000 else 1.0, "max_bin": 255,
        "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed, "data_random_seed": seed,
        "deterministic": True, "force_row_wise": True, "num_threads": NUM_THREADS, "verbose": -1,
    }
    params.update(overrides)
    return params


def plan(n_rows: int, n_vali_rows: int, n_queries: int, n_features: int, n_labels: int, seed: int,
         refit: bool = True, **overrides):
    """(params, max_rounds, patience, refit_allowed) for a dataset of this shape. If the time budget would only
    allow few rounds at learning rate 0.05, the rate is doubled rather than under-training."""
    params = rule_params(n_rows, n_queries, n_features, n_labels, seed, **overrides)
    cost = _round_cost_s(n_rows, n_features, params)
    max_rounds = int(FIT_BUDGET_S / cost)
    if max_rounds < 800 and "learning_rate" not in overrides:
        params["learning_rate"] = 0.1
    max_rounds = int(np.clip(max_rounds, 100, 3000))
    patience = int(np.clip(max_rounds // 10, 50, 200))
    return params, max_rounds, patience, refit


class _Deadline:
    """LightGBM callback: stop boosting once wall-clock passes `deadline` (safety net only)."""
    order = 30

    def __init__(self, deadline: float):
        self.deadline = deadline
        self.fired = False

    def __call__(self, env):
        if time.perf_counter() > self.deadline:
            self.fired = True
            import lightgbm as lgb
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list or [])


def fit_arrays(X, y, g, Xv, yv, gv, seed: int = 0, deadline: float = None, refit: bool = True, **overrides):
    """Steps 2-5 on in-memory arrays (g / gv = query group sizes, rows contiguous per query).
    Returns (booster, num_iteration to save, info dict)."""
    import lightgbm as lgb

    t0 = time.perf_counter()
    deadline = deadline if deadline is not None else t0 + TIME_BUDGET_S
    n_rows, D = X.shape
    n_vali = Xv.shape[0]
    n_labels = int(len(np.unique(np.concatenate([y, yv]))))
    params, max_rounds, patience, refit = plan(n_rows, n_vali, len(g), D, n_labels, seed, refit=refit, **overrides)
    info = {"params": {k: params[k] for k in ("num_leaves", "min_data_in_leaf", "learning_rate",
                                              "feature_fraction", "lambda_l2")},
            "max_rounds": max_rounds, "patience": patience}

    # step 3: fit on train, early-stop on vali nDCG@10
    dtrain = lgb.Dataset(X, label=y, group=g, free_raw_data=False)
    have_vali = n_vali > 0 and len(gv) > 0 and yv.max() > 0
    guard = _Deadline(t0 + 0.75 * (deadline - t0))
    callbacks, valid_sets = [guard], []
    if have_vali:
        valid_sets = [lgb.Dataset(Xv, label=yv, group=gv, reference=dtrain)]
        callbacks.append(lgb.early_stopping(patience, first_metric_only=True, verbose=False))
    booster = lgb.train(params, dtrain, num_boost_round=max_rounds, valid_sets=valid_sets, callbacks=callbacks)
    best = booster.best_iteration if (have_vali and booster.best_iteration > 0) else booster.current_iteration()
    best = max(int(best), 1)
    info.update(best_iteration=best, rounds_run=booster.current_iteration(), guard_fired=guard.fired,
                fit_seconds=round(time.perf_counter() - t0, 1))

    # step 4: refit on train + vali, round count scaled to the extra data, if the predicted cost fits. The
    # decision uses the size-based cost model only (deterministic); the wall-clock test is a safety net.
    if refit and have_vali and not guard.fired:
        rounds = int(round(best * (n_rows + n_vali) / n_rows))
        predicted_fit = info["rounds_run"] * _round_cost_s(n_rows, D, params)
        predicted = rounds * _round_cost_s(n_rows + n_vali, D, params)
        if predicted_fit + predicted <= REFIT_TOTAL_BUDGET_S and time.perf_counter() + predicted < deadline:
            t1 = time.perf_counter()
            dall = lgb.Dataset(np.vstack([X, Xv]), label=np.concatenate([y, yv]), group=list(g) + list(gv))
            refit_guard = _Deadline(deadline)
            final = lgb.train(params, dall, num_boost_round=rounds, callbacks=[refit_guard])
            info.update(refit_rounds=rounds, refit_seconds=round(time.perf_counter() - t1, 1),
                        refit_guard_fired=refit_guard.fired)
            return final, final.current_iteration(), info
        info["refit_skipped_predicted_s"] = round(predicted, 1)
    return booster, best, info


def train(train_path: str, vali_path: str, out_model_path: str, seed: int = 0) -> None:
    t0 = time.perf_counter()
    (X, y, g), (Xv, yv, gv) = load_pair(train_path, vali_path)
    booster, n_iter, _info = fit_arrays(X, y, g, Xv, yv, gv, seed=seed, deadline=t0 + TIME_BUDGET_S)
    booster.save_model(out_model_path, num_iteration=n_iter)
