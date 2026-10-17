"""
submission/pointwise_pairwise.py -- the pointwise / pairwise models (assignment Sections 4.1, 4.3 and 7).
Read docs/RANKNET_PRIMER.md first; no prior neural-network experience is assumed.

Both models use the SAME scoring network (harness/neural.py: ScoringNetwork), so the only difference between
them is the loss you write:
  pointwise: squared error between the network's score and the graded label
  pairwise : RankNet -- for a pair (i more relevant than j), loss =
             log(1 + exp(-(s_i - s_j))), i.e. -log sigmoid(s_i - s_j)

PROVIDED : ScoringNetwork, make_scorer, save_scorer, load_scorer (harness/neural.py), dev_ndcg, and the
           overall shape of train_model().
YOU WRITE: every `TODO(...)` below (roughly 100-150 lines in total):
  TODO(1) set_seed            TODO(2) pointwise_loss     TODO(3) make_pairs
  TODO(4) ranknet_loss        TODO(5) pointwise_step / pairwise_step
  TODO(6) the training loop inside train_model(): minibatching, per-epoch dev
          nDCG@10, early stopping.
ALLOWED  : PyTorch tensors, autograd, nn layers, activations, optimizers
           (Adam/SGD), random-seed utilities.
NOT ALLOWED: any ready-made ranking loss or LTR trainer (allRank, TF-Ranking,
           PT-Ranking, RankLib, ...) and LightGBM/XGBoost for these two
           models. torch's MarginRankingLoss is a HINGE loss, not RankNet.

TWO JOBS, KEEP THEM SEPARATE.
  1. Correctness (hand-checked): the functions below are tested by NAME and SIGNATURE (docs/GRADING.md) against
     their textbook definitions. Keep them exactly as declared -- in particular ranknet_loss must stay the plain,
     unweighted RankNet loss.
  2. Track P (leaderboard): your submitted pairwise_model.pt is scored on held-out data. To make it rank better you
     MAY change, in your own extra functions / train_model options: how pairs are sampled (e.g. favour large
     label gaps or top-ranked documents), pairs per query, epochs, learning rate, batch size, the MLP's width and
     depth (at most MAX_HIDDEN_LAYERS hidden layers of at most MAX_WIDTH units: pass e.g. hidden=[128, 64]), and
     per-pair weights such as |delta nDCG| (which turns RankNet into LambdaRank) -- write such weighting as a
     separate function (e.g. weighted_ranknet_loss), never inside ranknet_loss.
     You may NOT: train on anything but relevance labels (no distillation from LambdaMART or any other model's
     scores), or change the artifact format: staff load it with their own copy of harness/neural.py, and an
     artifact that breaks the bounds scores 0 on Track P. Declare the splits you trained and tuned on.

Artifacts to ship in submission/: pointwise_model.pt and pairwise_model.pt (see scripts/train_neural.py).
"""
import copy
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from harness.metrics import mean_ndcg_from_scores
from harness.preprocessing import Standardizer
# PROVIDED, and also what staff use to load your artifacts (assignment Section 7, Track P):
from harness.neural import MAX_HIDDEN_LAYERS, MAX_WIDTH, ScoringNetwork, load_scorer, make_scorer, save_scorer  # noqa: F401


def set_seed(seed: int) -> None:
    """TODO(1): make training reproducible -- seed Python's `random`, NumPy,
    and torch (torch.manual_seed). Staff check two runs with the same seed
    give identical weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pointwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """TODO(2): mean squared error between predicted scores and graded labels
    (both 1-D tensors of equal length). Returns a scalar tensor.
    Example: scores=[1, 2], labels=[0, 2] -> ((1-0)^2 + (2-2)^2) / 2 = 0.5"""
    return torch.mean((scores - labels.to(scores.dtype)) ** 2)


def make_pairs(qids, labels, pairs_per_query: int, rng: np.random.RandomState) -> np.ndarray:
    """TODO(3): build within-query preference pairs.

    qids  : one query id per ROW (any hashable; rows of a query may be anywhere)
    labels: one graded relevance label per row
    Return an int array of shape (P, 2) of ROW INDICES: column 0 is the
    strictly MORE relevant document of the pair, column 1 the less relevant one.
    Rules (all checked by staff): (a) both rows from the SAME query; (b) only
    pairs with different labels (ties carry no preference); (c) oriented
    more-relevant first; (d) at most `pairs_per_query` pairs per query -- sample
    without replacement using `rng` (a NumPy RandomState) so runs are
    reproducible. Never build the pairs of the whole TRAIN split at once.
    Example: qids=[a,a,a,b,b], labels=[2,0,1,1,1], pairs_per_query large ->
    the 3 pairs (0,1), (0,2), (2,1); query b has only ties -> no pairs."""
    labels = np.asarray(labels)
    out = []
    for rows in _query_rows(qids):                     # one query at a time: never all pairs of the split
        if len(rows) < 2:
            continue
        lab = labels[rows]
        i, j = np.triu_indices(len(rows), k=1)         # every unordered pair of this query, once
        keep = lab[i] != lab[j]                        # (b) ties carry no preference
        i, j = i[keep], j[keep]
        if len(i) == 0:
            continue
        if len(i) > pairs_per_query:                   # (d) cap, sampled without replacement
            pick = rng.choice(len(i), size=pairs_per_query, replace=False)
            i, j = i[pick], j[pick]
        first_better = lab[i] > lab[j]                 # (c) more relevant document first
        hi = np.where(first_better, i, j)
        lo = np.where(first_better, j, i)
        out.append(np.stack([rows[hi], rows[lo]], axis=1))   # (a) both indices come from this query's rows
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(out).astype(np.int64)


def _query_rows(qids) -> List[np.ndarray]:
    """Row indices of each query, queries in order of first appearance (rows need not be contiguous)."""
    groups: Dict = {}
    for r, q in enumerate(qids.tolist() if isinstance(qids, np.ndarray) else qids):
        groups.setdefault(q, []).append(r)
    return [np.asarray(rows, dtype=np.int64) for rows in groups.values()]


def ranknet_loss(s_preferred: torch.Tensor, s_other: torch.Tensor) -> torch.Tensor:
    """TODO(4): RankNet loss. For each pair, log(1 + exp(-(s_i - s_j))), where
    s_preferred = s_i is the score of the more relevant document. Average over
    the pairs (scalar tensor). Use a numerically stable form -- softplus(-d) or
    torch.logaddexp(0, -d) with d = s_i - s_j -- not a raw exp.
    Example: s_preferred=[2, 0], s_other=[0, 0] ->
    mean(log(1+e^-2), log 2) = mean(0.126928, 0.693147) = 0.410038"""
    d = s_preferred - s_other
    return torch.logaddexp(torch.zeros_like(d), -d).mean()


def pointwise_step(network: nn.Module, optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    """TODO(5): ONE optimisation step on a minibatch of documents: zero the
    gradients, forward pass, pointwise_loss, loss.backward(), optimizer.step().
    Return the loss as a float. (Forgetting optimizer.zero_grad() is the classic
    bug: gradients silently accumulate across batches.)"""
    optimizer.zero_grad()
    loss = pointwise_loss(network(x), y)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def pairwise_step(network: nn.Module, optimizer, x_preferred: torch.Tensor, x_other: torch.Tensor) -> float:
    """TODO(5): ONE optimisation step on a minibatch of PAIRS. x_preferred /
    x_other are (batch, n_features) feature tensors of each pair's more / less
    relevant document. Run both through the SAME network (shared weights),
    ranknet_loss on the two score vectors, backward, step. Return the loss as a
    float. Staff check one step on a pair raises s_preferred - s_other."""
    optimizer.zero_grad()
    loss = ranknet_loss(network(x_preferred), network(x_other))
    loss.backward()
    optimizer.step()
    return float(loss.item())


# ---- Track P extension: LambdaRank-style pair weights (NOT part of the graded textbook functions) -----------
#
# Plain RankNet treats every preference pair alike. LambdaRank (Burges 2010, MSR-TR-2010-82) multiplies each
# pair's RankNet loss by |delta nDCG|: how much nDCG would change if the two documents swapped places under the
# CURRENT scores. To have fresh ranks for that, a minibatch here is a set of whole QUERIES: all their documents go
# through the network, within-query pairs are formed with make_pairs, and the weights come from the batch's own
# (detached) scores. ranknet_loss above stays the plain, unweighted loss.

def weighted_ranknet_loss(s_preferred: torch.Tensor, s_other: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """sum_p w_p * log(1 + exp(-(s_i - s_j))) / sum_p w_p: RankNet with a non-negative weight per pair.
    With all weights equal this equals ranknet_loss."""
    d = s_preferred - s_other
    per_pair = torch.logaddexp(torch.zeros_like(d), -d)
    return (weights * per_pair).sum() / weights.sum().clamp_min(1e-12)


def delta_ndcg_weights(scores: torch.Tensor, labels: torch.Tensor, pairs: torch.Tensor, k: int = 10,
                       trunc: Optional[int] = None) -> torch.Tensor:
    """|delta nDCG@k| of swapping each pair (rows of `pairs` index into ONE query's `scores` / `labels`), with
    ranks taken from the current scores (ties by row order). Gain 2^label - 1, discount 1/log2(1 + rank); with
    `trunc` set, discounts below rank `trunc` are 0 (pairs entirely below it get weight 0). Normalised by the
    query's ideal DCG@k. Computed without gradient: the weights are constants for the backward pass."""
    with torch.no_grad():
        n = scores.shape[0]
        order = torch.argsort(-scores, stable=True)
        ranks = torch.empty(n, dtype=torch.float64)
        ranks[order] = torch.arange(1, n + 1, dtype=torch.float64)
        disc = 1.0 / torch.log2(1.0 + ranks)
        if trunc is not None:
            disc = torch.where(ranks <= trunc, disc, torch.zeros_like(disc))
        gain = torch.pow(2.0, labels.to(torch.float64)) - 1.0
        ideal = torch.sort(gain, descending=True).values[:k]
        idcg = (ideal / torch.log2(torch.arange(2, ideal.shape[0] + 2, dtype=torch.float64))).sum()
        if idcg <= 0:
            return torch.zeros(pairs.shape[0], dtype=torch.float32)
        i, j = pairs[:, 0], pairs[:, 1]
        return ((gain[i] - gain[j]).abs() * (disc[i] - disc[j]).abs() / idcg).to(torch.float32)


def lambdarank_query_step(network: nn.Module, optimizer, Xt: torch.Tensor, yt: torch.Tensor,
                          query_rows: List[np.ndarray], pairs_per_query: int, rng: np.random.RandomState,
                          trunc: Optional[int] = None) -> float:
    """ONE optimisation step on a minibatch of whole queries: score all their documents with the shared network,
    form within-query pairs with make_pairs (capped per query, sampled with `rng`), weight each pair by
    delta_ndcg_weights under the current scores, and minimise weighted_ranknet_loss."""
    sizes = [len(r) for r in query_rows]
    flat = torch.from_numpy(np.concatenate(query_rows))
    local_q = np.repeat(np.arange(len(query_rows)), sizes)
    labels = yt[flat]
    pairs = make_pairs(local_q, labels.numpy(), pairs_per_query, rng)   # indices into `flat`
    if len(pairs) == 0:
        return 0.0
    optimizer.zero_grad()
    scores = network(Xt[flat])
    pairs_t = torch.from_numpy(pairs)
    weights = torch.zeros(len(pairs), dtype=torch.float32)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    pair_q = local_q[pairs[:, 0]]                      # make_pairs emits each query's pairs contiguously
    bounds = np.searchsorted(pair_q, np.arange(len(query_rows) + 1))
    for qi in range(len(query_rows)):
        p0, p1 = bounds[qi], bounds[qi + 1]
        if p0 == p1:
            continue
        a, b = starts[qi], starts[qi + 1]
        weights[p0:p1] = delta_ndcg_weights(scores[a:b].detach(), labels[a:b], pairs_t[p0:p1] - a, trunc=trunc)
    if float(weights.sum()) <= 0:
        return 0.0
    loss = weighted_ranknet_loss(scores[pairs_t[:, 0]], scores[pairs_t[:, 1]], weights)
    loss.backward()
    optimizer.step()
    return float(loss.item())


# ---- provided helpers -------------------------------------------------------

def dev_ndcg(scorer: Callable[[np.ndarray], np.ndarray], X, y, qids) -> float:
    """PROVIDED. Mean nDCG@10 of `scorer` over the queries of (X, y, qids). It scores
    WHOLE queries and sorts them -- evaluation is not batched like training."""
    return mean_ndcg_from_scores(scorer(X), y, qids, k=10)


# ---- training ---------------------------------------------------------------

def train_model(kind: str, train, dev, *, seed: int = 0, epochs: int = 20, batch_size: int = 1024,
                lr: float = 1e-3, pairs_per_query: int = 300, patience: int = 3, hidden=64,
                verbose: bool = False, pair_weighting: str = "none", queries_per_batch: int = 16,
                ndcg_trunc: Optional[int] = None, weight_decay: float = 0.0, lr_decay: float = 1.0):
    """Train a 'pointwise' or 'pairwise' scorer. train / dev are (X_raw, y, qids)
    tuples. Returns (network, standardizer, history) with the best-dev-nDCG
    weights restored. The setup below is PROVIDED; the loop is yours (TODO(6)).

    `dev` is whatever split early stopping monitors: pass a query-disjoint slice of TRAIN (harness.cv) to keep
    the real DEV split unseen. Track P options (the defaults give the textbook RankNet / pointwise setup):
      pair_weighting="delta_ndcg" (pairwise only): minibatches of `queries_per_batch` whole queries, pairs
          (at most `pairs_per_query` per query) weighted by |delta nDCG@10| under the current scores, i.e.
          LambdaRank (lambdarank_query_step). `batch_size` is then unused; `ndcg_trunc` optionally zeroes the
          discount below that rank.
      weight_decay: Adam L2 penalty.   lr_decay: the learning rate is multiplied by this after every epoch."""
    if kind not in ("pointwise", "pairwise"):
        raise ValueError("kind must be 'pointwise' or 'pairwise'")
    if pair_weighting not in ("none", "delta_ndcg"):
        raise ValueError("pair_weighting must be 'none' or 'delta_ndcg'")
    X, y, q = train
    Xd, yd, qd = dev
    set_seed(seed)                                    # TODO(1) makes this meaningful
    standardizer = Standardizer().fit(X)              # fit on TRAIN only
    Xt = torch.from_numpy(standardizer.transform(X))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    network = ScoringNetwork(X.shape[1], hidden)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.RandomState(seed)
    lambdarank = kind == "pairwise" and pair_weighting == "delta_ndcg"
    query_rows = _query_rows(q) if lambdarank else None

    history = []
    best_ndcg, best_state, since_best = -np.inf, None, 0
    for epoch in range(1, epochs + 1):
        network.train()
        losses = []
        if lambdarank:
            order = rng.permutation(len(query_rows))             # reshuffle the queries every epoch
            for b in range(0, len(order), queries_per_batch):
                batch = [query_rows[i] for i in order[b:b + queries_per_batch]]
                losses.append(lambdarank_query_step(network, optimizer, Xt, yt, batch, pairs_per_query, rng,
                                                    trunc=ndcg_trunc))
        elif kind == "pairwise":
            pairs = make_pairs(q, y, pairs_per_query, rng)       # a fresh sample of pairs every epoch
            pairs = pairs[rng.permutation(len(pairs))]           # so one query's pairs do not sit together
            for b in range(0, len(pairs), batch_size):
                pb = torch.from_numpy(pairs[b:b + batch_size])
                losses.append(pairwise_step(network, optimizer, Xt[pb[:, 0]], Xt[pb[:, 1]]))
        else:
            rows = rng.permutation(len(yt))
            for b in range(0, len(rows), batch_size):
                rb = torch.from_numpy(rows[b:b + batch_size])
                losses.append(pointwise_step(network, optimizer, Xt[rb], yt[rb]))

        nd = dev_ndcg(make_scorer(network, standardizer), Xd, yd, qd)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else 0.0,
                        "dev_ndcg@10": nd})
        if verbose:
            print(f"  [{kind}] epoch {epoch:2d}  train_loss {history[-1]['train_loss']:.5f}  "
                  f"monitor nDCG@10 {nd:.4f}", flush=True)
        if nd > best_ndcg:
            best_ndcg, best_state, since_best = nd, copy.deepcopy(network.state_dict()), 0
        else:
            since_best += 1
            if since_best >= patience:
                break
        if lr_decay != 1.0:
            for group in optimizer.param_groups:
                group["lr"] *= lr_decay
    if best_state is not None:
        network.load_state_dict(best_state)
    return network, standardizer, history
