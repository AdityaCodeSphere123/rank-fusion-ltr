"""Pointwise (squared error) and pairwise (RankNet) scorers on the same MLP.

ranknet_loss is the plain unweighted loss. LambdaRank weights (|delta nDCG|)
are applied only in weighted_ranknet_loss, and only when train_model is called
with pair_weighting="delta_ndcg".
"""
import copy
import random
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from harness.metrics import mean_ndcg_from_scores
from harness.preprocessing import Standardizer
from harness.neural import MAX_HIDDEN_LAYERS, MAX_WIDTH, ScoringNetwork, load_scorer, make_scorer, save_scorer  # noqa: F401


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pointwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean squared error. scores and labels are 1-D and the same length."""
    return torch.mean((scores - labels.to(scores.dtype)) ** 2)


def make_pairs(qids, labels, pairs_per_query: int, rng: np.random.RandomState) -> np.ndarray:
    """Within-query pairs as (P, 2) row indices: more relevant, then less relevant.

    Same query only, unequal labels only, at most pairs_per_query per query,
    sampled without replacement from rng.
    """
    labels = np.asarray(labels)
    out = []
    for rows in _query_rows(qids):
        if len(rows) < 2:
            continue
        lab = labels[rows]
        i, j = np.triu_indices(len(rows), k=1)
        keep = lab[i] != lab[j]
        i, j = i[keep], j[keep]
        if len(i) == 0:
            continue
        if len(i) > pairs_per_query:
            pick = rng.choice(len(i), size=pairs_per_query, replace=False)
            i, j = i[pick], j[pick]
        first_better = lab[i] > lab[j]
        hi = np.where(first_better, i, j)
        lo = np.where(first_better, j, i)
        out.append(np.stack([rows[hi], rows[lo]], axis=1))
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(out).astype(np.int64)


def _query_rows(qids) -> List[np.ndarray]:
    """Row indices per query, in order of first appearance."""
    groups: Dict = {}
    for r, q in enumerate(qids.tolist() if isinstance(qids, np.ndarray) else qids):
        groups.setdefault(q, []).append(r)
    return [np.asarray(rows, dtype=np.int64) for rows in groups.values()]


def ranknet_loss(s_preferred: torch.Tensor, s_other: torch.Tensor) -> torch.Tensor:
    """Mean of log(1 + exp(-(s_i - s_j))). logaddexp avoids a raw exp."""
    d = s_preferred - s_other
    return torch.logaddexp(torch.zeros_like(d), -d).mean()


def pointwise_step(network: nn.Module, optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    optimizer.zero_grad()
    loss = pointwise_loss(network(x), y)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def pairwise_step(network: nn.Module, optimizer, x_preferred: torch.Tensor, x_other: torch.Tensor) -> float:
    """One step on a batch of pairs. Both sides go through the same network."""
    optimizer.zero_grad()
    loss = ranknet_loss(network(x_preferred), network(x_other))
    loss.backward()
    optimizer.step()
    return float(loss.item())


def weighted_ranknet_loss(s_preferred: torch.Tensor, s_other: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """RankNet weighted by a non-negative weight per pair. Equal weights match ranknet_loss."""
    d = s_preferred - s_other
    per_pair = torch.logaddexp(torch.zeros_like(d), -d)
    return (weights * per_pair).sum() / weights.sum().clamp_min(1e-12)


def delta_ndcg_weights(scores: torch.Tensor, labels: torch.Tensor, pairs: torch.Tensor, k: int = 10,
                       trunc: Optional[int] = None) -> torch.Tensor:
    """|delta nDCG@k| of swapping each pair, from the current scores. No gradient.

    Gain is 2^label - 1, discount 1/log2(1 + rank), divided by the query's ideal DCG@k.
    With trunc set, discounts past that rank are 0.
    """
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


def _sample_query_pairs(labels: np.ndarray, scores: Optional[np.ndarray], pairs_per_query: int,
                        rng: np.random.RandomState, mode: str) -> np.ndarray:
    """Pairs inside one query. mode 'gain' draws by |2^yi-2^yj|; 'hard' also
    upweights pairs the current scores get wrong."""
    n = len(labels)
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)
    i, j = np.triu_indices(n, k=1)
    keep = labels[i] != labels[j]
    i, j = i[keep], j[keep]
    if len(i) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    first_better = labels[i] > labels[j]
    hi = np.where(first_better, i, j)
    lo = np.where(first_better, j, i)
    if len(hi) > pairs_per_query:
        gain = np.abs(np.power(2.0, labels[hi].astype(np.float64)) - np.power(2.0, labels[lo].astype(np.float64)))
        if mode == "hard" and scores is not None:
            margin = np.clip(scores[lo] - scores[hi], -20.0, 20.0)
            gain = gain * (1.0 / (1.0 + np.exp(-margin)))
        gain = np.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(gain.sum())
        if total <= 0 or int((gain > 0).sum()) < pairs_per_query:
            pick = rng.choice(len(hi), size=pairs_per_query, replace=False)
        else:
            pick = rng.choice(len(hi), size=pairs_per_query, replace=False, p=gain / total)
        hi, lo = hi[pick], lo[pick]
    return np.stack([hi, lo], axis=1).astype(np.int64)


def approx_ndcg_loss(scores: torch.Tensor, labels: torch.Tensor, alpha: float = 8.0) -> torch.Tensor:
    """1 - smooth nDCG. Rank of a doc is 1 plus a soft count of docs scored above it."""
    gains = torch.pow(2.0, labels.to(scores.dtype)) - 1.0
    if float(gains.sum()) <= 0:
        return scores.sum() * 0.0
    diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    above = torch.sigmoid(-alpha * diff)
    above = above * (1.0 - torch.eye(scores.shape[0], device=scores.device, dtype=scores.dtype))
    ranks = 1.0 + above.sum(dim=1)
    dcg = torch.sum(gains / torch.log2(1.0 + ranks))
    ideal = torch.sort(gains, descending=True).values
    positions = torch.arange(1, ideal.shape[0] + 1, device=scores.device, dtype=scores.dtype)
    idcg = torch.sum(ideal / torch.log2(1.0 + positions))
    return 1.0 - dcg / idcg.clamp_min(1e-12)


def lambdarank_query_step(network: nn.Module, optimizer, Xt: torch.Tensor, yt: torch.Tensor,
                          query_rows: List[np.ndarray], pairs_per_query: int, rng: np.random.RandomState,
                          trunc: Optional[int] = None, pair_sampling: str = "uniform") -> float:
    """One step on whole queries: pairs from make_pairs, weights from the current scores."""
    sizes = [len(r) for r in query_rows]
    flat = torch.from_numpy(np.concatenate(query_rows))
    local_q = np.repeat(np.arange(len(query_rows)), sizes)
    labels = yt[flat]
    if pair_sampling == "uniform":
        pairs = make_pairs(local_q, labels.numpy(), pairs_per_query, rng)
    else:
        with torch.no_grad():
            scored = network(Xt[flat]).detach().numpy()
        blocks = []
        offset = 0
        lab_np = labels.numpy()
        for rows in query_rows:
            n = len(rows)
            local = _sample_query_pairs(lab_np[offset:offset + n], scored[offset:offset + n],
                                        pairs_per_query, rng, pair_sampling)
            if len(local):
                blocks.append(local + offset)
            offset += n
        pairs = np.concatenate(blocks) if blocks else np.zeros((0, 2), dtype=np.int64)
    if len(pairs) == 0:
        return 0.0
    optimizer.zero_grad()
    scores = network(Xt[flat])
    pairs_t = torch.from_numpy(pairs)
    weights = torch.zeros(len(pairs), dtype=torch.float32)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    pair_q = local_q[pairs[:, 0]]
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


def approx_ndcg_query_step(network: nn.Module, optimizer, Xt: torch.Tensor, yt: torch.Tensor,
                           query_rows: List[np.ndarray], alpha: float) -> float:
    """One step of smooth nDCG on whole queries. ranknet_loss is not used."""
    flat = torch.from_numpy(np.concatenate(query_rows))
    labels = yt[flat]
    optimizer.zero_grad()
    scores = network(Xt[flat])
    loss = scores.new_zeros(())
    offset = 0
    used = 0
    for rows in query_rows:
        n = len(rows)
        chunk = labels[offset:offset + n]
        if n > 1 and float(chunk.max()) > 0:
            loss = loss + approx_ndcg_loss(scores[offset:offset + n], chunk, alpha=alpha)
            used += 1
        offset += n
    if used == 0:
        return 0.0
    loss = loss / used
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
    optimizer.step()
    return float(loss.item())


def dev_ndcg(scorer: Callable[[np.ndarray], np.ndarray], X, y, qids) -> float:
    """Mean nDCG@10. Scores whole queries, then sorts."""
    return mean_ndcg_from_scores(scorer(X), y, qids, k=10)


def _standardizer_for(X: np.ndarray, scale: str) -> Standardizer:
    """'robust' uses median and IQR so a few huge features do not set the column scale."""
    if scale == "standard":
        return Standardizer().fit(X)
    if scale != "robust":
        raise ValueError("scale must be 'standard' or 'robust'")
    med = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    iqr = np.where(iqr < 1e-8, 1.0, iqr)
    return Standardizer(med.astype(np.float64), (iqr / 1.349).astype(np.float64))


def train_model(kind: str, train, dev, *, seed: int = 0, epochs: int = 20, batch_size: int = 1024,
                lr: float = 1e-3, pairs_per_query: int = 300, patience: int = 3, hidden=64,
                verbose: bool = False, pair_weighting: str = "none", queries_per_batch: int = 16,
                ndcg_trunc: Optional[int] = None, weight_decay: float = 0.0,                 lr_decay: float = 1.0,
                scale: str = "standard", pair_sampling: str = "uniform", approx_alpha: float = 8.0,
                swa_start: int = 0):
    """Train a pointwise or pairwise scorer. Restores the best dev-nDCG weights.

    dev is the early-stopping monitor; pass a query-disjoint slice of TRAIN so the
    real DEV split stays unseen. pair_weighting="delta_ndcg" (pairwise only) uses
    minibatches of whole queries and |delta nDCG| weights. pair_sampling="gain" or
    "hard" draws those pairs by label gain instead of uniformly. pair_weighting=
    "approx_ndcg" minimises a smooth nDCG on whole queries. The standardizer is
    fit on train only.
    """
    if kind not in ("pointwise", "pairwise"):
        raise ValueError("kind must be 'pointwise' or 'pairwise'")
    if pair_weighting not in ("none", "delta_ndcg", "approx_ndcg"):
        raise ValueError("pair_weighting must be 'none', 'delta_ndcg', or 'approx_ndcg'")
    if pair_sampling not in ("uniform", "gain", "hard"):
        raise ValueError("pair_sampling must be 'uniform', 'gain', or 'hard'")
    X, y, q = train
    Xd, yd, qd = dev
    set_seed(seed)
    standardizer = _standardizer_for(X, scale)
    Xt = torch.from_numpy(standardizer.transform(X))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    network = ScoringNetwork(X.shape[1], hidden)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.RandomState(seed)
    lambdarank = kind == "pairwise" and pair_weighting == "delta_ndcg"
    listwise = kind == "pairwise" and pair_weighting == "approx_ndcg"
    query_rows = _query_rows(q) if (lambdarank or listwise) else None

    history = []
    best_ndcg, best_state, since_best = -np.inf, None, 0
    swa_state, swa_n = None, 0
    for epoch in range(1, epochs + 1):
        network.train()
        losses = []
        if lambdarank or listwise:
            order = rng.permutation(len(query_rows))
            for b in range(0, len(order), queries_per_batch):
                batch = [query_rows[i] for i in order[b:b + queries_per_batch]]
                if listwise:
                    losses.append(approx_ndcg_query_step(network, optimizer, Xt, yt, batch, approx_alpha))
                else:
                    losses.append(lambdarank_query_step(
                        network, optimizer, Xt, yt, batch, pairs_per_query, rng,
                        trunc=ndcg_trunc, pair_sampling=pair_sampling))
        elif kind == "pairwise":
            pairs = make_pairs(q, y, pairs_per_query, rng)
            pairs = pairs[rng.permutation(len(pairs))]
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
        if swa_start and epoch >= swa_start:
            state = network.state_dict()
            if swa_state is None:
                swa_state = {k: v.detach().float().clone() for k, v in state.items()}
                swa_n = 1
            else:
                swa_n += 1
                for k, v in state.items():
                    swa_state[k] += (v.detach().float() - swa_state[k]) / swa_n
        if lr_decay != 1.0:
            for group in optimizer.param_groups:
                group["lr"] *= lr_decay
        if since_best >= patience and (not swa_start or epoch >= swa_start):
            break
    if swa_state is not None:
        network.load_state_dict(swa_state)
        swa_nd = dev_ndcg(make_scorer(network, standardizer), Xd, yd, qd)
        history.append({"epoch": 0, "train_loss": 0.0, "dev_ndcg@10": swa_nd, "swa": True})
        if verbose:
            print(f"  [{kind}] swa over {swa_n} epochs  monitor nDCG@10 {swa_nd:.4f}", flush=True)
        if swa_nd > best_ndcg:
            best_ndcg, best_state = swa_nd, copy.deepcopy(network.state_dict())
    if best_state is not None:
        network.load_state_dict(best_state)
    return network, standardizer, history
