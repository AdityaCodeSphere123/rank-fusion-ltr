"""Rank fusion over the named base-ranker scores. Higher is better on every channel.
Ties keep dict insertion order (the candidate order).
"""
from typing import Dict, Iterable, List, Optional


def _channels(base_scores: Dict[str, Dict[str, float]], channels: Optional[Iterable[str]]) -> List[str]:
    if channels is not None:
        return list(channels)
    first = next(iter(base_scores.values()), {})
    return list(first.keys())


def rrf_scores(base_scores: Dict[str, Dict[str, float]], channels: Optional[Iterable[str]] = None,
               c: float = 60.0) -> Dict[str, float]:
    """score(d) = sum_ch 1 / (c + rank_ch(d)), ranks 1-based, best score = rank 1."""
    fused = {d: 0.0 for d in base_scores}
    for ch in _channels(base_scores, channels):
        for d, r in _channel_ranks(base_scores, ch).items():
            fused[d] += 1.0 / (c + r)
    return fused


def _channel_ranks(base_scores: Dict[str, Dict[str, float]], channel: str) -> Dict[str, int]:
    """1-based rank in one channel. sorted() is stable, so ties keep candidate order."""
    ordered = sorted(base_scores, key=lambda d: -base_scores[d][channel])
    return {d: r for r, d in enumerate(ordered, start=1)}


def weighted_rrf_scores(base_scores: Dict[str, Dict[str, float]], weights: Dict[str, float],
                        c: float = 60.0) -> Dict[str, float]:
    """RRF with a weight per channel. Weight 0 skips the channel; all weights 1 is rrf_scores."""
    fused = {d: 0.0 for d in base_scores}
    for ch, w in weights.items():
        if w == 0:
            continue
        for d, r in _channel_ranks(base_scores, ch).items():
            fused[d] += w / (c + r)
    return fused


def minmax_normalize(values: Dict[str, float]) -> Dict[str, float]:
    """Min-max to [0, 1]. A constant channel becomes all zeros."""
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {d: 0.0 for d in values}
    return {d: (v - lo) / (hi - lo) for d, v in values.items()}


def combsum_scores(base_scores: Dict[str, Dict[str, float]],
                   channels: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """Sum of per-channel min-max scores over the candidate pool."""
    normalised = _normalised_channels(base_scores, channels)
    return {d: sum(norm[d] for norm in normalised.values()) for d in base_scores}


def _normalised_channels(base_scores: Dict[str, Dict[str, float]],
                         channels: Optional[Iterable[str]]) -> Dict[str, Dict[str, float]]:
    return {ch: minmax_normalize({d: s[ch] for d, s in base_scores.items()})
            for ch in _channels(base_scores, channels)}


def combmnz_scores(base_scores: Dict[str, Dict[str, float]],
                   channels: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """CombSUM times the number of channels with normalised score > 0.

    Every candidate has a raw score on every channel, so "appears in the list"
    is defined as a positive normalised score.
    """
    normalised = _normalised_channels(base_scores, channels)
    fused = {}
    for d in base_scores:
        vals = [norm[d] for norm in normalised.values()]
        fused[d] = sum(vals) * sum(1 for v in vals if v > 0)
    return fused
