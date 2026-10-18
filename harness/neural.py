"""
harness/neural.py -- the scoring network and the artifact format for pointwise_model.pt / pairwise_model.pt
(assignment Sections 4.3 and 7, Track P).

Staff load your .pt artifacts with THIS file (their own pristine copy, not yours), so:
  * the network is always a plain MLP: n_features -> hidden layers with ReLU -> 1 score;
  * at most MAX_HIDDEN_LAYERS hidden layers, each at most MAX_WIDTH units;
  * the artifact holds only {config, state_dict, standardizer} and loads with torch.load(weights_only=True).
An artifact that does not load under these rules scores 0 on Track P.
"""
from typing import Callable, List, Sequence, Union

import numpy as np
import torch
from torch import nn

from harness.preprocessing import Standardizer

MAX_HIDDEN_LAYERS = 3
MAX_WIDTH = 256


def _as_sizes(hidden: Union[int, Sequence[int]]) -> List[int]:
    sizes = [int(hidden)] if isinstance(hidden, (int, np.integer)) else [int(h) for h in hidden]
    if not 1 <= len(sizes) <= MAX_HIDDEN_LAYERS or any(not 1 <= h <= MAX_WIDTH for h in sizes):
        raise ValueError(f"hidden layer sizes {sizes} outside the allowed bounds: 1-{MAX_HIDDEN_LAYERS} layers "
                         f"of 1-{MAX_WIDTH} units")
    return sizes


class ScoringNetwork(nn.Module):
    """n_features -> hidden layer(s) with ReLU -> 1 score. `hidden` is one width (e.g. 64, the default) or a list
    of up to MAX_HIDDEN_LAYERS widths (e.g. [128, 64])."""

    def __init__(self, n_features: int, hidden: Union[int, Sequence[int]] = 64):
        super().__init__()
        self.hidden_sizes = _as_sizes(hidden)
        layers, width = [], n_features
        for h in self.hidden_sizes:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_scorer(network: nn.Module, standardizer: Standardizer) -> Callable[[np.ndarray], np.ndarray]:
    """Raw feature matrix -> numpy score vector (evaluation mode, no gradients)."""
    def scorer(X_raw: np.ndarray) -> np.ndarray:
        network.eval()
        with torch.no_grad():
            return network(torch.from_numpy(standardizer.transform(np.asarray(X_raw, dtype=np.float32)))).numpy()
    return scorer


def save_scorer(path: str, network: ScoringNetwork, standardizer: Standardizer, hidden=None) -> None:
    """Write the artifact. `hidden` is accepted for backward compatibility and ignored: the sizes come from the
    network itself."""
    torch.save({
        "config": {"n_features": network.net[0].in_features, "hidden": list(network.hidden_sizes)},
        "state_dict": network.state_dict(),
        "standardizer": standardizer.to_dict(),
    }, path)


def load_scorer(path: str) -> Callable[[np.ndarray], np.ndarray]:
    """Load an artifact written by save_scorer; raises if it breaks the bounds or does not match a plain MLP."""
    blob = torch.load(path, weights_only=True)
    cfg = blob["config"]
    network = ScoringNetwork(int(cfg["n_features"]), cfg["hidden"])
    network.load_state_dict(blob["state_dict"], strict=True)
    return make_scorer(network, Standardizer.from_dict(blob["standardizer"]))
