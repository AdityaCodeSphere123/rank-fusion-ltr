"""The artifact format staff use for Track P (harness/neural.py): bounded plain MLPs only."""
import numpy as np
import pytest
import torch

from harness.neural import ScoringNetwork, load_scorer, save_scorer
from harness.preprocessing import Standardizer


def _std(d):
    return Standardizer().fit(np.random.RandomState(0).randn(20, d).astype(np.float32))


@pytest.mark.parametrize("hidden", [64, [128, 64], [256, 256, 256]])
def test_roundtrip_within_bounds(tmp_path, hidden):
    net = ScoringNetwork(10, hidden)
    save_scorer(str(tmp_path / "m.pt"), net, _std(10))
    X = np.random.RandomState(1).randn(5, 10).astype(np.float32)
    assert load_scorer(str(tmp_path / "m.pt"))(X).shape == (5,)


@pytest.mark.parametrize("hidden", [[64, 64, 64, 64], 512, [0]])
def test_constructor_rejects_out_of_bounds(hidden):
    with pytest.raises(ValueError):
        ScoringNetwork(10, hidden)


def test_loader_rejects_tampered_artifacts(tmp_path):
    net = ScoringNetwork(10, 32)
    # config claims a 4-layer net
    torch.save({"config": {"n_features": 10, "hidden": [32, 32, 32, 32]}, "state_dict": net.state_dict(),
                "standardizer": _std(10).to_dict()}, tmp_path / "a.pt")
    with pytest.raises(ValueError):
        load_scorer(str(tmp_path / "a.pt"))
    # extra parameters smuggled into the state_dict
    sd = dict(net.state_dict())
    sd["extra.weight"] = torch.zeros(3)
    torch.save({"config": {"n_features": 10, "hidden": [32]}, "state_dict": sd,
                "standardizer": _std(10).to_dict()}, tmp_path / "b.pt")
    with pytest.raises(RuntimeError):
        load_scorer(str(tmp_path / "b.pt"))
