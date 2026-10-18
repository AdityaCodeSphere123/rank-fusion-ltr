"""SELF-CHECK for the parts YOU write (hand-checkable tests of assignment Section 4.3). These FAIL until
your TODOs are filled in; staff run a stricter version of the same checks. Not part of the CI gate."""
import numpy as np
import pytest
import torch

import submission.fusion as fusion
import submission.pointwise_pairwise as pp


def test_rrf_worked_example():
    base = {"a": {"x": 3.0, "y": 1.0}, "b": {"x": 2.0, "y": 3.0}, "c": {"x": 1.0, "y": 2.0}}
    s = fusion.rrf_scores(base, c=60.0)
    assert abs(s["a"] - (1 / 61 + 1 / 63)) < 1e-12 and abs(s["b"] - (1 / 62 + 1 / 61)) < 1e-12


def test_combsum_or_combmnz():
    base = {"a": {"x": 1.0, "y": 10.0}, "b": {"x": 3.0, "y": 20.0}, "c": {"x": 2.0, "y": 30.0}}
    done = False
    try:
        s = fusion.combsum_scores(base)
        assert abs(s["a"] - 0.0) < 1e-9 and abs(s["b"] - 1.5) < 1e-9 and abs(s["c"] - 1.5) < 1e-9
        done = True
    except NotImplementedError:
        pass
    try:
        m = fusion.combmnz_scores(base)
        # a has normalised score 0 in both channels -> appears in 0 -> 0 ; b,c appear in 2 -> 3.0
        assert abs(m["a"]) < 1e-9 and abs(m["b"] - 3.0) < 1e-9 and abs(m["c"] - 3.0) < 1e-9
        done = True
    except NotImplementedError:
        pass
    assert done, "implement combsum_scores or combmnz_scores"


def test_pointwise_loss():
    assert abs(pp.pointwise_loss(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 2.0])).item() - 0.5) < 1e-6


def test_ranknet_loss():
    v = pp.ranknet_loss(torch.tensor([2.0, 0.0]), torch.tensor([0.0, 0.0])).item()
    assert abs(v - 0.410038) < 1e-5
    big = pp.ranknet_loss(torch.tensor([-1000.0]), torch.tensor([1000.0])).item()
    assert np.isfinite(big) and abs(big - 2000.0) < 1e-3  # numerically stable


def test_make_pairs():
    rng = np.random.RandomState(0)
    pairs = pp.make_pairs(list("aaabb"), [2, 0, 1, 1, 1], 100, rng)
    assert sorted(map(tuple, pairs.tolist())) == [(0, 1), (0, 2), (2, 1)]
    labels = [2, 0, 1, 1, 1]
    capped = pp.make_pairs(list("aaabb"), labels, 2, np.random.RandomState(0))
    assert len(capped) == 2


def test_one_step_raises_margin():
    pp.set_seed(0)
    net = pp.ScoringNetwork(4, 8)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    xp, xo = torch.randn(1, 4), torch.randn(1, 4)
    before = (net(xp) - net(xo)).item()
    pp.pairwise_step(net, opt, xp, xo)
    assert (net(xp) - net(xo)).item() > before


def test_seed_reproducible():
    from harness.feature_io import read_svmlight_ltr, stack
    tr = stack(read_svmlight_ltr("data/toy/train.txt"))
    dv = stack(read_svmlight_ltr("data/toy/vali.txt"))
    n1, _, _ = pp.train_model("pairwise", tr, dv, seed=3, epochs=2)
    n2, _, _ = pp.train_model("pairwise", tr, dv, seed=3, epochs=2)
    for a, b in zip(n1.parameters(), n2.parameters()):
        assert torch.equal(a, b)
