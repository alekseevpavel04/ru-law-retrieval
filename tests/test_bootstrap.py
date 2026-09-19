import numpy as np
import pytest

from rlr.eval.bootstrap import paired_bootstrap


def test_identical_systems_zero_delta():
    a = np.array([0.1, 0.5, 1.0, 0.0])
    r = paired_bootstrap(a, a, n=500)
    assert r["delta"] == 0.0
    assert r["ci_low"] == 0.0 and r["ci_high"] == 0.0


def test_clear_improvement_is_significant():
    rng = np.random.default_rng(1)
    b = rng.uniform(0, 0.5, size=300)
    a = b + 0.2
    r = paired_bootstrap(a, b, n=2000)
    assert r["delta"] == pytest.approx(0.2)
    assert r["ci_low"] > 0.15 and r["p_value"] < 0.001


def test_ci_contains_delta_and_is_reproducible():
    rng = np.random.default_rng(2)
    a, b = rng.uniform(size=100), rng.uniform(size=100)
    r1, r2 = paired_bootstrap(a, b, n=1000), paired_bootstrap(a, b, n=1000)
    assert r1 == r2
    assert r1["ci_low"] <= r1["delta"] <= r1["ci_high"]


def test_length_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap(np.ones(3), np.ones(4))
