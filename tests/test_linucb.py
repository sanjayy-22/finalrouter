import numpy as np

from router.linucb import LinUCB


def test_incremental_inverse_matches_explicit():
    rng = np.random.default_rng(0)
    d, lam = 5, 2.0
    pol = LinUCB(["a"], d, alpha=0.5, lam=lam)
    A, b = lam * np.eye(d), np.zeros(d)
    for _ in range(50):
        x, r = rng.normal(size=d), rng.normal()
        pol.update("a", x, r)
        A += np.outer(x, x)
        b += r * x
    assert np.allclose(pol.A_inv["a"], np.linalg.inv(A))
    assert np.allclose(pol.theta("a"), np.linalg.solve(A, b))


def test_learns_context_dependent_best_arm():
    rng = np.random.default_rng(1)
    true = {"small": np.array([0.9, 0.1]), "large": np.array([0.2, 0.8])}  # small wins on ctx0, large on ctx1
    pol = LinUCB(list(true), d=2, alpha=0.5)
    for _ in range(600):
        x = np.eye(2)[rng.integers(2)]
        m = pol.select(x)
        pol.update(m, x, float(true[m] @ x + rng.normal(0, 0.1)))
    assert pol.select(np.array([1.0, 0.0])) == "small"
    assert pol.select(np.array([0.0, 1.0])) == "large"


def test_exploration_bonus_shrinks_with_data():
    pol = LinUCB(["a"], 3, alpha=1.0)
    x = np.array([1.0, 0.0, 1.0])
    before = pol.scores(x)["a"]
    for _ in range(20):
        pol.update("a", x, 0.0)
    assert pol.scores(x)["a"] < before


def test_tie_break_is_model_order():
    assert LinUCB(["x", "y", "z"], 2).select(np.ones(2)) == "x"


def test_state_roundtrip(tmp_path):
    pol = LinUCB(["a", "b"], 3, alpha=0.7)
    pol.update("b", np.array([1.0, 2.0, 3.0]), 0.5)
    p = tmp_path / "s.json"
    pol.save(str(p))
    pol2 = LinUCB.load(str(p))
    x = np.array([0.3, -1.0, 2.0])
    assert pol2.scores(x) == pol.scores(x)
    assert pol2.counts == {"a": 0, "b": 1}
