import numpy as np
import pytest

from router.features import CATEGORIES, FeatureExtractor, difficulty_from_outcomes
from router.reward import RewardConfig, compute_reward, fit_ranges

CFG = RewardConfig(w_quality=1.0, w_cost=0.5, w_latency=0.5, cost_range=(0, 10), latency_range=(0, 1000))


def test_reward_formula_and_clipping():
    assert compute_reward(1.0, 0, 0, False, CFG) == pytest.approx(1.0)
    assert compute_reward(1.0, 5, 500, False, CFG) == pytest.approx(0.5)
    assert compute_reward(1.0, 50, 5000, False, CFG) == pytest.approx(0.0)   # clipped to 1


def test_failure_gets_zero_quality_but_keeps_penalties():
    assert compute_reward(0.9, 10, 1000, True, CFG) == pytest.approx(-1.0)


def test_missing_quality_is_deferred():
    assert compute_reward(None, 1, 1, False, CFG) is None


def test_cheap_unusable_answer_cannot_beat_good_answer():
    assert compute_reward(0.0, 0, 0, False, CFG) < compute_reward(1.0, 10, 1000, False, CFG) + 1e-9


def test_fit_ranges():
    rows = [{"cost": c, "latency_ms": 10 * c} for c in range(101)]
    r = fit_ranges(rows)
    assert r["cost_range"] == (5.0, 95.0) and r["latency_range"] == (50.0, 950.0)


def test_feature_vector_shape_and_names():
    fx = FeatureExtractor(difficulty={"coding": 0.8})
    x = fx({"prompt": "```python\ndef f(x):\n  return x\n```\nfix the bug", "task_category": "coding"})
    assert x.shape == (fx.dim,) == (len(fx.names()),)
    named = dict(zip(fx.names(), x))
    assert named["cat=coding"] == 1.0 and named["has_code"] == 1.0
    assert named["difficulty"] == 0.8 and named["bias"] == 1.0
    assert sum(named[f"cat={c}"] for c in CATEGORIES) == 1.0


def test_ablation_flags_reduce_dim():
    full = FeatureExtractor()
    small = FeatureExtractor(use_category=False, use_cues=False)
    assert small.dim == full.dim - len(CATEGORIES) - 2
    assert small({"prompt": "hi", "task_category": "qa"}).shape == (small.dim,)


def test_difficulty_prior_uses_reference_model_only():
    rows = [{"model_id": "s", "task_category": "qa", "quality_score": 1.0},
            {"model_id": "s", "task_category": "qa", "quality_score": 0.0},
            {"model_id": "l", "task_category": "qa", "quality_score": 0.0}]
    assert difficulty_from_outcomes(rows, "s") == {"qa": 0.5}
