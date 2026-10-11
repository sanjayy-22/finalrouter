"""Adaptive LLM model routing with contextual bandits (LinUCB)."""

from router.baselines import FixedPolicy, RandomPolicy, RuleBasedPolicy
from router.features import CATEGORIES, FeatureExtractor
from router.linucb import LinUCB
from router.reward import RewardConfig, compute_reward, fit_ranges

__all__ = [
    "CATEGORIES",
    "FeatureExtractor",
    "FixedPolicy",
    "LinUCB",
    "RandomPolicy",
    "RewardConfig",
    "RuleBasedPolicy",
    "compute_reward",
    "fit_ranges",
]
