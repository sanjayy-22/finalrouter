"""Non-learning baseline policies: small-only, large-only, random, rule-based."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


class FixedPolicy:
    learns = False

    def __init__(self, model: str, name: str | None = None):
        self.model = model
        self.name = name or f"fixed:{model}"

    def select(self, x, query: Mapping | None = None) -> str:
        return self.model

    def update(self, model, x, reward) -> None:
        pass


class RandomPolicy:
    name = "random"
    learns = False

    def __init__(self, models: Sequence[str], seed: int = 0):
        self.models = list(models)
        self.rng = np.random.default_rng(seed)

    def select(self, x, query: Mapping | None = None) -> str:
        return self.models[int(self.rng.integers(len(self.models)))]

    def update(self, model, x, reward) -> None:
        pass


class RuleBasedPolicy:
    """Routes by task category; rules are fixed on the development split before testing."""

    name = "rule_based"
    learns = False

    def __init__(self, rules: Mapping[str, str], default: str):
        self.rules = dict(rules)
        self.default = default

    def select(self, x, query: Mapping | None = None) -> str:
        cat = (query or {}).get("task_category")
        return self.rules.get(cat, self.default)

    def update(self, model, x, reward) -> None:
        pass

    @classmethod
    def from_dev_outcomes(cls, rows, reward_fn, models: Sequence[str], default: str) -> "RuleBasedPolicy":
        """For each category, pick the model with the highest mean dev reward."""
        acc: dict[tuple[str, str], list[float]] = {}
        for r in rows:
            rew = reward_fn(r)
            if rew is not None:
                acc.setdefault((r["task_category"], r["model_id"]), []).append(rew)
        rules = {}
        for cat in {c for c, _ in acc}:
            means = {m: float(np.mean(acc[(cat, m)])) for m in models if (cat, m) in acc}
            if means:
                rules[cat] = max(means, key=means.get)
        return cls(rules, default)
