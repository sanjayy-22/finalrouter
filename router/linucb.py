"""Disjoint LinUCB (Li et al., WWW 2010) with an incrementally maintained inverse.

Per model a:  A_a = lam*I + sum x x^T,  b_a = sum r x,  theta_a = A_a^-1 b_a
score:        p_a = theta_a^T x + alpha * sqrt(x^T A_a^-1 x)
"""

from __future__ import annotations

import json
from typing import Mapping, Sequence

import numpy as np


class LinUCB:
    name = "linucb"
    learns = True

    def __init__(self, models: Sequence[str], d: int, alpha: float = 1.0, lam: float = 1.0):
        if lam <= 0:
            raise ValueError("lam must be > 0")
        self.models = list(models)
        self.d = d
        self.alpha = alpha
        self.lam = lam
        self.A_inv = {m: np.eye(d) / lam for m in self.models}
        self.b = {m: np.zeros(d) for m in self.models}
        self.counts = {m: 0 for m in self.models}

    def theta(self, model: str) -> np.ndarray:
        return self.A_inv[model] @ self.b[model]

    def scores(self, x: np.ndarray) -> dict[str, float]:
        out = {}
        for m in self.models:
            Ai = self.A_inv[m]
            mean = float(self.theta(m) @ x)
            bonus = float(np.sqrt(max(x @ Ai @ x, 0.0)))
            out[m] = mean + self.alpha * bonus
        return out

    def select(self, x: np.ndarray, query: Mapping | None = None) -> str:
        s = self.scores(x)
        # ties broken by the order of `models` (documented, deterministic)
        return max(self.models, key=lambda m: (s[m], -self.models.index(m)))

    def update(self, model: str, x: np.ndarray, reward: float) -> None:
        Ai = self.A_inv[model]
        Ax = Ai @ x
        # Sherman-Morrison: (A + x x^T)^-1 = A^-1 - (A^-1 x)(x^T A^-1) / (1 + x^T A^-1 x)
        self.A_inv[model] = Ai - np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.b[model] = self.b[model] + reward * x
        self.counts[model] += 1

    # ---- persistence (used by the live service) ----
    def state_dict(self) -> dict:
        return {
            "models": self.models, "d": self.d, "alpha": self.alpha, "lam": self.lam,
            "A_inv": {m: self.A_inv[m].tolist() for m in self.models},
            "b": {m: self.b[m].tolist() for m in self.models},
            "counts": self.counts,
        }

    @classmethod
    def from_state_dict(cls, s: Mapping) -> "LinUCB":
        obj = cls(s["models"], s["d"], s["alpha"], s["lam"])
        obj.A_inv = {m: np.asarray(v, dtype=float) for m, v in s["A_inv"].items()}
        obj.b = {m: np.asarray(v, dtype=float) for m, v in s["b"].items()}
        obj.counts = dict(s["counts"])
        return obj

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.state_dict(), f)

    @classmethod
    def load(cls, path: str) -> "LinUCB":
        with open(path) as f:
            return cls.from_state_dict(json.load(f))
