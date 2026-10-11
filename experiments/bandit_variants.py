"""Alternative contextual-bandit policies to compare against disjoint LinUCB.

* LinTS               - linear Thompson sampling (Gaussian posterior per model)
* EpsilonGreedyRidge  - ridge-regression estimate, explore uniformly with prob. epsilon
* SlidingWindowLinUCB - LinUCB fitted on the last W observations only (non-stationary workloads;
                        sliding-window idea from Garivier & Moulines, ALT 2011)
"""

from __future__ import annotations

from collections import deque
from typing import Mapping, Sequence

import numpy as np

from router.linucb import LinUCB


class LinTS(LinUCB):
    name = "lin_ts"

    def __init__(self, models: Sequence[str], d: int, v: float = 0.3, lam: float = 1.0, seed: int = 0):
        super().__init__(models, d, alpha=0.0, lam=lam)
        self.v = v
        self.rng = np.random.default_rng(seed)

    def select(self, x: np.ndarray, query: Mapping | None = None) -> str:
        best, best_s = None, -np.inf
        for m in self.models:
            cov = self.v ** 2 * self.A_inv[m]
            cov = (cov + cov.T) / 2
            theta = self.rng.multivariate_normal(self.theta(m), cov, method="cholesky")
            s = float(theta @ x)
            if s > best_s:
                best, best_s = m, s
        return best


class EpsilonGreedyRidge(LinUCB):
    name = "eps_greedy"

    def __init__(self, models: Sequence[str], d: int, epsilon: float = 0.1, lam: float = 1.0, seed: int = 0):
        super().__init__(models, d, alpha=0.0, lam=lam)
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def select(self, x: np.ndarray, query: Mapping | None = None) -> str:
        if self.rng.random() < self.epsilon:
            return self.models[int(self.rng.integers(len(self.models)))]
        return super().select(x, query)


class SlidingWindowLinUCB(LinUCB):
    name = "sw_linucb"

    def __init__(self, models: Sequence[str], d: int, alpha: float = 1.0, lam: float = 1.0, window: int = 300):
        super().__init__(models, d, alpha=alpha, lam=lam)
        self.window = window
        self.hist: deque = deque()

    def update(self, model: str, x: np.ndarray, reward: float) -> None:
        self.hist.append((model, x, reward))
        self.counts[model] += 1
        dropped = None
        if len(self.hist) > self.window:
            dropped = self.hist.popleft()[0]
        for m in {model, dropped} - {None}:
            A, b = self.lam * np.eye(self.d), np.zeros(self.d)
            for mm, xx, rr in self.hist:
                if mm == m:
                    A += np.outer(xx, xx)
                    b += rr * xx
            self.A_inv[m] = np.linalg.inv(A)
            self.b[m] = b
