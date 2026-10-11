"""R = w_q*Q - w_c*C~ - w_l*L~  (paper Eq. 6-7).

* Q in [0, 1]; failed / invalid / timed-out responses get Q = 0 but keep their cost and latency.
* C~ and L~ are min-max normalized with ranges fixed on the development split, clipped to [0, 1].
* Missing quality (no validated feedback yet) -> reward is None and the observation is not used
  for an update (deferred). This rule is reported with results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass
class RewardConfig:
    w_quality: float = 1.0
    w_cost: float = 0.3
    w_latency: float = 0.2
    cost_range: tuple[float, float] = (0.0, 1.0)
    latency_range: tuple[float, float] = (0.0, 1.0)  # milliseconds
    notes: str = field(default="weights are placeholders; fix and document before the main comparison")

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(min(max((v - lo) / (hi - lo), 0.0), 1.0))


def compute_reward(
    quality: float | None,
    cost: float,
    latency_ms: float,
    failed: bool,
    cfg: RewardConfig,
) -> float | None:
    if failed:
        q = 0.0
    elif quality is None:
        return None
    else:
        q = float(min(max(quality, 0.0), 1.0))
    c = _norm(cost, *cfg.cost_range)
    l = _norm(latency_ms, *cfg.latency_range)
    return cfg.w_quality * q - cfg.w_cost * c - cfg.w_latency * l


def reward_of_row(row: Mapping, cfg: RewardConfig) -> float | None:
    return compute_reward(row.get("quality_score"), float(row.get("cost") or 0.0),
                          float(row.get("latency_ms") or 0.0), bool(row.get("failed")), cfg)


def fit_ranges(rows: Sequence[Mapping], lo_pct: float = 5, hi_pct: float = 95) -> dict:
    """Percentile-based normalization ranges. Call on development rows only."""
    cost = np.array([float(r.get("cost") or 0.0) for r in rows])
    lat = np.array([float(r.get("latency_ms") or 0.0) for r in rows])
    return {
        "cost_range": (float(np.percentile(cost, lo_pct)), float(np.percentile(cost, hi_pct))),
        "latency_range": (float(np.percentile(lat, lo_pct)), float(np.percentile(lat, hi_pct))),
    }
