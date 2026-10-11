"""Query -> context vector x.

Features use only information available before generation. The default set is
small and interpretable; an optional embedding projection can be appended.
"""

from __future__ import annotations

import math
import re
from typing import Callable, Mapping, Sequence

import numpy as np

CATEGORIES: tuple[str, ...] = (
    "qa",            # general question answering
    "reasoning",     # reasoning and multi-step problems
    "coding",        # coding and debugging
    "summarization",
    "writing",       # text transformation and writing
    "extraction",    # information extraction
    "instruction",   # instruction following
)

_CODE_RE = re.compile(
    r"```|\bdef\s+\w+\(|\bclass\s+\w+|\bimport\s+\w+|#include|\bfunction\s*\w*\(|"
    r"\bTraceback\b|\bstack trace\b|\bcompile error\b|=>|\breturn\s",
    re.IGNORECASE,
)
_REASONING_RE = re.compile(
    r"\bstep[- ]by[- ]step\b|\bprove\b|\bderive\b|\bcalculate\b|\bhow many\b|\bsolve\b|"
    r"\bwhy\b|\bexplain\b|\boptimi[sz]e\b|\bdebug\b|\bcompare\b",
    re.IGNORECASE,
)


def approx_tokens(text: str) -> int:
    """Cheap token-count proxy (whitespace words * 1.3). Replace with a tokenizer if needed."""
    return int(math.ceil(len(text.split()) * 1.3))


class EmbeddingProjector:
    """PCA projection of sentence embeddings, fitted on development queries only."""

    def __init__(self, embed: Callable[[str], np.ndarray], n_components: int = 8):
        self.embed = embed
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, texts: Sequence[str]) -> "EmbeddingProjector":
        E = np.stack([np.asarray(self.embed(t), dtype=float) for t in texts])
        self.mean_ = E.mean(axis=0)
        _, s, vt = np.linalg.svd(E - self.mean_, full_matrices=False)
        k = min(self.n_components, vt.shape[0])
        self.components_ = vt[:k]
        proj = (E - self.mean_) @ self.components_.T
        self.scale_ = np.maximum(np.abs(proj).max(axis=0), 1e-9)
        return self

    def __call__(self, text: str) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("EmbeddingProjector must be fitted before use")
        z = (np.asarray(self.embed(text), dtype=float) - self.mean_) @ self.components_.T
        return np.clip(z / self.scale_, -1.0, 1.0)


class FeatureExtractor:
    """Builds x = [length, category one-hot, has_code, reasoning_cue, difficulty, (embedding), 1].

    Feature groups can be switched off for ablation (experiment E4).
    """

    def __init__(
        self,
        categories: Sequence[str] = CATEGORIES,
        max_tokens: int = 2048,
        difficulty: Mapping[str, float] | None = None,
        projector: EmbeddingProjector | None = None,
        use_length: bool = True,
        use_category: bool = True,
        use_cues: bool = True,
        use_difficulty: bool = True,
    ):
        self.categories = tuple(categories)
        self.max_tokens = max_tokens
        self.difficulty = dict(difficulty or {})
        self.projector = projector
        self.use_length = use_length
        self.use_category = use_category
        self.use_cues = use_cues
        self.use_difficulty = use_difficulty

    @property
    def dim(self) -> int:
        d = 1  # bias
        d += 1 if self.use_length else 0
        d += len(self.categories) if self.use_category else 0
        d += 2 if self.use_cues else 0
        d += 1 if self.use_difficulty else 0
        d += self.projector.n_components if self.projector is not None else 0
        return d

    def names(self) -> list[str]:
        out = []
        if self.use_length:
            out.append("log_length")
        if self.use_category:
            out += [f"cat={c}" for c in self.categories]
        if self.use_cues:
            out += ["has_code", "reasoning_cue"]
        if self.use_difficulty:
            out.append("difficulty")
        if self.projector is not None:
            out += [f"emb{i}" for i in range(self.projector.n_components)]
        out.append("bias")
        return out

    def __call__(self, query: Mapping) -> np.ndarray:
        text = query["prompt"]
        cat = query.get("task_category")
        parts: list[float] = []
        if self.use_length:
            parts.append(min(math.log1p(approx_tokens(text)) / math.log1p(self.max_tokens), 1.0))
        if self.use_category:
            parts += [1.0 if cat == c else 0.0 for c in self.categories]
        if self.use_cues:
            parts.append(1.0 if _CODE_RE.search(text) else 0.0)
            parts.append(1.0 if _REASONING_RE.search(text) else 0.0)
        if self.use_difficulty:
            parts.append(float(self.difficulty.get(cat, 0.5)))
        x = np.asarray(parts, dtype=float)
        if self.projector is not None:
            x = np.concatenate([x, self.projector(text)])
        return np.concatenate([x, [1.0]])


def difficulty_from_outcomes(rows: Sequence[Mapping], reference_model: str) -> dict[str, float]:
    """Per-category difficulty prior = 1 - mean quality of `reference_model` on dev rows.

    Must be computed from the development split only, to avoid test leakage.
    """
    sums: dict[str, list[float]] = {}
    for r in rows:
        if r["model_id"] != reference_model or r.get("quality_score") is None:
            continue
        q = 0.0 if r.get("failed") else float(r["quality_score"])
        sums.setdefault(r["task_category"], []).append(q)
    return {c: 1.0 - float(np.mean(v)) for c, v in sums.items()}
