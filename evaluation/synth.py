"""Synthetic queries + full-information outcomes for testing the pipeline WITHOUT GPUs.

Every row is tagged "synthetic": true. These numbers are NOT experimental results and must
never be reported in the paper.

usage: python -m evaluation.synth --out-dir data/synthetic --per-category 200 --seed 0
"""

from __future__ import annotations

import argparse

import numpy as np

from evaluation.io import write_jsonl
from router.features import CATEGORIES

MODELS = [  # (model_id, size_B) -- mirrors serving/models.json
    ("qwen3-4b", 4.0), ("qwen3-8b", 8.0), ("qwen3-14b", 14.0)]

# assumed P(correct) per (category, model) at average difficulty -- purely illustrative
BASE_SKILL = {
    "qa":            (0.70, 0.76, 0.80),
    "reasoning":     (0.45, 0.62, 0.75),
    "coding":        (0.40, 0.55, 0.70),
    "summarization": (0.78, 0.80, 0.82),
    "writing":       (0.75, 0.78, 0.80),
    "extraction":    (0.82, 0.84, 0.85),
    "instruction":   (0.65, 0.72, 0.78),
}
PROMPTS = {
    "qa": "What is the capital of country #{i}?",
    "reasoning": "Solve step by step: a train travels {i} km in 3 hours; how many km per hour?",
    "coding": "```python\ndef f{i}(xs):\n    return sorted(xs)[0]\n```\nDebug this function so it returns the maximum.",
    "summarization": "Summarize the following report #{i} in three sentences: " + "lorem ipsum " * 40,
    "writing": "Rewrite this paragraph #{i} in a formal tone: hey team, the launch slipped a week.",
    "extraction": "Extract the invoice number and total from: Invoice INV-{i} total $1{i}.00",
    "instruction": "List exactly three fruits #{i}, in lowercase, separated by commas.",
}


def generate(per_category: int, seed: int, dev_frac: float = 0.3):
    rng = np.random.default_rng(seed)
    queries, outcomes = [], []
    for cat in CATEGORIES:
        for i in range(per_category):
            qid = f"{cat}-{i:04d}"
            difficulty = float(rng.beta(2, 2))                      # hidden per-query difficulty
            long_prompt = rng.random() < 0.3
            prompt = PROMPTS[cat].format(i=i) + (" Please explain why in detail." * 8 if long_prompt else "")
            split = "dev" if rng.random() < dev_frac else "test"
            queries.append({"query_id": qid, "prompt": prompt, "task_category": cat, "split": split,
                            "scorer": "exact", "reference": None, "synthetic": True})
            in_tok = int(len(prompt.split()) * 1.3)
            for (mid, size), base in zip(MODELS, BASE_SKILL[cat]):
                # harder queries hurt small models more
                p = np.clip(base - (difficulty - 0.5) * (0.9 - 0.04 * size), 0.02, 0.98)
                failed = rng.random() < 0.01
                quality = 0.0 if failed else float(rng.random() < p)
                out_tok = int(rng.normal(180, 40) * (1.3 if cat in ("reasoning", "coding") else 1.0))
                out_tok = max(out_tok, 10)
                latency = (40 + 2.2 * size) * out_tok / 10 + rng.normal(0, 30) + (8 * size if long_prompt else 0)
                outcomes.append({
                    "query_id": qid, "model_id": mid, "task_category": cat, "split": split,
                    "response": "", "quality_score": None if failed else quality, "failed": failed,
                    "error": "SyntheticTimeout" if failed else None,
                    "latency_ms": float(max(latency, 50.0)), "input_tokens": in_tok, "output_tokens": out_tok,
                    "cost": (in_tok + out_tok) / 1000 * size, "synthetic": True,
                })
    return queries, outcomes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/synthetic")
    ap.add_argument("--per-category", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    q, o = generate(a.per_category, a.seed)
    write_jsonl(f"{a.out_dir}/queries.jsonl", q)
    write_jsonl(f"{a.out_dir}/outcomes.jsonl", o)
    print(f"wrote {len(q)} synthetic queries and {len(o)} outcomes to {a.out_dir}")


if __name__ == "__main__":
    main()
