"""Build data/queries.jsonl from GSM8K (MIT license, github.com/openai/grade-school-math).

Samples N problems from the official test split with a fixed seed and assigns a dev/test split.

usage:
  python -m data.build_gsm8k --n 300 --dev-frac 0.3 --seed 0 --out data/queries_gsm8k.jsonl
  python -m data.build_gsm8k --input path/to/test.jsonl ...      # offline copy
"""

from __future__ import annotations

import argparse
import json
import urllib.request

import numpy as np

from evaluation.io import write_jsonl

URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

PROMPT = ("Solve the following math word problem. Reason step by step, then give the final answer "
          "on the last line in the form '#### <number>'.\n\nProblem: {question}")


def load(path: str | None) -> list[dict]:
    if path:
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    with urllib.request.urlopen(URL, timeout=60) as r:
        return [json.loads(l) for l in r.read().decode("utf-8").splitlines() if l.strip()]


def build(rows: list[dict], n: int, dev_frac: float, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))[:n]
    n_dev = int(round(n * dev_frac))
    out = []
    for rank, i in enumerate(idx):
        r = rows[int(i)]
        final = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append({
            "query_id": f"gsm8k-test-{int(i):04d}",
            "prompt": PROMPT.format(question=r["question"]),
            "task_category": "reasoning",
            "split": "dev" if rank < n_dev else "test",
            "scorer": "numeric",
            "reference": final,
            "source": "gsm8k/test",
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--dev-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/queries_gsm8k.jsonl")
    a = ap.parse_args()
    rows = load(a.input)
    qs = build(rows, min(a.n, len(rows)), a.dev_frac, a.seed)
    write_jsonl(a.out, qs)
    n_dev = sum(q["split"] == "dev" for q in qs)
    print(f"GSM8K test rows: {len(rows)} -> wrote {len(qs)} queries ({n_dev} dev / {len(qs) - n_dev} test) to {a.out}")


if __name__ == "__main__":
    main()
