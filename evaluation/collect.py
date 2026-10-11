"""Step 3: run every query on every model through vLLM's OpenAI-compatible API and log outcomes.

usage:
  python -m evaluation.collect --queries data/queries.jsonl --models serving/models.json \
      --out data/outcomes.jsonl [--judge serving/judge.json]

Resumable: (query_id, model_id) pairs already in --out are skipped.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Mapping

from evaluation.io import append_jsonl, iter_jsonl, read_jsonl
from evaluation.scoring import score


class ServerDown(RuntimeError):
    """Model server unreachable. Not logged as a model failure, so a resumed run retries it."""


def call_model(model: Mapping, prompt: str, decoding: Mapping) -> dict:
    body = {
        "model": model["name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": decoding.get("temperature", 0.0),
        "max_tokens": decoding.get("max_tokens", 1024),
        # Qwen3 thinking mode must be fixed for the whole study (vLLM forwards this to the chat template)
        "chat_template_kwargs": {"enable_thinking": bool(decoding.get("enable_thinking", False))},
    }
    if "seed" in decoding:
        body["seed"] = decoding["seed"]
    req = urllib.request.Request(model["base_url"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=decoding.get("timeout_s", 120)) as r:
            data = json.load(r)
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = data.get("usage") or {}
        return {"response": data["choices"][0]["message"]["content"] or "", "latency_ms": latency_ms,
                "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
                "failed": False, "error": None}
    except urllib.error.URLError as e:
        if isinstance(e.reason, ConnectionRefusedError):
            raise ServerDown(str(e)) from e     # infrastructure, not the model's fault: retry later
        return {"response": "", "latency_ms": (time.perf_counter() - t0) * 1000, "input_tokens": None,
                "output_tokens": None, "failed": True, "error": f"{type(e).__name__}: {e}"}
    except (TimeoutError, KeyError, ValueError) as e:
        return {"response": "", "latency_ms": (time.perf_counter() - t0) * 1000, "input_tokens": None,
                "output_tokens": None, "failed": True, "error": f"{type(e).__name__}: {e}"}


def compute_cost(model: Mapping, out: Mapping) -> float:
    """Compute-cost proxy: (input+output tokens)/1000 * model's relative cost factor (documented in models.json)."""
    toks = (out.get("input_tokens") or 0) + (out.get("output_tokens") or 0)
    return toks / 1000.0 * float(model.get("cost_per_1k_tokens", 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.models))
    judge = json.load(open(args.judge)) if args.judge else None
    done = set()
    try:
        done = {(r["query_id"], r["model_id"]) for r in iter_jsonl(args.out)}
    except FileNotFoundError:
        pass

    queries = read_jsonl(args.queries)
    for model in cfg["models"]:                      # model-major order: one server warm at a time
        for q in queries:
            if (q["query_id"], model["model_id"]) in done:
                continue
            try:
                out = call_model(model, q["prompt"], cfg.get("decoding", {}))
            except ServerDown as e:
                print(f"{model['model_id']}: server unreachable ({e}); skipping this model, rerun to resume")
                break
            quality = None if out["failed"] else score(q, out["response"], judge)
            append_jsonl(args.out, {
                "query_id": q["query_id"], "model_id": model["model_id"],
                "task_category": q.get("task_category"), "split": q.get("split"),
                **out, "cost": compute_cost(model, out), "quality_score": quality,
            })
            print(f"{model['model_id']:>10} {q['query_id']:>12} q={quality} {out['latency_ms']:.0f}ms"
                  + (f" ERROR {out['error']}" if out["failed"] else ""))


if __name__ == "__main__":
    main()
