"""Step 6: replay-based comparison of routing policies on a full-information outcome log.

Each policy sees the same query stream; on each query it picks one model and only that model's
logged outcome is revealed (bandit feedback). Learning policies may warm up on the dev split;
metrics are computed on the test split only.

usage:
  python -m evaluation.replay --queries data/queries.jsonl --outcomes data/outcomes.jsonl \
      --models serving/models.json --seeds 0 1 2 3 4 --alpha 1.0 --out results/main
  # workload shift (E5): first phase uses group A categories, then group B
  python -m evaluation.replay ... --shift qa,extraction,writing,summarization:reasoning,coding,instruction
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from evaluation.io import read_jsonl
from router.baselines import FixedPolicy, RandomPolicy, RuleBasedPolicy
from router.features import FeatureExtractor, difficulty_from_outcomes
from router.linucb import LinUCB
from router.reward import RewardConfig, fit_ranges, reward_of_row


@dataclass
class Setup:
    """Everything fitted on the dev split (never on test)."""
    models: list[str]
    queries: dict[str, dict]
    table: dict[tuple[str, str], dict]           # (query_id, model_id) -> outcome row
    cfg: RewardConfig
    fx: FeatureExtractor
    rules: RuleBasedPolicy
    dev_ids: list[str]
    test_ids: list[str]
    excluded: int = 0
    extra: dict = field(default_factory=dict)

    def reward(self, qid: str, model: str) -> float | None:
        return reward_of_row(self.table[(qid, model)], self.cfg)


def build_setup(queries: Sequence[Mapping], outcomes: Sequence[Mapping], models: Sequence[str],
                cfg: RewardConfig | None = None, fx_kwargs: Mapping | None = None) -> Setup:
    qmap = {q["query_id"]: dict(q) for q in queries}
    table = {(o["query_id"], o["model_id"]): o for o in outcomes if o["model_id"] in models}
    dev_rows = [o for o in outcomes if o.get("split") == "dev" and o["model_id"] in models]
    if cfg is None:
        cfg = RewardConfig(**fit_ranges(dev_rows))
    # keep only queries where every model has a usable (non-None) reward
    complete, excluded = [], 0
    for qid in qmap:
        if all((qid, m) in table and reward_of_row(table[(qid, m)], cfg) is not None for m in models):
            complete.append(qid)
        else:
            excluded += 1
    fx = FeatureExtractor(difficulty=difficulty_from_outcomes(dev_rows, models[0]), **(fx_kwargs or {}))
    rules = RuleBasedPolicy.from_dev_outcomes(dev_rows, lambda r: reward_of_row(r, cfg), models, default=models[-1])
    dev = [q for q in complete if qmap[q].get("split") == "dev"]
    test = [q for q in complete if qmap[q].get("split") == "test"]
    return Setup(list(models), qmap, table, cfg, fx, rules, dev, test, excluded)


def default_policies(s: Setup, alpha: float, lam: float, seed: int) -> dict[str, Callable[[], object]]:
    return {
        "small_only": lambda: FixedPolicy(s.models[0], "small_only"),
        "large_only": lambda: FixedPolicy(s.models[-1], "large_only"),
        "random": lambda: RandomPolicy(s.models, seed=seed),
        "rule_based": lambda: s.rules,
        "linucb": lambda: LinUCB(s.models, s.fx.dim, alpha=alpha, lam=lam),
    }


def make_stream(s: Setup, seed: int, shift: tuple[set, set] | None = None) -> list[str]:
    rng = np.random.default_rng(seed)
    ids = list(s.test_ids)
    rng.shuffle(ids)
    if shift is None:
        return ids
    a, b = shift
    return [q for q in ids if s.queries[q]["task_category"] in a] + \
           [q for q in ids if s.queries[q]["task_category"] in b]


def run_policy(policy, s: Setup, stream: Sequence[str], warmup: Sequence[str] = ()) -> list[dict]:
    for qid in warmup:                      # learning on dev data, not scored
        x = s.fx(s.queries[qid])
        m = policy.select(x, s.queries[qid])
        r = s.reward(qid, m)
        if r is not None:
            policy.update(m, x, r)
    steps = []
    for qid in stream:
        q = s.queries[qid]
        x = s.fx(q)
        m = policy.select(x, q)
        row = s.table[(qid, m)]
        r = s.reward(qid, m)
        if r is not None:
            policy.update(m, x, r)
        best = max(s.reward(qid, k) for k in s.models)
        steps.append({
            "query_id": qid, "task_category": q["task_category"], "model": m, "reward": r, "oracle": best,
            "quality": 0.0 if row.get("failed") else row.get("quality_score"),
            "latency_ms": row.get("latency_ms"), "cost": row.get("cost"), "failed": bool(row.get("failed")),
        })
    return steps


def summarize(steps: Sequence[Mapping], models: Sequence[str]) -> dict:
    rew = np.array([st["reward"] for st in steps], dtype=float)
    orc = np.array([st["oracle"] for st in steps], dtype=float)
    lat = np.array([st["latency_ms"] for st in steps], dtype=float)
    qual = [st["quality"] for st in steps if st["quality"] is not None]
    share = {m: float(np.mean([st["model"] == m for st in steps])) for m in models}
    return {
        "n": len(steps),
        "mean_quality": float(np.mean(qual)) if qual else None,
        "mean_latency_ms": float(lat.mean()),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "mean_cost": float(np.mean([st["cost"] for st in steps])),
        "failure_rate": float(np.mean([st["failed"] for st in steps])),
        "cumulative_reward": float(rew.sum()),
        "mean_reward": float(rew.mean()),
        "regret": float((orc - rew).sum()),
        "selection_share": share,
    }


def per_category(steps: Sequence[Mapping], models: Sequence[str]) -> dict:
    out = {}
    for cat in sorted({st["task_category"] for st in steps}):
        sub = [st for st in steps if st["task_category"] == cat]
        out[cat] = {"n": len(sub), "mean_reward": float(np.mean([st["reward"] for st in sub])),
                    "selection_share": {m: float(np.mean([st["model"] == m for st in sub])) for m in models}}
    return out


def evaluate(s: Setup, seeds: Sequence[int], alpha: float = 1.0, lam: float = 1.0, warmup: bool = True,
             shift: tuple[set, set] | None = None, policy_factory=default_policies) -> dict:
    runs: dict[str, list[dict]] = {}
    traces: dict[str, list[list[float]]] = {}
    cats: dict[str, list[dict]] = {}
    for seed in seeds:
        stream = make_stream(s, seed, shift)
        warm = list(s.dev_ids)
        np.random.default_rng(seed + 10_000).shuffle(warm)
        for name, make in policy_factory(s, alpha, lam, seed).items():
            pol = make()
            steps = run_policy(pol, s, stream, warm if (warmup and getattr(pol, "learns", False)) else ())
            runs.setdefault(name, []).append(summarize(steps, s.models))
            traces.setdefault(name, []).append([st["reward"] for st in steps])
            cats.setdefault(name, []).append(per_category(steps, s.models))
    agg = {}
    for name, rs in runs.items():
        agg[name] = {}
        for k in ("mean_quality", "mean_latency_ms", "p95_latency_ms", "mean_cost", "failure_rate",
                  "cumulative_reward", "mean_reward", "regret"):
            vals = np.array([r[k] for r in rs if r[k] is not None], dtype=float)
            agg[name][k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
        agg[name]["selection_share"] = {m: float(np.mean([r["selection_share"][m] for r in rs])) for m in s.models}
    cat_agg = {}
    for name, cs in cats.items():
        cat_agg[name] = {c: {"mean_reward": float(np.mean([x[c]["mean_reward"] for x in cs])),
                             "selection_share": {m: float(np.mean([x[c]["selection_share"][m] for x in cs]))
                                                 for m in s.models}}
                         for c in cs[0]}
    return {"seeds": list(seeds), "alpha": alpha, "lam": lam, "warmup": warmup,
            "shift": [sorted(shift[0]), sorted(shift[1])] if shift else None,
            "n_test": len(s.test_ids), "n_dev": len(s.dev_ids), "excluded_queries": s.excluded,
            "reward_config": s.cfg.to_dict(), "features": s.fx.names(),
            "policies": agg, "per_category": cat_agg,
            "reward_traces": {k: np.mean(np.array(v, dtype=float), axis=0).tolist() for k, v in traces.items()}}


def markdown_tables(res: Mapping, models: Sequence[str]) -> str:
    """Table VII / VIII layouts from the paper."""
    def f(d, k, p=3):
        v = d[k]
        return f"{v['mean']:.{p}f} ± {v['std']:.{p}f}"
    lines = ["| Policy | Mean Quality | Mean Latency (ms) | p95 Latency (ms) | Cost / Request | Failure Rate | "
             "Cumulative Reward | Regret |", "|" + "---|" * 8]
    for name, d in res["policies"].items():
        lines.append(f"| {name} | {f(d,'mean_quality')} | {f(d,'mean_latency_ms',0)} | {f(d,'p95_latency_ms',0)} | "
                     f"{f(d,'mean_cost')} | {f(d,'failure_rate')} | {f(d,'cumulative_reward',1)} | {f(d,'regret',1)} |")
    if "linucb" in res["per_category"]:
        fixed = [p for p in res["per_category"] if p in ("small_only", "large_only", "rule_based")]
        lines += ["", "| Task Category | " + " | ".join(models) + " | LinUCB mean reward − best fixed |",
                  "|" + "---|" * (len(models) + 2)]
        for cat, d in res["per_category"]["linucb"].items():
            best_fixed = max(res["per_category"][p][cat]["mean_reward"] for p in fixed)
            shares = " | ".join(f"{d['selection_share'][m]:.2f}" for m in models)
            lines.append(f"| {cat} | {shares} | {d['mean_reward'] - best_fixed:+.3f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--outcomes", required=True)
    ap.add_argument("--models", required=True, help="models.json (order: small -> large)")
    ap.add_argument("--reward", default=None, help="JSON with RewardConfig fields; ranges fitted on dev if omitted")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--shift", default=None, help="catsA:catsB, comma-separated")
    ap.add_argument("--out", default="results/run")
    a = ap.parse_args()

    models = [m["model_id"] for m in json.load(open(a.models))["models"]]
    cfg = None
    if a.reward:
        rc = json.load(open(a.reward))
        cfg = RewardConfig(**{k: tuple(v) if k.endswith("_range") else v for k, v in rc.items()})
    s = build_setup(read_jsonl(a.queries), read_jsonl(a.outcomes), models, cfg)
    shift = None
    if a.shift:
        A, B = a.shift.split(":")
        shift = (set(A.split(",")), set(B.split(",")))
    res = evaluate(s, a.seeds, a.alpha, a.lam, not a.no_warmup, shift)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    md = markdown_tables(res, models)
    (out / "tables.md").write_text(md + "\n")
    print(md)
    print(f"\n{res['n_test']} test / {res['n_dev']} dev queries, {res['excluded_queries']} excluded -> {out}")


if __name__ == "__main__":
    main()
