"""Experiment runners for the paper's E2 (reward weights), E4 (ablation), E5 (workload shift),
E6 (exploration sensitivity) and a bandit-variant comparison.

usage:
  python -m experiments.sweeps --queries Q --outcomes O --models serving/models.json --out results/sweeps [--only alpha,weights,...]

On data/synthetic/* the output only exercises the code: it is NOT an experimental result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.io import read_jsonl
from evaluation.replay import build_setup, default_policies, evaluate
from experiments.bandit_variants import EpsilonGreedyRidge, LinTS, SlidingWindowLinUCB
from router.linucb import LinUCB
from router.reward import RewardConfig, fit_ranges

SHIFT = ({"qa", "extraction", "summarization", "writing"}, {"reasoning", "coding", "instruction"})


def _row(name, d):
    return {"setting": name, **{k: d[k]["mean"] for k in ("mean_reward", "mean_quality", "mean_cost",
                                                            "mean_latency_ms", "regret")},
            "share": d["selection_share"]}


def only(names):
    def factory(s, alpha, lam, seed):
        return {k: v for k, v in default_policies(s, alpha, lam, seed).items() if k in names}
    return factory


def run_alpha(Q, O, models, seeds):
    out = []
    s = build_setup(Q, O, models)
    for a in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
        res = evaluate(s, seeds, alpha=a, policy_factory=only({"linucb"}))
        out.append(_row(f"alpha={a}", res["policies"]["linucb"]))
    return out


def run_weights(Q, O, models, seeds, alpha):
    dev = [o for o in O if o.get("split") == "dev"]
    ranges = fit_ranges(dev)
    out = []
    for wc, wl in ((0.0, 0.0), (0.1, 0.05), (0.3, 0.2), (0.6, 0.4), (1.0, 0.6)):
        s = build_setup(Q, O, models, RewardConfig(w_cost=wc, w_latency=wl, **ranges))
        res = evaluate(s, seeds, alpha=alpha)
        best_fixed = max(("small_only", "large_only", "rule_based"),
                         key=lambda p: res["policies"][p]["mean_reward"]["mean"])
        r = _row(f"w_cost={wc}, w_lat={wl}", res["policies"]["linucb"])
        r["best_fixed"] = best_fixed
        r["best_fixed_reward"] = res["policies"][best_fixed]["mean_reward"]["mean"]
        out.append(r)
    return out


def run_ablation(Q, O, models, seeds, alpha):
    out = []
    variants = {"full": {}, "no_length": {"use_length": False}, "no_category": {"use_category": False},
                "no_cues": {"use_cues": False}, "no_difficulty": {"use_difficulty": False},
                "bias_only": {"use_length": False, "use_category": False, "use_cues": False, "use_difficulty": False}}
    for name, kw in variants.items():
        s = build_setup(Q, O, models, fx_kwargs=kw)
        res = evaluate(s, seeds, alpha=alpha, policy_factory=only({"linucb"}))
        out.append(_row(name, res["policies"]["linucb"]))
    return out


def variant_factory(window):
    def factory(s, alpha, lam, seed):
        return {
            "linucb": lambda: LinUCB(s.models, s.fx.dim, alpha=alpha, lam=lam),
            "lin_ts": lambda: LinTS(s.models, s.fx.dim, v=0.3, lam=lam, seed=seed),
            "eps_greedy": lambda: EpsilonGreedyRidge(s.models, s.fx.dim, epsilon=0.1, lam=lam, seed=seed),
            "sw_linucb": lambda: SlidingWindowLinUCB(s.models, s.fx.dim, alpha=alpha, lam=lam, window=window),
            "rule_based": lambda: s.rules,
        }
    return factory


def run_variants(Q, O, models, seeds, alpha, shift):
    s = build_setup(Q, O, models)
    res = evaluate(s, seeds, alpha=alpha, shift=SHIFT if shift else None, warmup=not shift,
                   policy_factory=variant_factory(window=200))
    rows = [_row(k, v) for k, v in res["policies"].items()]
    if shift:  # mean reward in the 100 steps after the shift point
        n_a = sum(1 for q in s.test_ids if s.queries[q]["task_category"] in SHIFT[0])
        for r in rows:
            tr = res["reward_traces"][r["setting"]]
            r["post_shift_100"] = float(np.mean(tr[n_a:n_a + 100]))
            r["pre_shift_last_100"] = float(np.mean(tr[n_a - 100:n_a]))
    return rows


def to_md(title, rows):
    keys = [k for k in rows[0] if k != "share"]
    lines = [f"### {title}", "", "| " + " | ".join(keys) + " | selection share |", "|" + "---|" * (len(keys) + 1)]
    for r in rows:
        vals = [f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys]
        share = ", ".join(f"{m.split('-')[-1]}={v:.2f}" for m, v in r["share"].items())
        lines.append("| " + " | ".join(vals) + f" | {share} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--outcomes", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--only", default="alpha,weights,ablation,variants,shift")
    ap.add_argument("--out", default="results/sweeps")
    a = ap.parse_args()
    Q, O = read_jsonl(a.queries), read_jsonl(a.outcomes)
    models = [m["model_id"] for m in json.load(open(a.models))["models"]]
    todo = set(a.only.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sections, data = [], {}
    if "alpha" in todo:
        data["alpha"] = run_alpha(Q, O, models, a.seeds); sections.append(to_md("E6: exploration coefficient α", data["alpha"]))
    if "weights" in todo:
        data["weights"] = run_weights(Q, O, models, a.seeds, a.alpha); sections.append(to_md("E2: reward weights (LinUCB)", data["weights"]))
    if "ablation" in todo:
        data["ablation"] = run_ablation(Q, O, models, a.seeds, a.alpha); sections.append(to_md("E4: feature ablation (LinUCB)", data["ablation"]))
    if "variants" in todo:
        data["variants"] = run_variants(Q, O, models, a.seeds, a.alpha, False); sections.append(to_md("Bandit variants, stationary stream", data["variants"]))
    if "shift" in todo:
        data["shift"] = run_variants(Q, O, models, a.seeds, a.alpha, True); sections.append(to_md("E5: workload shift (no warm-up)", data["shift"]))
    (out / "sweeps.json").write_text(json.dumps(data, indent=2))
    md = "\n\n".join(sections)
    (out / "sweeps.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
