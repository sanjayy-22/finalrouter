import numpy as np

from evaluation.replay import build_setup, evaluate, run_policy, make_stream
from evaluation.synth import generate
from router.baselines import FixedPolicy
from router.reward import RewardConfig

MODELS = ["qwen3-4b", "qwen3-8b", "qwen3-14b"]


def test_end_to_end_on_synthetic():
    q, o = generate(per_category=30, seed=3)
    s = build_setup(q, o, MODELS)
    res = evaluate(s, seeds=[0, 1])
    assert set(res["policies"]) == {"small_only", "large_only", "random", "rule_based", "linucb"}
    assert res["policies"]["small_only"]["selection_share"]["qwen3-4b"] == 1.0
    for d in res["policies"].values():
        assert d["regret"]["mean"] >= -1e-9          # oracle is a per-query upper bound
    assert res["n_test"] + res["n_dev"] + res["excluded_queries"] == len(q)


def test_only_chosen_models_outcome_is_used():
    q, o = generate(per_category=10, seed=4)
    s = build_setup(q, o, MODELS)
    steps = run_policy(FixedPolicy("qwen3-8b"), s, make_stream(s, 0))
    for st in steps:
        assert st["reward"] == s.reward(st["query_id"], "qwen3-8b")
        assert st["oracle"] >= st["reward"]


def test_rules_are_fitted_on_dev_only():
    cfg = RewardConfig(w_cost=0, w_latency=0)
    queries = [{"query_id": f"q{i}", "prompt": "x", "task_category": "qa", "split": "dev" if i < 5 else "test"}
               for i in range(10)]
    outcomes = []
    for qd in queries:
        dev = qd["split"] == "dev"
        for m in MODELS:   # dev: small is best; test: large is best
            good = (m == "qwen3-4b") if dev else (m == "qwen3-14b")
            outcomes.append({"query_id": qd["query_id"], "model_id": m, "task_category": "qa",
                             "split": qd["split"], "quality_score": 1.0 if good else 0.0, "failed": False,
                             "latency_ms": 1.0, "cost": 1.0})
    s = build_setup(queries, outcomes, MODELS, cfg)
    assert s.rules.rules == {"qa": "qwen3-4b"}
