# Comparison: routing policies and bandit variants (synthetic smoke test)

> **Synthetic data only.** Everything below comes from `evaluation/synth.py`, a hand-made outcome
> generator with assumed per-category skill levels. It tests that the code works and shows what the
> analysis will look like. **These are not experimental results and must not be reported in the paper.**

Reproduce:
```bash
python -m evaluation.synth --out-dir data/synthetic --per-category 200 --seed 0
python -m evaluation.replay  --queries data/synthetic/queries.jsonl --outcomes data/synthetic/outcomes.jsonl \
    --models serving/models.json --out results/synthetic_main
python -m experiments.sweeps --queries data/synthetic/queries.jsonl --outcomes data/synthetic/outcomes.jsonl \
    --models serving/models.json --out results/synthetic_sweeps
```
(1,400 queries; 377 dev / 1,023 test; 5 seeds; reward weights 1.0 / 0.3 / 0.2 unless stated.)

## What the variants are

| Policy | Exploration mechanism | Expected strength | Expected weakness |
|---|---|---|---|
| LinUCB (proposed) | optimism: θᵀx + α·√(xᵀA⁻¹x) | deterministic, inspectable, one knob (α) | over-explores if α is too large; assumes stationarity |
| Linear Thompson sampling | sample θ from posterior N(θ̂, v²A⁻¹) | often strong empirically; randomized | results vary by seed; v needs tuning |
| ε-greedy ridge | uniform random with prob. ε | simplest | keeps exploring forever at rate ε |
| Sliding-window LinUCB (W=200) | LinUCB on recent data only | adapts to drift | forgets useful data in stationary periods |
| Rule-based (category → best dev model) | none | strong when category explains most variance | can't use within-category signals |

## What the synthetic runs revealed (about the method, not about Qwen3)

1. **Reward weights decide the winner, not the algorithm.** With cost/latency weights at 0, large-only is the best
   fixed policy; at 0.1/0.05 rule-based is; at ≥0.3/0.2 small-only is. LinUCB's model mix follows the weights
   (14B share falls from 0.73 to 0.00). → E2 must be presented as a curve, and the headline comparison must name the
   weights it uses.
2. **LinUCB did not beat the best fixed policy here.** On every weight setting LinUCB's mean reward was 0.01–0.05
   below the best fixed policy. In this generator, task category explains most of the variance, which is exactly
   the case where a per-category rule is hard to beat. Real data may differ — this is what E1/E3 must test, and the
   paper already avoids claiming a win.
3. **Per-query hindsight regret is dominated by noise.** With binary quality, the per-query oracle "knows" which
   model got lucky on each query, so even the best policy has regret ≈ 290 over 1,023 queries. A more informative
   reference for the paper is regret against the **best fixed policy per category in hindsight**. *(Not yet
   implemented — candidate next step.)*
4. **α in the range 0.5–1.0 did best; α = 0 (pure greedy) was worst.** Differences are small relative to seed
   variation, so E6 needs ≥5 seeds and intervals.
5. **Ablation: the bias-only router (no features) was worst**, but removing any single group changed little —
   features overlap (category already implies most of difficulty/cues). Real prompts with more within-category
   variation should make this more informative.
6. **Workload shift: the "post-shift reward" metric is confounded.** Reward drops after the shift for every policy,
   including rule-based, because the second phase contains harder categories. Adaptation should instead be
   measured as *regret per step relative to the best fixed policy for the new mix*, in a window after the shift.
   *(Candidate next step.)*
7. **Variants vs LinUCB** (stationary): TS, ε-greedy and sliding-window were all within ~0.03 mean reward of
   LinUCB; none is clearly better on this data. Keep LinUCB as the primary method (paper), report TS as a secondary
   comparison if real data shows a gap.

## Recommendation

Keep LinUCB as the proposed method. Before running on real outcomes:
- add category-level hindsight regret and the post-shift regret window (items 3 and 6);
- fix the reward weights from an explicit argument (e.g. relative GPU cost) before looking at test results;
- treat the rule-based policy as the strongest baseline, not random.

## Next experiments to consider
- Warm-start comparison: no warm-up vs. dev warm-up vs. warm-start from a supervised predictor.
- Embedding features (`EmbeddingProjector`) — matters only if within-category variation is large.
- Budget-constrained selection (cf. PILOT) — choose the best model subject to a running cost budget.
