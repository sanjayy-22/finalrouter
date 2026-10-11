# finalrouter — Adaptive LLM Model Routing with Contextual Bandits

A LinUCB router picks one of several LLMs (Qwen3-4B / 8B / 14B) for each query, balancing response
quality, cost and latency, and learns from feedback. The router is ordinary software; it never
changes model weights. The accompanying paper is in [`paper/`](paper/).

## How it fits together

```
query ─► router/features.py ─► router/linucb.py ─► model (vLLM) ─► evaluation/scoring.py
            x_t                    pick a_t                             quality Q
                                      ▲                                     │
                                      └──── update(A_a, b_a) ◄── router/reward.py: R = w_q·Q − w_c·C̃ − w_l·L̃
```

| Path | What it is |
|---|---|
| `router/` | features, LinUCB, baselines (small-only, large-only, random, rule-based), reward |
| `evaluation/collect.py` | runs every query on every model via vLLM, scores and logs outcomes (resumable) |
| `evaluation/scoring.py` | exact / multiple-choice / numeric / contains / unit-test / LLM-judge scorers |
| `evaluation/replay.py` | compares policies on the outcome log (bandit feedback), prints paper Tables VII–VIII |
| `evaluation/synth.py` | synthetic outcomes for testing without GPUs (**not** results) |
| `experiments/` | reward-weight, α, ablation, workload-shift sweeps; Thompson sampling / ε-greedy / sliding-window variants |
| `service/app.py` | live FastAPI router: `/route`, `/feedback`, `/stats`, audit log |
| `serving/` | vLLM launch script and model config |
| `notes/` | design notes and what the synthetic runs revealed |

## Quick start (no GPU)

```bash
pip install -r requirements.txt
python -m pytest
python -m evaluation.synth --out-dir data/synthetic
python -m evaluation.replay --queries data/synthetic/queries.jsonl \
    --outcomes data/synthetic/outcomes.jsonl --models serving/models.json --out results/synthetic
```

## Pilot: GSM8K (real questions)

`data/queries_gsm8k.jsonl` holds 300 problems sampled (seed 0) from the official GSM8K test split
([openai/grade-school-math](https://github.com/openai/grade-school-math), MIT), 90 dev / 210 test.
The numeric scorer reproduces 1319/1319 reference answers. On a machine with one ≥40 GB GPU:

```bash
bash scripts/run_gsm8k_pilot.sh      # serves 4B, 8B, 14B in turn, collects, then runs the replay comparison
```

## Real run (GPU machine)

```bash
pip install vllm
./serving/launch_vllm.sh 4b            # or all; one at a time on a single GPU
python -m evaluation.collect --queries data/queries.jsonl --models serving/models.json \
    --out data/outcomes.jsonl [--judge serving/judge.json]
python -m evaluation.replay --queries data/queries.jsonl --outcomes data/outcomes.jsonl \
    --models serving/models.json --seeds 0 1 2 3 4 --out results/main          # E1, E3
python -m experiments.sweeps --queries data/queries.jsonl --outcomes data/outcomes.jsonl \
    --models serving/models.json --out results/sweeps                          # E2, E4, E5, E6
ROUTER_MODELS=serving/models.json uvicorn service.app:app --port 9000          # live demo
```

## Rules that keep results honest
- Normalization ranges, difficulty priors, rule-based rules and hyperparameters come from the **dev** split only;
  results are reported on **test**.
- Qwen3 thinking mode is fixed for the whole study (`enable_thinking` in `serving/models.json`).
- The LLM judge must not be one of the routed models, and is validated on ~100 hand-graded responses.
- `unittest` scoring executes model-written code: run collection in a sandbox/container.
- Synthetic outputs never go into the paper.
