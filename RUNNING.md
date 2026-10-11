# Running the GSM8K pilot on one RTX 4090 (24 GB)

## Why FP8
| Model | bf16 weights | FP8 weights | Fits on 24 GB? |
|---|---|---|---|
| Qwen3-4B | ~8 GB | ~4 GB | bf16 ✔ / FP8 ✔ |
| Qwen3-8B | ~16 GB | ~8 GB | bf16 ✔ (tight) / FP8 ✔ |
| Qwen3-14B | ~30 GB | ~15 GB | bf16 ✘ / FP8 ✔ |

Weights ≈ parameters × bytes per parameter; vLLM also needs room for the KV cache.
`serving/models_4090.json` uses Qwen's FP8 checkpoints for **all three** models so precision is the same
across the pool (mixing bf16 small models with an FP8 large model would confound the comparison).
vLLM documents FP8 (W8A8) support on Ada Lovelace GPUs such as the 4090.
Record "FP8" in the paper's Table I.

## 1. Machine setup
- Linux, or Windows with **WSL2** (vLLM does not run natively on Windows).
- Recent NVIDIA driver; check with `nvidia-smi`.
- ~40 GB free disk for the three FP8 checkpoints plus the Python environment.

```bash
git clone https://github.com/sanjayy-22/finalrouter && cd finalrouter
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt vllm
python -m pytest            # should pass
```

Check that each FP8 repo exists before running (names in `serving/models_4090.json`):
`https://huggingface.co/Qwen/Qwen3-4B-FP8`, `.../Qwen3-8B-FP8`, `.../Qwen3-14B-FP8`.

## 2. Run
```bash
MODELS_JSON=serving/models_4090.json bash scripts/run_gsm8k_pilot.sh
```
What it does:
1. builds `data/queries_gsm8k.jsonl` (300 GSM8K test problems, 90 dev / 210 test, seed 0);
2. for each model in turn: starts vLLM, waits until it is ready, sends all 300 problems, stops it;
3. runs the replay comparison of small-only, large-only, random, rule-based and LinUCB.

First run downloads the checkpoints. The collection step is resumable: if it stops, run the same command again.

## 3. Outputs
| File | Contents |
|---|---|
| `data/outcomes_gsm8k_models_4090.jsonl` | one row per (question, model): answer, correct or not, latency, tokens, cost |
| `results/gsm8k_pilot_models_4090/tables.md` | policy comparison (paper Table VII layout) |
| `results/gsm8k_pilot_models_4090/results.json` | all metrics, per seed averages, reward traces |
| `logs_vllm_<port>.txt` | vLLM server logs |

Quick accuracy check per model:
```bash
python - <<'PY'
import json, collections
acc = collections.defaultdict(list)
for l in open("data/outcomes_gsm8k_models_4090.jsonl"):
    r = json.loads(l); acc[r["model_id"]].append(0 if r["failed"] else r["quality_score"] or 0)
for m, v in acc.items(): print(m, f"{sum(v)/len(v):.3f}", len(v))
PY
```

## 4. Troubleshooting
| Symptom | Fix |
|---|---|
| CUDA out of memory when vLLM starts | `MAX_LEN=2048` or `GPU_UTIL=0.85` before the command |
| "vLLM exited, see logs_vllm_8003.txt" | read that log; usually memory, a wrong model name, or an FP8 kernel/driver issue |
| FP8 model fails to load | update vLLM and the NVIDIA driver; fallback: an AWQ 4-bit build of the same model for **all three** sizes |
| Many answers marked wrong despite correct reasoning | open a few rows in the outcomes file; the model must end with `#### <number>` |
| WSL2 runs out of RAM while loading | raise the WSL memory limit in `%UserProfile%\.wslconfig` |

## 5. Send back
Push `data/outcomes_gsm8k_models_4090.jsonl` and `results/` (they are git-ignored, so use `git add -f`),
or paste `tables.md`, and the results can be analysed and written into the paper.
