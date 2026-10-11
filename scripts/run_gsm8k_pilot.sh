#!/usr/bin/env bash
# GSM8K pilot: serve each Qwen3 model in turn with vLLM, collect outcomes, then compare routers.
# Needs one GPU with enough memory for Qwen3-14B (~30 GB of bf16 weights + KV cache), e.g. A100 40GB / L40S 48GB.
# usage: bash scripts/run_gsm8k_pilot.sh
set -euo pipefail
cd "$(dirname "$0")/.."

python -m data.build_gsm8k --n 300 --dev-frac 0.3 --seed 0 --out data/queries_gsm8k.jsonl

declare -A PORT=( [Qwen/Qwen3-4B]=8001 [Qwen/Qwen3-8B]=8002 [Qwen/Qwen3-14B]=8003 )
for MODEL in Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B; do
  P=${PORT[$MODEL]}
  echo "=== serving $MODEL on :$P"
  vllm serve "$MODEL" --port "$P" --served-model-name "$MODEL" --max-model-len 8192 \
      --gpu-memory-utilization 0.90 > "logs_vllm_${P}.txt" 2>&1 &
  PID=$!
  until curl -sf "http://localhost:$P/v1/models" > /dev/null; do
    kill -0 $PID 2>/dev/null || { echo "vLLM exited, see logs_vllm_${P}.txt"; exit 1; }
    sleep 5
  done
  python -m evaluation.collect --queries data/queries_gsm8k.jsonl --models serving/models.json \
      --out data/outcomes_gsm8k.jsonl     # models whose server is down are skipped and retried next pass
  kill $PID; wait $PID 2>/dev/null || true
done

python -m evaluation.replay --queries data/queries_gsm8k.jsonl --outcomes data/outcomes_gsm8k.jsonl \
    --models serving/models.json --seeds 0 1 2 3 4 --out results/gsm8k_pilot
