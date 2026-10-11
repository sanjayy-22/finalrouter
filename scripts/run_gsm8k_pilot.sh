#!/usr/bin/env bash
# GSM8K pilot: serve each model in turn with vLLM, collect outcomes, then compare routers.
#
#   bash scripts/run_gsm8k_pilot.sh                                   # bf16 models, needs ~40 GB+ GPU
#   MODELS_JSON=serving/models_4090.json bash scripts/run_gsm8k_pilot.sh   # RTX 4090 (24 GB): FP8 models
#
# Env: MODELS_JSON (model config), MAX_LEN (context length, default 4096), GPU_UTIL (default 0.90)
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS_JSON="${MODELS_JSON:-serving/models.json}"
MAX_LEN="${MAX_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.90}"
OUT_SUFFIX="$(basename "$MODELS_JSON" .json)"
OUTCOMES="data/outcomes_gsm8k_${OUT_SUFFIX}.jsonl"

python -m data.build_gsm8k --n 300 --dev-frac 0.3 --seed 0 --out data/queries_gsm8k.jsonl

# "name port" per line, in config order (small -> large)
mapfile -t MODELS < <(python -c "import json,sys; [print(m['name'], m['port']) for m in json.load(open(sys.argv[1]))['models']]" "$MODELS_JSON")

for entry in "${MODELS[@]}"; do
  read -r MODEL P <<< "$entry"
  echo "=== serving $MODEL on :$P"
  vllm serve "$MODEL" --port "$P" --served-model-name "$MODEL" --max-model-len "$MAX_LEN" \
      --gpu-memory-utilization "$GPU_UTIL" > "logs_vllm_${P}.txt" 2>&1 &
  PID=$!
  until curl -sf "http://localhost:$P/v1/models" > /dev/null; do
    kill -0 $PID 2>/dev/null || { echo "vLLM exited, see logs_vllm_${P}.txt"; exit 1; }
    sleep 5
  done
  # only the model being served answers; the others are skipped and picked up on their turn
  python -m evaluation.collect --queries data/queries_gsm8k.jsonl --models "$MODELS_JSON" --out "$OUTCOMES"
  kill $PID; wait $PID 2>/dev/null || true
  sleep 5   # let the GPU memory be released before the next model
done

python -m evaluation.replay --queries data/queries_gsm8k.jsonl --outcomes "$OUTCOMES" \
    --models "$MODELS_JSON" --seeds 0 1 2 3 4 --out "results/gsm8k_pilot_${OUT_SUFFIX}"
