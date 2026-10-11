#!/usr/bin/env bash
# Launch one OpenAI-compatible vLLM server per candidate model.
# Approximate bf16 weight memory: 4B ~8 GB, 8B ~16 GB, 14B ~30 GB (+ KV cache).
# With one GPU, run them one at a time: collect.py processes models in order and is resumable.
#   ./serving/launch_vllm.sh 4b     # or 8b / 14b / all
set -euo pipefail
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_LEN="${MAX_LEN:-8192}"

serve() {  # name port
  vllm serve "$1" --port "$2" --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAX_LEN" \
    --served-model-name "$1" &
}

case "${1:-all}" in
  4b)  serve Qwen/Qwen3-4B 8001 ;;
  8b)  serve Qwen/Qwen3-8B 8002 ;;
  14b) serve Qwen/Qwen3-14B 8003 ;;
  all) serve Qwen/Qwen3-4B 8001; serve Qwen/Qwen3-8B 8002; serve Qwen/Qwen3-14B 8003 ;;
  *) echo "usage: $0 [4b|8b|14b|all]"; exit 1 ;;
esac
wait
