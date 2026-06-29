#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf}"
THREADS="${THREADS:-4}"
LOG_DIR="${LOG_DIR:-experiments/results}"
BENCH="${BENCH:-BitNet/build/bin/llama-bench}"

mkdir -p "${LOG_DIR}"

for prompt_tokens in 256 512 1024; do
  for generated_tokens in 32 128 256; do
    echo "prompt_tokens=${prompt_tokens} generated_tokens=${generated_tokens} threads=${THREADS}"
    "${BENCH}" \
      -m "${MODEL}" \
      -n "${generated_tokens}" \
      -ngl 0 \
      -b 1 \
      -t "${THREADS}" \
      -p "${prompt_tokens}" \
      -r 5 \
      > "${LOG_DIR}/bitnet_p${prompt_tokens}_n${generated_tokens}_t${THREADS}.log" 2>&1
  done
done
