#!/usr/bin/env bash
# Evaluate the BF16 baseline and the 8-bit QLoRA model on all benchmarks.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="src:${PYTHONPATH:-}"
CONFIG="${1:-configs/eval_config.yaml}"

python -m qvaidya.evaluate --config "$CONFIG" --no-adapter --bf16 --tag bf16_base
python -m qvaidya.evaluate --config "$CONFIG" --tag 8bit_qlora
