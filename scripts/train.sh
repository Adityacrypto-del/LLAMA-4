#!/usr/bin/env bash
# Fine-tune Llama-4 Scout with 8-bit QLoRA adapters.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python -m qvaidya.train --config "${1:-configs/train_config.yaml}"
