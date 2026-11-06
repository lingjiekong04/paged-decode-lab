#!/usr/bin/env bash
# Source from any directory. Does not activate or modify any other environment.
PAGED_DECODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PAGED_DECODE_ROOT
export PYTHONPATH="$PAGED_DECODE_ROOT/src:$PAGED_DECODE_ROOT/vendor/nano-vllm${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="$PAGED_DECODE_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PAGED_DECODE_ROOT/.cache/torchinductor"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
