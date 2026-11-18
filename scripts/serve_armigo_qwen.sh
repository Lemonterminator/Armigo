#!/usr/bin/env bash
set -euo pipefail

# Run in the WSL environment provisioned by the local vLLM launcher skill.
runtime="${ARMIGO_VLLM_RUNTIME:-$HOME/.local/share/vllm-qwen}"
port="${1:-8038}"
export CUDA_HOME="$runtime/venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$runtime/venv/bin:$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$runtime/linklib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$runtime/linklib:$CUDA_HOME/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

exec "$runtime/venv/bin/vllm" serve nvidia/Qwen3.8-27B-NVFP4 \
  --revision 482ca0f3832238542f8f5295dde86b5f22711d80 \
  --served-model-name qwen3.8-vllm --host 127.0.0.1 --port "$port" \
  --max-model-len 4096 --max-num-seqs 1 --gpu-memory-utilization 0.76 \
  --kv-cache-dtype nvfp4 --no-enable-prefix-caching --enforce-eager \
  --limit-mm-per-prompt '{"image":1,"video":0}'
