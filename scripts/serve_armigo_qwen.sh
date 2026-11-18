#!/usr/bin/env bash
set -euo pipefail

# Run in the WSL environment provisioned by the local vLLM launcher skill.
# Every knob is an environment variable; the first argument is still the port (default 8038).
#   ARMIGO_VLLM_RUNTIME       vLLM runtime directory                         (~/.local/share/vllm-qwen)
#   ARMIGO_VLLM_MAX_LEN       --max-model-len                                (4096)
#   ARMIGO_VLLM_MAX_SEQS      --max-num-seqs; 4 lets parallel episode workers share the server   (4)
#   ARMIGO_VLLM_GPU_UTIL      --gpu-memory-utilization; 0.76 leaves ~7 GB of the 32 GB RTX 5090 for the
#                             detector process. Raise it only after checking nvidia-smi with the detector loaded.  (0.76)
#   ARMIGO_VLLM_PREFIX_CACHE  1 = prefix caching on. The decision prompts share a fixed system prefix.  (1)
#   ARMIGO_VLLM_EAGER         1 = --enforce-eager (no CUDA graphs). Eager is the memory-safe default: graph capture
#                             costs extra GPU memory and start-up time. Set 0 to A/B decode latency once there is
#                             headroom (the decision answers are ~10 tokens, so prefill dominates and the gain is small).  (1)
#   ARMIGO_VLLM_EXTRA         extra arguments appended verbatim
runtime="${ARMIGO_VLLM_RUNTIME:-$HOME/.local/share/vllm-qwen}"
port="${1:-8038}"
max_len="${ARMIGO_VLLM_MAX_LEN:-4096}"
max_seqs="${ARMIGO_VLLM_MAX_SEQS:-4}"
gpu_util="${ARMIGO_VLLM_GPU_UTIL:-0.76}"
export CUDA_HOME="$runtime/venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$runtime/venv/bin:$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$runtime/linklib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$runtime/linklib:$CUDA_HOME/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

args=(
  --revision 482ca0f3832238542f8f5295dde86b5f22711d80
  --served-model-name qwen3.8-vllm --host 127.0.0.1 --port "$port"
  --max-model-len "$max_len" --max-num-seqs "$max_seqs" --gpu-memory-utilization "$gpu_util"
  --kv-cache-dtype nvfp4
  --limit-mm-per-prompt '{"image":1,"video":0}'
)
if [ "${ARMIGO_VLLM_PREFIX_CACHE:-1}" = "1" ]; then
  args+=(--enable-prefix-caching)
else
  args+=(--no-enable-prefix-caching)
fi
if [ "${ARMIGO_VLLM_EAGER:-1}" = "1" ]; then
  args+=(--enforce-eager)
fi
# shellcheck disable=SC2206
extra=(${ARMIGO_VLLM_EXTRA:-})

exec "$runtime/venv/bin/vllm" serve nvidia/Qwen3.8-27B-NVFP4 "${args[@]}" "${extra[@]}"
