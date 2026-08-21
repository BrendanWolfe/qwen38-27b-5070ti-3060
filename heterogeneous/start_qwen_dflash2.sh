#!/bin/bash
# Experimental 32k DFlash2 profile. This is separate from the frozen 140k MTP
# launcher because DFlash2 needs BF16 KV and additional PP auxiliary-state relay.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MAX_LEN=${MAX_LEN:-32768}
export GPU_UTIL=${GPU_UTIL:-0.915}
export KV=${KV:-bf16}
export MAX_SEQS=${MAX_SEQS:-4}
export SPEC=dflash2
export DFLASH_TOKENS=${DFLASH_TOKENS:-7}
export VLLM_DFLASH_CUDAGRAPH=${VLLM_DFLASH_CUDAGRAPH:-0}

# Manual sizing prevents startup profiling from assigning ~2.9 GiB to rank 1,
# then exhausting the 12 GiB card when target CUDA graphs are captured. This
# pool provides 33,506 tokens with the current model and vLLM build.
export EXTRA_ARGS="--kv-cache-memory=${DFLASH_KV_MEMORY:-2500000000} ${EXTRA_ARGS:-}"

exec "$DIR/start_qwen.sh"
