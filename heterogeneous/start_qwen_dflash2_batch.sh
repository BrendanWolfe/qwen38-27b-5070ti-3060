#!/bin/bash
# Concurrent DFlash2 profile, measured on the RTX 5070 Ti + RTX 3060 pair:
# KVarN 4-bit keys / 2-bit values, reversed pipeline, four scheduler seats,
# 147,456 API context in a 151,503-token pool.
#
# A 140,028-token needle retrieval passed and left /health=200. Four distinct
# 4k requests completed with no preemption (three resident, one queued; 94.3%
# peak cache). Do not reuse the solo profile's 2.8 GB pin here: it booted and
# passed one 150k request, then OOMed rank 1 on a 24 MiB allocation at C4.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPU_IDS=${GPU_IDS:-1,0}
export PP_LAYERS=${PP_LAYERS:-30,34}
export MAX_LEN=${MAX_LEN:-147456}
export GPU_UTIL=${GPU_UTIL:-0.915}
export KV=${KV:-kvarn}
export MAX_SEQS=${MAX_SEQS:-4}
export SPEC=dflash2
export DFLASH_TOKENS=${DFLASH_TOKENS:-7}
export SPEC_ATTN=${SPEC_ATTN:-0}
export VLLM_DFLASH_CUDAGRAPH=${VLLM_DFLASH_CUDAGRAPH:-0}
export EXTRA_ARGS="--kv-cache-memory=${DFLASH_KV_MEMORY:-2400000000} ${EXTRA_ARGS:-}"

exec "$DIR/start_qwen.sh"
