#!/bin/bash
# Single-stream DFlash2 long-context profile, measured on the RTX 5070 Ti +
# RTX 3060 pair: KVarN 4-bit keys / 2-bit values, reversed pipeline, one seat,
# 200,192 API context in a 202,174-token pool.
#
# Validated with a 190,050-token needle retrieval (PASS, 592 prompt tok/s) and
# /health=200 afterward. The 2.8 GB cache pin is for this one-seat shape only;
# four concurrent requests need the smaller pin in start_qwen_dflash2_batch.sh.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPU_IDS=${GPU_IDS:-1,0}
export PP_LAYERS=${PP_LAYERS:-30,34}
export MAX_LEN=${MAX_LEN:-200192}
export GPU_UTIL=${GPU_UTIL:-0.915}
export KV=${KV:-kvarn}
export MAX_SEQS=${MAX_SEQS:-1}
export SPEC=dflash2
export DFLASH_TOKENS=${DFLASH_TOKENS:-7}
export SPEC_ATTN=${SPEC_ATTN:-0}
export VLLM_DFLASH_CUDAGRAPH=${VLLM_DFLASH_CUDAGRAPH:-0}
export EXTRA_ARGS="--kv-cache-memory=${DFLASH_KV_MEMORY:-2800000000} ${EXTRA_ARGS:-}"

exec "$DIR/start_qwen.sh"
