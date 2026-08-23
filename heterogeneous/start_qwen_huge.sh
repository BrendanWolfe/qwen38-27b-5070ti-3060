#!/bin/bash
# Qwen3.8's full native 262,144-token context on the 5070 Ti + 3060 pair.
#
# FP8 KV is 2 KB per token per full-attention layer, and the binding rank (the
# 5070 Ti, which holds 11 of the model's 16 full-attention layers) has ~4.2 GiB
# for KV -- enough for ~156k tokens, not 262k. vLLM 0.27.1's
# int4_per_token_head cache is about half that per token, which is what makes
# the native context fit: a measured 284,234-token pool at MAX_LEN=262144.
#
# The per-token-head quant modes exist only in the Triton attention backend,
# which is much slower than FlashInfer at long context. This is a mode for
# requests that would not otherwise fit, not a faster one -- keep
# start_qwen_stable.sh as the general server and switch to this only when a
# request exceeds its window. See heterogeneous/README.md for measurements.
#
# Untested past what README.md reports; KVarN (kvarn/, CTX=huge upstream) is
# denser still but has never been run on the V2 model runner that MTP+PP needs.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPU_IDS=${GPU_IDS:-0,1}
export PP_LAYERS=${PP_LAYERS:-44,20}
export MAX_LEN=${MAX_LEN:-262144}
export KV=${KV:-int4pth}
export SPEC=${SPEC:-mtp}
export DRAFT_TOKENS=${DRAFT_TOKENS:-3}
export MAX_SEQS=${MAX_SEQS:-4}

exec "$DIR/start_qwen.sh"
