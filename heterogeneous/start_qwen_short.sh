#!/bin/bash
# Experimental short-context profile. The llama-swap stable model uses
# start_qwen_stable.sh; tune this wrapper without changing that profile.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MAX_LEN=${MAX_LEN:-65536}
export GPU_UTIL=${GPU_UTIL:-0.91}
# FP8 was substantially faster than BF16 KV in both 32k and 65k tests.
export KV=${KV:-fp8}
export MAX_SEQS=${MAX_SEQS:-4}
export SPEC=${SPEC:-mtp}
export DRAFT_TOKENS=${DRAFT_TOKENS:-3}

exec "$DIR/start_qwen.sh"
