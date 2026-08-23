#!/bin/bash
# Single-stream long-context profile: KVarN 4/2-bit KV, ONE request at a time,
# 160k context in a 214,468-token pool at the stable profile's decode rate.
#
# This is the one shape in which KVarN wins on this pair. Measured against the
# other MTP profiles (bench/real_rep.sh, 8 realistic 1,024-token prompts at
# concurrency 1, greedy):
#
#   profile              KV       slots  ctx    pool      decode
#   start_qwen_solo.sh   KVarN      1    160k   214,468   73.5 tok/s, 35.6 ms/step
#   start_qwen_stable.sh fp8        8    140k   155,978   73.6-75.5 tok/s
#   start_qwen_huge.sh   int4pth    4    262k   284,234   batch only (~112 tok/s prefill at depth)
#
# So: 37% more pool and 20k more context than stable, at the same decode rate
# and the same tokens/step (2.51 against 2.51-2.62) -- but it serves ONE stream.
# That is not a tuning choice, it is what KVarN costs here. Because KVarN
# compresses KV to 840 B/token/layer, vLLM has to LENGTHEN the attention block
# to 2048 tokens so the attention page still covers the fixed ~1.6 MB
# Gated-DeltaNet page (fp8 gets an 800-token block for the same reason). Every
# scheduler slot then pays that coarser granularity, and the per-slot cost is
# steep -- normalised to a common 3.0 GiB of pool:
#
#   1 slot  179,621 tokens     4 slots  126,844     8 slots  96,172
#
# Use start_qwen_stable.sh for anything with concurrent users: it totals
# 301 tok/s over 8 streams where this profile is capped at one. Use this when a
# single long document has to fit and decode must stay at full speed, and
# start_qwen_huge.sh when the request is longer than 160k and you can wait.
#
# Needs kvarn/install.sh (which also applies kvarn-pp-pool-budget.patch --
# without it KVarN pins max_num_seqs to 1 on its own, for the wrong reason).
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPU_IDS=${GPU_IDS:-0,1}
export PP_LAYERS=${PP_LAYERS:-44,20}
export MAX_LEN=${MAX_LEN:-160000}
export KV=${KV:-kvarn}
export SPEC=${SPEC:-mtp}
export DRAFT_TOKENS=${DRAFT_TOKENS:-3}
export MAX_SEQS=${MAX_SEQS:-1}

exec "$DIR/start_qwen.sh"
