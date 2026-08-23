#!/bin/bash
# Single-stream long-context profile: KVarN 4/2-bit KV, ONE request at a time,
# a ~215,000-token pool at the stable profile's decode rate.
#
# This is the one shape in which KVarN wins on this pair. Measured against the
# other MTP profiles (bench/real_rep.sh, 8 realistic 1,024-token prompts at
# concurrency 1, greedy):
#
#   profile              KV       slots  ctx    pool      decode
#   start_qwen_solo.sh   KVarN      1    143k   215,883   73.5 tok/s, 35.6 ms/step
#   start_qwen_stable.sh fp8        8    140k   155,978   73.6-75.5 tok/s
#   start_qwen_huge.sh   int4pth    4    262k   284,234   batch only (~112 tok/s prefill at depth)
#
# Read that honestly: against stable this buys 38% more POOL and almost no extra
# max context (143,360 against 140,000), at the same decode rate and the same
# tokens/step (2.51 against 2.51-2.62), and it serves ONE stream. The pool is
# what it is for -- a single long document over many turns, where the extra
# 60,000 tokens of pool hold the prefix cache that makes turn 2 cheap. If you
# want more max context take start_qwen_huge.sh; if you want users, stable.
#
# WHY 143,360 AND NOT MORE. The pool is ~215k either way (it is memory divided
# by bytes-per-token, and does not move with MAX_LEN); MAX_LEN only caps the
# longest single request, and that cap is set by whichever rank has least KV per
# full-attention layer. Rank 1's profiled PEAK ACTIVATION is bimodal between
# identical launches -- 0.32 GiB or ~1.22 GiB, the swing README already notes --
# and it decides which rank binds:
#
#   rank1 peak_act 0.32  ->  rank1 kv ~1.7 GiB, rank 0 binds  ->  supports 258,048
#   rank1 peak_act 1.22  ->  rank1 kv  0.82 GiB, rank 1 binds ->  supports 143,360
#
# So the ceiling swings 1.8x between identical starts. 143,360 (= 70 x 2048, a
# whole number of KVarN blocks) is the WORST case, verified to boot on a run
# that profiled the unfavourable way. MAX_LEN=258048 works when rank 1 profiles
# low and fails outright when it does not -- it is a gamble, not a setting.
#
# --kv-cache-memory does NOT fix this: it is an absolute per-GPU byte count
# applied to every rank, and with 11 full-attention layers on rank 0 against 5 on
# rank 1 any single value either starves rank 0 or overcommits rank 1. Upstream
# pins KV_MEM on one card for exactly this determinism; the trick does not carry
# to an uneven pipeline.
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
export MAX_LEN=${MAX_LEN:-143360}
export KV=${KV:-kvarn}
export SPEC=${SPEC:-mtp}
export DRAFT_TOKENS=${DRAFT_TOKENS:-3}
export MAX_SEQS=${MAX_SEQS:-1}

exec "$DIR/start_qwen.sh"
