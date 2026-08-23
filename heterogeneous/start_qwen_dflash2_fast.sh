#!/bin/bash
# DFlash2, reversed pipeline, 24/40 split, BF16 KV. The fastest general profile
# on this pair and a strict improvement on start_qwen_dflash2.sh: quicker per
# step, quicker at prefill, a slightly larger pool, and an UNQUANTIZED KV cache.
#
# Three things make it, all measured:
#
# 1. Reversed pipeline (GPU_IDS=1,0). vLLM puts the LM head, sampler and drafter
#    on the LAST rank. DFlash2's drafter is ~1.2 GiB and runs eagerly, so leaving
#    it on the 3060 is expensive: reversal is worth -12.6% step time here against
#    MTP's -7.7%.
#
# 2. A 24/40 split. At concurrency 1 a pipeline does not overlap, so the step is
#    the SUM of the stages: a layer on the 3060 costs 0.21GB/0.360 = 0.58 ms and
#    saves only 0.21/0.896 = 0.23 ms on the 5070 Ti. Measured 0.48 ms/layer
#    across 26/24/20/16 -- it is monotone, nothing balances, and the only limit
#    is 5070 Ti memory. The old 26/38 split maximised context, not speed.
#
# 3. BF16 KV -- for CUDA graphs, not for precision. Under speculative decoding
#    vLLM consults the attention backend's AttentionCGSupport. FlashInfer (which
#    fp8 forces, since FlashAttention cannot do fp8 on sm120/sm86) reports
#    UNIFORM_SINGLE_TOKEN_DECODE, so every fp8 run is silently downgraded to
#    piecewise CUDA graphs -- the line is in the server log:
#      "CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for
#       attention backend FlashInferBackend; setting cudagraph_mode=PIECEWISE"
#    FLASH_ATTN keeps FULL_AND_PIECEWISE, and BF16 is what selects it here.
#
# Measured, bench/real_rep.sh, 3 reps each, concurrency 1, reversed 24/40:
#   KV       ms/step    tok/step     decode tok/s   pool      graphs
#   fp8      35.5-35.6  3.04-3.21    88.1-92.9      45,085    piecewise
#   int8pth  34.1-34.6  3.12-3.29    93.5-99.6      44,387    full
#   BF16     33.9-34.0  3.15-3.27    95.2-98.8      34,539    full   <- this
#
# and prefill at the same split, which is where BF16 separates from int8pth:
#   depth    fp8         int8pth     BF16
#    4,251   1075 tok/s  1010 tok/s  1141 tok/s
#   16,804   1167        932         1181
#   29,355   1016        715         1008
#
# So BF16 matches or beats fp8 at every prefill depth AND decodes 4.5% quicker,
# while int8pth buys the same graph win but gives up 30% of prefill at 29k.
# BF16 is also the only one of the three that does not quantize the cache at all.
#
# The one thing fp8 still wins is pool: 45,085 against 34,539. Both cover a full
# 32k request, but fp8 allows 1.38x concurrency there against BF16's 1.05x, so
# for several long requests in flight use KV=fp8 DFLASH_KV_MEMORY=1750000000.
#
# For maximum decode speed at reduced context:
#   PP_LAYERS=20,44 KV=int8pth MAX_LEN=16384 DFLASH_KV_MEMORY=1300000000
#   -> 32.6 ms/step, 97.4-103.2 tok/s, 18,770-token pool.
#   PP_LAYERS=16,48 KV=int8pth MAX_LEN=8192 DFLASH_TOKENS=5 MAX_SEQS=2
#   DFLASH_KV_MEMORY=1000000000 -> 30.3-30.6 ms/step, 98.6-104.2 tok/s, 10,436.
#
# WHY THIS STAYS AT 32k WHILE start_qwen_dflash2.sh RUNS 88k. Reversing the
# pipeline puts the drafter, LM head and sampler on the 5070 Ti, and that is
# worth +13.4% decode -- but only at short context, and it does NOT free
# memory. It moves the bottleneck: the 5070 Ti now carries 40 of 64 layers AND
# the drafter (10.62 GiB of weights, ~1.4 GiB spare) while the 3060 holds 24
# layers and idles on 2.67 GiB. Measured, every row probed with a real prompt
# near MAX_LEN followed by a health check:
#
#   split  drafter   cap    KV    MAX_LEN  pool      decode   prefill  long prompt
#   44,20  3060      3.2G   fp8    88,000   97,962   85.77    991/s    survives
#   28,36  5070 Ti   3.2G   fp8    88,000  117,886   87.60    761/s    survives
#   24,40  5070 Ti   2.15G  bf16   32,768   34,539   97.24    957/s    survives  <- shipped
#   24,40  5070 Ti   2.6G   fp8    60,000   85,123   90.66    --       OOM-KILLED
#   24,40  5070 Ti   3.2G   fp8    88,000  117,886   --       --       dies at startup
#
# So the 13% is a SHORT-CONTEXT win. At 88k the reversed split has to give
# layers back to the 3060 to fit, and the advantage collapses to +2.1% decode
# while costing 23% of prefill (761 against 991 tok/s -- 105 s to first token on
# an 80k prompt against 81 s). For long context the forward split is the better
# machine; this profile is for short prompts where 97 tok/s is the point.
#
# DO NOT raise DFLASH_KV_MEMORY here. 2.6 GB boots, benches at 90.66 tok/s, and
# then dies with a 500 partway into a 55k prompt. The shipped 2.15 GB is
# verified to survive a 30,038-token prompt.

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPU_IDS=${GPU_IDS:-1,0}
export PP_LAYERS=${PP_LAYERS:-24,40}
export MAX_LEN=${MAX_LEN:-32768}
export GPU_UTIL=${GPU_UTIL:-0.915}
export KV=${KV:-bf16}
export MAX_SEQS=${MAX_SEQS:-4}
export SPEC=dflash2
export DFLASH_TOKENS=${DFLASH_TOKENS:-7}
export VLLM_DFLASH_CUDAGRAPH=${VLLM_DFLASH_CUDAGRAPH:-0}

# Pinned for the same reason as start_qwen_dflash2.sh: startup profiling
# otherwise mis-sizes the pool and CUDA graph capture then exhausts a card.
# BF16 needs 1.9 GiB to cover a full 32k window at this split; 2.15 GB leaves
# headroom without OOMing during capture.
export EXTRA_ARGS="--kv-cache-memory=${DFLASH_KV_MEMORY:-2150000000} ${EXTRA_ARGS:-}"

exec "$DIR/start_qwen.sh"
