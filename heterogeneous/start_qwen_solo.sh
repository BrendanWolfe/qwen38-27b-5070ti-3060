#!/bin/bash
# Single-stream long-context profile: KVarN 4/2-bit KV, ONE request at a time,
# the model's full native 262,144 context in a 296,974-token pool at the batch
# profile's decode rate.
#
# This is the one shape in which KVarN wins on this pair. Measured against the
# other MTP profiles (bench/real_rep.sh, 8 realistic 1,024-token prompts at
# concurrency 1, greedy):
#
#   profile              KV       slots  ctx    pool      decode
#   start_qwen_solo.sh   KVarN      1    262k   296,974   72.4-73.6 tok/s, 34.8-35.4 ms/step
#   start_qwen_batch.sh  fp8        8    140k   155,978   73.6-75.5 tok/s
#   start_qwen_huge.sh   int4pth    4    262k   284,234   batch only (~112 tok/s prefill at depth)
#
# Against batch that is 90% more pool and 122k more context at the same decode
# rate and the same tokens/step (2.48-2.52 against 2.51-2.62) -- for ONE stream.
# It also holds a larger pool than start_qwen_huge.sh at the same 262,144
# context (296,974 against 284,234), so if you want the native context on a
# single stream this is the better of the two; huge keeps 4 slots.
#
# WHY THE FULL 262,144, AND WHY THAT IS NOT OBVIOUS. The pool is memory divided
# by bytes-per-token; MAX_LEN only caps the longest single request. What is NOT
# obvious is that the two are coupled through the memory PROFILE, and the
# coupling is NON-MONOTONIC. Measured, GPUs verified idle before each run:
#
#   MAX_LEN    rank1 peak_act   result
#   143,360        1.22 GiB     boots, 220,943-token pool
#   200,704        1.22 GiB     REFUSES to start (estimates 182,272)
#   245,760        1.22 GiB     REFUSES to start (estimates 149,504)
#   262,144        0.30 GiB     boots, 296,974-token pool
#
# So the model's native 262,144 is both the longest context AND the largest pool,
# while two shorter settings between do not boot at all. profile_run() calls
# _dummy_sampler_run() on the LAST pipeline rank only -- the 3060 -- and that
# dummy sampler allocation is what moves, 1.22 GiB against 0.30. Every other
# input is byte-identical across these runs: same max_num_batched_tokens (2048),
# same cudagraph capture sizes, same 2048-token attention block, same 1.45%
# mamba padding, same startup free memory. WHY max_model_len changes that
# allocation is NOT explained here -- only that it does, deterministically.
#
# Deterministic is the other half of it: three consecutive runs at 262,144 from
# verified-idle GPUs returned bit-identical numbers (296,974 tokens, peak_act
# 0.30/0.30). Earlier notes in this repo called this a run-to-run swing; it is
# not. Comparing runs that differed in MAX_LEN made a deterministic effect look
# like noise. If you change MAX_LEN here, re-measure rather than interpolate --
# a value between 143,360 and 262,144 may simply refuse to start.
#
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPU_IDS=${GPU_IDS:-0,1}
export PP_LAYERS=${PP_LAYERS:-44,20}
export MAX_LEN=${MAX_LEN:-262144}
export KV=${KV:-kvarn}
export SPEC=${SPEC:-mtp}
export DRAFT_TOKENS=${DRAFT_TOKENS:-3}
export MAX_SEQS=${MAX_SEQS:-1}

exec "$DIR/start_qwen.sh"
