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
#   start_qwen_batch.sh  fp8        8    147k   147,456+  73.6-75.5 tok/s
#   start_qwen_huge.sh   int4pth    4    262k   284,234   batch only (~112 tok/s prefill at depth)
#
# Against batch that is roughly twice the pool and 115k more context at the same
# decode rate and the same tokens/step (2.48-2.52 against 2.51-2.62) -- for ONE
# stream. FP8 cannot reach this context at any MAX_LEN; it refuses above ~148k.
# It also holds a larger pool than start_qwen_huge.sh at the same 262,144
# context (296,974 against 284,234), so if you want the native context on a
# single stream this is the better of the two; huge keeps 4 slots.
#
# WHY THE FULL 262,144, AND WHY THAT IS NOT OBVIOUS. The pool is memory divided
# by bytes-per-token; MAX_LEN only caps the longest single request. What is NOT
# obvious is that the two are coupled through the startup memory PROFILE, and
# that the profile is not reproducible. vLLM profiles each pipeline rank
# independently, and each rank lands on a low (~0.3 GiB) or high (~1.2 GiB)
# activation peak; the pool follows, and so does whether the server starts at
# all. On the eight-slot batch profile an unchanged command has returned pools
# of 146,086 / 157,500 / 184,891 tokens and refusal thresholds of 148,096 /
# 159,744 / ~189,000. That is gotcha 38, and it is the thing to know before
# touching MAX_LEN anywhere in this repo.
#
# This profile appears to sit on the low branch. Measured at 262,144, GPUs
# verified idle before each run: EIGHT consecutive boots, all peak_act 0.3/0.3,
# all 296,974 tokens, no failures. Single runs at three shorter MAX_LEN values
# each drew the high branch instead:
#
#   MAX_LEN    rank peaks    result
#   143,360    1.22 / -      boots, 220,943-token pool
#   200,704    1.22 / -      refuses to start (estimates 182,272)
#   245,760    1.22 / -      refuses to start (estimates 149,504)
#   262,144    0.30 / 0.30   boots, 296,974-token pool  (x8)
#
# Read that table carefully: n=1 per shorter row against n=8 here. It is enough
# to justify shipping 262,144 -- it is simultaneously the longest context, the
# largest pool, and the only setting with repeat evidence behind it -- and it is
# NOT enough to conclude that 200,704 always fails or that MAX_LEN causes the
# branch. Those three rows may simply be unlucky draws.
#
# An earlier version of this comment claimed the effect was deterministic and
# driven by MAX_LEN, on the strength of three identical runs at 262,144. Three
# agreeing runs are three draws from the same branch, not a proof; the batch
# profile draws differently from an identical command. If you change MAX_LEN
# here, re-measure SEVERAL times rather than interpolating from one boot.
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

# RETRY, because 262,144 is above the worst startup-profiling draw (gotcha 38).
# This profile asks for more context than the high-activation branch can size a
# pool for, so a launch that draws it dies in ~20 s with "estimated maximum
# model length is 143360" instead of serving. The branch is redrawn on every
# launch, so the fix is to launch again: observed 8 clean boots, then a refusal,
# then clean boots again.
#
# The alternative was to drop MAX_LEN to the worst draw, which is 143,360 --
# that would throw away the entire point of the profile to avoid a retry that
# costs 20 seconds. A failed attempt dies at KV sizing, before graph capture,
# and a successful boot is ~27 s with a warm compile cache, so four attempts fit
# inside llama-swap's healthCheckTimeout (180 s by default; raise it if you have
# a cold torch.compile cache).
#
# Only this refusal is retried. Any other exit -- a real OOM, a bad flag, or the
# SIGTERM llama-swap sends to unload the model -- propagates on the first try.
ATTEMPTS=${SOLO_BOOT_ATTEMPTS:-4}
LOG="$(mktemp -t qwen-solo-boot.XXXXXX)"
trap 'rm -f "$LOG"' EXIT
trap 'exit 143' TERM INT

for n in $(seq 1 "$ATTEMPTS"); do
  : > "$LOG"
  set +e
  "$DIR/start_qwen.sh" 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ] && grep -q 'estimated maximum model length' "$LOG" \
     && [ "$n" -lt "$ATTEMPTS" ]; then
    echo "start_qwen_solo: startup profiling drew the high-activation branch (attempt $n/$ATTEMPTS); relaunching to redraw it. See gotcha 38." >&2
    continue
  fi
  exit "$rc"
done
