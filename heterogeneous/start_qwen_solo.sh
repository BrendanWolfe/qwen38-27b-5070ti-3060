#!/bin/bash
# Single-stream long-context profile: KVarN 4/2-bit KV, ONE request at a time,
# the model's full native 262,144 context in a 296,974-token pool at the batch
# profile's decode rate.
#
# This is the one shape in which KVarN wins on this pair. Measured against the
# other MTP profiles (bench/real_rep.sh, 8 realistic 1,024-token prompts at
# concurrency 1, greedy):
#
#   profile                KV       slots  ctx    pool      decode
#   start_qwen_solo.sh     KVarN      1    262k   296,974   72.4-73.6 tok/s, 34.8-35.4 ms/step
#   start_qwen_batch.sh    fp8        8    147k   147,456+  73.6-75.5 tok/s
#   KV=int4pth MAX_SEQS=4  int4pth    4    262k   284,234   batch only (~112 tok/s prefill at depth)
#
# Against batch that is roughly twice the pool and 115k more context at the same
# decode rate and the same tokens/step (2.48-2.52 against 2.51-2.62) -- for ONE
# stream. FP8 cannot reach this context at any MAX_LEN; it refuses above ~148k.
# It also holds a larger pool at the same 262,144 context than the int4
# per-token-head mode this profile replaced (296,974 against 284,234), so if
# you want the native context on a single stream this is the better of the two;
# int4pth is still reachable as KV=int4pth MAX_SEQS=4 on start_qwen.sh, and
# keeps its 4 slots.
#
# WHY THE RETRY. A cold exact torch.compile shape can run compilation/autotuning
# inside vLLM's memory-profile window. The profiler charges its temporary
# scratch as peak activation memory and can reject 262k even though the scratch
# is gone before serving. The failed launch warms the exact artifact; a relaunch
# then profiles the real steady-state footprint. Gotcha 40 records the evidence.
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

# RETRY, because 262,144 can exceed the falsely reduced pool on a cold compiled
# shape. The sizing refusal happens after the missing artifact is populated, so
# the next launch measures the warmed path.
#
# The alternative was to drop MAX_LEN to the cold-profile floor, 143,360 --
# that would throw away the entire point of the profile to avoid a retry that
# costs 20 seconds. A failed attempt dies at KV sizing, before graph capture,
# and a successful boot is ~27 s with a warm compile cache, so four attempts fit
# inside llama-swap's healthCheckTimeout (180 s by default; raise it if you have
# a cold torch.compile cache).
#
# Only this refusal is retried. Any other exit -- a real OOM, a bad flag, or the
# SIGTERM llama-swap sends to unload the model -- propagates on the first try.
ATTEMPTS=${SOLO_BOOT_ATTEMPTS:-4}
# Use the model volume, not /tmp: /tmp is quota-limited on this host.
REPO="$(dirname "$DIR")"
LOG="$(mktemp "$REPO/.qwen-solo-boot.XXXXXX")"
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
    echo "start_qwen_solo: compile scratch was charged to KV sizing (attempt $n/$ATTEMPTS); relaunching with the warmed compile cache. See gotcha 40." >&2
    continue
  fi
  exit "$rc"
done
