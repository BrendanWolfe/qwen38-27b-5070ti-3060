#!/bin/bash
# Long-context concurrent profile: MTP-3, eight scheduler seats, and KVarN
# 4-bit-key / 2-bit-value KV at Qwen3.8's full 262,144-token native context.
#
# Measured on the RTX 5070 Ti + RTX 3060 pair at the default 44/20 split:
# 289,641-token pool; a 240,035-token needle passed at ~684 prefill tok/s and
# /health remained 200.
#
# A cold exact torch.compile shape can allocate compiler/autotuner scratch
# inside vLLM's memory-profile window. vLLM mislabels that transient as peak
# activation memory and may reject 262k before serving. The failed attempt
# populates the compile cache, so retry only that sizing refusal. Other errors
# and normal termination propagate immediately.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"

export GPU_IDS=${GPU_IDS:-0,1}
export PP_LAYERS=${PP_LAYERS:-44,20}
export MAX_LEN=${MAX_LEN:-262144}
export KV=${KV:-kvarn}
export SPEC=${SPEC:-mtp}
export DRAFT_TOKENS=${DRAFT_TOKENS:-3}
export MAX_SEQS=${MAX_SEQS:-8}

ATTEMPTS=${KVARN_BATCH_BOOT_ATTEMPTS:-4}
# Use the model volume, not /tmp: /tmp is quota-limited on this host.
LOG="$(mktemp "$REPO/.qwen-kvarn-batch-boot.XXXXXX")"
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
    echo "start_qwen_kvarn_batch: compile scratch was charged to KV sizing (attempt $n/$ATTEMPTS); relaunching with the warmed compile cache." >&2
    continue
  fi
  exit "$rc"
done
