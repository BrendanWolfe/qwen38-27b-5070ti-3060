#!/bin/bash
# Reversed-pipeline short-context MTP profile: 32k, 7.7% quicker per step than
# the stable 140k profile.
#
# The stable profile puts the 5070 Ti first (target layers 0-43) and the 3060
# last (44-63). vLLM puts the LM head, the sampler and the MTP drafter on the
# LAST pipeline rank, so in that arrangement the slow card reads, every step,
# its 20 target layers *plus* the 248k-vocab LM head and three chained MTP
# draft passes -- about 5.8 GB against the 5070 Ti's 9.9 GB, at 360 GB/s
# against 896. At concurrency 1 a pipeline does not overlap, so step time is
# the sum of the two stages and the small card was costing ~16 of ~35 ms.
#
# Reversing the order moves the head and the drafter onto the fast card while
# keeping the same 44/20 target-layer counts per GPU (and therefore the same
# 11/5 split of full-attention layers). Measured: 35.1 -> 32.4 ms/step,
# 73.6-75.5 -> 78.7-82.6 tok/s, with tokens-per-step unchanged (2.51-2.62), so
# the gain is step time and not acceptance.
#
# The cost is context. The last rank also carries the sampling/logits peak
# (1.24 GiB here against 0.32 on the first rank), so the 5070 Ti gives up about
# 3.15 GiB of KV: 4.23 -> 1.08 GiB, which is a 33,363-token pool. That is why
# this is a separate short-context profile and not a change to the stable one.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPU_IDS=${GPU_IDS:-1,0}
export PP_LAYERS=${PP_LAYERS:-20,44}
export MAX_LEN=${MAX_LEN:-32768}
export KV=${KV:-fp8}
export SPEC=${SPEC:-mtp}
export DRAFT_TOKENS=${DRAFT_TOKENS:-3}
export MAX_SEQS=${MAX_SEQS:-4}

exec "$DIR/start_qwen.sh"
