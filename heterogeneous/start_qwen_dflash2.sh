#!/bin/bash
# Experimental DFlash2 profile. This is separate from the frozen 147k MTP
# profile so cache/backend experiments cannot change what llama-swap serves.
#
# 88k of context, not the 32k this profile shipped with. Two changes got it
# there, and neither cost decode speed (85.17 tok/s against bf16's 85.50 on
# bench/real_rep.sh at C1, acceptance 3.26 against 3.19):
#
#   KV=fp8 instead of bf16     fp8 is half the bytes per token, and on THIS
#                              profile the CUDA-graph downgrade it causes
#                              (FlashInfer reports UNIFORM_SINGLE_TOKEN_DECODE,
#                              so the mode drops to PIECEWISE) costs nothing
#                              measurable. Do not generalise that to the
#                              reversed-pipeline profile, where bf16 measured
#                              95-99 against fp8's 88-93.
#   DFLASH_KV_MEMORY 2.5->3.2  the shipped 2.5 GB was conservative.
#
# 3.2 GB IS THE CEILING, AND THE FAILURE MODE IS NASTY. --kv-cache-memory takes
# memory that vLLM would otherwise leave for compiled prefill on the 3060, so
# too large a value does NOT fail at startup -- it boots, sizes a bigger pool,
# captures graphs, answers short prompts, and then dies with a 500 partway into
# the first long prompt (torch.OutOfMemoryError in the compiled prefill forward,
# failing on as little as 18 MiB). Measured, each probe a real prompt near
# MAX_LEN followed by a health check:
#
#   cap    KV    MAX_LEN  pool     probe    result
#   2.5G   bf16   32,768   33,506   30,046  survives  (what this shipped as)
#   2.9G   fp8    40,000   68,280   36,056  survives
#   3.2G   fp8    48,000   80,914   44,035  survives
#   3.2G   fp8    60,000   87,192   54,999  survives
#   3.2G   fp8    72,000   92,751   65,035  survives
#   3.2G   fp8    88,000   97,962   80,035  survives  <- shipped
#   3.4G   fp8    48,000   86,125   44,035  OOM-KILLED mid-request
#   3.6G   bf16   45,000   53,152   40,000  OOM-KILLED mid-request
#
# So do not raise DFLASH_KV_MEMORY on the strength of a clean boot: a boot only
# proves the pool was sized, and the thing that kills you happens later. Every
# row above was checked by sending a prompt near MAX_LEN and confirming the
# server still answered /health afterwards.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MAX_LEN=${MAX_LEN:-88000}
export GPU_UTIL=${GPU_UTIL:-0.915}
export KV=${KV:-fp8}
export MAX_SEQS=${MAX_SEQS:-4}
export SPEC=dflash2
export DFLASH_TOKENS=${DFLASH_TOKENS:-7}
export VLLM_DFLASH_CUDAGRAPH=${VLLM_DFLASH_CUDAGRAPH:-0}

# Manual sizing prevents startup profiling from assigning ~2.9 GiB to rank 1,
# then exhausting the 12 GiB card when target CUDA graphs are captured. This
# pool provides 33,506 tokens with the current model and vLLM build.
export EXTRA_ARGS="--kv-cache-memory=${DFLASH_KV_MEMORY:-3200000000} ${EXTRA_ARGS:-}"

exec "$DIR/start_qwen.sh"
