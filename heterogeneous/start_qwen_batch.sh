#!/bin/bash
# General-purpose CONCURRENT profile for an RTX 5070 Ti (16 GiB) + RTX 3060
# (12 GiB): FP8 KV, MTP-3, eight scheduler slots, 140k of context. This is the
# profile llama-swap runs, and it is a full standalone copy of start_qwen.sh
# rather than a wrapper, so later experiments in the shared launcher cannot
# change what the served model does. Use start_qwen_solo.sh instead when one
# stream needs the full native 262k context.
#
# Pipeline parallelism is intentional. These cards have different capacities and
# no peer-to-peer PCIe access; tensor parallelism would split every layer evenly
# and communicate several times per layer. PP permits an uneven layer split and
# only transfers activations at the stage boundary.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
cd "$REPO"

if [ -z "${MODEL:-}" ] && [ -d "$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast" ]; then
  MODEL=$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast
fi
MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}
PORT=${PORT:-18020}
GPU_IDS=${GPU_IDS:-0,1}
PP_LAYERS=${PP_LAYERS:-44,20}
MAX_LEN=${MAX_LEN:-140000}
# Eight scheduler slots, not four. Decode on this pair is bound by weight
# bandwidth, so extra sequences ride along in the same step nearly for free:
# 4 -> 8 slots is +41% aggregate throughput at eight concurrent streams
# (213.9 -> 301.0 tok/s) and changes nothing at one to four (73.0/115.7/213.9
# against the four-slot 73.1/120.8/212.7). The only cost is CUDA graph memory,
# 0.13 -> 0.14 GiB, about 1.3k tokens of pool -- the ~1 GiB pool swing between
# identical starts is rank 1's profiled activation peak (0.32 or 1.24 GiB) and
# is unrelated to this. Past eight the curve flattens but latency does not:
# C12 is +13% for 82 ms ITL, C16 is +28% for a ~3 s TTFT. See
# heterogeneous/README.md.
MAX_SEQS=${MAX_SEQS:-8}
# Leave enough unprofiled memory for compiled prefill workspaces. At 0.93 a
# 5k-token harness request could need another 56 MiB, OOM a PP worker, and
# strand its peers. 0.91 still supports the required 131k context.
GPU_UTIL=${GPU_UTIL:-0.91}
API_SERVERS=${API_SERVERS:-1}
KV=${KV:-fp8}
PREFIX_CACHE=${PREFIX_CACHE:-1}
ASYNC_SCHED=${ASYNC_SCHED:-1}
SPEC=${SPEC:-mtp}
DRAFT_TOKENS=${DRAFT_TOKENS:-3}

[ "$GPU_IDS" = "0,1" ] || echo "WARN: PP_LAYERS is ordered by CUDA_VISIBLE_DEVICES; verify the faster/larger card is first." >&2
[ "$PP_LAYERS" = "44,20" ] || echo "INFO: using custom pipeline layer partition $PP_LAYERS (must total 64)." >&2

if [ "$KV" = "bf16" ]; then
  KV_ARGS="--attention-backend FLASH_ATTN --kv-cache-dtype bfloat16"
else
  KV_ARGS="--kv-cache-dtype fp8"
fi

PREFIX_ARGS=""
if [ "$PREFIX_CACHE" = "1" ]; then
  PREFIX_ARGS="--enable-prefix-caching --mamba-cache-mode align"
fi
ASYNC_ARGS=$([ "$ASYNC_SCHED" = "1" ] && echo --async-scheduling || echo --no-async-scheduling)

SPEC_ARGS=""
if [ "$SPEC" = "mtp" ]; then
  # PR #46994 implements MTP+PP on the V2 model runner.
  export VLLM_USE_V2_MODEL_RUNNER=1
  SPEC_ARGS="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$DRAFT_TOKENS,\"draft_sample_method\":\"${DRAFT_SAMPLE:-probabilistic}\"}"
  CG=${CG:-$((MAX_SEQS * (DRAFT_TOKENS + 1)))}
else
  CG=${CG:-$MAX_SEQS}
fi

# W4A8 remains opt-in: it accelerates large batched GEMMs but is not useful for
# the latency-oriented C1 MTP workload, and unlike speculation it changes quality.
INT8_ACT=${INT8_ACT:-}
INT8_LAYERS=${INT8_LAYERS:-mlp}

# Keep CUDA's logical order stable on a mixed-GPU host. GPU_IDS and PP_LAYERS
# are interpreted in this order.
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
# JIT extensions must be built for both consumer Blackwell and Ampere.
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.6;12.0"}
# CUDA 13.3 rejects this host's GCC 16. Use the installed supported compiler
# for FlashInfer/torch JIT extensions while leaving the system default alone.
if command -v gcc-15 >/dev/null 2>&1 && command -v g++-15 >/dev/null 2>&1; then
  export CC=${CC:-$(command -v gcc-15)}
  export CXX=${CXX:-$(command -v g++-15)}
  export CUDAHOSTCXX=${CUDAHOSTCXX:-$CXX}
fi
export VLLM_PP_LAYER_PARTITION="$PP_LAYERS"
export PATH="$REPO/venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export VLLM_USE_FLASHINFER_SAMPLER=0
# vLLM distinguishes an unset dtype (W4A16) from the invalid empty string.
# Export these only when W4A8 is explicitly requested.
if [ -n "$INT8_ACT" ]; then
  export VLLM_MARLIN_INPUT_DTYPE="$INT8_ACT"
  export VLLM_MARLIN_INT8_INCLUDE_RE="$INT8_LAYERS"
else
  unset VLLM_MARLIN_INPUT_DTYPE VLLM_MARLIN_INT8_INCLUDE_RE
fi

# Direct launches use api_key.txt by default. A trusted local proxy such as
# llama-swap can explicitly disable child authentication with NO_API_KEY=1.
if [ "${NO_API_KEY:-0}" = "1" ]; then
  unset VLLM_API_KEY
elif [ -z "$VLLM_API_KEY" ] && [ -f "$REPO/api_key.txt" ]; then
  export VLLM_API_KEY="$(cat "$REPO/api_key.txt")"
fi

# Keep vLLM in its own process group. If its API/engine process crashes, reap
# every PP worker rather than leaving an orphan spinning on a GPU.
SERVER_PID=""
cleanup() {
  set +e
  if [ -n "$SERVER_PID" ] && kill -0 -- "-$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null
    for _ in 1 2 3 4 5; do
      kill -0 -- "-$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT

setsid venv/bin/vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port "$PORT" \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend mp \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --api-server-count "$API_SERVERS" \
  --language-model-only \
  $KV_ARGS \
  $PREFIX_ARGS \
  --mamba-ssm-cache-dtype float16 \
  $ASYNC_ARGS \
  --max-num-batched-tokens 2048 \
  $SPEC_ARGS \
  --compilation-config "{\"max_cudagraph_capture_size\":$CG,\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\"]}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-force-include-usage \
  --enable-per-request-metrics \
  --enable-prompt-tokens-details \
  ${EXTRA_ARGS} &
SERVER_PID=$!
wait "$SERVER_PID"
