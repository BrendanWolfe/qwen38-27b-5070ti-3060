#!/bin/bash
# General-purpose CONCURRENT profile for an RTX 5070 Ti (16 GiB) + RTX 3060
# (12 GiB): FP8 KV, MTP-3, eight scheduler slots, 147k of context. This is the
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
MAX_LEN=${MAX_LEN:-147456}
# Eight scheduler slots, not four. Decode on this pair is bound by weight
# bandwidth, so extra sequences ride along in the same step nearly for free:
# 4 -> 8 slots is +41% aggregate throughput at eight concurrent streams
# (213.9 -> 301.0 tok/s) and changes nothing at one to four (73.0/115.7/213.9
# against the four-slot 73.1/120.8/212.7). The only cost is CUDA graph memory,
# 0.13 -> 0.14 GiB, about 1.3k tokens of pool -- the much larger pool swing
# between starts comes from cold compiler scratch being charged during memory
# profiling (gotcha 43), not this. Past eight the curve flattens but latency does not:
# C12 is +13% for 82 ms ITL, C16 is +28% for a ~3 s TTFT. See
# heterogeneous/README.md.
MAX_SEQS=${MAX_SEQS:-8}
# Leave enough unprofiled memory for compiled prefill workspaces. At 0.93 a
# 5k-token harness request could need another 56 MiB, OOM a PP worker, and
# strand its peers. 0.91 still supports the required 131k context.
#
# MAX_LEN is 147,456 rather than 140,000 because 147,456 is the most this
# profile can ask for even when compiler scratch pollutes startup profiling.
# Cold exact compile artifacts can add ~0.9 GiB to the reported activation
# peak, so the pool from otherwise identical commands
# ranges over 146,086 / 157,500 / 184,891 tokens, and the refusal threshold
# over 148,096 / 159,744 / ~189,000. This frozen profile keeps the cold-safe
# floor instead of adding retry behavior. See gotcha 43.
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

# The V2 model runner captures decode graphs in multiples of k+1 tokens, so the
# derived CG covers MAX_SEQS requests -- but never more than 64 query tokens'
# worth. This profile's default is 8x4 = 32 and the cap is inert; it is here
# because the overrides this repo invites are not inert. Upstream measured DFLASH_TOKENS=15 MAX_SEQS=8, i.e. 128,
# booting, capturing its graphs, answering /health 200 and then dying on the
# first concurrent batch with torch.OutOfMemoryError inside the engine
# (docs/gotchas.md, gotcha 38). Past the cap the oversized batches run piecewise
# instead of not at all. Set CG explicitly to override.
SPEC_ARGS=""
if [ "$SPEC" = "mtp" ]; then
  # PR #46994 implements MTP+PP on the V2 model runner.
  export VLLM_USE_V2_MODEL_RUNNER=1
  SPEC_ARGS="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$DRAFT_TOKENS,\"draft_sample_method\":\"${DRAFT_SAMPLE:-probabilistic}\"}"
  CG=${CG:-$((MAX_SEQS * (DRAFT_TOKENS + 1) > 64 ? 64 : MAX_SEQS * (DRAFT_TOKENS + 1)))}
else
  CG=${CG:-$MAX_SEQS}
fi

# An explicit CUDAGRAPH_MODE is honoured on every path. Unlike upstream these
# profiles never pin the capture mode: the residue bug that PIECEWISE works
# around is fixed in the runner here (the prefill discriminator in
# get_uniform_token_count, README "Bug B"), so vLLM's own choice stands. But a fix you cannot A/B is a fix you
# cannot check -- without this knob there is no way to ask the same server for
# the other capture mode and sweep residues under both.
CG_MODE=""
[ -n "${CUDAGRAPH_MODE:-}" ] && CG_MODE=",\"cudagraph_mode\":\"$CUDAGRAPH_MODE\""

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
# expandable_segments needs CUDA VMM, which WSL2's paravirt driver rejects during
# Marlin repack -- and it does not present as an allocator problem. Default it off
# there; see the long note in single-user/start_qwen.sh. Native Linux, which is
# what this mode is built for, is untouched: gotcha 3 applies and turning it off
# costs the top of the GPU_UTIL range, which this pair does not have to spare.
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null || [ -n "${WSL_DISTRO_NAME:-}" ]; then
  ALLOC_DEFAULT=expandable_segments:False
  [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && echo \
    "WSL detected: PYTORCH_CUDA_ALLOC_CONF=$ALLOC_DEFAULT (VMM breaks Marlin repack under the paravirt driver; set it explicitly to override)"
else
  ALLOC_DEFAULT=expandable_segments:True
fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-$ALLOC_DEFAULT}
export VLLM_USE_FLASHINFER_SAMPLER=0
# vLLM distinguishes an unset dtype (W4A16) from the invalid empty string.
# Export these only when W4A8 is explicitly requested.
if [ -n "$INT8_ACT" ]; then
  export VLLM_MARLIN_INPUT_DTYPE="$INT8_ACT"
  export VLLM_MARLIN_INT8_INCLUDE_RE="$INT8_LAYERS"
else
  unset VLLM_MARLIN_INPUT_DTYPE VLLM_MARLIN_INT8_INCLUDE_RE
fi

# Vision. --language-model-only drops the vision tower cleanly -- no weights
# loaded, 0.858 GiB on this checkpoint (gotcha 9) -- and stays the default.
# VISION=1 keeps it for a client that sends images.
#
# It gets a knob rather than being left to EXTRA_ARGS because the alternative
# regresses silently: countering a hardcoded --language-model-only with
# --no-language-model-only depends on which flag argparse saw last, and when it
# loses, images are still accepted and still billed as prompt tokens while the
# model answers from placeholder embeddings. The two flags VISION=1 adds have no
# such conflict and EXTRA_ARGS, expanded after them, still wins.
#
# UNMEASURED on this pair, and the reason to expect it to cost more here than on
# one card: model.visual.* loads on the FIRST pipeline rank, which under the
# 44/20 split is the 5070 Ti -- the rank that binds the KV pool. So the tower's
# 0.858 GiB and the encoder's profiled peak both come out of the scarce side.
# The pixel cap is shipped rather than left to the processor default for that
# second reason: vLLM profiles the encoder at the largest image it will accept.
# 2097152 px = 2048 image tokens.
if [ "${VISION:-0}" = 1 ]; then
  VISION_ARGS='--limit-mm-per-prompt {"image":{"count":1}} --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":2097152}}'
else
  VISION_ARGS="--language-model-only"
fi

# fp16 activations do not work with the speculative path here, and every way of
# finding that out is late and cryptic: the split-KV verify kernel hardcodes
# tl.bfloat16 for the query cast and the dot accumulate
# (patches/spec-decode-attn.patch), so it fails to COMPILE at the first
# attention -- "Both operands must be same dtype. Got bf16 and fp16", surfaced
# as a triton CompilationError that never names the dtype you set. Under PP that
# arrives as a worker dying mid-init, which is harder to read still. SPEC=none is
# the escape hatch; this is a limit of these kernels, not of the checkpoint.
case " ${EXTRA_ARGS:-} " in
  *" --dtype float16 "*|*" --dtype=float16 "*|*" --dtype fp16 "*|*" --dtype=fp16 "*|*" --dtype half "*|*" --dtype=half "*)
    if [ "${SPEC:-mtp}" != "none" ]; then
      echo "--dtype float16 needs SPEC=none: this repo's speculative path is bf16-only." >&2
      echo "  the split-KV verify attention casts to tl.bfloat16 (patches/spec-decode-attn.patch)," >&2
      echo "  so it fails to compile at the first attention." >&2
      exit 1
    fi ;;
esac

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
  ${VISION_ARGS} \
  $KV_ARGS \
  $PREFIX_ARGS \
  --mamba-ssm-cache-dtype float16 \
  $ASYNC_ARGS \
  --max-num-batched-tokens 2048 \
  $SPEC_ARGS \
  --compilation-config "{\"max_cudagraph_capture_size\":$CG${CG_MODE},\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\"]}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-force-include-usage \
  --enable-per-request-metrics \
  --enable-prompt-tokens-details \
  ${EXTRA_ARGS} &
SERVER_PID=$!
wait "$SERVER_PID"
