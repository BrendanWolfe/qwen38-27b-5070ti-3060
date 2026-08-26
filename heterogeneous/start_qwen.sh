#!/bin/bash
# Experimental two-GPU mode for an RTX 5070 Ti (16 GiB) + RTX 3060 (12 GiB).
#
# Pipeline parallelism is intentional. These cards have different capacities and
# no peer-to-peer PCIe access; tensor parallelism would split every layer evenly
# and communicate several times per layer. PP permits an uneven layer split and
# only transfers activations at the stage boundary.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# flashinfer-cubin (the no-nvcc route) publishes 0.6.13 against flashinfer-python
# 0.6.16.post3; without this the import refuses the pair (upstream #35). Inert if
# only one of the two is installed. NOTE for this pair: `has_flashinfer()` is what
# the DFlash2 selector consults, and it wants nvcc on PATH *or* flashinfer-cubin --
# a bare flashinfer-python passes `import flashinfer` and still leaves the selector
# on torch.topk at about half speed, with one INFO line. verify.sh tests the real
# predicate; heterogeneous/SETUP.md has the install line.
export FLASHINFER_DISABLE_VERSION_CHECK=1

# No profile here enables the OffloadingConnector, so this normally finds nothing;
# it is here for the EXTRA_ARGS path. A dead engine leaves its region behind as
# /dev/shm/vllm_offload_*.mmap and the next boot dies with OSError: Bad address,
# which under a restart policy loops (upstream #33 found 70 ghosts after one host
# OOM). Unlink regions no live process maps. VLLM_OFFLOAD_KEEP_SHM=1 skips it.
if [ "${VLLM_OFFLOAD_KEEP_SHM:-0}" != 1 ]; then
  for f in /dev/shm/vllm_offload_*.mmap; do
    [ -e "$f" ] || continue
    grep -lqs "$f" /proc/[0-9]*/maps 2>/dev/null || { echo "[start_qwen] removing stale offload region $f"; rm -f "$f"; }
  done
fi
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
# otherwise identical starts is cold compiler scratch being charged to rank
# 1's memory profile (0.32 GiB warm or 1.24 GiB cold), not this. Past eight the
# curve flattens but latency does not:
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

[ "$GPU_IDS" = "0,1" ] || echo "INFO: PP_LAYERS is ordered by CUDA_VISIBLE_DEVICES, not by physical device id; verify the split is what you meant." >&2
[ "$PP_LAYERS" = "44,20" ] || echo "INFO: using custom pipeline layer partition $PP_LAYERS (must total 64)." >&2

# Tensor parallelism through EXTRA_ARGS is a trap on THIS pair, which is why the
# launcher pins --pipeline-parallel-size 2 rather than leaving the topology open.
# Upstream now measures TP=2 as a clear win on two matched 3090s at PCIe 4.0 x8
# (+16-35% decode, their #40); none of that carries here. TP splits every layer
# evenly, so the 12 GiB 3060 would have to hold half of a model the 16 GiB card
# is already the constraint on, and it communicates several times per layer over
# a link that is ONE lane wide (the 3060 negotiates x1; see the hardware note in
# heterogeneous/README.md). PP moves activations once per stage boundary and
# permits the uneven 44/20 split, which is the whole reason this fork exists.
case " ${EXTRA_ARGS:-} " in *"--tensor-parallel-size"*|*" -tp "*)
  echo "WARNING: EXTRA_ARGS requests tensor parallelism, and this launcher also passes" \
       "--pipeline-parallel-size 2. On this pair TP is the wrong axis: unequal cards" \
       "and a PCIe x1 link to the 3060. Every measured profile here is PP-only." >&2
;; esac

# A --kv-cache-memory pin is applied PER WORKER (vllm/v1/worker/gpu_worker.py), so
# under PP=2 the number below is reserved on EACH card -- 2.8 GB is 2.8 GB on the
# 5070 Ti and 2.8 GB of the 3060's 12 GiB. That is deliberate on the DFlash2
# profiles and it is also why their pins do not transfer between profiles: a pin
# skips memory profiling entirely, so passing its sizing check says nothing about
# whether either rank still has room for graph capture, prefill workspaces and
# NCCL (gotcha 44 -- three configurations booted, reported a healthy pool, served
# /health 200 and then died). 2,800,000,000 is the largest pin validated here,
# and only at MAX_SEQS=1.
KV_PIN=$(printf %s " ${EXTRA_ARGS:-} " | sed -En "s/.*--kv-cache-memory[= ]([0-9]+).*/\1/p")
if [ -n "$KV_PIN" ] && [ "$KV_PIN" -gt 2800000000 ]; then
  echo "WARNING: --kv-cache-memory=$KV_PIN is above the largest pin validated on this" \
       "pair (2,800,000,000, at one seat). The pin is per PP rank, and the extra pool" \
       "comes out of the headroom each rank needs AFTER sizing succeeds. Validate with" \
       "a real long prompt followed by /health, not with a clean boot (gotcha 44)." >&2
fi

if [ "$KV" = "bf16" ]; then
  # The fastest KV mode for DFlash2, and not because it is the most accurate.
  # Under spec-decode vLLM consults the attention backend's AttentionCGSupport:
  # FlashInfer (which fp8 forces here) reports UNIFORM_SINGLE_TOKEN_DECODE and is
  # downgraded to piecewise CUDA graphs, while FLASH_ATTN keeps FULL_AND_PIECEWISE.
  # BF16 is what selects FLASH_ATTN on this hardware, since FlashAttention cannot
  # take fp8 on sm120/sm86. Costs density (4 KB/token/layer, so ~a third less pool
  # than fp8) and nothing else. See heterogeneous/README.md.
  KV_ARGS="--attention-backend FLASH_ATTN --kv-cache-dtype bfloat16"
elif [ "$KV" = "int4pth" ]; then
  # ~1 KB/token/layer against fp8's 2 KB, so the pool roughly doubles and the
  # model's native 262,144 fits. vLLM's per-token-head quant modes live only in
  # the Triton attention backend, which is markedly slower than FlashInfer at
  # long context -- this is a mode for requests that would not otherwise fit,
  # not a faster one. See docs/long-context.md and heterogeneous/README.md.
  KV_ARGS="--attention-backend TRITON_ATTN --kv-cache-dtype int4_per_token_head"
elif [ "$KV" = "fp8fa" ]; then
  # FP8 KV on FlashAttention: half BF16's 4 KB/token/layer, and unlike the
  # per-token-head modes it does not force the slower Triton backend. This is
  # the DFlash2 density option that keeps prefill fast.
  KV_ARGS="--attention-backend FLASH_ATTN --kv-cache-dtype fp8"
elif [ "$KV" = "kvarn" ]; then
  # KVarN 4-bit keys / 2-bit values per 128-token tile (kvarn/, run
  # kvarn/install.sh once). 840 B/token/layer against fp8's 2048 and int4pth's
  # ~1024, and unlike the per-token-head modes it is its own attention backend
  # rather than a Triton-only cache dtype.
  #
  # This is the first configuration in which KVarN has run under MTP+PP at all
  # (heterogeneous/README.md used to say it never had). It boots, serves and
  # answers correctly. It is still NOT the right choice on this pair, and the
  # reason is the per-slot cost rather than the per-token one -- measured, same
  # bench/real_rep.sh as every other row:
  #
  #   KV=kvarn  32k  MAX_SEQS=1   171,239-token pool   74.6 tok/s, 35.2 ms/step
  #   KV=kvarn  32k  MAX_SEQS=8    78,220-token pool
  #   KV=fp8   147k  MAX_SEQS=8   147,456-token floor  73.6-75.5 tok/s (batch)
  #   KV=int4pth 262k MAX_SEQS=4  284,234-token pool
  #
  # So at one slot KVarN beats the batch profile's pool at matching decode, and
  # at eight it holds half of it: KVarN forces a 2048-token attention block to
  # match the GDN page, and every scheduler slot pays a full aligned page. The
  # density win is real per token and is spent on slots. At 262k with 4 slots it
  # does not fit at all (2.48 GiB of pool against the 2.52 GiB one request
  # needs), so four-slot 262k stays int4pth territory: KV=int4pth MAX_SEQS=4
  # here, which is what start_qwen_huge.sh wrapped before start_qwen_solo.sh
  # replaced it. KVarN is wired into that profile; int4pth is an override only.
  #
  # NOTE this mode is only usable at all because of kvarn/kvarn-pp-pool-budget.patch:
  # stock KVarN charges rank 0 the whole checkpoint and pins max_num_seqs to 1.
  KV_ARGS="--kv-cache-dtype kvarn_k4v2_g128 --block-size 128"
  export KVARN_POOL_MEM_FRAC=${KVARN_POOL_MEM_FRAC:-0.15}
elif [ "$KV" = "int8pth" ]; then
  # The fastest KV mode for speculative decoding on this pair, and not for the
  # reason the name suggests. Under spec-decode vLLM consults the attention
  # backend's AttentionCGSupport: FlashInfer reports UNIFORM_SINGLE_TOKEN_DECODE,
  # so an fp8 run is silently downgraded to piecewise CUDA graphs, while
  # TRITON_ATTN reports ALWAYS and keeps FULL_AND_PIECEWISE. Triton cannot take
  # plain fp8 here (native fp8e4nv needs SM89+; the 3060 is sm86), but
  # int8_per_token_head is not gated -- so it is the only dense-KV mode that keeps
  # full graphs on both cards. Costs some prefill speed; see heterogeneous/README.md.
  KV_ARGS="--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head"
else
  KV_ARGS="--kv-cache-dtype fp8"
fi

PREFIX_ARGS=""
if [ "$PREFIX_CACHE" = "1" ]; then
  PREFIX_ARGS="--enable-prefix-caching --mamba-cache-mode align"
  # KVarN runs --block-size 128; the prefix hash unit must match its tile or
  # cache hits land off a tile boundary and corrupt the pool.
  [ "$KV" = "kvarn" ] && PREFIX_ARGS="--prefix-match-unit 128 $PREFIX_ARGS"
fi
ASYNC_ARGS=$([ "$ASYNC_SCHED" = "1" ] && echo --async-scheduling || echo --no-async-scheduling)

# The V2 model runner captures decode graphs in multiples of k+1 tokens, so the
# derived CG covers MAX_SEQS requests -- but never more than 64 query tokens'
# worth. Every profile here is far under that (8x4 batch, 4x8 dflash2, 1x4 solo)
# and the cap is inert by default; it is here because the overrides this repo
# invites are not inert. Upstream measured DFLASH_TOKENS=15 MAX_SEQS=8, i.e. 128,
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
elif [ "$SPEC" = "dflash2" ]; then
  export VLLM_USE_V2_MODEL_RUNNER=1
  DRAFT=${DRAFT:-$REPO/models/Qwen3.8-27B-DFlash2-W4A16}
  [ -f "$DRAFT/model.safetensors" ] || {
    echo "DFlash2 drafter missing: run venv/bin/python prepare/fetch_dflash2.py" >&2
    exit 1
  }
  DRAFT_TOKENS=${DFLASH_TOKENS:-7}
  export VLLM_DFLASH2_LOOKUP=${LOOKUP:-1}
  # CHAIN=1 turns on drafter-free n-gram chains
  # (patches/dflash2-ngram-chains.patch, upstream #38): while a request keeps
  # reproducing its own context, whole verify blocks come from history alone and the
  # drafter's forward AND its CUDA-graph replay are skipped until the first rejected
  # token. Requires the lookup lane above; greedy requests only.
  #
  # OFF by default, as upstream ships it, and UNMEASURED on this pair. Two reasons it
  # is worth trying here specifically, and one reason to be careful:
  #   - this box is bandwidth-bound, and a skipped drafter forward is skipped weight
  #     traffic, which is the quantity that sets step time here;
  #   - the drafter sits on the LAST pipeline rank (gotcha 44), so on a chain step
  #     that rank's work disappears rather than being merely cheaper.
  #   - but the chain needs a single request in the batch, so it can only engage on
  #     the solo profiles (MAX_SEQS=1) -- at MAX_SEQS=4 the batch DFlash2 profile
  #     will simply never enter it. Upstream measured +7% on a copy workload and flat
  #     elsewhere on ONE card with no PP; do not assume either number transfers.
  # VLLM_DFLASH2_CHAIN_MINMATCH (8) and _CHAIN_LOG_SEC (30) tune entry and logging.
  export VLLM_DFLASH2_CHAIN=${CHAIN:-0}
  # The split-KV verify attention is bf16-KV only. KVarN brings its own dequant
  # path, so the two cannot both be on -- upstream runs SPEC_ATTN=0 for exactly
  # this pair of settings (SPEC=dflash2 CTX=huge). Default it off here rather
  # than letting KV=kvarn inherit the 1 that every other KV mode wants. Still
  # overridable, but there is no known reason to.
  #
  # This combination is now measured on this pair, not just guarded: reversed
  # 30,34 with a 2.8 GB (2.61 GiB) pinned pool serves 200,192 context at
  # 83.06-86.63 tok/s e2e; a 190,057-token needle passes. It does NOT reach
  # 262k -- DFlash2's 1+k aligned state pages cost more than KVarN's density
  # saves. See heterogeneous/README.md, "DFlash2 on KVarN".
  if [ "$KV" = "kvarn" ]; then
    export VLLM_SPEC_DECODE_ATTN=${SPEC_ATTN:-0}
  else
    export VLLM_SPEC_DECODE_ATTN=${SPEC_ATTN:-1}
  fi
  # The shared target lm_head is Marlin-quantized. Its candidate GEMM currently
  # fails inside DFlash's private CUDA graph, so keep only the drafter eager;
  # target-model compilation and CUDA graphs remain enabled.
  export VLLM_DFLASH_CUDAGRAPH=${VLLM_DFLASH_CUDAGRAPH:-0}
  export VLLM_SPEC_DECODE_ATTN_QMAX=${VLLM_SPEC_DECODE_ATTN_QMAX:-$((DRAFT_TOKENS + 1))}
  SPEC_ARGS="--speculative-config {\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$DRAFT_TOKENS}"
  CG=${CG:-$((MAX_SEQS * (DRAFT_TOKENS + 1) > 64 ? 64 : MAX_SEQS * (DRAFT_TOKENS + 1)))}
else
  CG=${CG:-$MAX_SEQS}
fi

# An explicit CUDAGRAPH_MODE is honoured on every path. Unlike upstream these
# profiles never pin the capture mode: the residue bug that PIECEWISE works
# around is fixed in the runner here (the prefill discriminator in
# get_uniform_token_count, README "Bug B"), so vLLM's own choice stands and
# bf16/int8pth keep their full graphs. But a fix you cannot A/B is a fix you
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
  # VISION_OFFLOAD keeps the tower's weights in pinned host RAM and copies each module
  # to the GPU for the duration of its own forward
  # (patches/vision-tower-cpu-offload.patch). It answers the paragraph above directly:
  # the 0.858 GiB that would otherwise land on the first pipeline rank -- the rank that
  # binds the KV pool -- stops landing there, and 8.5 MiB of patch/pos embedding stays
  # resident so visual.device still reports cuda. Default ON here for the same reason
  # upstream defaults it on, and with more force: both cards are smaller than the 24 GiB
  # it was measured on, and gotcha 44 puts rank 0's survival boundary at or below
  # 1.57 GiB spare on the 3060.
  #
  # UNMEASURED on this pair, and the cost is NOT upstream's. They measured +36 ms per
  # image moving ~891 MiB at PCIe 4.0 x16 (296 -> 333 ms per forward). Rank 0 here is
  # the 5070 Ti at x16 under the default 44/20, so that number should roughly transfer
  # -- but the DFlash2 profiles run GPU_IDS=1,0, which puts rank 0, and therefore the
  # tower, on the 3060's ONE PCIe lane. At Gen4 x1 (~1.5-2 GB/s in practice) the same
  # copy is order half a second per image rather than 36 ms. So: leave it on for the
  # 44/20 profiles, and on the reversed DFlash2 profiles treat it as buying 0.86 GiB of
  # 3060 headroom at a per-image price you should measure before serving images at rate.
  [ "${VISION_OFFLOAD:-1}" = 1 ] && export VLLM_VISION_CPU_OFFLOAD_GB=${VLLM_VISION_CPU_OFFLOAD_GB:-1}
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
