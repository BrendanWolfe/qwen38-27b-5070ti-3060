# RTX 5070 Ti + RTX 3060 (heterogeneous pipeline-parallel mode)

This directory runs the repository's W4A16 Qwen3.8-27B across **two unequal
consumer GPUs**: a 16 GiB RTX 5070 Ti (`sm120`, CUDA logical device 0) and a
12 GiB RTX 3060 (`sm86`, device 1). It is runtime-tested with vLLM 0.27.1, the
repository's fast model variant, and both speculative decoders (MTP and
DFlash2).

Two setups are supported:

| profile | launcher | speculation | KV | context | measured decode |
|---|---|---|---|---|---|
| stable / general | `start_qwen_stable.sh` | MTP-3, FP8 weights + FP8 KV | FP8 | 140k (146,847-token pool) | **83.8-84.9 tok/s** |
| short-context | `start_qwen_dflash2.sh` | DFlash2 (7 drafts, 1 pass) | BF16 | 32k (33,506-token pool) | **91.7 tok/s** avg (77–159 t/s by workload) |

`start_qwen.sh` is the shared, tunable launcher both profiles wrap.

## Why pipeline parallelism

`nvidia-smi topo -m` reports the cards connected through the PCIe host bridge
(`PHB`) with P2P reads unsupported. Tensor parallelism would split weights
evenly despite the unequal cards and communicate several times per layer.
Pipeline parallelism permits an uneven layer assignment and transfers
activations only at the stage boundary.

The `44,20` split (`VLLM_PP_LAYER_PARTITION`) keeps 44 target layers on the
faster/larger 5070 Ti and 20 layers plus the drafter on the 3060. A tested
`46,18` split reduced the KV pool from ~160k to ~132k tokens with no speed
gain, so `44,20` remains the default.

## Everything that was modified to make this work

### New files in this repository

- `heterogeneous/start_qwen.sh` — shared launcher: uneven PP, toolchain fixes,
  MTP/DFlash2/no-spec selection, process-group cleanup, no-auth mode.
- `heterogeneous/start_qwen_stable.sh` — frozen MTP snapshot used by the
  stable llama-swap model, so later experiments cannot change its behavior.
- `heterogeneous/start_qwen_dflash2.sh` — 32k DFlash2 profile (BF16 KV,
  manually sized cache, eager drafter; see below).
- `heterogeneous/llama-swap.example.yaml` — the two llama-swap model entries.
- `patches/vllm-pr46994-mtp-pp.patch` — MTP + pipeline parallelism (below).
- `patches/zz-dflash2-pipeline-parallel.patch` — DFlash2 + pipeline
  parallelism (below). The `zz-` prefix keeps it last in the `patches/*.patch`
  glob because it touches files earlier patches also patch.
- `docker-compose.yml` gained a `hetero` profile (two-GPU reservation) and
  `docker/entrypoint.sh` a `hetero` command.

### Host and toolchain fixes (all in `start_qwen.sh`)

- `CUDA_DEVICE_ORDER=PCI_BUS_ID` so `GPU_IDS`/`PP_LAYERS` ordering is stable
  across reboots on a mixed-GPU host.
- `TORCH_CUDA_ARCH_LIST="8.6;12.0"` so JIT extensions cover both Ampere and
  consumer Blackwell.
- `CC/CXX/CUDAHOSTCXX` pinned to GCC 15: CUDA 13.3 rejects this host's GCC 16
  for FlashInfer/torch JIT extensions.
- `VLLM_MARLIN_INPUT_DTYPE`/`VLLM_MARLIN_INT8_INCLUDE_RE` are **unset** unless
  `INT8_ACT` is explicitly requested — vLLM distinguishes an unset dtype
  (W4A16) from the invalid empty string, and an empty export breaks Marlin.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and
  `VLLM_USE_FLASHINFER_SAMPLER=0`.
- `--language-model-only` (the VL wrapper's encoder is unused here),
  `--mamba-ssm-cache-dtype float16` (FP16 recurrent state), four request slots,
  async scheduling, `--max-num-batched-tokens 2048`.
- Tool calling and accurate per-request metrics:
  `--reasoning-parser qwen3 --enable-auto-tool-choice
  --tool-call-parser qwen3_coder --enable-force-include-usage
  --enable-per-request-metrics --enable-prompt-tokens-details`. Without the
  first three, `"auto"` tool choice fails; without the last three, llama-swap
  cannot record token counts and request-level speeds.
- Process-group cleanup: vLLM is launched via `setsid` and the wrapper traps
  EXIT/TERM/INT to kill the whole group. Before this, a CUDA OOM killed the
  API process while orphaned PP workers kept spinning at 100% GPU.
- `NO_API_KEY=1` disables child authentication for a trusted local proxy
  (llama-swap); direct launches still read `api_key.txt`.

### Memory headroom: why `GPU_UTIL=0.91`

At 0.93, a ~5k-token request triggered `torch.OutOfMemoryError` in compiled
prefill (`Tried to allocate 56.00 MiB`) because startup profiling does not
include compiled-prefill or transient DeltaNet speculative workspaces. The
5070 Ti is the binding constraint. 0.91 still yields a measured **146,847**
FP8-KV token pool — enough for the required 131k request with real headroom.
Typical idle allocation is ~15.2 GB on the 5070 Ti and ~11.1 GB on the 3060.

### MTP + pipeline parallelism: `patches/vllm-pr46994-mtp-pp.patch`

Stock vLLM 0.27.1 cannot run MTP with PP. This is a local backport of upstream
PR #46994 for the V2 model runner, plus the Qwen3.5 hybrid-Mamba fixes needed
to actually run it:

- pads and relays sampled/draft-token broadcasts between pipeline ranks,
- runs the Qwen MTP projection only on the last pipeline stage and passes the
  last stage's target intermediate tensors to the drafter,
- loads the MTP module's own embedding table where required,
- casts `index_fill_` indices to int64 in `mamba_hybrid.py` (PP long prefill
  crashed with int32 indices).

The launcher selects the V2 runner (`VLLM_USE_V2_MODEL_RUNNER=1`) when
`SPEC=mtp`. Three drafts: MTP-3 is soak-tested with FP8 KV; MTP-4 has known
FlashInfer failures when concurrent requests finish at different times.

### DFlash2 + pipeline parallelism: `patches/zz-dflash2-pipeline-parallel.patch`

Upstream vLLM 0.27.1 flatly refuses DFlash with PP (`method "dflash" with
pipeline parallel is not supported`). This patch adds the missing machinery:

- **Auxiliary hidden-state relay** (`qwen3_next.py`): DFlash2 needs target
  hidden states from layers 5, 19, 33, 47, 61 — three of which live on the
  first stage. They now ride across the stage boundary inside
  `IntermediateTensors` next to `hidden_states`/`residual`, in configured
  target-layer order, and the second stage prepends them to its own captures.
- **Factory expansion** (`qwen3_5.py`, `model_runner.py`): the
  intermediate-tensor factories are expanded for the incoming aux states, and
  the multimodal wrapper's cached factory (frozen at construction) is
  refreshed after aux-layer setup.
- **PP-local drafter** (`config/speculative.py`, `model_runner.py`): the
  complete small DFlash drafter lives only on the last target rank, so it is
  validated with a local PP size of 1 and the runner permits `dflash` with PP.
- **No duplicate vocab tables** (`qwen3_dflash.py`, `dflash/utils.py`): the
  checkpoint has no private embeddings/LM head, so the drafter is constructed
  with `PPMissingLayer` placeholders and then shares the target's modules —
  avoids ~2.4 GiB temporary BF16 allocations that OOMed the 12 GiB card.
- **`VLLM_DFLASH_CUDAGRAPH=0`** (`dflash/speculator.py`): runs only the
  drafter eagerly. Required here because the shared target LM head is
  Marlin-quantized and its candidate GEMM crashes inside DFlash's private CUDA
  graph capture (`torch_call_dispatcher ... aten::empty` in
  `marlin_gemm`). Target-model compilation and CUDA graphs stay enabled.

### DFlash2 memory sizing (the non-obvious part)

With PP, vLLM's automatic KV profiling assigns the same cache budget to both
ranks, but rank 1 (the 3060) also carries the drafter, its KV, and the target
CUDA graphs. Auto-sizing over-allocated (~2.9 GiB KV) and then NCCL failed to
allocate a 40-byte broadcast buffer — the card was full. `start_qwen_dflash2.sh`
therefore sets `--kv-cache-memory=2500000000` (`DFLASH_KV_MEMORY`), which
yields a measured 33,506-token pool at `MAX_LEN=32768`, `GPU_UTIL=0.915`.

## Measured results

### Stable MTP profile

A repeated 512-token greedy technical response improved from **32.6 tok/s**
(conservative non-speculative baseline) to **83.8-84.9 tok/s**. A real
**130,916-token prompt** completed in 165.5 s and the server stayed healthy.
Quality matches the repository's known fast-variant results:

- perplexity: **8.0943** over 32,635 English, Danish, and code tokens
- GSM8K: **96.5%** over 200 questions, greedy, thinking disabled

These are local before/after measurements, not directly comparable to the
RTX 3090 benchmark suite in the root README.

### Short-context experiments (raw logs under `bench/results/short-context/`)

Identical 32k random-token requests with zero MTP acceptance: FP8 KV decoded
at **25.7 tok/s** versus **14.7** with BF16. At 65k with 96.2% MTP acceptance:
**83.5 tok/s** FP8 versus **28.7** BF16. Cold 65k prefill favored BF16 (36.1 s
versus 62.9 s) — BF16 can reduce one-shot long prefill but is not the faster
decode profile on this pair.

Eight realistic short prompts (English, Danish, code; up to 1,024 output
tokens, one at a time):

- FP8, no speculation: **36.95 tok/s**
- FP8, MTP-3: **75.45 tok/s** (54.8% draft-token acceptance)
- `MAX_SEQS=1`: **75.25 tok/s** — no gain from fewer slots
- greedy vs probabilistic draft selection: no measurable difference
- **BF16, DFlash2-7: 91.7 tok/s** (3.34 accepted tokens per target step)

Context-window caps alone do not make matrix math faster; their value is
enabling cache/backend experiments. FP8 MTP remains the stable general and
long-context profile; DFlash2 is the fastest tested short-context profile.
Expected crossover: DFlash2 wins below ~8k prompts for general chat/coding and
at much longer contexts for copy/quote/edit workloads (its lookup drafter
proposes straight from the prompt).

### DFlash2 correctness

The 32k DFlash2 profile passed the repository's OpenAI API smoke suite
**12/12** (greedy and seeded determinism, logprobs, n=2, stop strings,
JSON-schema structured output, penalties, streaming, thinking mode,
completions echo+logprobs, a 21,755-token prompt, and the expected
`thinking_token_budget` 400 on the V2 runner), and Qwen tool-call parsing
returned a valid `get_weather({"city": "Paris"})` call.

### DFlash2 speed is workload-dependent

Decode speed is not a single number. Per-request llama-swap telemetry (the
profile's own `--enable-per-request-metrics` output) on real traffic at
8–25k context:

| speed band | observed | what the requests were |
|---|---:|---|
| **77–89 t/s** | 77.5, 80.4, 81.3, 83.2, 83.8, 86.5, 88.6 | general chat/generation from context |
| **106–159 t/s** | 106.4, 125.5, 158.7 | long generations and context-reproducing work |

Two artifacts to ignore: 3–64-token requests measure 15–49 t/s because queue +
TTFT dominate their duration, not decode; and the same ~9k context produces
both 77 t/s and 159 t/s, which is acceptance, not context length. The spread is
DFlash2 doing what it was built for: its 7-token proposal is verified whole
when the continuation is predictable, and the lookup drafter
(`VLLM_DFLASH2_LOOKUP=1`, on by default) proposes straight from the prompt
when the model is quoting, editing, or copying — the 106–159 t/s rows. The
91.7 t/s benchmark is an average over eight mixed realistic prompts; treat it
as the midpoint of a workload-shaped range, with the high end on reproduction
and editing.

## llama-swap integration

`heterogeneous/llama-swap.example.yaml` holds the two model entries (also
what runs on the origin host):

- `vllm-speed/qwen3.8-27b` — stable 140k MTP (`start_qwen_stable.sh`)
- `vllm-speed/qwen3.8-27b-dflash2` — experimental 32k DFlash2
  (`start_qwen_dflash2.sh`)

Each entry launches through `/usr/bin/env PORT='${PORT}' NO_API_KEY=1` (the
`env` indirection avoids a llama-swap multi-argument `cd` parsing bug), keeps
`useModelName: "qwen3.8-27b"` (the vLLM served name), and uses `ttl: 900` with
`unloadTimeout: 60`. llama-swap owns launch, swap, unload, and persistent
SQLite activity metrics; the launcher flags above make those metrics
per-request accurate. Engine-wide log lines like `Accepted: N tokens` are
interval aggregates and are not safely assignable per request under
concurrency.

## Bring-up

Complete the common preparation in the root README (including the fast
variant; plus `prepare/fetch_dflash2.py` for the DFlash2 profile), apply every
patch in `patches/` in glob order, then:

```bash
bash verify.sh --no-server          # checks all patches, incl. the two PP ones
bash heterogeneous/start_qwen.sh    # or one of the profile wrappers
```

Docker:

```bash
docker compose --profile hetero up -d
docker compose logs -f hetero
```

## Useful overrides

| variable | default | purpose |
|---|---:|---|
| `GPU_IDS` | `0,1` | CUDA order; the 5070 Ti must be first for the default split |
| `PP_LAYERS` | `44,20` | target layers per pipeline rank; total must be 64 |
| `MODEL` | fast variant if present | target model directory |
| `SPEC` | `mtp` | `mtp`, `dflash2`, or `none` for a diagnostic baseline |
| `DRAFT_TOKENS` | `3` | MTP depth; three is the stable FP8 long-context setting |
| `DFLASH_TOKENS` | `7` | DFlash2 verify block; the checkpoint was trained for 7 |
| `DFLASH_KV_MEMORY` | `2500000000` | manual per-rank KV bytes for the DFlash2 profile |
| `VLLM_DFLASH_CUDAGRAPH` | `0` in the wrapper | `1` re-enables the drafter's private CUDA graph (crashes with the shared quantized LM head) |
| `MAX_LEN` | `140000` / `32768` | API context limit (stable / DFlash2 wrappers) |
| `MAX_SEQS` | `4` | scheduler slots and CUDA-graph sizing; 1 showed no gain |
| `GPU_UTIL` | `0.91` / `0.915` | headroom for compiled prefill and speculative workspaces |
| `KV` | `fp8` / `bf16` | FP8 for MTP (faster decode); BF16 for DFlash2 |
| `PREFIX_CACHE` | `1` | cache shared prefixes and resume hybrid recurrent state |
| `ASYNC_SCHED` | `1` | asynchronous vLLM scheduler |
| `INT8_ACT` | empty | W4A16 target; int8 activation GEMMs do not help batch size one |
| `NO_API_KEY` | unset | `1` disables child auth for a trusted local proxy |

Examples:

```bash
# Frozen stable profile (what llama-swap runs)
bash heterogeneous/start_qwen_stable.sh

# Experimental 32k DFlash2 profile
bash heterogeneous/start_qwen_dflash2.sh

# Known-correct non-speculative fallback
SPEC=none MAX_SEQS=8 GPU_UTIL=0.95 bash heterogeneous/start_qwen.sh

# Disable prefix caching to maximize unrelated one-shot request capacity
PREFIX_CACHE=0 bash heterogeneous/start_qwen.sh
```

## Remaining limitations

- DFlash2+PP is an experimental local vLLM patch, not upstream support; the
  drafter's private CUDA graph is disabled for compatibility with the shared
  quantized LM head. MTP+PP is a backport of upstream PR #46994.
- The V2 runner does not support the `thinking_token_budget` request field;
  normal thinking controls and `chat_template_kwargs.enable_thinking` work.
- A full 140k request nearly fills the stable cache, so only one such request
  can be resident. Several shorter requests can run concurrently.
- Pipeline parallelism still includes the slower 3060 in every model step and
  cannot behave like one unified 28 GiB GPU.
- Keep `GPU_UTIL=0.91` for the stable MTP profile; startup profiling does not
  include compiled-prefill or transient DeltaNet speculative workspace.
- If llama-swap listens on a LAN interface with no auth, restrict network
  access at the firewall.

## Research basis

- vLLM 0.27.1 recommends PP for uneven splits and GPUs without NVLink.
- `VLLM_PP_LAYER_PARTITION` provides manual stage assignment.
- Qwen3.5's model implementation supports pipeline-stage intermediate tensors,
  which the DFlash2 patch extends to auxiliary hidden states.
- Upstream PR #46994 supplies the missing MTP+PP V2-runner behavior backported
  here.
- Consumer Blackwell requires `sm120` JIT support; the launcher builds for both
  `8.6` and `12.0` and uses GCC 15 with CUDA 13.3.
