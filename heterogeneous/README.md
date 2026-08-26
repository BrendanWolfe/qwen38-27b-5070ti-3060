# RTX 5070 Ti + RTX 3060 (heterogeneous pipeline-parallel mode)

This directory runs the repository's W4A16 Qwen3.8-27B across **two unequal
consumer GPUs**: a 16 GiB RTX 5070 Ti (`sm120`, CUDA logical device 0) and a
12 GiB RTX 3060 (`sm86`, device 1). It is runtime-tested with vLLM 0.27.1, the
repository's fast model variant, and both speculative decoders (MTP and
DFlash2).

Four setups are supported:

| profile | launcher | speculation | KV | context | measured result |
|---|---|---|---|---|---|
| batch / general | `start_qwen_batch.sh` | MTP-3, FP8 weights + FP8 KV | FP8 | 147k (147,456-token pool floor) | **73.6-75.5 tok/s** C1, **210 tok/s** aggregate at C4 |
| solo / long | `start_qwen_solo.sh` | MTP-3, one scheduler slot | KVarN 4/2-bit | **262k** (296,974-token pool) | **72.4-73.6 tok/s**, 34.8-35.4 ms/step, C1 only |
| DFlash2 / batch | `start_qwen_dflash2_batch.sh` | DFlash2-7, four scheduler seats | KVarN 4/2-bit | **147k** (151,503-token pool) | C4 completes; three requests resident at 4k |
| DFlash2 / solo | `start_qwen_dflash2_solo.sh` | DFlash2-7, one scheduler seat | KVarN 4/2-bit | **200k** (202,174-token pool) | **83.1–86.6 tok/s**, C1 only |

`start_qwen.sh` is the shared, tunable launcher every profile wraps.

The batch profile's headline number is quoted two ways in this file because
both are true and they answer different questions. A single stream decodes at
73.6-75.5 tok/s on eight mixed realistic prompts (`bench/real_rep.sh`); an
earlier repeated-512-token measurement read 83.8-84.9 on easier text. Four
concurrent streams total 210 tok/s, because a pipeline only overlaps when
there is more than one request in it — see "Concurrency" below.

## Why pipeline parallelism

`nvidia-smi topo -m` reports the cards connected through the PCIe host bridge
(`PHB`) with P2P reads unsupported. Tensor parallelism would split weights
evenly despite the unequal cards and communicate several times per layer.
Pipeline parallelism permits an uneven layer assignment and transfers
activations only at the stage boundary.

The `44,20` split (`VLLM_PP_LAYER_PARTITION`) keeps 44 target layers on the
faster/larger 5070 Ti and 20 layers plus the drafter on the 3060. A tested
`46,18` split reduced the KV pool from ~160k to ~132k tokens with no speed
gain, so `44,20` remains the default. Why moving two layers changed nothing,
and what does move the number, is in "Where the step time goes" below.

### The 3060 is on a PCIe x1 link

On this host the small card is in an x1-wired slot:

```
$ cat /sys/bus/pci/devices/0000:07:00.0/{current,max}_link_width
1
16
```

(The 5070 Ti at `0000:01:00.0` reports 16/16. Both report 2.5 GT/s at idle;
link *speed* trains up under load, link *width* does not.)

This is worth knowing and is not currently a bottleneck. A decode step moves
about 160 KB across the boundary — hidden states and residual for the ~8 tokens
in flight — which is microseconds even at ~1 GB/s. A 2,048-token prefill chunk
moves ~42 MB, so the 130,916-token request pays roughly 3 s of link time inside
its 165 s. The link does cap things worth wanting later: raising
`--max-num-batched-tokens` above 2048 to cut long-prompt TTFT, and the
five-tensor auxiliary relay DFlash2 needs during prefill. If a wider slot is
free, use it; nothing here depends on it.

## Everything that was modified to make this work

### New files in this repository

- `heterogeneous/start_qwen.sh` — shared launcher: uneven PP, toolchain fixes,
  MTP/DFlash2/no-spec selection, process-group cleanup, no-auth mode.
- `heterogeneous/start_qwen_batch.sh` — frozen MTP snapshot used by the
  general llama-swap model, so later experiments cannot change its behavior.
- `heterogeneous/start_qwen_dflash2_batch.sh` — 147k, four-seat DFlash2
  profile (KVarN KV, reversed pipeline, manually sized cache).
- `heterogeneous/start_qwen_dflash2_solo.sh` — 200k, one-seat DFlash2
  profile (KVarN KV, reversed pipeline, manually sized cache).
- `heterogeneous/start_qwen_solo.sh` — 262k single-stream KVarN MTP profile
  (one scheduler slot, 4-bit keys / 2-bit values).
- `heterogeneous/llama-swap.example.yaml` — the four llama-swap model entries.
- `patches/vllm-pr46994-mtp-pp.patch` — MTP + pipeline parallelism (below).
- `patches/vllm-pr52297-gdn-common-metadata.patch` — upstream PR #52297,
  backported. No measurable speed change here, but +6.2% KV pool; see
  "Where the step time goes".
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
allocate a 40-byte broadcast buffer — the card was full. Both DFlash2 wrappers
therefore set `--kv-cache-memory` (`DFLASH_KV_MEMORY`) rather than trusting
auto-sizing.

That cap originally shipped at 2.5 GB, which was conservative: the later FP8
sweep ran 3.2 GB for a 97,962-token pool at `MAX_LEN=88000` — 2.7x the context
the profile first offered, at the same decode rate (85.2 tok/s against bf16's
85.5, acceptance 3.26 against 3.19). The current wrappers use denser KVarN and
the separately validated 2.4/2.8 GB pins below.

**3.2 GB is the ceiling and the failure mode is a trap.** `--kv-cache-memory`
takes memory vLLM would otherwise leave for compiled prefill on the 3060, so an
over-large value does not fail at startup. It boots, sizes a *larger* pool,
captures graphs, answers short prompts, and then dies with a 500 partway into
the first long prompt — `torch.OutOfMemoryError` in the compiled prefill
forward, failing on as little as 18 MiB. 3.4 GB and 3.6 GB both do this. The
launcher header records the full table; every row was checked by sending a
prompt near `MAX_LEN` and confirming `/health` still answered afterwards. A
clean boot proves only that the pool was sized.

## Measured results

### Batch MTP profile

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
enabling cache/backend experiments. FP8 MTP remains the general and
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

## Where the step time goes

Every number in this section is `bench/real_rep.sh` — eight mixed realistic
prompts, 1,024 output tokens each, one at a time, three repeats. It reports
**ms per model step** and **tokens per step**, which is what to compare when
two configurations look close: end-to-end tok/s drifts between sessions,
ms/step does not.

The model is dense (27B, 64 layers, no experts), so a decode step reads every
weight. That makes the step time predictable from bandwidth alone. Quantized
weights are ~14.7 GB, ~0.21 GB per layer, plus ~0.64 GB each for the
248,320-entry embedding table and LM head. The 5070 Ti runs at ~896 GB/s and
the 3060 at ~360 GB/s, and **at concurrency 1 a pipeline does not overlap** —
rank 0 runs, then rank 1 runs, so the step is the sum of the two stages:

| batch profile, per decode step | bytes read | at that card's bandwidth |
|---|---:|---:|
| 5070 Ti (rank 0): 44 target layers + embedding | 9.9 GB | 11.0 ms |
| 3060 (rank 1): 20 target layers | 4.2 GB | 11.7 ms |
| 3060: 248k-vocab LM head, verify pass | 0.64 GB | 1.8 ms |
| 3060: 3 chained MTP passes (layer + draft head) | ~0.96 GB | 2.7 ms |
| **modelled total** | | **~27 ms** |
| measured | | **35.2 ms** |

Two things fall out of that table. The first is that the 3060 holds a third of
the model and accounts for roughly **16 of the 27 modelled milliseconds**,
because vLLM puts the LM head, the sampler and the MTP drafter on the *last*
pipeline rank and that was the slow card. The second is that ~8 ms of the
measured step is neither weight reads nor bandwidth inefficiency — it is
speculation overhead: rejection sampling, the multi-query verify attention,
the draft loop, and per-step metadata.

This also explains the `46,18` result. Moving two layers to the fast card is
worth 2 x 0.35 = 0.7 ms, about 2%, which is inside the noise — while costing
28k tokens of pool. The split was never the lever; *what sits on the last
rank* is.

### What this round of tuning actually changed

| change | ms/step | tok/step | KV pool | verdict |
|---|---:|---:|---:|---|
| batch, as shipped | 35.2 | 2.54-2.58 | see below | baseline |
| + PR #52297 (GDN metadata hoist) | 35.1 | 2.54-2.58 | not measurable | no speed change |
| + reversed pipeline, 32k | **32.4** | 2.51-2.62 | 33,363 | -7.7% step time, -78% pool |

The KV pool column is deliberately empty for the first two rows. The pool from
an unchanged command varies by up to 38,805 tokens between launches (gotcha
38), which is larger than any effect either change could have had, so a
one-run-each comparison of pools measures the lottery and nothing else.

**PR #52297 does not speed this machine up.** Upstream measured +61% output
tok/s at concurrency 1 on Qwen3.6-35B-A3B + DFlash on an H200, from cutting
`GDNAttentionMetadataBuilder.build()` from ~900us to ~300us per cache group.
That is a fixed CPU-side saving, and it lands against a GPU step three times
longer here, on a 9800X3D that is not the bottleneck, with async scheduling
already hiding part of it. Measured: 35.2 -> 35.1 ms/step, 0.3%, with
byte-identical tokens per step.

It is still worth keeping — a CPU-side saving that costs nothing is worth
having, and the hoisted helper is verified against the 0.27.1 inline code over
4,002 random batches. `patches/vllm-pr52297-gdn-common-metadata.patch`
documents the backport.

This file previously claimed the patch also bought 9,131 tokens of pool
(146,847 -> 155,978) by lowering the profiled peak activation. That is
withdrawn. Both numbers are ordinary draws from the startup-profiling lottery
in gotcha 40, which spans 146,086 to 188,769 tokens on this profile with no
code change at all. The patch may or may not help the pool; one run each
cannot tell, and nothing here has measured it properly.

### Reversing the pipeline is a short-context win only

Putting the drafter, LM head and sampler on the 5070 Ti is worth **+13.4%**
decode on DFlash2 (97.24 against 85.77 tok/s, ITL 33.2 against 37.5) with
acceptance length unchanged at 3.30 against 3.28 — so it is pure placement, not
a quality difference.

It does **not** free memory, which is the intuitive but wrong reason to reach
for it. It relocates the pressure: reversed, the 5070 Ti carries 40 of 64
layers *and* the drafter (10.62 GiB of weights, ~1.4 GiB spare) while the 3060
holds 24 layers and idles on 2.67 GiB. Forward, the 3060 carries 20 layers plus
the drafter's 2.69 GiB and has 1.44 GiB spare. Either way one card is tight.

Every row below was probed with a real prompt near `MAX_LEN` and a health check
afterwards:

| split | drafter | cap | KV | MAX_LEN | pool | decode | prefill | long prompt |
|---|---|---:|---|---:|---:|---:|---:|---|
| 44,20 | 3060 | 3.2G | fp8 | 88,000 | 97,962 | 85.77 | 991/s | survives |
| 28,36 | 5070 Ti | 3.2G | fp8 | 88,000 | 117,886 | 87.60 | 761/s | survives |
| 24,40 | 5070 Ti | 2.15G | bf16 | 32,768 | 34,539 | **97.24** | 957/s | survives |
| 24,40 | 5070 Ti | 2.6G | fp8 | 60,000 | 85,123 | 90.66 | — | OOM-killed |
| 24,40 | 5070 Ti | 3.2G | fp8 | 88,000 | 117,886 | — | — | dies at startup |

To hold 88k the reversed split must give layers back to the 3060, and the
advantage collapses to +2.1% decode while costing **23% of prefill** — 105 s to
first token on an 80k prompt against 81 s. The two tested DFlash2
configurations are different operating points. The current DFlash2 wrappers
use the later KVarN measurements below instead.

**Reversing the pipeline is worth 7.7%.** `GPU_IDS=1,0 PP_LAYERS=20,44` gives
each card the same number of target layers as before — and therefore the same
11/5 split of full-attention layers and the same KV bytes per token — but
makes the 5070 Ti the last rank, so the LM head, the sampler and the MTP
drafter move onto it. 35.1 -> 32.4 ms/step, 73.6-75.5 -> 78.7-82.6 tok/s,
tokens per step unchanged. It can be reproduced manually with
`GPU_IDS=1,0 PP_LAYERS=20,44`.

It cannot become the general profile, because the last rank also carries the
sampling and logits peak — 1.24 GiB against the first rank's 0.32 GiB — so the
5070 Ti gives up ~3.15 GiB of KV (4.23 -> 1.08 GiB) and the pool collapses to
33,363 tokens. At `MAX_LEN=100000` it will not start at all. The 3060 idles at
6.5 GiB of its 12 GiB in this arrangement, which is the honest description of
the trade: the pair's memory is used worse so that its bandwidth is used
better.

### Concurrency

Pipeline parallelism exists to overlap stages, and at one request it cannot.
Measured on the batch profile, 512 output tokens, `vllm bench serve`:

| concurrency | aggregate output tok/s | vs C1 | mean ITL | mean TTFT |
|---|---:|---:|---:|---:|
| 1 | 73.0 | 1.00x | 34.6 ms | 170 ms |
| 2 | 115.7 | **1.58x** | 40.8 ms | 225 ms |
| 4 | 213.9 | **2.93x** | 43.6 ms | 325 ms |
| 6 | 258.4 | **3.54x** | 56.4 ms | 389 ms |
| 8 | 301.0 | **4.12x** | 61.2 ms | 438 ms |

Concurrency is the largest free win on this pair and it needs no patch: an
agent or editor that issues parallel calls gets most of it automatically. Note
what it does *not* say — a single stream is not faster, and the earlier
finding that `MAX_SEQS=1` buys nothing (75.25 against 75.45 tok/s) still
stands, because that experiment only ever ran one request at a time.

**The slot count was the binding limit, so the default is now `MAX_SEQS=8`.**
The C1-C4 rows above reproduce the earlier four-slot measurements almost
exactly (73.1 / 120.8 / 212.7 at four slots), so eight slots change nothing
about how the server behaves at low load; they only add the C6 and C8 rows,
which were previously unreachable. That is +41% aggregate throughput at eight
streams for free. Decode here is bound by weight bandwidth, not by math, so
additional sequences ride along in a step that was already reading every
weight.

Nothing else gated it. `max_concurrent_batches` is already 3 (PP=2 plus async
scheduling on the V2 runner, `config/vllm.py`), `max_cudagraph_capture_size` is
derived as `MAX_SEQS * (DRAFT_TOKENS + 1)` so the captured shapes grow with the
ceiling (capped at 64 query tokens since the upstream merge — inert at every
profile here, see gotcha 38), and `long_prefill_token_threshold` is 0 with decode tokens placed in
the batch ahead of prefill chunks, so concurrent streams are not starved by one
long prefill. The GDN recurrent state — which `docs/optimizations.md` records
as the real concurrency bound on this architecture — is already handled by
`--mamba-ssm-cache-dtype float16` in the launch command.

The measured cost of the higher ceiling is CUDA graph memory: 0.13 -> 0.14 GiB,
about 1.3k tokens of pool. It is *not* responsible for the pool varying between
starts. That swing is rank 1's profiled activation peak, which comes out at
either 0.32 GiB or 1.24 GiB (the sampling/logits peak) depending on the draw,
and it happens at four slots too — six repeated starts gave 184,130 tokens
three times out of three at four slots and 146,847 / 157,500 / 184,130 / 184,130
at eight. Every draw still clears the 140,000 the profile requires, but pin
`--kv-cache-memory` before comparing pools across configurations.

Past eight the curve flattens and latency does not, which is why 8 and not 16
is the default. At `MAX_SEQS=16`: C8 is 309.5 tok/s (the same as eight slots,
within noise), C12 is 341.6 for 82.0 ms ITL, and C16 is 385.2 but with mean
TTFT at **2,955 ms** as 64 prompts queue behind a 2,048-token prefill budget.
`MAX_SEQS=16` is the right override for batch or throughput work, where TTFT
does not matter; it costs nothing at low load.

**Seats are admissions; residency is pool-shaped.** Upstream measured this on
one 3090 and the mechanism applies here too: `MAX_SEQS` decides how many
requests are *admitted*, but every request that is actually *resident* reserves
its `1+k` recurrent-state slots out of the same pool before it stores a single
token of context — 0.88 GiB at DFlash2 `k=7` against 0.44 GiB at MTP `k=4`.
Past the point where the pool runs out, extra seats queue and then preempt. On
one 3090 at `CTX=fast` that ceiling is six DFlash2 residents and eight MTP, and
raising `MAX_SEQS` there buys nothing.

This pair's answer came out the other way — eight seats being +41% — because
the constraint binds much later here: MTP-3 has the smallest state page of the
three (`k+2 = 5`) and the batch pool is 147k+ tokens, so the C8 row above is
measured throughput with nothing preempted, not an admission count. Both
statements hold; the *number* is what does not transfer. If you change `SPEC`,
`DRAFT_TOKENS` or `KV` and want the residency for that shape rather than the
seat count, `bench/conc_ladder.py` now reports it directly — decode aggregate
over the window in which all N streams are genuinely decoding, `ms/pass`,
preemptions, peak pool occupancy and the per-stream spread, on salted prompts
that cannot hit the prefix cache.

Read gotcha 39 the same way. It found `MAX_SEQS > 12` *fatal* at `CTX=huge` on
a 24 GiB 3090 — boots, captures, serves `/health`, then dies on the first
prompt because the per-seat allocations ate the headroom the first prefill
needs. That is a VRAM budget rather than a shape, so it is specific to a card
and a pinned pool, and the `MAX_SEQS=16` row above is not a refutation of it:
that run is the batch profile, whose pool comes from `GPU_UTIL` rather than
from a pinned `--kv-cache-memory`. The profile here that *is* pinned is
DFlash2, at `DFLASH_KV_MEMORY=3200000000` and four seats, and it has never been
run above four. Raising seats on a pinned pool is the case to check before
trusting.

### 262k on int4 per-token-head KV

This is the route to the full native context that `start_qwen_solo.sh`
replaced. It no longer has a launcher of its own; reach it with
`KV=int4pth MAX_LEN=262144 MAX_SEQS=4 bash heterogeneous/start_qwen.sh`. It is
still the only 262k shape here that keeps four scheduler slots, which is the
one thing solo gives up, so the measurements stay.

Read those four slots as admissions, not as four concurrent 262k requests: the
pool is 284,234 tokens, so one request at full depth is the whole of it. What
the slots buy is a server that accepts one very long request *and* three
ordinary ones — which solo, at `MAX_SEQS=1`, cannot do at all. That is the
distinction the concurrency section above draws, and it is the reason to keep
this shape rather than a reason to prefer it at depth.

`--kv-cache-dtype int4_per_token_head --attention-backend TRITON_ATTN` holds a
**284,234-token pool** at `MAX_LEN=262144`, which FP8 cannot reach at any
`MAX_LEN` (it refuses above ~148k). The binding
rank is the 5070 Ti with 11 of the 16 full-attention layers; int4 per-token-head
is about half FP8's 2 KB per token per layer, which is the whole trick. MTP,
pipeline parallelism, the V2 model runner and the Triton backend all start
together, which was the open question — the repo's earlier
`int4_per_token_head` results were batch-mode on the V1 runner with no
speculation.

The cost is prefill, and it is much larger than "somewhat slower". Measured on
this pair, with the KV pool as the clock:

| prompt | prefill rate | wall time |
|---|---|---|
| 32k | ~840 tok/s | 38.2 s end to end, needle **PASS** |
| ~225k (instantaneous) | **~112 tok/s** | ~4,460 tokens per 40 s |

At that depth a full 240k prompt takes roughly **35 minutes** before the first
token, so the profile is a batch tool, not an interactive one. A 240k needle
run was started and abandoned partway through the first depth for exactly that
reason: the rate is the answer, and confirming recall would have cost another
hour of GPU time. What is verified is that the profile boots with a
284,234-token pool, generates correctly, and passes a 32k needle; **recall
beyond 32k is unverified here.**

Use `start_qwen_batch.sh` as the general server and switch to this only when a
request will not fit, and only when you can wait.

KVarN (`kvarn/`, `CTX=huge` in single-user mode) is denser still — ~840 B per
token per layer against int4 per-token-head's ~1 KB. This section used to say
it had never been run on the V2 model runner that MTP+PP requires and was "the
next thing to try"; it has since been tried, and it works — that is
`start_qwen_solo.sh`, a 296,974-token pool at the same 262,144 context. It cost
`kvarn/kvarn-pp-pool-budget.patch`, and it buys the denser pool by dropping to
a single scheduler slot: KVarN forces a 2048-token attention block to match the
GDN page, and every slot pays a full aligned page. At four slots it does not
fit at 262k at all, which is why the int4pth numbers above are still the
four-slot answer.

### DFlash2 on KVarN: 200,192 context, measured

Upstream added a KVarN route under DFlash2 (`SPEC=dflash2 CTX=huge`, 245,760 on
one 24 GiB 3090). On this pair the combination works, but the limiting term is
structural: DFlash2 holds `1+k` recurrent-state slots per request and under
KVarN every slot pays a full **aligned** page — 2176 tokens at
`DFLASH_TOKENS=7` (17x128; the 2048 figure quoted elsewhere is the k=3 block,
and the size follows the speculator, gotcha 37).

**Pipeline geometry is the large lever.** Reversing the devices puts the
DFlash2 drafter, LM head and sampler on the 5070 Ti. The earlier `28,36` split
left too little real headroom on that rank for a larger pinned pool; moving two
target layers back to the 3060 (`30,34`) transfers about 0.46 GiB of weights
away from the 5070 Ti. The measured exchange rate is 0.2307 GiB per target
layer, from 6.46 GiB / 28 layers on the 3060.

The resulting measured command is:

```
GPU_IDS=1,0 PP_LAYERS=30,34 SPEC=dflash2 DFLASH_TOKENS=7 SPEC_ATTN=0 \
  KV=kvarn MAX_SEQS=1 MAX_LEN=200192 \
  EXTRA_ARGS=--kv-cache-memory=2800000000 bash heterogeneous/start_qwen.sh
```

The pin is 2.8 GB decimal, or 2.61 GiB. The pool was **202,174 tokens** on two
independent boots, bit-identical. A 32k needle and a **190,057-token** needle
both passed, with `/health` returning 200 after each. The decode benchmark also
left `/health` at 200:

| | previous measured KVarN | this |
|---|---:|---:|
| geometry | `1,0` / `28,36` | `1,0` / `30,34` |
| pinned KV | 2.0 GiB | 2.8 GB / 2.61 GiB |
| context | 122,880 | **200,192** |
| pool | 123,407 | **202,174** (2/2 identical) |
| `bench/real_rep.sh` C1 x3, e2e | 88.83-88.95 tok/s | **83.06 / 83.83 / 86.63 tok/s** |
| decode-only | 91.4-91.5 tok/s | **85.7 / 86.4 / 89.2 tok/s** |
| tokens/step | 3.29 | 3.27 / 3.27 / 3.27 |
| ms/step | 37.0 | 39.4 / 39.1 / 37.8 |
| needle | PASS @ 32,040 and 110,055 | PASS @ 32k and **190,057** |

That is 63% more validated context than the previous 122,880 result. Decode is
modestly lower at the longer context, as expected; the result is a capacity
trade, not a speed win.

**A pinned pool makes sizing reproducible, not execution safe.** Passing the
KV-cache sizing check, reporting the expected pool and answering `/health` 200
do not prove the configuration can run. Three higher-pressure configurations
made it that far and then died during CUDA graph capture, first prefill, or a
later NCCL allocation. `--kv-cache-memory` skips profiling entirely, so the
`available` value in the sizing path describes the selected pool budget, not
real per-rank free-VRAM headroom. Survival is gated by free VRAM after weights
and the pin: this sweep found that rank 1 (the 5070 Ti) needs roughly 2.86 GiB
or more spare, while rank 0 (the 3060) fails somewhere at or below 1.57 GiB
(the lower threshold was not fully bracketed). See gotcha 41.

Every candidate therefore needs a real prompt and a health check afterwards;
a clean boot and an initial health check are only setup. `SPEC_ATTN=0` also
remains mandatory: the split-KV verify attention is bf16-KV only and KVarN
brings its own dequant path.

### KVarN on the batch profiles: longer context, with different constraints

KVarN does help the eight-seat MTP batch profile. The original 70k-capped test
reported a 106,400-token pool, but that was one high-activation startup draw
and was incorrectly treated as a capacity comparison. At the same shipped
`44,20` geometry with `MAX_SEQS=8`, a low-activation draw booted at the full
**262,144-token** native context with a **289,641-token pool**. A
**240,035-token** retrieval passed at 684 prompt tok/s and left `/health` at
200. The eight-seat tail pool adds only 14 fp16 slots over the one-seat shape
(40 instead of 26), so it does not erase KVarN's per-token density win.

Startup remains the constraint: high-activation draws in this sweep refused
245,760 and 262,144 with estimated limits of 124,928--135,168, while the next
262,144 draw booted cleanly. The FP8 batch launcher therefore remains the
reliable default until the KVarN batch route gets the solo launcher's bounded
retry behavior; the result is not evidence that FP8 holds more context.

DFlash2 is different. With the solo profile's reversed `30,34` geometry,
`MAX_SEQS=4`, a 2.4 GB pin and `MAX_LEN=147456`, KVarN produced a reproducible
**151,503-token pool**. A **140,028-token** retrieval passed at 664 prompt
tok/s and left `/health` at 200. Four distinct ~4k requests also completed
without preemption; three were resident while one queued, peaking at 94.3% of
the pool. This is `start_qwen_dflash2_batch.sh`.

The pin is part of the profile, not spare capacity. At 2.8 GB the same
four-seat server booted with a 183,370-token pool and passed one 150,054-token
retrieval, then died at C4 when rank 1 needed another 24 MiB. The 2.8 GB pin is
safe only in the one-seat `start_qwen_dflash2_solo.sh`, where a fresh
190,050-token retrieval passed in 321.1 s and `/health` remained 200.

### DFlash2: KV dtype, pipeline order and where the layers go

The shipped DFlash2 profile assumed BF16 KV. That assumption was never tested
and it is wrong; testing it turned up two further effects that matter more. All
rows are `bench/real_rep.sh`, 3 reps, 8 realistic 1,024-token prompts at
concurrency 1, pool pinned via `--kv-cache-memory`.

| order | KV | backend | graphs | ms/step | tok/step | e2e tok/s | pool |
|---|---|---|---|---|---|---|---|
| forward 44/20 | BF16 | FlashAttn | full | 37.5 | 3.13–3.23 | 83.4–86.0 | 33,506 |
| forward 44/20 | FP8 | FlashInfer | piecewise | 38.5 | 3.12–3.22 | 81.1–83.5 | 53,683 |
| forward 44/20 | int8pth | Triton | full | 37.6–37.9 | 3.03–3.19 | 80.4–84.8 | 52,986 |
| reversed 26/38 | FP8 | FlashInfer | piecewise | 36.6 | 3.23–3.31 | 88.3–90.4 | **56,704** |
| reversed 24/40 | FP8 | FlashInfer | piecewise | 35.5 | 3.04–3.21 | 88.1–92.9 | 45,085 |
| reversed 26/38 | BF16 | FlashAttn | full | 34.9 | 3.13–3.53 | 92.5–104.3 | 35,277 |
| **reversed 24/40** | **BF16** | FlashAttn | full | **33.9–34.0** | 3.15–3.27 | **95.2–98.8** | 34,539 |
| reversed 24/40 | int8pth | Triton | full | **34.1–34.6** | 3.12–3.29 | 93.5–99.6 | 44,387 |
| reversed 20/44 | FP8 | FlashInfer | piecewise | 33.8–34.0 | 3.01–3.29 | 89.0–96.6 | 17,497 |
| reversed 20/44 | int8pth | Triton | full | 32.6 | 3.08–3.26 | 97.4–103.2 | 18,770 |
| reversed 16/48 † | int8pth | Triton | full | **30.3–30.6** | 2.94–3.09 | 98.6–104.2 | 10,436 |

† `DFLASH_TOKENS=5 MAX_SEQS=2`, `MAX_LEN=8192` — see the memory note below.

Four things follow.

**Reversal pays roughly twice as much for DFlash2 as for MTP** — −12.6% step
time at 20/44 against MTP's −7.7%. That is the drafter: DFlash2's is ~1.2 GiB
and runs eagerly because its private CUDA graph is disabled, so leaving it on
the 3060 costs more than MTP's three small chained passes. (Re-enabling that
graph with `VLLM_DFLASH_CUDAGRAPH=1` now starts without the Marlin failure the
flag was added for, and is worth nothing: 36.5 ms against 36.6 over 3 reps.)

**The fewest layers the 3060 can hold is the fastest split.** At concurrency 1 a
pipeline does not overlap, so the step is the *sum* of the stages. A layer on
the 3060 costs 0.21 GB / 0.360 GB/s = 0.58 ms and saves only 0.21 / 0.896 =
0.23 ms on the 5070 Ti, so moving one off is worth ~0.35 ms in theory and 0.48
measured (24 → 20 → 16). Nothing about the split balances; it is monotone, and
the only limit is 5070 Ti memory. That limit bites twice, because the
full-attention layers are every 4th one: at 24/40 the 5070 Ti holds 10 of the
16, at 16/48 it holds 12, so KV cost per token rises as the split falls. Hence
the pool collapsing 45,085 → 18,770 → 10,436 while the step time drops. The
memory to attack is the aligned recurrent-state page, which scales with the
verify block (`DFLASH_TOKENS`) and *not* with `MAX_SEQS` — gotcha 33. Dropping
`DFLASH_TOKENS` 7 → 5 is what makes the 16/48 row start at all.

**FP8 KV silently costs you full CUDA graphs under speculative decoding.** This
is why the int8pth rows beat their FP8 neighbours, and it is not about density.
Under spec-decode vLLM consults the attention backend's `AttentionCGSupport`:
FlashInfer reports `UNIFORM_SINGLE_TOKEN_DECODE`, so the mode is downgraded —
every FP8 server log contains *"CUDAGraphMode.FULL_AND_PIECEWISE is not
supported with spec-decode for attention backend FlashInferBackend; setting
cudagraph_mode=PIECEWISE"* — while `TRITON_ATTN` reports `ALWAYS` and keeps
full graphs. FlashAttention cannot do FP8 on these cards (*"FP8 KV cache
requires FA3 on SM90 or FA4 on SM100"*; they are sm120 and sm86) and Triton
cannot either (native `fp8e4nv` needs SM89+, the 3060 is sm86), so
`int8_per_token_head` is the only dense-KV mode that keeps full graphs on
**both** cards. It works here only because the repo already carries
`hybrid-sw-block-promote.patch` (without it int8 KV costs *more* memory than
bf16) and `spec-decode-int8-kv.patch`.

**But int8pth trades prefill for decode, and the trade gets worse with depth.**
Measured at the 24/40 split:

| prompt depth | FP8 | int8pth | BF16 |
|---|---|---|---|
| 4,251 | 1075 tok/s | 1010 tok/s | **1141 tok/s** |
| 16,804 | 1167 tok/s | 932 tok/s | **1181 tok/s** |
| 29,355 | **1016 tok/s** | 715 tok/s | 1008 tok/s |

int8pth's decode win is ~0.36 ms per output token, so it only repays that TTFT
cost after roughly 700 output tokens at a 4k prompt, 10,000 at 17k and 34,000 at
29k. (Its quality is fine — GSM8K 96.0% at 24/40, n=200 greedy, against the
repository's 96.0–96.5% baseline — the problem is purely prefill.)

**BF16 is the answer, and it wins on every axis except pool.** It selects
FLASH_ATTN, which keeps full graphs just as Triton does, but without Triton's
prefill penalty: at a matched 24/40 split BF16 is quicker than FP8 at prefill at
two of three depths and tied at the third, *and* decodes 4.5% quicker (33.9–34.0
against 35.5–35.6 ms). It is also the only one of the three that does not
quantize the KV cache at all, so it needs no quality argument. The cost is
density — 4 KB per token per layer means a 34,539-token pool against FP8's
45,085. Both cover a full 32k request; FP8 allows 1.38x concurrency there
against BF16's 1.05x, which is the only reason left to choose it.
For manually tuned short-context runs, **BF16 at 24/40** is the fastest tested
combination.

**None of this transfers to the batch FP8 MTP profile.** The same piecewise
downgrade fires there — it is FP8/FlashInfer too — but switching it to int8pth
is a net loss, measured same-session at 44/20, `MAX_LEN=140000`:

| batch profile, FP8 | ms/step | tok/step | e2e tok/s | pool |
|---|---|---|---|---|
| FP8 (as shipped) | 35.0 | 2.56–2.61 | 74.7–75.9 | **184,130** |
| int8pth | 34.5 | 2.50–2.63 | 74.1–77.8 | 145,240 |

Only −1.4% step time, against −21% of the pool, before counting a Triton
prefill penalty that is worst exactly where this profile is used. Two reasons
the win shrinks: MTP-3 verifies 4 query tokens where DFlash2-7 verifies 8, so
there is less per-step launch overhead for full graphs to remove; and at long
context int8pth is *less* dense, not more, because its per-head fp32 scale
breaks the page divisibility (2112 vs 2080 B/token — gotcha 27). The batch
profile stays on FP8.

This also corrects an earlier claim in this file, that the per-token-head modes
are "worth the Triton prefill penalty only when nothing else fits, never for
speed." The table above already contradicted it — int8pth was quicker than FP8
in both comparable rows — and the reason is the CUDA graph mode, not the cache
dtype. What remains true is the *density* claim: below ~2 KB per token per
layer more compression buys no context, because the Mamba/GDN state page binds
(FP8 53,683 vs int8pth 52,986 at 44/20), and `MambaDType` only admits
float32/float16/bfloat16, so that floor cannot be lowered.

## llama-swap integration

`heterogeneous/llama-swap.example.yaml` holds the four model entries (also
what runs on the origin host):

- `vllm-speed/qwen3.8-27b-batch` — general 147k MTP (`start_qwen_batch.sh`),
  aliased from `vllm-speed/qwen3.8-27b`
- `vllm-speed/qwen3.8-27b-solo` — 262k single-stream KVarN (`start_qwen_solo.sh`)
- `vllm-speed/qwen3.8-27b-dflash2-batch` — DFlash2, 147k KVarN, four seats
- `vllm-speed/qwen3.8-27b-dflash2-solo` — DFlash2, 200k KVarN, one seat

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
| `GPU_IDS` | `0,1` | CUDA order; `1,0` reverses the pipeline |
| `PP_LAYERS` | `44,20` | target layers per pipeline rank; total must be 64 |
| `MODEL` | fast variant if present | target model directory |
| `SPEC` | `mtp` | `mtp`, `dflash2`, or `none` for a diagnostic baseline |
| `DRAFT_TOKENS` | `3` | MTP depth; three is the FP8 long-context setting |
| `DFLASH_TOKENS` | `7` | DFlash2 verify block; the checkpoint was trained for 7 |
| `DFLASH_KV_MEMORY` | `2400000000` / `2800000000` | manual per-rank KV bytes for DFlash2 batch / solo; the solo pin OOMs at C4 |
| `VLLM_DFLASH_CUDAGRAPH` | `0` in the wrapper | `1` re-enables the drafter's private CUDA graph (crashes with the shared quantized LM head) |
| `MAX_LEN` | `147456` / `262144` / `147456` / `200192` | API context limit (MTP batch / MTP solo / DFlash2 batch / DFlash2 solo) |
| `MAX_SEQS` | `8` | scheduler slots and CUDA-graph sizing; 8 is +41% aggregate throughput at C8 over the old 4 and identical at C1-C4. `16` for batch work (up to 385 tok/s at C16, but ~3 s TTFT) |
| `GPU_UTIL` | `0.91` / `0.915` | headroom for compiled prefill and speculative workspaces |
| `KV` | `fp8` / `bf16` / `fp8fa` / `int4pth` / `int8pth` / `kvarn` | FP8 for MTP (faster decode); BF16 for DFlash2; `int4pth` is int4 per-token-head on the Triton backend, which is what makes 262k fit |
| `PREFIX_CACHE` | `1` | cache shared prefixes and resume hybrid recurrent state |
| `ASYNC_SCHED` | `1` | asynchronous vLLM scheduler |
| `INT8_ACT` | empty | W4A16 target; int8 activation GEMMs do not help batch size one |
| `NO_API_KEY` | unset | `1` disables child auth for a trusted local proxy |
| `VISION` | `0` | `1` keeps the vision tower instead of `--language-model-only`. Untested on this pair: `model.visual.*` loads on rank 0, the card that binds the KV pool, so its 0.858 GiB and the encoder's profiled peak both come out of the scarce side |
| `CUDAGRAPH_MODE` | unset | forces `PIECEWISE` / `FULL_AND_PIECEWISE` instead of vLLM's choice. Nothing here pins it — the runner fix makes that unnecessary — but A/B-ing the capture mode on one server needs the knob |
| `CG` | `MAX_SEQS * (DRAFT_TOKENS + 1)`, capped at 64 | captured decode-graph size. Raise `VLLM_V2_CUDAGRAPH_MEM_MIB` with it if you override past the cap |

Examples:

```bash
# Frozen general profile (what llama-swap runs)
bash heterogeneous/start_qwen_batch.sh

# Four-seat 147k DFlash2/KVarN profile
bash heterogeneous/start_qwen_dflash2_batch.sh

# Single-stream 200k DFlash2/KVarN profile
bash heterogeneous/start_qwen_dflash2_solo.sh

# Full native 262k context on ONE stream, at the batch profile's decode rate
bash heterogeneous/start_qwen_solo.sh

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
- A full 147k request nearly fills the batch cache, so only one such request
  can be resident. Several shorter requests can run concurrently.
- Pipeline parallelism still includes the slower 3060 in every model step and
  cannot behave like one unified 28 GiB GPU. At concurrency 1 it does not even
  overlap the two stages, so a single stream pays the sum of both; throughput
  scales 4.12x to eight concurrent streams and no further without giving up
  TTFT.
- The reversed pipeline trades context for step time and cannot exceed ~33k. It
  leaves ~5 GiB idle on the 3060; that memory has no use in that arrangement.
- `start_qwen_solo.sh` serves exactly one request at a time. A second concurrent
  request queues behind the first; use the batch profile when that matters.
- The 262k profile is Triton-backend only and its prefill is much slower than
  the batch profile's. It is a fits-or-does-not-fit mode, not a faster one.
- Keep `GPU_UTIL=0.91` for the batch MTP profile; startup profiling does not
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
