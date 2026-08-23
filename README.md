# Fork notes — RTX 5070 Ti + RTX 3060 heterogeneous mode

This is a fork of [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090).
Everything below the `---` is the upstream README, unchanged. This fork adds an
experimental **two-GPU pipeline-parallel** path (`heterogeneous/`) that serves
the same W4A16 Qwen3.8-27B across a 16 GiB RTX 5070 Ti (`sm120`) and a 12 GiB
RTX 3060 (`sm86`). Pipeline parallelism is used instead of tensor parallelism
because the cards are unequal and have no P2P path; the 44/20 layer split keeps
most of the target on the faster card and transfers activations only at the
stage boundary. The conversion of the upstream 3090 codebase into this
two-GPU setup was produced with the assistance of **gpt-5.6-sol**.

## Five setups

| profile | launcher | KV | context | measured decode |
|---|---|---|---|---|
| **batch / general** | `heterogeneous/start_qwen_batch.sh` | FP8 | 147k (147,456-token pool floor) | **73.6–75.5 tok/s** C1; **210 tok/s** at 4 concurrent |
| **solo / long** | `heterogeneous/start_qwen_solo.sh` | KVarN 4/2-bit | **262k** (296,974-token pool) | **72.4–73.6 tok/s**, one stream only |
| **short-context** | `heterogeneous/start_qwen_dflash2.sh` | BF16 | 32k (33,506-token pool) | **83.4–86.0 tok/s** (77–159 t/s by workload) |
| **fastest DFlash2** | `heterogeneous/start_qwen_dflash2_fast.sh` | BF16 | 32k (34,539-token pool) | **95.2–98.8 tok/s**, 33.9 ms/step |
| **huge context** | `heterogeneous/start_qwen_huge.sh` | int4 | **262k** (284,234-token pool) | batch only — prefill falls to ~112 tok/s at depth |

The decode column is `bench/real_rep.sh` — 8 realistic 1,024-token prompts at
concurrency 1, 3 reps — so the rows are comparable to each other. DFlash2's
speed is strongly workload-dependent (77–159 tok/s across the range); the
earlier headline of 91.7 tok/s came from a different prompt mix and is not
comparable to these.

All five use the repository's fast variant (int4-GPTQ lm_head + MTP module) and
vLLM 0.27.1's V2 model runner. Quality is unchanged by speculation: perplexity
8.0943, GSM8K 96.5%, and a real 130,916-token prompt completes on the batch
profile.

## Why this is faster than a two-GPU GGUF setup (and what it costs)

On the **same two cards**, this fork measured **~84 tok/s** stable (up to
**~159 tok/s** with DFlash2 on reproduction/editing workloads) against **30–40
tok/s** for a two-GPU unsloth UD-Q5_K_M MTP setup. It is faster for mechanical
reasons, not luck:

- **Fewer weight bytes per token.** W4A16 Marlin is ~14.7 GB versus ~18.5 GB
  for Q5_K_M at ~5.5 bpw; at the DRAM-bandwidth regime these cards live in,
  fewer bytes is directly faster decode — before speculation.
- **vLLM's pipeline parallelism overlaps the cross-card traffic.** Without
  P2P, every token moves activations and KV over PCIe; vLLM's async NCCL
  point-to-point overlaps that with compute, where a GGUF device split
  serializes more of it.
- **Speculation amortizes the per-token PCIe/DRAM cost.** MTP-3 lands ~2.8–3.3
  tokens per verify step and DFlash2 up to ~7 with the lookup drafter, so the
  cross-card and memory cost is paid once per step instead of once per token.
- **FP8 KV halves KV traffic** versus the higher-precision KV a Q5 GGUF run
  keeps.
- **It is a serving stack, not an inference loop**: CUDA graphs, Triton/
  FlashAttention kernels for the hybrid-Mamba architecture, per-request
  metrics, and working Qwen tool calling (12/12 API smoke suite).

The context win is just as large: with Q5 weights the pair has only ~9.5 GB
left for KV (roughly 20–40k tokens), while this fork's FP8 pool holds at
least **147,456 tokens**, and the KVarN and int4 profiles reach Qwen3.8's full
native **262,144** — 131k+ context is only possible on the vLLM path.

Honest tradeoffs versus the GGUF setup:

- **Quant quality.** 4-bit vs 5.5-bit is a real delta, measured small: IFBench
  78.3 vs 79.5 unquantized, GSM8K 96.5%, perplexity 8.0943. Speculation is
  lossless — the quant is the only lossy layer.
- **Maintenance surface.** vLLM is pinned at 0.27.1 with 15 patches, two of
  them bespoke to this PP setup; upgrades are projects. GGUF stacks update
  routinely.
- **Known workaround.** The DFlash2 drafter runs eagerly
  (`VLLM_DFLASH_CUDAGRAPH=0`) because its shared quantized LM-head GEMM crashes
  inside DFlash's private CUDA graph — some speed is left on the table.
- **Throughput shape.** The 3060 bounds every step and a full 147k request
  occupies the cache alone. A single stream pays both pipeline stages in
  series, so concurrency is where this pair is strong rather than weak:
  eight streams total 301 tok/s, 4.12x one stream, for 77% more per-token
  latency. The batch profile now ships eight scheduler slots for this reason.

## What changed

- **Single-stream 262k on KVarN** — `heterogeneous/start_qwen_solo.sh` runs the
  model's full native 262,144-token context in a 296,974-token pool at the
  batch profile's decode rate, by spending all of the KV budget on one
  scheduler slot and 4-bit-key/2-bit-value KV. It is the first profile here to
  run KVarN under MTP and pipeline parallelism at all, and it holds a larger
  pool than `heterogeneous/start_qwen_huge.sh` at the same context.
- **Reversing the pipeline is worth 7.7%** — putting the LM head, sampler and
  drafter on the 5070 Ti instead of the 3060 costs ~3.15 GiB of KV on the
  bigger card, so it only pays where context is cheap:
  `heterogeneous/start_qwen_dflash2_fast.sh` is the profile that ships it.
  `heterogeneous/start_qwen_huge.sh` reaches 262,144 tokens on int4
  per-token-head KV. Both are measured in
  [heterogeneous/README.md](heterogeneous/README.md).
- **DFlash2 on FP8 KV and a reversed pipeline** —
  `heterogeneous/start_qwen_dflash2_fast.sh`. The shipped DFlash2 profile's
  assumption that it needs BF16 KV was untested and wrong: FP8 works, nearly
  doubles the pool, and combined with putting the eager 1.2 GiB drafter on the
  5070 Ti it is quicker as well. Also measured: below ~2 KB/token/layer the
  Mamba state page binds the pool, so int8/int4 per-token-head KV buys no
  context over FP8 while costing Triton-backend prefill speed.
- **GDN metadata hoist** — `patches/vllm-pr52297-gdn-common-metadata.patch`,
  upstream PR #52297 backported. It does not make this pair faster (0.3%), but
  it lowers the profiled activation peak and buys 9,131 tokens of KV pool.
- **Bug B, the uniform-decode prefill dispatch** —
  `patches/zzz-bugb-uniform-prefill.patch`, upstream's `a75ee4b` lifted out of
  `kvarn/kvarn-v2-runner.patch` so it applies without installing KVarN. vLLM's
  `get_uniform_token_count` has no prefill discriminator, so a request whose
  final prefill chunk is exactly `decode_query_len` tokens is dispatched as a
  uniform decode batch and the captured spec-verify CUDA graph is replayed over
  prompt tokens: dflash2 degenerates into repetition, mtp returns an empty
  answer with `finish_reason=stop`. It needs a prefix-cache hit, a FULL capture
  and speculation together — which these profiles have by default and upstream
  single-user does not, since it defaults `PREFIX_CACHE=0` and this fork
  defaults it to 1. Pinning `cudagraph_mode=PIECEWISE` is the upstream
  workaround; here it would undo the reason `bf16` and `int8pth` are the
  preferred KV modes (they are the ones that keep full graphs under
  spec-decode), so the runner is fixed instead. The failure rate on this pair
  is **not** measured — upstream's "one prompt length in every 128" is a
  consequence of `--prefix-match-unit 128`, which these profiles do not set.
- **MTP + pipeline parallelism** — `patches/vllm-pr46994-mtp-pp.patch`, a
  backport of upstream PR #46994 (MTP on the V2 runner under PP) plus the
  hybrid-Mamba int64 `index_fill_` fix PP long prefill needs. It is applied by
  the normal `patches/*.patch` loop from the README setup.
- **DFlash2 + pipeline parallelism** — `patches/zz-dflash2-pipeline-parallel.patch`
  (the `zz-` prefix keeps it last in the patch glob, which matters because it
  touches files other patches also modify). Upstream 0.27.1 refuses DFlash with
  PP outright. The patch relays the target's auxiliary hidden states (layers
  5/19/33 from stage 1, 47/61 from stage 2) across the boundary, validates the
  complete drafter as PP-local on the last rank, shares the target's embedding
  table and quantized LM head instead of allocating multi-GiB BF16 temporaries,
  and adds `VLLM_DFLASH_CUDAGRAPH=0` (the shared Marlin LM head's candidate GEMM
  crashes inside DFlash's private CUDA graph, so the drafter runs eagerly while
  the target keeps compilation and graphs).
- **Launchers** — `heterogeneous/start_qwen.sh` (shared, tunable),
  `start_qwen_batch.sh` (frozen MTP snapshot), `start_qwen_solo.sh` (262k
  single-stream KVarN) and `start_qwen_dflash2.sh` (32k DFlash2). They encode
  the host/toolchain fixes this pair needed: `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `TORCH_CUDA_ARCH_LIST="8.6;12.0"`, GCC 15 for CUDA 13.3's JIT, unsetting
  `VLLM_MARLIN_INPUT_DTYPE` for W4A16, `GPU_UTIL=0.91` (0.93 OOMed compiled
  prefill), process-group cleanup under `setsid`, `NO_API_KEY` mode, and the
  vLLM flags for accurate per-request metrics and Qwen tool calling.
- **llama-swap** — the two model entries in
  `heterogeneous/llama-swap.example.yaml`:
  `vllm-speed/qwen3.8-27b-batch`, `vllm-speed/qwen3.8-27b-solo` (262k) and `vllm-speed/qwen3.8-27b-dflash2`
  (unprefixed IDs kept as aliases).
- **Docker** — a `hetero` compose profile (two-GPU reservation) and a `hetero`
  entrypoint command.

Full details, benchmarks, the DFlash2 memory-sizing trap, and limitations are
in [heterogeneous/README.md](heterogeneous/README.md); the two-GPU work is also
summarised at the end of [docs/optimizations.md](docs/optimizations.md).
Step-by-step instructions for someone starting from scratch on this exact GPU
pair (download, model creation, patches, run, llama-swap) are in
[heterogeneous/SETUP.md](heterogeneous/SETUP.md).

---
# Qwen3.8-27B on one RTX 3090

![Stock vLLM against this repo, same card, same prompts](docs/media/demo.gif)

Serving setup for [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on a
single 24 GB consumer GPU with vLLM. 150k token context, OpenAI-compatible
API with key auth, and two ready-made configs depending on what you're doing:

| | [batch/](batch/) | [single-user/](single-user/) |
|---|---|---|
| for | API backends, pipelines, many concurrent requests | one or a few people chatting |
| aggregate, 64 concurrent (128 in / 512 out) | **~1,035 tok/s** steady-state decode, 948 end-to-end (~1,222 / 1,042 with all layers int8) | n/a (8 slots) |
| single-stream (C1) decode rate, realistic prompts | 46 tok/s | MTP: **121** tok/s at default sampling, **120** greedy (`CTX=fast`, 64k; 96 / 102 with `CTX=long`, 150k). DFlash2 (`SPEC=dflash2`): **127** default, **130** greedy |
| reproducing its own context (quoting a document, applying an edit) | 46 tok/s | **381 tok/s** at 25k context — 15.0 tokens per verify step, drafted straight from the prompt (`SPEC=dflash2` + `DFLASH_TOKENS=15`) |
| trick | 16-bit recurrent state + int8 tensor-core GEMMs | MTP speculation with 4 cheap drafts, a draft vocabulary that covers what the model says, calibrated int4 lm_head/drafter, split-KV verify attention; optionally DFlash2 (7 drafts in one pass, int4-requantized, vLLM PR #52816 backported) with a verify block the context fills |

<sub>Single-stream numbers re-measured 2026-08-22 on current main with
`bash bench/run_benchmarks.sh single` — `vllm bench serve`, the 8 prompts in
`bench/prompts_real.jsonl`, 1024 output tokens, C1, decode rate taken as
`C / mean TPOT`. Quote them against that harness: a client with a different output
length is not measuring the same thing, and mixing the two is how
[#3](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/3) got confusing.</sub>

Both modes share one install — the mode is just which launch script you run.
Speculation wins below ~8 concurrent users, plain batching above. Numbers are
`vllm bench serve` on an RTX 3090 at a 250 W power limit. If the card is yours
alone, the fastest configuration is three environment variables away:
[If you are the only user](#if-you-are-the-only-user-do-this).

Prefill is a separate budget from either: ~1,810 tok/s at 1k inputs in batch
mode (~1,210 single-user), ~1,000 tok/s at 100k, so a 100k prompt costs ~100 s
of TTFT ([full matrix](batch/README.md#prefill)). How each number was won:
[docs/optimizations.md](docs/optimizations.md).

## Quick start

Docker (recommended — image build, model download and requantization, then
the server; the API is OpenAI-compatible on port 18020):

```bash
git clone https://github.com/syv-ai/qwen38-27b-rtx3090 && cd qwen38-27b-rtx3090
docker compose --profile single up -d      # one or a few users; or --profile batch
```

The server listens on `0.0.0.0` and is unauthenticated unless you give it a key.
For anything past your own machine, add one first — everything reads it from
`.env` or `api_key.txt`, and nothing needs it otherwise:

```bash
echo "VLLM_API_KEY=$(openssl rand -hex 24)" > .env
```

Or by hand in a venv (same steps: model download, requantization, vLLM
patches, `verify.sh`) — see [Setup](#setup). Then pick a mode:
[batch/](batch/) for throughput, [single-user/](single-user/) for latency.

### If you are the only user, do this

The command above starts the conservative default — MTP speculation, 8 request
slots, 64k context, 120 tok/s greedy at C1. Three settings are worth more than
every other knob in this repo put together:

```bash
printf 'SPEC=dflash2\nDFLASH_TOKENS=15\nPREFIX_CACHE=1\n' >> .env
docker compose --profile single up -d
```

or, in the venv install:

```bash
venv/bin/python prepare/fetch_dflash2.py   # once, 1.2 GB (Docker's prepare step does it for you)
SPEC=dflash2 DFLASH_TOKENS=15 PREFIX_CACHE=1 bash single-user/start_qwen.sh
```

`SPEC=dflash2` swaps Qwen's MTP head for the DFlash2 block drafter: 7 tokens
proposed in one pass instead of 4 chained ones. `DFLASH_TOKENS=15` then lets the
target verify 16 tokens per step — the drafter still proposes the 7 it was
trained for, and the remaining positions are filled from the request's own
context, which costs nothing to draft and is exactly right whenever the answer
quotes the prompt. `PREFIX_CACHE=1` keeps the document you already sent, both
its attention KV and its recurrent state. One request at a time, greedy, RTX
3090 at 250 W:

| decode | MTP (default) | `SPEC=dflash2` | `+ DFLASH_TOKENS=15` |
|---|---|---|---|
| 8 real chat prompts | 118 tok/s | 132 | **133** |
| reproducing a 25k-token document | n/a* | 260 | **382** |
| request slots / context | 8 / 64k | 8 / 64k | 4 / 56k |

<sub>\* drafting from the context only exists in `SPEC=dflash2`. The two right
columns are one server session, where run-to-run greedy divergence is ±3-5%;
reproduce them with `venv/bin/python bench/labd_bench.py <tag> --ctx 20000`.</sub>

`PREFIX_CACHE=1` is orthogonal to the other two and worth as much again in a
chat client: a second turn against that same 25k-token document takes 0.56 s to
first token instead of 22.4 s, with the answers unchanged token for token.

All of it is lossless: speculative decoding samples the same distribution as no
speculation at all, the prefix cache resumes recurrent state rather than
approximating it, and GSM8K reads 96.0-96.5% across the three columns. What
`DFLASH_TOKENS=15` costs is half the request slots and 8k of context — that is
the whole reason it is opt-in, and why the default stays where it is for anyone
serving more than a few people. Every other knob: [single-user/](single-user/).

### DFlash2 at 240k: `CTX=huge` (KVarN) also combines with `SPEC=dflash2`

```bash
bash kvarn/install.sh                # applies kvarn-v2-runner.patch as its second stage
SPEC=dflash2 CTX=huge PREFIX_CACHE=1 bash single-user/start_qwen.sh
```

Where `CTX=long` doubles the DFlash2 pool with int8 KV (138k), the KVarN cache
takes the same idea further: 268k tokens of pool at 245760 max-model-len, on the
same pinned budget. No kernel work — the KVarN Triton kernels run unmodified on
the V2 runner; the seven fixes in `kvarn/kvarn-v2-runner.patch` are allocator and
geometry logic (the patch header walks through them, including an upstream vLLM
bug in the mamba align resume path, and a NaN path in the DFlash2 candidate
selector that KVarN noise exposes on verbatim-reproduction content). Two
machines, both RTX 3090 at 250 W, `bench/labd_bench.py --ctx 20000` — the
contributor's WSL2 box and this repo's bare-metal one, which do not agree on
decode rate and do agree on everything else:

| `SPEC=dflash2 CTX=huge PREFIX_CACHE=1` | WSL2 | bare metal |
|---|---|---|
| copy (reproduction) | 130 tok/s, 7.8 tok/step | 164 tok/s, 7.83 tok/step |
| code / edit / quote / summary / qa | 89 / 65 / 44 / 38 / 36 | 109 / 83 / 58 / 51 / 43 |
| all six tasks together | 53 tok/s, 3.0 tok/step | 67 tok/s, 3.15 tok/step |
| verbatim reproduction, 25k document | correct | 1,150 / 1,150 chars |
| KV capacity at 245760 max-model-len | 268,169 tokens | 268,169 tokens |
| GSM8K exact-match (thinking off) | 97.0% (n=200) | 95.2% (n=600), 95.0% (n=200) |
| 100k-deep needle, both turns | correct | — |
| turn 2 over a 100k cached prefix | 4.7 s (vs 169 s cold) | — |

<sub>Context for the GSM8K column: every configuration this repo already ships
reads 95.0-96.5% on the same 200-question harness ([docs/quality.md](docs/quality.md)),
and 95.0% is the batch-mode default. 95.2% at n=600 (±0.9 points) therefore sits
inside the band rather than below it — which is the useful comparison, since this
mode inherits KVarN's lossy 4/2-bit cache and should be judged against the other
lossy configurations rather than against bf16. Repeat runs of the reproduction
check on bare metal are bit-identical (same step count, same 1,150 characters),
which is the property that was missing before `PIECEWISE` — see below.</sub>

One caveat to the "all of it is lossless" paragraph above: the speculation here
is still exact, but this mode inherits KVarN's 4/2-bit KV cache, which is lossy —
the same trade `CTX=huge` already makes (deep-needle retrieval passes at 200k).
On WSL2, set `VLLM_WSL2_ENABLE_PIN_MEMORY=1` — the V2 runner needs pinned
memory, and vLLM leaves it off by default there; its UVA buffers work fine on
the paravirt driver.

One knob this mode sets for you: `cudagraph_mode=PIECEWISE`. Prefix caching and
a *captured* (FULL) verify step do not currently mix on this path. On WSL2 that
showed up as acceptance collapsing to about one token per step; on bare metal it
also **corrupted the output** — special-token ids leaking into the stream, a
different failure on every run, 1 of 1,176 characters matching the source
instead of all of them. It is the capture rather than the drafter: eager is
clean, `LOOKUP=0` is not, forcing a fixed verify-block length is not, and
PIECEWISE — which keeps the compiled graphs and leaves only the multi-query
verify uncaptured — restores both the speed and the correctness on both
machines. So `CTX=huge` runs PIECEWISE and keeps prefix caching, which is worth
having: turn 2 over a cached 100k document costs 4.7 s against 169 s cold.

`CUDAGRAPH_MODE=FULL_AND_PIECEWISE` switches the capture back for anyone hunting
the root cause. Treat that as unsafe rather than merely slower — the corruption
above is what it does on bare metal.

Two limits worth knowing before you point this at anything.
**It is a single-user mode by configuration, and the knob is `MAX_SEQS`.** Fire 8
concurrent requests at `CTX=huge` and the server runs **two**, with the other six
queued — because this mode sets `MAX_SEQS=2`. That is a deliberate default for
long-document sessions, not an engine limit and not a property of the block
verify, so `MAX_SEQS=8` lifts it: peak 5 concurrent on the same 8-stream test,
with the KV pool **unchanged at 268,169 tokens** (a recurrent-state slot costs
~8 MiB, so the slots are close to free; what scales with the pool is the verify
block length, not the slot count). Raising it grows the captured decode graphs,
which is why the default stays low rather than because it would cost you context.
And `DFLASH_TOKENS=15` does not boot at 240k on 24 GB — the pinned-buffer
arithmetic in [docs/long-context.md](docs/long-context.md) says why — so
reproduction mode and 240k are mutually exclusive; the default 7 is what runs.

It is a trade rather than a free win: dropping the full decode graphs costs
short-prompt throughput. Same box, same script, only the capture toggled,
three runs each:

| | `copy` @25k | de | en | code |
|---|---|---|---|---|
| `FULL_AND_PIECEWISE` | 38 tok/s (1.97/step) | 78 | 125 | 202 |
| PIECEWISE (default here) | **132 tok/s (7.83/step)** | 74 | 102 | 176 |

3.5x on the long shared prefix this mode exists for, 13-18% off short-prompt
decode. **That 13-18% is short prompts only, and it does not generalise** — past
8k the two capture modes are within noise of each other on bare metal (111.8 vs
109.3 tok/s at 8k, 78.2 vs 86.1 at 16k, 68.9 vs 73.3 at 32k, 58.4 vs 56.0 at 50k,
unique prompts, one server per mode). Under GPU passthrough on a VM the same
comparison costs 2-3x, reported in [#13](https://github.com/syv-ai/qwen38-27b-rtx3090/pull/13)
and consistent with the uncaptured verify being launch-bound: launches that are
nearly free here are not free there.

The capture mode is fixed at boot, so `CTX=huge` takes the trade that matches what
it is for — **for every speculator, `mtp` included**. `CUDAGRAPH_MODE=FULL_AND_PIECEWISE`
switches back; treat it as unsafe. What it does is corrupt one prompt length in
every 128, and only for a request that hits the prefix cache: `dflash2` collapses
to degenerate repetition, `mtp` stops dead and returns an empty answer. The broken residue is
a function of the draft count (`R = 117 + k`, the same for both speculators),
which is why scoping this workaround to `dflash2` was wrong — `SPEC=mtp CTX=huge`
shipped with the same bug, and
piecewise capture costs it nothing measurable (87.8/86.1/70.4/63.5 tok/s captured
against 93.5/83.8/70.3/59.6 piecewise over 8k-50k). Gotcha 37 in
[docs/gotchas.md](docs/gotchas.md) has the residue table and `bench/bugb_sweep.py`
reproduces it; the hunt for the boundary case is in the PR thread.

### More than one GPU

Everything here is written for one 24 GB card, and that is the only configuration
measured in this README. It is not the only one that works: `--tensor-parallel-size`
goes through untouched, via `EXTRA_ARGS`.

```bash
EXTRA_ARGS="--tensor-parallel-size 2" bash single-user/start_qwen.sh
```

Reported working on **2x RTX 5060 Ti 16 GB** by
[@antonybudianto](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/22) — two cards
that could not hold this model individually. I have one 3090, so every multi-GPU
number in the issues is a user report rather than something I have reproduced, and
the tuning here (the pinned `KV_MEM`, the graph budget, `MAX_SEQS`) is sized for a
single card and is probably not right for yours. If you run it on two, numbers in
[#7](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/7) are welcome.

## Benchmarks

Full tables per mode in [batch/README.md](batch/README.md) and
[single-user/README.md](single-user/README.md); quality in
[docs/quality.md](docs/quality.md). Reproduce any of it with
`bash bench/run_benchmarks.sh batch|single` against your own server.

### vs. ninfer-3090

[ninfer-3090](https://github.com/Don-Chad/ninfer-3090) is a standalone C++/CUDA engine
that publishes cohort benchmarks for this model on this card. Theirs are 1,024-token
answers from 29-34-token prompts, greedy, MTP3, int8 KV, prefix reuse off, an
8,192-token context window, and **thinking on** at `reasoning_effort=medium`, so their
1,024 tokens include reasoning. Ours are 8 realistic chat prompts (English, Danish,
code), 1,024-token answers, model-default sampling, thinking off:

| Cohort | ninfer-3090 (MTP3) | this repo, batch | single-user, MTP | single-user, DFlash2 |
|---|---|---|---|---|
| C1 | 71.00 tok/s | 45.5 | 111.1 | **121.8** |
| C2 | 90.66 tok/s | 86.3 | 191.8 | **195.5** |
| C4 | 100.28 tok/s | 168.3 | 268.5 | **278.9** |
| C8 | 165.33 tok/s | 324.9 | **407.3** | 389.9 |
| C64 (128 in / 512 out) | not supported | **~1,035** | — | — |

Decode rate, C × 1000 / mean TPOT. All four of our columns were re-measured together
on the current stack with `bench/run_benchmarks.sh`, keeping the second run after each
restart as the script advises; greedy instead of default sampling reads
131.2 / 214.6 / 285.7 / 405.5 for DFlash2. Run-to-run spread on the same server is
5-8%, so treat one-decimal differences between the three right-hand columns as noise —
C1 and C8 are where the modes genuinely separate.

Theirs is the **decode** column of their table; their end-to-end column reads
70.19 / 89.43 / 97.89 / 161.28, and an earlier version of this table quoted *those*
against our decode rate, which was not like-for-like. What still is not like-for-like,
in their favour and ours: their C1 is a single prompt in a single run with no error
bars, thinking is on for them and off for us, and they publish no power limit or driver
version — ours is an RTX 3090 pinned at 250 W. Peak VRAM is comparable (23.0 vs
22.1 GiB at C8). The gap is mostly vLLM's continuous batching plus the memory this
repo's requantization frees up.

### Quality

The whole stack is quantized, so the honest question is what it costs. Short
version: **IFBench 78.3** prompt-level strict vs 79.5 for the unquantized model
(one point), **perplexity 8.09** on 33k held-out tokens, **GSM8K 96.5%** (200
questions, greedy). Speculative decoding — MTP, DFlash2 and the lookup drafter —
is exact by construction and changes none of it; the int8-activation steps in
batch mode are the only knobs that trade accuracy for speed, and they cost
0.9-3.7% perplexity depending on how far you push them. Per-configuration
tables: [docs/quality.md](docs/quality.md).

### Why this isn't just `vllm serve`

Nine things, from requantizing both embedding matrices to drafting straight out
of the prompt — one line each, then the reasoning and measurements, in
[docs/optimizations.md](docs/optimizations.md).

### What each step buys

Measured cumulatively on the 3090, 64 concurrent, 128 in / 512 out, `vllm bench
serve` random dataset:

| step | what it does | e2e output tok/s | steady-state decode |
|---|---|---|---|
| W4A16 AutoRound body (as published) + fp8 KV | int4 Marlin kernels, 66.7k-token pool | 370 (48 conc, 256/256) | — |
| + lm_head / embed_tokens int8 | 2.6 GB of cache pages back | 516 | ~585 (37 requests resident) |
| + fp16 recurrent state | 64 requests resident, half the state traffic | 707 | ~830 |
| + int8 activations, MLP (default) | int8 tensor cores on 74% of the FLOPs | 942 | ~1,094 |
| + int8 activations, everything (`INT8_LAYERS=.`, needs `GPU_UTIL=0.95`) | | 1,042 | ~1,222 |

And single-stream on realistic prompts (single-user mode, T = model default /
greedy):

| step | tok/s | tokens per step | draft acceptance, position 0 |
|---|---|---|---|
| no speculation | 46 / 46 | 1.0 | — |
| MTP-2 as shipped (bf16 drafter, full head, fp32 state) | 66 / 79 | 2.1 / 2.4 | 65% / 80% |
| MTP-4, int8 drafter, 40k draft head, fp16 state | 78 / 99 | 2.2 / 2.7 | 58% / 70% |
| + probabilistic draft sampling (`CTX=fast`, k=4) | 90 / 98 | 2.6 / 2.7 | 69% / 70% |
| same with 3 drafts on FlashInfer/fp8 KV (`CTX=long`, 150k) | 84 / 89 | 2.5 / 2.4 | 69% / 71% |
| + sampler patch, split-KV verify attention | 93 / 99 | 2.6 / 2.6 | 69% / 70% |
| + draft vocab counted over the model's own outputs | 107 / 109 | 2.9 / 2.9 | 74% / 74% |
| + GPTQ-int4 lm_head (calibrated) | 109 / 112 | 2.8 / 2.8 | 73% / 73% |
| + GPTQ-int4 MTP module (**fast variant, shipped**) | **~114 / 118-124** | 2.8 / 2.9-3.0 | 74% / 77% |
| DFlash2 block drafter instead of MTP (`SPEC=dflash2`, int4-requantized) | **118 / 126** | 3.14 / 3.34 | ~75% / ~78% |
| + drafting from the context (`LOOKUP=1`, on by default) | **130** at C1, up to **259** where the model reproduces its context | 3.3-7.8 | |
| + a 16-token verify block the context fills (`DFLASH_TOKENS=15`) | **133** at C1, up to **381** reproducing context | 3.4-15.0 | |

(Steps 4-6 are the same 8-prompt protocol; greedy is deterministic for a
given server and request order but differs between configs and even with
prefix-cache hits, so single runs carry ±3-5% on tokens/step —
`bench/run_benchmarks.sh single` reproduces 111.1 / 120.0 tok/s decode at C1,
the best repeats read 119 / 124.)
Going deeper (k=5) loses again: 106 / 105. k=4 is the knee, but on vLLM
0.27.1's FlashInfer backend (needed for fp8 KV, i.e. for 150k context) four
drafts crash the engine with an illegal memory access as soon as one request
finishes while another is mid-generation — club-3090 reports the same "n=4
eventually dies, n=3 stable" pattern — so `CTX=long` drafts 3 and gives up
~7%; `CTX=fast` (FlashAttention, bf16 KV, ~64k context, the default) keeps k=4
and is also the only backend the split-KV attention patch applies to.

Two things that did *not* help, measured rather than assumed: fine-tuning the
MTP head on the model's own outputs (KL halves, greedy top-1 on response
tokens unchanged; `drafter/README.md`), and retuning Marlin's tile
configuration for M ≤ 16 on sm86 (3-7% per GEMM in isolation,
nothing measurable end to end — the remaining gap to peak bandwidth is the
memory system's ramp on 16-92 MB reads, not the kernel).

## Setup

You need: a 24 GB Ampere or newer NVIDIA card, a recent driver, Python 3.12,
~40 GB disk. Everything below is CPU-safe to run while the GPU does other
things. (Or skip the venv and use the container: [docs/docker.md](docs/docker.md).)

```bash
git clone https://github.com/syv-ai/qwen38-27b-rtx3090 ~/qwen-serving
cd ~/qwen-serving

python3 -m venv venv
venv/bin/pip install vllm huggingface_hub hf_transfer ninja

# model, ~19.5 GB
HF_HUB_ENABLE_HF_TRANSFER=1 venv/bin/hf download \
  dbirks/Qwen3.8-27B-W4A16-AutoRound \
  --local-dir models/Qwen3.8-27B-W4A16-AutoRound

# requantize lm_head + embeddings + the MTP draft module (CPU only, a few minutes)
venv/bin/python prepare/quant_lm_head.py models/Qwen3.8-27B-W4A16-AutoRound
venv/bin/python prepare/quant_embed.py   models/Qwen3.8-27B-W4A16-AutoRound
venv/bin/python prepare/quant_mtp.py     models/Qwen3.8-27B-W4A16-AutoRound
# 40k-token draft head for single-user mode (uses the shipped id list)
venv/bin/python prepare/build_draft_vocab.py models/Qwen3.8-27B-W4A16-AutoRound \
  --ids prepare/draft_vocab_ids.json
# single-user "fast" variant (~1 GB from the Hub, hardlinks the rest): int4-GPTQ
# lm_head + drafter; single-user/start_qwen.sh picks it up automatically
venv/bin/python prepare/fetch_fast_variant.py
# optional: the W4A16 DFlash2 block drafter (1.2 GB) for SPEC=dflash2 single-user mode
venv/bin/python prepare/fetch_dflash2.py

# patch vllm (all written against 0.27.1; reapply after upgrades)
for p in patches/*.patch; do
  patch -p1 -d venv/lib/python3.12/site-packages/vllm < $p
done
# optional: the KVarN 4/2-bit KV cache for 262k context (docs/long-context.md)
bash kvarn/install.sh

# api key — optional, but the server binds 0.0.0.0 and is open without one
openssl rand -hex 24 > api_key.txt
```

Then `bash verify.sh --no-server` — it checks the venv and vLLM version, that
every patch in `patches/` is actually applied, and that the model has been
requantized (lm_head, embeddings, MTP module, draft head). Then pick a mode
and follow its README:

- **[batch/](batch/)** — throughput. `bash batch/start_qwen.sh`
- **[single-user/](single-user/)** — latency. `bash single-user/start_qwen.sh`

First start takes a few minutes (torch.compile, CUDA graph capture, flashinfer
JIT). Test it:

```bash
curl http://localhost:18020/v1/chat/completions \
  -H "Authorization: Bearer $(cat api_key.txt 2>/dev/null)" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b",
       "messages": [{"role": "user", "content": "hej"}],
       "chat_template_kwargs": {"enable_thinking": false}}'
```

Qwen recommends temperature 0.7 / top_p 0.8 for instruct mode, and 1.0 / 0.95
with thinking enabled (the default).

Tool calling works over the same endpoint — send `tools` with `tool_choice:
"auto"` and the reply carries `tool_calls`. Both launchers set
`--enable-auto-tool-choice --tool-call-parser qwen3_coder`; the parser has to
read Qwen's XML call format, which is what this model's chat template emits —
not the JSON that `hermes` reads. `TOOLS=0` turns it off.

To check the numbers on your own card: `bash verify.sh` (also probes the live
server and prints which attention backend and KV pool it came up with), then
`bash bench/run_benchmarks.sh batch` or `... single` reproduces the tables
above against the running server (`--prefill` and `--long` add the prefill
matrix and the long-context rows), `bash bench/real_rep.sh <tag> 3 0` repeats
the single-stream row, and `python bench/quality_battery.py <tag>` the
perplexity / GSM8K rows.

## The rest

| | |
|---|---|
| [docs/optimizations.md](docs/optimizations.md) | Every optimization in full: why it was needed, what it measured, which patch implements it. Includes the two speculative-decoding modes (MTP and DFlash2) and the lookup drafter. |
| [docs/gotchas.md](docs/gotchas.md) | 18 things that each cost us hours — read before debugging something that looks like a vLLM bug. |
| [docs/quality.md](docs/quality.md) | IFBench, perplexity and GSM8K per configuration. |
| [docs/docker.md](docs/docker.md) | The container image, and an independent WSL2 reproduction. |
| [docs/long-context.md](docs/long-context.md) | 262k context with the KVarN 4/2-bit KV cache, what vLLM's own per-token-head KV modes are worth here, and how to run the DFlash2 drafter past 64k (`CTX=long`, 114-139k — worth it only for context reproduction). |
| [batch/](batch/) · [single-user/](single-user/) | The two serving modes: full benchmark tables, every env knob, systemd units. |
| [prepare/](prepare/) | The one-time model-preparation scripts run by [Setup](#setup) (and by `docker compose run --rm prepare`). |
| [drafter/](drafter/) | How the draft vocabulary, the int4 drafters and the DFlash2 requantization were built — including what did not work. |
| [kvarn/](kvarn/) | The KVarN 4/2-bit KV cache port. |

## License

Apache-2.0, same as the model.
