# Qwen3.8-27B on one RTX 3090

![Stock vLLM against this repo, same card, same prompts](docs/media/demo.gif)

Serving setup for [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on a
single 24 GB consumer GPU with vLLM. 150k token context, OpenAI-compatible
API with key auth, and two ready-made configs depending on what you're doing:

| | [batch/](batch/) | [single-user/](single-user/) |
|---|---|---|
| for | API backends, pipelines, many concurrent requests | one or a few people chatting |
| aggregate, 64 concurrent (128 in / 512 out) | **~1,094 tok/s** steady-state decode, 942 end-to-end (~1,222 / 1,042 with all layers int8) | n/a (8 slots) |
| single-stream (C1) decode rate, realistic prompts | 46 tok/s | MTP: **114** tok/s at default sampling, **118** greedy (`CTX=fast`, 64k; 85 / 89 with `CTX=long`, 150k). DFlash2 (`SPEC=dflash2`): **122** default, **132** greedy |
| reproducing its own context (quoting a document, applying an edit) | 46 tok/s | **381 tok/s** at 25k context — 15.0 tokens per verify step, drafted straight from the prompt (`SPEC=dflash2` + `DFLASH_TOKENS=15`) |
| trick | 16-bit recurrent state + int8 tensor-core GEMMs | MTP speculation with 4 cheap drafts, a draft vocabulary that covers what the model says, calibrated int4 lm_head/drafter, split-KV verify attention; optionally DFlash2 (7 drafts in one pass, int4-requantized, vLLM PR #52816 backported) with a verify block the context fills |

Both modes share one install — the mode is just which launch script you run.
An experimental heterogeneous two-GPU path for an RTX 5070 Ti + RTX 3060 is
in [heterogeneous/](heterogeneous/). Its uneven pipeline-parallel MTP mode is
runtime-tested at about 84 tok/s single-stream. Its frozen stable profile uses
a 140k context limit with measured 146,847-token KV capacity (up from 32.6
tok/s on its conservative non-speculative baseline); separate 32k and 65k
profiles are available for experiments. An experimental DFlash2+PP 32k profile
reached **91.7 tok/s** on eight realistic prompts and passed the 12/12 API smoke
suite. The general numbers below are
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
echo "VLLM_API_KEY=$(openssl rand -hex 24)" > .env
docker compose --profile single up -d      # one or a few users; or --profile batch
```

Or by hand in a venv (same steps: model download, requantization, vLLM
patches, `verify.sh`) — see [Setup](#setup). Then pick a mode:
[batch/](batch/) for throughput, [single-user/](single-user/) for latency.

### If you are the only user, do this

The command above starts the conservative default — MTP speculation, 8 request
slots, 64k context, 118 tok/s greedy at C1. Three settings are worth more than
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

## Benchmarks

Full tables per mode in [batch/README.md](batch/README.md) and
[single-user/README.md](single-user/README.md); quality in
[docs/quality.md](docs/quality.md). Reproduce any of it with
`bash bench/run_benchmarks.sh batch|single` against your own server.

### vs. ninfer-3090

[ninfer-3090](https://github.com/Don-Chad/ninfer-3090) publishes cohort benchmarks
for this model on this card, using C concurrent requests of random tokens. Random
tokens are a bad yardstick for speculative decoding — acceptance swings between 80%
and near zero with the sample — so ours are 8 realistic chat prompts (English,
Danish, code), 1,024-token answers, model-default sampling:

| Cohort | ninfer-3090 (MTP3, random tokens) | this repo, batch | single-user, MTP | single-user, DFlash2 |
|---|---|---|---|---|
| C1 | 70.19 tok/s | 45.4 | 113.6 | **122.1** |
| C2 | 89.43 tok/s | 82.6 | **194.0** | 191.4 |
| C4 | 97.89 tok/s | 165.6 | 258.6 (289.2 with `CTX=long`) | **286.7** |
| C8 | 161.28 tok/s | 298.5 | **379.9** (409.0 with `CTX=long`) | 373.7 |
| C64 (128 in / 512 out) | not supported | **~1,094** | — | — |

Decode rate (C × 1000 / mean TPOT), best of the runs we have, from
`bench/run_benchmarks.sh`; greedy instead of default sampling reads
131.9 / 209.6 / 309.6 / 390.6 for DFlash2. The C64 and DFlash2 columns are from the
current stack, the other two from earlier runs; ninfer's published figures are their
own protocol. Peak VRAM is comparable to theirs (23.0 vs 22.1 GiB at C8) — the gap is
mostly vLLM's continuous batching plus the memory this repo's requantization frees up.

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
`bench/run_benchmarks.sh single` reproduces 113.6 / 118.3 tok/s decode at C1,
the best repeats read 115 / 124.)
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

# api key
openssl rand -hex 24 > api_key.txt
```

Then `bash verify.sh --no-server` — it checks the venv and vLLM version, that
every patch in `patches/` is actually applied, and that the model has been
requantized (lm_head, embeddings, MTP module, draft head). Then pick a mode
and follow its README:

- **[batch/](batch/)** — throughput. `bash batch/start_qwen.sh`
- **[single-user/](single-user/)** — latency. `bash single-user/start_qwen.sh`
- **[heterogeneous/](heterogeneous/)** — experimental RTX 5070 Ti + RTX 3060 pipeline-parallel serving. `bash heterogeneous/start_qwen.sh`

First start takes a few minutes (torch.compile, CUDA graph capture, flashinfer
JIT). Test it:

```bash
curl http://localhost:18020/v1/chat/completions \
  -H "Authorization: Bearer $(cat api_key.txt)" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b",
       "messages": [{"role": "user", "content": "hej"}],
       "chat_template_kwargs": {"enable_thinking": false}}'
```

Qwen recommends temperature 0.7 / top_p 0.8 for instruct mode, and 1.0 / 0.95
with thinking enabled (the default).

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
| [batch/](batch/) · [single-user/](single-user/) | The two single-GPU serving modes: full benchmark tables, every env knob, systemd units. |
| [heterogeneous/](heterogeneous/) | Experimental uneven pipeline-parallel mode for an RTX 5070 Ti + RTX 3060, with compatibility research and limitations. |
| [prepare/](prepare/) | The one-time model-preparation scripts run by [Setup](#setup) (and by `docker compose run --rm prepare`). |
| [drafter/](drafter/) | How the draft vocabulary, the int4 drafters and the DFlash2 requantization were built — including what did not work. |
| [kvarn/](kvarn/) | The KVarN 4/2-bit KV cache port. |

## License

Apache-2.0, same as the model.
