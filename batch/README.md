# Batch mode

For serving many concurrent requests: API backends, data processing pipelines,
eval runs. Tuned for aggregate tokens per second, not per-request latency.

## Benchmarks

`vllm bench serve`, random dataset, 256 requests, RTX 3090 at 250 W, default
config (fp16 recurrent state, int8 activations on the MLP GEMMs):

| workload | steady-state decode | output tok/s (e2e) | median TPOT | median TTFT |
|---|---|---|---|---|
| 256/256, 1 concurrent | 46 | 46 | 21.7 ms | 230 ms |
| 128/512, 64 concurrent | **~1,094** | 942 | 62.2 ms | 3.0 s* |
| 256/256, 64 concurrent | ~1,094 | 673 | 81.4 ms | 3.4 s* |

(Re-measured on the current stack; two passes each, within 0.4% of one another. The 128/512
row read 876 when this repo was first published — the difference is everything that landed
since. `INT8_LAYERS=.` — int8 activations on every linear, not just the MLP — reaches
**1,042 tok/s** e2e and ~1,222 steady-state decode, but needs `GPU_UTIL=0.95`: it OOMs inside
the GDN chunk kernel at the 0.972 default. Quality cost of that row: +3.7% perplexity,
[docs/quality.md](../docs/quality.md).)

*TTFT at saturation is queue time — the bench fires all requests at once.

"Steady-state decode" is what the engine logs as generation throughput once
all 64 requests are resident and no prefill is interleaved; the e2e number
includes everybody's prefill and the ramp. Older configs, same protocol:

| config | e2e 128/512 | e2e 256/256 | steady decode |
|---|---|---|---|
| W4A16, fp32 state (only 37 of 64 requests could run) | 516 | 393 | ~585 |
| W4A16, fp16 state | 707 | 491 | ~830 |
| int8 activations, gate/up only (`INT8_LAYERS=gate_up`) | 787 | 572 | ~930 |
| int8 activations, MLP (default) | 876 | 642 | ~1,050 |
| int8 activations, all linears (`INT8_LAYERS=.`) | 1,025 | 752 | ~1,150 |

Quality of each row is in the [quality tables](../docs/quality.md).

Cohort protocol on realistic prompts (C concurrent chat requests, 1,024-token
answers, model-default sampling; comparable to the
[single-user tables](../single-user/README.md)):

| Cohort | e2e throughput | decode throughput | mean TTFT |
|---|---|---|---|
| C1 | 45.4 tok/s | 45.6 tok/s | 110 ms |
| C2 | 81.8 tok/s | 83.7 tok/s | 197 ms |
| C4 | 153.8 tok/s | 162.7 tok/s | 343 ms |
| C8 | 298.4 tok/s | 321.0 tok/s | 638 ms |

Below ~C8, single-user mode's speculative decoding is faster (2.5× at C1 with
the fast variant); batch mode pulls ahead from C8 and keeps scaling to C64.
Re-measured after the single-user sampler / attention patches went in
(2026-08-18): 882 / 635 tok/s e2e at 64 concurrent, cohorts 44.7 / 82.6 /
165.6 / 298.5 — unchanged within noise; the patches only touch small-batch
paths.

### Prefill

Measured with 1-token outputs so it's pure prompt processing. Batch mode
default (int8 tensor cores on the MLP GEMMs):

| input length | conc 1 | conc 4 | conc 8-16 | single-request TTFT |
|---|---|---|---|---|
| 1k | 1,812 tok/s | 1,820 | 1,806 | 0.56 s |
| 4k | 1,758 | 1,757 | 1,747 | 2.3 s |
| 16k | 1,595 | 1,601 | 1,599 | 10.3 s |
| 64k | 1,182 | 1,184 | — | 55 s |
| 100k | 997 | — | — | 103 s |

W4A16 kernels (single-user mode, or batch mode with `INT8_ACT=` unset) for
comparison — the int8 path is +50% at 1k, tapering to +25% at 100k as the 16
attention layers take a bigger share:

| input length | conc 1 | conc 4 | conc 8-16 | single-request TTFT |
|---|---|---|---|---|
| 1k | 1,210 tok/s | 1,215 | 1,213 | 0.85 s |
| 4k | 1,185 | 1,185 | 1,183 | 3.5 s |
| 16k | 1,112 | 1,117 | 1,116 | 14.7 s |
| 64k | 906 | 908 | — | 72 s |
| 100k | 795 | — | — | 129 s |

Two things to plan capacity around. First, concurrency does nothing for
prefill: chunked prefill feeds everything through the same 2,048-token
per-step budget, so prompt processing is a fixed resource the whole server
shares, and queueing is linear (four 16k prompts at once means the last one
waits ~40 s). Second, the falloff with length is mild — ~45% from 1k to 100k
on the int8 path, ~34% on W4A16 — because just 16 of 64 layers pay quadratic
attention; this is one of the places the hybrid architecture genuinely helps.

## Shared prompts: `PREFIX_CACHE=1`

If every request carries the same system prompt, few-shot block or document — the normal
shape of an API backend — turn on prefix caching (`--enable-prefix-caching
--mamba-cache-mode align`, opt-in upstream for hybrid models). The attention KV of the shared
prefix is reused and the recurrent (GDN) state resumes from the last cached block boundary,
so the prefix is prefilled once instead of once per request:

| 64 requests sharing a 5,820-token system prompt, concurrency 32 | wall | output | median latency | p90 |
|---|---|---|---|---|
| default | 222.2 s | 10.6 tok/s | 94.9 s | 155.3 s |
| `PREFIX_CACHE=1` | **16.9 s** | **133.9 tok/s** | **8.0 s** | **15.2 s** |

It costs ~14% of the KV pool (223,821 → 193,298 tokens, 1.29x concurrency at 150k instead of
1.49x) — one extra recurrent-state page per request — and nothing on workloads with no shared
prefix (870 tok/s on the 128 in / 512 out row, i.e. unchanged). Answers are identical; the
state resume is exact.

## Setup

Do the [common setup](../README.md#setup) first (venv, model download,
requantization, vLLM patches). Then:

```bash
bash batch/start_qwen.sh
```

Or in Docker (image build, model prep and the same knobs via `.env` — see the
[docs/docker.md](../docs/docker.md)):

```bash
docker compose --profile batch up -d
```

Or as a service:

```bash
mkdir -p ~/.config/systemd/user
cp batch/qwen-serving.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now qwen-serving
loginctl enable-linger $USER
```

First start takes a few minutes (torch.compile, CUDA graph capture, flashinfer
JIT). Watch `qwen.log` in the repo root; it's up when you see
"HTTP server started".

## Knobs

All overridable as env vars, defaults in the script:

| var | default | notes |
|---|---|---|
| `PREFIX_CACHE` | 0 | 1 = reuse a shared prompt prefix across requests (see above): 13x on shared-prompt workloads, ~14% smaller KV pool |
| `KV` | `fp8` | `kvarn` switches to the KVarN 4/2-bit KV cache (run `bash kvarn/install.sh` once): 262k context, ~2× the token pool, slower decode — see the main README's "262k context" section |
| `INT8_ACT` | `int8` | int8 activations on the Marlin GEMMs (int8 tensor cores, weights stay int4). Empty string = plain W4A16 |
| `INT8_LAYERS` | `mlp` | regex on the layer name that gets int8 activations. `gate_up` for the gentle variant, `.` for everything, or a hand-picked list from `bench/act_calib.py` |
| `MAX_SEQS` | 64 | scheduler slots; with fp16 state ~70 short requests fit the page pool |
| `MAX_LEN` | 150000 | max context. Raising it much past this fails startup, the pool can't hold a longer request |
| `TOOLS` | 1 | tool/function calling (`--enable-auto-tool-choice --tool-call-parser`). `TOOL_PARSER` (`qwen3_coder`) must match the XML call format this model's chat template emits — `hermes` parses the JSON a Qwen model does *not* produce here, and fails silently. 0 = off, and `tool_choice: "auto"` then 400s |
| `PORT` | 18020 | |
| `GPU_UTIL` | 0.972 | do not raise, see gotchas in the main README. Use 0.93 when you want `prompt_logprobs` (quality checks) |

## Verify you're getting the numbers

```bash
bash verify.sh                       # install + patches + model + live server sanity
bash bench/run_benchmarks.sh batch   # the tables above (--prefill / --long for the rest)
```

or by hand:

```bash
OPENAI_API_KEY=$(cat api_key.txt) venv/bin/vllm bench serve \
  --host 127.0.0.1 --port 18020 \
  --model models/Qwen3.8-27B-W4A16-AutoRound \
  --served-model-name qwen3.8-27b \
  --dataset-name random --random-input-len 128 --random-output-len 512 \
  --num-prompts 256 --max-concurrency 64
```

Run it twice, keep the second number. The first run after a restart includes
JIT warmup and reads 30-50% low. Then run `bench/quality_battery.py` — a
throughput number from a server that emits garbage is worth nothing, and the
int8 path taught us that the hard way.
