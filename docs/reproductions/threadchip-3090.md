# Reproduction: RTX 3090, native Python 3.14, Ubuntu 26.04

*August 2026. Native venv, no container. `verify.sh --no-server` clean, all fifteen patches
applied. Every run below discards a warmup pass, per the README's own instruction.*

[← back to Python 3.14](../python-314.md) · [← main README](../../README.md)

## Hardware and stack

| | |
|---|---|
| GPU | RTX 3090, 24 GiB, driver 610.43.02, 250 W |
| OS / Python | Ubuntu 26.04, Python **3.14.4** (system), `python3.14-dev` installed |
| vLLM | 0.27.1 + all fifteen `patches/`, torch 2.13.0+cu130, Triton 3.7.1 |
| model | `dbirks/Qwen3.8-27B-W4A16-AutoRound` + repo requantization + fast variant + DFlash2 drafter |
| KVarN | installed (`kvarn/install.sh`) |

## `bench/run_benchmarks.sh single`, greedy — vs the README

| cohort | ours e2e | ours decode | README |
|---|---:|---:|---:|
| C1 | **122.46** | 125.9 | 121.8 |
| C2 | 203.13 | 231.7 | 195.5 |
| C4 | 270.91 | 320.3 | 278.9 |
| C8 | 372.62 | 466.5 | 389.9 |

Within a few percent throughout. **The README's table reproduces on Python 3.14.**

## Context profiles as measured

| profile | KV dtype | max_model_len | pool | perplexity¹ |
|---|---|---:|---:|---:|
| `CTX=fast` | bf16 | 65,536 | 68,605–72,992² | 3.1174 |
| `CTX=long` | int8 per-token-head | 131,072 | 136,429 | 3.1100 |
| `CTX=huge` | KVarN 4/2-bit | 245,760 | 268,169 | 3.1107 |

¹ Teacher-forced, identical 12,000-character passage, echo+logprobs. Only the KV dtype
varies. **All three within 0.24%.**
² Varies with `SPEC`/`DFLASH_TOKENS`; both values observed.

## Aggregate tok/s, context × concurrency

`SPEC=dflash2 DFLASH_TOKENS=7 PREFIX_CACHE=1`, 256-token answers, prefix warmed before each
cohort so no row is measuring cache warmth instead of concurrency.

| profile | depth | C1 | C2 | C4 | C8 |
|---|---|---:|---:|---:|---:|
| fast | shallow | 137.3 | 239.5 | **321.3** | 300.6 |
| fast | deep (44k) | 66.8 | 99.7 | 19.2 | 33.4 |
| long | shallow | 121.1 | 230.0 | 304.8 | 311.0 |
| long | deep (44k) | 47.2 | 63.7 | 69.5 | 70.8 |
| huge | shallow | 134.1 | 220.9 | 157.6 | 232.2 |
| huge | deep (44k) | 43.6 | 50.1 | 53.3 | 52.8 |

`CTX=huge` costs roughly 10% shallow and 8% deep against `long`, for 1.9× the window. At
depth the larger pools hold up where `fast` does not.

## Prefix cache

Same 96,370-token document, three consecutive turns, `PREFIX_CACHE=1`:

```
turn 1:  232.9 s   ← pays the prefill
turn 2:    5.9 s
turn 3:    6.0 s
```

The README predicts *"5.9 s afterwards"* for a 112k document. Reproduced exactly.

## KV quality at 4/2 bits

Perplexity is a weak instrument here — it scores the model's fit to text it is *looking at*,
over a short window, which is not where a quantised cache fails. Needle-in-a-haystack on
`CTX=huge` (KVarN 4/2-bit), one distinctive fact planted at varying depth in a long
document, exact-match recall:

| prompt tokens | needle at | recall |
|---:|---:|---|
| 29,653 | 25% | HIT |
| 96,368 | 50% | HIT |
| 168,542 | 75% | HIT |
| **218,085** | 50% | **HIT** |

**Caveat, stated because the table would otherwise imply more than it shows:** the bf16
profile cannot exceed 65,536 tokens, so only the first rung has a like-for-like control.
This demonstrates that KVarN recalls correctly at 218k — not that it equals bf16 there.

## Not reproduced

A community report of **~208 accepted tok/s** on a 3090 at `CTX=long` with a basic prompt
did not reproduce here: **107.2 tok/s median**, same posted configuration
(`SPEC=dflash2 DFLASH_TOKENS=7 PREFIX_CACHE=1 CTX=long`), four consecutive runs within 0.2%
of each other. Reproduction-shaped prompts do climb, consistent with `LOOKUP` drafting:

| task | tok/s |
|---|---:|
| free generation | 102.7 |
| quote a passage back verbatim | 125.5 |
| repeat a passage with a substitution | 156.6 |

The original report says "peaked at around 208", and a peak is not a median; the prompt was
not published. Recorded as unreproduced rather than disputed.
