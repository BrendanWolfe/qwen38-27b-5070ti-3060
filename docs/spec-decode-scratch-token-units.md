# The scratch-sizing fix for #46's int4 speculative attention: measurements and verification

Companion to `patches/spec-decode-scratch-token-units.patch`. The PR description says what
changed and why; this file carries the conditions behind every number and the detail of how
it was checked.

## Conditions

Qwen3.8-27B-W4A16-AutoRound with its DFlash2 drafter (`Qwen3.8-27B-DFlash2-W4A16`), int4
per-token-head KV cache (`single-user/alternative.sh`), draft depth 7, `max_num_seqs=1`, CUDA
graphs on, `--prefix-match-unit 848`. RTX 3090, native Ubuntu 26.04, Python 3.14, vLLM
0.27.1, torch 2.13.0+cu130, CUDA 13.0, driver 610.43.02. The independent numerical check ran
on an RTX 4090 under WSL2.

## The defect, in numbers

`seq_threshold_3D` is `128 // num_heads_kv`; this model has 4 KV heads, so 32. With CUDA
graphs on, that 32 is replaced by the closest entry in the graph capture list, which at
`max_num_seqs=1` is `[1, 2, 4, 8]`: the buffers get 8 rows. The dispatch log for one
6,747-token request, with the multi-query kernel enabled and debug output on:

| batch size (query tokens) | CUDA graphs on | fully eager | cause |
|---|---|---|---|
| 1, 2, 4, 8 | 3D kernel | 3D kernel | |
| 9 | **2D kernel** | 3D kernel | 9 tokens, 8 rows |
| 24 and larger | 2D kernel | 2D kernel | over the 16-token per-sequence limit (intended) |

The same request, same config; only the graph mode differs. Holding the mode fixed and moving
only the row count flips the 9-token batch in both directions (eager with 8 rows falls back;
graph mode with 16 rows serves it), so the row count is the deciding predicate, and the graph
mode is what shrinks the row count.

## Cost

One token row is `heads x 16 segments x 256 (padded head dim) x fp32` plus two small
max/expsum buffers: 396,288 bytes for the target model (24 query heads) and 528,384 bytes for
the drafter (32 query heads). There are 4 live allocations: one attention backend instance
per attention shape, two shapes, two instances each. Not one per layer (the model has 64
layers, 16 of them full attention).

| `max_num_seqs` | token rows | total across 4 allocations |
|---|---|---|
| 1 (this repo's default) | 16 | 28.22 MiB (14.11 MiB more than the 8-row allocation) |
| 8 | 128 | 225.76 MiB |
| 32 (the `seq_threshold_3D` ceiling) | 512 | about 903 MiB |

The 8 and 32 rows are arithmetic from the measured per-row cost; only `max_num_seqs=1` and
`max_num_seqs=8` were booted. The boot log prints the derived capacity with its inputs, so
the cost is auditable per deployment; there is no silent cap.

## Throughput: a null result, and why it is expected

Metric: decode-only tokens per second from `usage.completion_tokens` (one SSE chunk is one
speculative step, not one token, so chunk counts are not used), generating 384 tokens after a
NASA history prompt. Boots alternate patched and unpatched so drift cannot align with the arm.
Every run keeps its dispatch log, which shows which kernel served the 9-token batches.

| prompt | boots per arm | patched | unpatched | difference |
|---|---|---|---|---|
| 6,747 tokens | 3 | 110.77 tok/s | 110.63 tok/s | +0.13% |
| 68,013 tokens | 2 | 44.6 tok/s | 44.6 tok/s | 0.0% |

The spread across the three patched boots at 6,747 tokens was 0.54%, so both differences sit
inside same-configuration noise. The reason is arithmetic. At draft depth 7 the ordinary
verify batch is 8 tokens, 8 is a capture size that already ran the 3D kernel, and the 9 to 16
token band this fixes was 0.66% of attention calls in the 68k run: 16 of 2,432 (152 decode
steps across 16 int4 attention layers). Those batches occur because the drafter's n-gram
chain extension occasionally proposes more than the base 7 tokens. So #46's gain on 8-token
batches is unaffected, and only the 9 to 16 band was falling back.

A deeper draft would put that band in the common case. The drafter checkpoint is trained at
depth 7, so any other depth is off-design for it, and the one we tried (11) also exceeded 24
GB at 262k context on this card. That regime is untested here, and the patch stands on
correctness rather than on a speed claim.

## Verification

**Property test**, `bench/mq3d_capacity_property.py`: every batch the eligibility rule accepts
must fit the derived allocation. It sweeps `max_num_seqs` 1 to 256, `seq_threshold_3D` 1 to
128, `max_num_batched_tokens` 8 to 131,072, per-sequence query length 1 to 32, and ragged
random query-length vectors: 238,276 eligible batches over 4,032 configurations, 0 violations.
`--mutate seq-rows` (the pre-fix sizing) fails 129,498 of them; `--mutate snapped-rows` (what
shipped) fails 140,008. A property test that has never been shown to fail is decoration.

**Independent numerical check**, written and run on the 4090 by a second author who did not
write the patch, against a separate fp32 attention over a separate dequantisation of the
cached bytes: 9 of 9 cases. The cases that matter: a 17-token single sequence (rejected by
the per-sequence limit, capacity never consulted) against a 16+1 two-sequence batch (the same
17 tokens, admitted); a padded zero-length sequence with NaN-prefilled scratch (nothing
leaked); an over-capacity batch checked through `mq3d_breach_state()`. Reverting the file to
sequence-unit sizing makes every multi-query case breach and leaves exactly the one-token and
17-token cases green, which is the original defect's signature. Scope: the reference reads the
cached bytes the production write path produced, so it bounds the attention and dispatch, not
the cache write.

**Live at `max_num_seqs=8`** (post-fix only; the pre-fix build was not run at this setting):
capacity 128 rows, and a batch of 64 tokens across 8 sequences is served by the 3D kernel.

**Mutations.** Allocation re-expressed in sequence units: the construction assert fires and
the engine refuses to boot. Declared capacity shrunk to 8: the 9-token batch breaches, the
counter reads 1 after the first and 16 at the end of the run, one diagnostic line is written,
and the first specimen is unchanged by later breaches.

**Patch application.** The full patch set, this one included, applies without failure or
fuzz to the stock vLLM 0.27.1 wheel in the Dockerfile's order. That is an apply check, not a
build; the same set booted and served on the 3090 for every measurement above.

## Known nit

The boot log states the capacity formula as a sentence. The inputs and the result are printed
and can be recomputed, which is how the second mutation was caught, but the sentence could go
stale relative to the code. Follow-up: print inputs and result only.
