# MR draft — speculative decoding at depth, the CPU tier, and two fixes

> **DRAFT — not filed.** This is the assembled body for the single upstream MR
> covering both boxes (RTX 4090 / WSL2 / Windows 11 · RTX 3090 / native
> Ubuntu). Michael files it. Every planned measurement on both boxes is in;
> every number below traces to the bench record and the 3090 seat has
> reviewed its own rows (no misquotes found); the load-bearing claims are
> measured on both boxes.

## Summary

This MR carries two fixes, one kernel, one measurement correction, and the
operating doctrine that fell out of measuring them:

1. **OffloadingConnector fix for non-UVA platforms** (`patches/offload-wsl2-devptr.patch`)
   — the CPU offload tier crashes with an illegal memory access on WSL2;
   root-caused to host-VA dereference in the Triton swap kernel, fixed with
   device-pointer translation, validated by falsification bracket on both
   platforms. Likely genuinely upstream-vLLM (any WSL2 user of the
   OffloadingConnector, any model).
2. **Prefix-cache zero-reuse on int4 KV + DFlash2** — a one-flag fix
   (`--prefix-match-unit 848`) with the two-site contradiction analysis that
   explains why it was invisible under int8.
3. **An int4 MQ-3D spec-decode kernel** — 33-line dispatch change + reducer
   guard; true end-to-end ~3.3× on deep int4 spec decode (22.3 → ~73 tok/s).
   Held behind six promotion gates (listed in §3); included as a separate,
   cuttable commit.
4. **A measurement correction that affects any SSE-based benchmark of
   speculative decoding** (§1) — it reversed our own initial "speculation
   loses at depth" conclusion.
5. **Doctrine**: the two-wing serving design, retention capacity vs.
   round-robin, MAX_SEQS as designed geometry, and operator notes.

## 1. The unit inversion — a correction that affects every SSE chunk-rate benchmark

With speculative decoding on, **SSE chunks are spec steps, not tokens** (~3
tokens per chunk at n7 acceptance). Any benchmark that counts chunks/s is
undercounting spec-on throughput ~3× while counting spec-off correctly — a
systematic bias *against* speculation. Fix: request
`stream_options.include_usage` and read `usage.completion_tokens`; report
emitted-tokens-per-step alongside as the receipt. Both boxes reproduced the
inversion independently. All spec-on rates in this MR are token-true.

## 2. The token-true deep map (72.6k-token context, 4090/WSL2)

| Config | decode tok/s (deep) |
|---|---|
| int4 KV + MQ-3D kernel, DFlash2 n7 | **70.5–75.2** |
| long (int8 KV), DFlash2 n7 | 64–68 |
| huge (KVarN 4/2-bit), DFlash2 n7 | ~60–63 |
| int4 KV, spec off | 35.55 |
| int4 KV, stock #42 dispatch, DFlash2 n7 | 22.3 |

**Speculation wins ~2× at depth with the right kernel** — the pre-correction
verdict ("spec loses at 72k") was the unit inversion plus the stock int4
dispatch, compounded. The stock dispatch genuinely loses (0.63× vs spec-off);
the kernel turns that into a 2× win.

## 3. The int4 MQ-3D kernel

Stock #42's spec-decode path launches the int4 attention kernel once per query
position (2D grid); the MQ-3D form dispatches all q positions in one 3D launch
with a reducer guard. 33 lines. Measured: 8.14 → 25.2–25.4 steps/s at 72.6k
(×2.74 step rate), true e2e 22.3 → ~73 tok/s (~3.3×). Ships as
`patches/spec-decode-int4-kv-mq3d.patch` with the dispatch **opt-in**
(`VLLM_INT4_MQ_3D=1`; default off) because six promotion gates remain open
before it should become the default: 72k operator/logit oracle · full-length
exactness · per-position logit comparison · eager+captured+q=9 parity ·
shallow crossover (shallow shows −16%; wants a seq_len floor) ·
capture-aware dual-variant switching. Its own commit, cuttable without
touching the rest.

## 4. Prefix-cache zero-reuse on int4 + DFlash2 — the one-flag fix

Symptom: int4 KV + DFlash2 gets **zero** prefix-cache hits (every request
re-prefills; 62s TTFT on a warm 72.6k whale). The 2×2 cross-box isolation:
int8+DFlash hits, int4−DFlash hits — the interaction is int4 × DFlash2.

Cause: the kvarn-v2 port's two sites carry contradictory assumptions. The GCD
computation *excludes* SW groups ("their block is a divisor of the primary by
construction" — true), yielding hash unit 1696; the SW guard then demands the
SW block (848) be a *multiple* of that hash unit — 848 % 1696 ≠ 0, guaranteed
false by the exclusion itself. Invisible under int8 where 864 = 864 satisfies
both clauses.

Fix: the port's own escape hatch, **`--prefix-match-unit 848`** (the GCD) —
every clause then passes by arithmetic (1696 % 848 = 0, 848 % 848 = 0, both
read clauses). Verified warm-whale TTFT 62s → 4.3s. **Rule: the
prefix-match unit must equal the SW block, and it is per-(model, kv-dtype,
draft-length)** — 848 is this model's number, not a constant.

## 5. Retention — capacity and round-robin are different questions

Measured on the 3090 (int8/long, three distinct contexts, 111,904 naive
tokens = 82% of a 136,429-token pool):

- **Capacity** ("what the pool holds"): recheck the *newest* context only →
  retained, 2.24s vs 50.35s cold (22×). Capacity ≈ **1.3 contexts**.
- **Round-robin** ("what the workload sees"): recheck A, B, C in order →
  **0/3**, every recheck ~1.0× cold. The sweep's own prefills evict the next
  context before it is asked.

The tax is mamba-dominated (fixed-size per-seq state pages) plus drafter
groups. Measured bounds: on the 4090 (int4 geometry, 300,583-token pool),
two 72.6k contexts do **not** both stay warm (K=2 no-tier round-robin =
0/2), so real per-context cost exceeds 150k tokens at 72.6k — a multiplier
above 2.07× at that depth. Consequence for the engine's own banner: its
number is **right only at full length** (proved by the needle probe, §7).
At shorter or cached contexts the real cost per context is substantially
larger than its token count — and a naive constant-plus-linear model
computed from the engine's own spec bytes still *under*counts the measured
cost (we falsified our own corrective banner patch against this data; the
gap mechanism — drafter group pages, per-group block rounding, or
prefix-retention granularity — is an open question this MR flags rather
than answers).

**Product sentence:** with the CPU offload tier (§6), the same round-robin
pattern goes **0/3 → 3/3** (three 72.6k whales all recheck in 4.2–4.3s on
the 4090 — tier restores, not GPU-warm hits, and that is the point: the
GPU pool holds ~one such context warm, and the tier serves the rest at
interactive latency instead of a 62s recompute). Measured under true
concurrency: three simultaneous agents, each on its own 72.6k context,
firing repeatedly — every round completes with **zero recomputes**;
unqueued tier restores land in 5–7s, the LRU-rotated seat pays ~16s
restoring under load, against 62s cold. Three deep agents on one 24 GB
card is a serving pattern this tier makes real.

## 6. The OffloadingConnector fix (the patch in this MR)

Full write-up: `docs/wsl2-4090.md` § "CPU offload tier under WSL2". Short
form: the Triton `swap_blocks` kernel dereferences **host** VAs of the
`cudaHostRegistered` /dev/shm mmap. Native UVA makes host VA == device VA by
coincidence; WSL2 maps the region at a different device address and the
kernel faults. The fix calls `cudaHostGetDevicePointer` after registration,
carries the delta on the CPU views, and adds it where
`compute_sub_block_ptrs` builds addresses. Delta is 0 on native — no behavior
change. All failure branches **raise** (a platform without the device pointer
refuses to start rather than run silent-unsafe) — which makes a live patched
engine itself evidence the query ran: the fix is not only safer but more
falsifiable than a warn-and-continue shape.

Falsification bracket: WSL2 with fix = 3 whales warm, 0 IMAs; fix removed =
IMA ×3, first prime dies. Native = three arms byte-identical at 52.4 tok/s,
805 MB moved through the tier live, loader resolves via `CDLL(None)`, rc=0,
delta=0. **The bracket is trustworthy only for v3.** v1's native arms
"passed" inert: `CDLL("libcudart.so")` throws on standard pip installs (the
wheels ship only *versioned* sonames at wheel paths), so v1's happy path
never executed on native — and discovering that is what exposed v2's
would-be regression (its raise would have refused most native boxes). v3
resolves from the process image first and raises on every failure, so a
live patched engine is itself the execution proof — the evidence behind
§9's falsifiability claim.

Production-form follow-on (tracked, not in this MR): typed two-pointer
separation instead of a delta attribute, transfer records, restored-state
checksums, lifecycle/teardown coverage.

## 7. Two wings, and MAX_SEQS as designed geometry

- **Spec wing** (single/low concurrency, DFlash2 on): wins ≤C4–C8. long C4 =
  517 agg (4090).
- **Batch wing** (spec off, batch geometry): takes over ~C10+. 4090
  fp8-block-800: 676 agg @ C16. 3090 shipped `fast` @ `GPU_UTIL=0.972`
  (boots native; WSL2 refuses this setting): ms/pass 23.3 → 28.2 across
  C1–C16 — a 21% spread over a 16× batch — with **zero preemptions at every
  level**, where the spec profile took 2 at N=8. *The wings differ in what
  they are willing to have go wrong, not only in speed.* (The two boxes'
  batch rows are different profiles — kept as separate rows, never a ratio.)
- **MAX_SEQS is designed geometry, not a free knob**: graph budget (3090:
  N=4 aggregate +6.1% just from MS=8→4) *and* worst-case admission bound
  (huge MS=2 shows 44.7% KV at N=2, yet 2 × 245,760 > the 268,169 pool —
  and that arithmetic is naive-token, i.e. optimistic, per §5's tax). **The
  two gauges lie in opposite directions** — kv% understates worst-case
  admission risk, the concurrency banner overstates multi-context capacity
  by 2×+ (measured bound, §5; honest at N=1 — see the needle row below) —
  so an operator reading both sees headroom twice and tunes into a cliff.
  Size MAX_SEQS to the workload's real prompt distribution, never to a
  gauge.
- **Needle-at-max-len (3090)**: a single request climbing to **234,158
  tokens — 95.3% of max_model_len — completes with exact needle recall**
  (the last 4.7% untested: the corpus ran out, not the engine). So the
  banner is honest about *one*, which sharpens the criticism rather than
  softening it: the banner is not wrong about capacity but about **what
  capacity means**. `pool / max_model_len` is a true statement about a
  single live request and a 2×+ overstatement (measured bound, §5) the
  moment the question is concurrent or cached contexts, where the 9-group
  page tax applies. Same number, honest in one regime and misleading in the
  other, with nothing on the line to say which regime you are in.
- **Politeness knob**: `--max-num-batched-tokens 512` makes minnows live
  (~5s) under a whale prefill for ~15% whale tax (prefill head-of-line
  blocking is the mechanism).
- **Pinned prefixes**: pin one whale prefix per box and it holds.

## 8. Operator notes / gotcha ledger

- The 8 GB `/dev/shm` offload region **outlives the process** (confirmed
  empirically on both boxes) and holds RAM while the box idles — clean at
  shutdown as well as startup; on a small-RAM box the stranded region is the
  next boot's OOM.
- Docker: `--ipc host` required — the 64 MB default shm makes the region's
  `madvise(MADV_POPULATE_WRITE)` fail with EFAULT at boot.
- `alternative.sh` silently ignores `SPEC=off`; a no-nvcc image silently
  runs spec-off. Check the boot banner, not the launch command.
- `GPU_UTIL=0.972` never fits under WSL2 (compositor + WDDM overhead);
  fine on headless native. fp8 MAX_LEN geometry ≠ int4's numbers.
- cudart loading: see §6's loader note.
- Two independent `sed` failures in one week (a silent no-op edit; an
  alternation filter under non-`-E` sed that printed nothing while the run
  was fine): **instruments persist raw output (`--json` always); the
  presentation layer is allowed to fail.**

## 9. Methodology

Pre-registered predictions before each arm (two held across the campaign;
six mechanism stories died on instruments built to kill them). One rule
this campaign paid for four times, in four different ways: **a retention
measurement is only valid when nothing else can produce a fast recheck.**
A tier restore, a shared prefix between ladder rungs, and pool residue
from a prior rung all read as "retained" through a latency threshold — a
latency threshold is not a mechanism discriminator. Corollaries: retention
numbers are valid only with the offload tier off (stated as precondition,
not assumption); ladder rungs must share no prefix (a descending ladder
with fixed salts is contaminated at exactly the rung being hunted, while
the same code read ascending is safe); and the boundary rung must
reproduce **in isolation** — a fresh boot at that length alone — before
the number is real. The
falsification bracket as the standard of "fixed": reproduce the failure,
apply the fix, watch it pass, remove the fix, watch it fail again. And the
campaign's capstone, paid for twice: **a pass from an unexecuted branch is
indistinguishable from a pass — only the execution count tells you which you
have.** (v1 of the offload fix "validated" on four arms whose happy path
never ran; v3 was redesigned so every failure branch raises and a live
engine is itself the proof of execution.) The same discipline killed one of
our own contributions before it could ship: a banner patch printing
model-derived warm-context capacity passed its internal-consistency check,
then failed its pre-registered tier-off falsification (predicted 2/2,
measured 0/2) — it is parked on an experiment branch as mechanism data, not
merged, because a wrong honesty line is worse than none. Corrections
throughout the docs are struck in place, not cleaned — the record keeps its
own history legible.
