# Setup guide — RTX 5070 Ti + RTX 3060 (from zero to serving)

Step-by-step for someone with the same pair: a **16 GiB RTX 5070 Ti** and a
**12 GiB RTX 3060**. By the end you will have five working profiles:

| profile | what it is | context | KV | single-stream decode |
|---|---|---|---|---|
| `start_qwen_batch.sh` | general-purpose, MTP-3, 8 slots | 147k (147,456-token pool floor) | FP8 | **~74 tok/s** (210 tok/s at 4 concurrent) |
| `start_qwen_kvarn_batch.sh` | long-context MTP-3, 8 slots | **262k** (289,641-token pool) | KVarN 4/2-bit | 240,035-token needle passed; **684 prefill tok/s** |
| `start_qwen_solo.sh` | full native context, one stream | **262k** (296,974-token pool) | KVarN 4/2-bit | **~73 tok/s** |
| `start_qwen_dflash2_batch.sh` | DFlash2, 4 scheduler seats | **147k** (151,503-token pool) | KVarN 4/2-bit | C4 completes; 3 resident at 4k |
| `start_qwen_dflash2_solo.sh` | DFlash2, one stream | **200k** (202,174-token pool) | KVarN 4/2-bit | **~83–87 tok/s** |

Everything is driven by this fork (`BrendanWolfe/qwen38-27b-5070ti-3060`) of
[syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) —
the upstream repo does **not** contain the two-GPU patches. The model
preparation below follows the upstream setup; the run/verify steps are the
fork's additions. If something breaks, `bash verify.sh --no-server` names the
missing piece.

## 0. What you need

- The two GPUs, both visible to the host, with a driver new enough for
  consumer Blackwell (R570+). CUDA 13.x is what this was tested with.
- Linux, **Python 3.12**, ~60 GB free disk, and ≥24 GB RAM (32 GB comfortable —
  checkpoint load wants ~19 GB free).
- `gcc-15`/`g++-15` (or another host compiler CUDA 13.3 accepts) for the
  FlashInfer/torch JIT builds on first start. Fedora:
  `sudo dnf install gcc15 gcc15-c++`. Ubuntu 25.04+:
  `sudo apt install gcc-15 g++-15`.
- Internet for ~22 GB of Hugging Face downloads.

**Check the GPU order first** — the default 44/20 pipeline split assumes the
5070 Ti is CUDA device 0 and the 3060 is device 1 (the launchers force
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, so this is PCIe slot order):

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

If your 3060 is slot 0, either move the cards or launch with
`GPU_IDS=1,0 PP_LAYERS=44,20` (keep the 5070 Ti first in `GPU_IDS`).

## 1. Clone this fork and create the venv

```bash
git clone https://github.com/BrendanWolfe/qwen38-27b-5070ti-3060 ~/qwen-serving
cd ~/qwen-serving

python3.12 -m venv venv
venv/bin/pip install -r docker/requirements.txt
```

`docker/requirements.txt` pins the exact stack the numbers were measured on:
`vllm==0.27.1` (which itself pins torch 2.13.0/cu130, triton 3.7.1,
flashinfer-python 0.6.16.post3), `transformers==5.15.0`, `compressed-tensors==0.17.0`,
`huggingface_hub==1.27.0`, `hf_transfer`, `ninja`. **Do not install an unpinned
`vllm`** — the patches are written against 0.27.1.

If you will run either DFlash2 profile, add the cubin package:

```bash
venv/bin/pip install flashinfer-cubin==0.6.13
```

`flashinfer-python` alone is not enough. vLLM only *uses* FlashInfer where
`has_flashinfer()` is true, and that predicate wants `nvcc` on `PATH` **or**
`flashinfer-cubin` installed; with neither, the DFlash2 candidate selector falls
back to `torch.topk` at roughly half speed and says so in exactly one INFO line
(upstream #35). It is not hypothetical on a machine without the CUDA toolkit —
this fork's own box had `flashinfer-python 0.6.16.post3`, no `nvcc`, and
`has_flashinfer() == False`, so **the DFlash2 numbers in
`heterogeneous/README.md` were measured on the torch.topk fallback** and should
improve once the cubins are installed. `verify.sh` now tests the real predicate
instead of a bare import. Do not fix the version mismatch by downgrading
`flashinfer-python`: that drags torch back and breaks vLLM's C extension — the
launchers export `FLASHINFER_DISABLE_VERSION_CHECK=1` instead.

The FlashInfer *attention* backend that `KV=fp8` selects is a different code
path and works without any of this, which is why the MTP profiles were never
affected.

## 2. Download the base model (~19.5 GB)

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 venv/bin/hf download \
  dbirks/Qwen3.8-27B-W4A16-AutoRound \
  --local-dir models/Qwen3.8-27B-W4A16-AutoRound
```

## 3. Create the model (requantization + fast variant + drafter)

The published W4A16 quant ships two 2.5 GB bf16 embedding matrices and an
unquantized MTP draft module — too big for 24 GB and unusable with the PP
patches. These scripts fix the model in place on the CPU (a few minutes each),
and each backs up what it rewrites next to the original:

```bash
M=models/Qwen3.8-27B-W4A16-AutoRound
venv/bin/python prepare/quant_lm_head.py $M      # lm_head int8 group-128, in place
venv/bin/python prepare/quant_embed.py   $M      # embed_tokens likewise (untied)
venv/bin/python prepare/quant_mtp.py     $M      # mtp.* draft module -> int8
venv/bin/python prepare/build_draft_vocab.py $M --ids prepare/draft_vocab_ids.json
venv/bin/python prepare/fetch_fast_variant.py    # ~1 GB: int4-GPTQ lm_head + drafter
venv/bin/python prepare/fetch_dflash2.py         # ~1.2 GB: W4A16 DFlash2 block drafter
```

- `fetch_fast_variant.py` assembles `models/Qwen3.8-27B-W4A16-AutoRound-fast`
  (hardlinks the base shards, downloads the int4 lm_head/drafter tensors from
  the Hub). The launchers prefer it automatically when present. **Run it.**
- `fetch_dflash2.py` downloads `models/Qwen3.8-27B-DFlash2-W4A16` — only needed
  for the DFlash2 profile, but cheap; run it.

## 4. Apply the vLLM patches

```bash
for p in patches/*.patch; do
  patch -p1 -d venv/lib/python3.12/site-packages/vllm < $p
done
```

This loop applies all 23 patches in filename order — including the four this
fork adds. Order matters, which is why the DFlash2 one is named
`zz-dflash2-pipeline-parallel.patch`: it must be last because it edits files
earlier patches also touch. `vllm-pr46994-mtp-pp.patch` (MTP + pipeline
parallelism) uses the same conventional paths as the rest, so it applies in
the loop too.

Six of those arrived from upstream in the 2026-08-26 merge and all six apply
cleanly on top of this fork's stack (verified by dry-run against a fully
patched tree). Two of them do nothing on this hardware and are carried only so
`verify.sh` and upstream stay in step: `marlin-repack-staged-sm80.patch`
defaults on for compute capability **8.0 exactly** (this pair is sm120 and
sm86), and `offload-dflash-eagle-groups.patch` fixes the OffloadingConnector,
which no profile here enables. The other four are live:
`xgrammar-spec-terminated.patch` (tool calls were returning HTTP errors for
valid output when a verify window ran past the grammar's end — every profile
here ships `--enable-auto-tool-choice`), `vision-tower-cpu-offload.patch`
(`VISION_OFFLOAD`, below), `int4-kv-per-token-head.patch` (unblocks
`KV=int4pth` under `SPEC=dflash2`; `KV=int4pth` under MTP is unchanged, since
the padded-page branch it adds only fires when a promoted drafter layer pads
the pages) and `dflash2-ngram-chains.patch` (`CHAIN=1`, off by default).

## 5. Verify the install

```bash
bash verify.sh --no-server
```

It checks Python/vLLM versions, that **every** patch is applied, that
FlashInfer is usable by vLLM rather than merely importable, and that the model
was requantized (lm_head, embeddings, MTP module, draft head, fast variant,
drafter). It should print `verify: OK (0 failures)` — KVarN is an optional
warning you can ignore. A `flashinfer unusable` FAIL means step 1's
`flashinfer-cubin` line was skipped; it only costs you the DFlash2 profiles.

## 6. API key (direct launches only)

```bash
openssl rand -hex 24 > api_key.txt
```

llama-swap (step 8) runs with `NO_API_KEY=1` and needs no key.

## 7. Run

### Batch MTP profile (general purpose, 147k context)

```bash
bash heterogeneous/start_qwen_batch.sh
```

- First start takes a few minutes: model load, torch.compile (cached after the
  first run), CUDA graph capture, FlashInfer JIT. Subsequent starts are faster.
- It listens on `0.0.0.0:18020`, served model name `qwen3.8-27b`. It is ready
  when `/health` answers:
  ```bash
  curl -sf http://127.0.0.1:18020/health && echo READY
  curl http://127.0.0.1:18020/v1/chat/completions \
    -H "Authorization: Bearer $(cat api_key.txt)" -H "Content-Type: application/json" \
    -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hej"}],
         "chat_template_kwargs":{"enable_thinking":false}}'
  ```
- Expected: **74–75 tok/s** single-stream on realistic prompts (an earlier
  repeated-512-token measurement read 84; both are in
  [README.md](README.md)), and an FP8 KV pool of at least 147,456 tokens —
  often more after the exact compile shape is warm (gotcha 43). Keep it in a `tmux` session or a systemd unit for long runs.

### DFlash2 batch profile (four scheduler seats)

```bash
bash heterogeneous/start_qwen_dflash2_batch.sh
```

This uses DFlash2-7 with KVarN 4/2 KV, a reversed `30,34` pipeline, four
scheduler seats, and a manually pinned 2.4 GB cache. It exposes 147,456 context
in a measured 151,503-token pool. A 140,028-token retrieval probe passed and
four distinct 4k requests completed without preemption; three were resident
while the fourth queued. Do not raise the pin to the solo profile's 2.8 GB:
that configuration passed one 150k request but OOMed rank 1 at concurrency 4.

### DFlash2 solo profile (one stream, 200k)

```bash
bash heterogeneous/start_qwen_dflash2_solo.sh
```

The one-seat variant keeps the same KVarN/reversed-pipeline layout and spends
the freed runtime headroom on a 2.8 GB cache: 200,192 context in a measured
202,174-token pool. A 190,050-token retrieval probe passed and `/health`
remained 200 afterward.

### Non-speculative fallback (diagnostics)

```bash
SPEC=none MAX_SEQS=8 GPU_UTIL=0.95 bash heterogeneous/start_qwen.sh
```

## 8. (Optional) llama-swap front door

Install llama-swap, then merge the four model entries from
`heterogeneous/llama-swap.example.yaml` into `~/.config/llama-swap/config.yaml`
under `models:`. The entries:

- `vllm-speed/qwen3.8-27b-batch` — general 147k MTP (`start_qwen_batch.sh`);
  keeps `vllm-speed/qwen3.8-27b` as an alias so existing clients keep working
- `vllm-speed/qwen3.8-27b-dflash2-batch` — DFlash2, 147k context
  (`start_qwen_dflash2_batch.sh`)
- `vllm-speed/qwen3.8-27b-dflash2-solo` — single-stream DFlash2, 200k context
  (`start_qwen_dflash2_solo.sh`)
- `vllm-speed/qwen3.8-27b-solo` — 262k single-stream KVarN (`start_qwen_solo.sh`)

If you rename a launcher, remember that llama-swap holds absolute paths — grep
your live config, not just this repo.

Notes that matter:

- Each entry launches through `/usr/bin/env PORT='${PORT}' NO_API_KEY=1 …` —
  do not drop the `env` indirection; llama-swap's multi-argument command
  parsing breaks on the launcher's `cd` otherwise.
- `useModelName` stays `qwen3.8-27b` (the vLLM served name).
- `ttl: 900` / `unloadTimeout: 60` unload an idle model after 15 minutes.
- The launchers already pass the vLLM flags llama-swap needs for accurate
  per-request metrics (`--enable-force-include-usage`,
  `--enable-per-request-metrics`, `--enable-prompt-tokens-details`).
- llama-swap is **unauthenticated** and the launchers bind `0.0.0.0` — put a
  firewall in front if you expose the LAN.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `patch: … REJECTED` | vLLM not 0.27.1 (reinstall from `docker/requirements.txt`), or patches applied twice (re-extract/reinstall). |
| startup `ValueError: … KV cache is needed, larger than the available` | `DFLASH_KV_MEMORY` too small for `MAX_LEN`, or too much GPU memory already used (check `nvidia-smi`). |
| `torch.OutOfMemoryError` mid-request on batch profile | `GPU_UTIL` too high. 0.91 is the safe value; startup profiling does not include compiled-prefill workspaces. |
| `Failed to CUDA calloc async 40 bytes` (NCCL) on DFlash2 | rank 1 ran out — lower `DFLASH_KV_MEMORY` or `GPU_UTIL`. |
| Marlin `torch_call_dispatcher … aten::empty` during capture | DFlash's private CUDA graph with the shared quantized LM head — keep `VLLM_DFLASH_CUDAGRAPH=0` (the wrapper sets it). |
| orphaned processes after a crash | the launchers run vLLM under `setsid` and kill the whole process group on exit; check `nvidia-smi` for stuck workers and `pkill -f 'vllm serve'` if needed. |
| tool calls fail with `"auto" tool choice requires …` | the launcher already adds `--enable-auto-tool-choice --tool-call-parser qwen3_coder`; don't remove them. |
| slow first start / JIT errors | missing `gcc-15` or `TORCH_CUDA_ARCH_LIST` — the launcher exports `8.6;12.0` itself; install the compiler from step 0. |

After any vLLM upgrade, reapply the patches (`verify.sh --no-server` will tell
you) and re-run `verify.sh` — the fork's two patches are written against
0.27.1 like the rest.
