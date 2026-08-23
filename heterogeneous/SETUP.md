# Setup guide — RTX 5070 Ti + RTX 3060 (from zero to serving)

Step-by-step for someone with the same pair: a **16 GiB RTX 5070 Ti** and a
**12 GiB RTX 3060**. By the end you will have five working profiles:

| profile | what it is | context | KV | single-stream decode |
|---|---|---|---|---|
| `start_qwen_batch.sh` | general-purpose, MTP-3, 8 slots | 140k (155,978-token pool) | FP8 | **~74 tok/s** (210 tok/s at 4 concurrent) |
| `start_qwen_solo.sh` | full native context, one stream | **262k** (296,974-token pool) | KVarN 4/2-bit | **~73 tok/s** |
| `start_qwen_dflash2.sh` | short-context DFlash2 | 32k (33,506-token pool) | BF16 | **~92 tok/s** avg (77–159 t/s by workload) |
| `start_qwen_dflash2_fast.sh` | reversed-pipeline DFlash2 | 32k (34,539-token pool) | BF16 | **~97 tok/s**, 33.9 ms/step |
| `start_qwen_huge.sh` | full native context | **262k** (284,234-token pool) | int4 | batch only — ~112 tok/s prefill at depth |

Everything is driven by this fork (`BrendanWolfe/qwen38-27b-rtx3090`) of
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
git clone https://github.com/BrendanWolfe/qwen38-27b-rtx3090 ~/qwen-serving
cd ~/qwen-serving

python3.12 -m venv venv
venv/bin/pip install -r docker/requirements.txt
```

`docker/requirements.txt` pins the exact stack the numbers were measured on:
`vllm==0.27.1` (which itself pins torch 2.13.0/cu130, triton 3.7.1,
flashinfer-python 0.6.16.post3), `transformers==5.15.0`, `compressed-tensors==0.17.0`,
`huggingface_hub==1.27.0`, `hf_transfer`, `ninja`. **Do not install an unpinned
`vllm`** — the patches are written against 0.27.1.

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

This loop applies all 15 patches in filename order — including the two this
fork adds. Order matters, which is why the DFlash2 one is named
`zz-dflash2-pipeline-parallel.patch`: it must be last because it edits files
earlier patches also touch. `vllm-pr46994-mtp-pp.patch` (MTP + pipeline
parallelism) uses the same conventional paths as the rest, so it applies in
the loop too.

## 5. Verify the install

```bash
bash verify.sh --no-server
```

It checks Python/vLLM versions, that **every** patch is applied, and that the
model was requantized (lm_head, embeddings, MTP module, draft head, fast
variant, drafter). It should print `verify: OK (0 failures)` — KVarN is an
optional warning you can ignore.

## 6. API key (direct launches only)

```bash
openssl rand -hex 24 > api_key.txt
```

llama-swap (step 8) runs with `NO_API_KEY=1` and needs no key.

## 7. Run

### Batch MTP profile (general purpose, 140k context)

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
  [README.md](README.md)), ~155,978-token
  FP8 KV pool. Keep it in a `tmux` session or a systemd unit for long runs.

### DFlash2 profile (short context, fastest decode)

```bash
bash heterogeneous/start_qwen_dflash2.sh
```

Same endpoint/port/model name, but BF16 KV, 32k context, and DFlash2 drafting
(~92 tok/s average; real traffic measures **77–159 t/s** depending on the
workload — the high end on long generations and on copy/quote/edit tasks,
where the lookup drafter proposes straight from the prompt). It pins
`--kv-cache-memory=2500000000` on purpose: vLLM's
automatic sizing over-allocates the 3060 rank (which also carries the drafter)
until NCCL cannot even allocate a 40-byte buffer. If you change `MAX_LEN`,
re-tune `DFLASH_KV_MEMORY` — too low fails startup with a "KV cache is needed,
larger than the available" error, too high OOMs rank 1 during graph capture.

### Non-speculative fallback (diagnostics)

```bash
SPEC=none MAX_SEQS=8 GPU_UTIL=0.95 bash heterogeneous/start_qwen.sh
```

## 8. (Optional) llama-swap front door

Install llama-swap, then merge the two model entries from
`heterogeneous/llama-swap.example.yaml` into `~/.config/llama-swap/config.yaml`
under `models:`. The entries:

- `vllm-speed/qwen3.8-27b` — general 140k MTP (`start_qwen_batch.sh`)
- `vllm-speed/qwen3.8-27b-dflash2` — DFlash2 profile

`start_qwen_solo.sh` and `start_qwen_huge.sh` are not wired into llama-swap by
default; add them the same way if you want them swappable.

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
