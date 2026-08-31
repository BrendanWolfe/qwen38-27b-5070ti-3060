"""Fetch the uncensored (abliterated) W4A16 checkpoint and say what to run next:
philbert440/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ, an AWQ quant of the
de-refused Qwen3.8-27B with the vision tower and the grafted MTP head preserved.

  venv/bin/python prepare/fetch_uncensored.py [dst_dir]   # default models/Qwen3.8-27B-Uncensored-W4A16

~18.6 GB. It is NOT servable on 24 GB as it ships, for the same reason the base
model is not (bf16 lm_head, bf16 embeddings, bf16 MTP module) -- but the three
prepare/quant_*.py scripts cannot fix this one: it is a single 18.6 GB shard and
its body is asymmetric AWQ. prepare/quant_heads_stream.py handles both; this
script prints the exact commands when the download finishes.
"""
import os, sys
from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
REPO = "philbert440/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
D = (args[0] if args else os.path.join(ROOT, "models", "Qwen3.8-27B-Uncensored-W4A16")).rstrip("/")
os.makedirs(D, exist_ok=True)

# chat_template.jinja is not *.json and the server needs it: without it vLLM falls back
# to the tokenizer's built-in template, which is not the one this checkpoint was tuned
# with (and does not emit the XML tool-call format --tool-call-parser qwen3_coder reads).
snapshot_download(REPO, local_dir=D,
                  allow_patterns=["*.json", "*.jinja", "*.safetensors", "README.md"])

rel = os.path.relpath(D, ROOT)
print(f"\nuncensored checkpoint downloaded: {rel}")
print("it is not servable yet -- requantize the heads (CPU only, ~15 min, needs ~20 GB free disk\n"
      "for the .bak-orig originals):\n"
      f"  venv/bin/python prepare/quant_heads_stream.py {rel}\n"
      f"  venv/bin/python prepare/build_draft_vocab.py {rel} --ids prepare/draft_vocab_ids.json\n"
      f"then:  MODEL=$PWD/{rel} bash single-user/start_qwen.sh")
