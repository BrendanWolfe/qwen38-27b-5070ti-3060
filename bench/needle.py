"""Needle-in-a-haystack retrieval check for the long-context profiles.

Builds a haystack of varied real text (this repo's docs, then vLLM's source, which is
long and non-repeating), measures its length with the SERVED model's own tokenizer, and
buries a unique fact at a given fractional depth. Then asks for the fact back and checks
the answer contains it.

Why varied text: a haystack built by repeating one document hands a lookup drafter free
matches and flatters retrieval as well -- the same reason bench/make_long_corpus.py
appends vLLM's source rather than repeating the head.

  venv/bin/python bench/needle.py --port 18021 --ctx 32000 --depths 0.1,0.5,0.9

Exit code 0 if every probe passed. --ctx is in TOKENS of the assembled prompt, and the
script trims to hit it within a few hundred tokens. --model-len must match the server's
max_model_len: the completion budget is derived from it, because prompt+max_tokens over
the limit is a 400, not a truncation.

MEASURED, start_qwen_solo.sh (KVarN 4-bit key / 2-bit value, one slot, 262,144 context,
296,974-token pool). 10/10 retrieved:

  ctx        depth 0.1        depth 0.5        depth 0.9      prefill
  32,000     PASS             PASS             PASS           1,234-1,852 tok/s
  64,000     PASS             PASS             PASS           1,167-1,788 tok/s
  128,000    PASS             PASS             PASS             910-911 tok/s
  258,041    PASS (0.05)                                          662 tok/s

So the 4/2-bit KV does not cost retrieval anywhere in the window, including at
258,041 tokens -- within 4k of the model's full native context. What it does cost is
prefill: 1,236 -> 911 -> 662 tok/s as the window grows, which is attention against an
ever-longer cache, not a KVarN effect. Budget ~6.5 minutes to first token on a full
258k prompt.
"""
import argparse, glob, os, random, sys, time
import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def corpus_chunks():
    seen = set()
    for pat in (f"{REPO}/*.md", f"{REPO}/docs/*.md", f"{REPO}/*/*.md"):
        for p in sorted(glob.glob(pat)):
            if p in seen: continue
            seen.add(p)
            try: yield open(p, errors="replace").read()
            except OSError: pass
    sp = glob.glob(f"{REPO}/venv/lib/python*/site-packages/vllm")
    if sp:
        for p in sorted(glob.glob(f"{sp[0]}/**/*.py", recursive=True)):
            if "test" in p: continue
            try: yield open(p, errors="replace").read()
            except OSError: pass

def build(tok, target):
    parts, n = [], 0
    for c in corpus_chunks():
        parts.append(c); n += len(tok.encode(c, add_special_tokens=False))
        if n > target * 1.15: break
    text = "\n\n".join(parts)
    lo, hi = 0, len(text)
    while lo < hi - 200:                      # binary search on characters
        mid = (lo + hi) // 2
        if len(tok.encode(text[:mid], add_special_tokens=False)) < target: lo = mid
        else: hi = mid
    return text[:lo]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=18020)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--tokenizer", default=f"{REPO}/models/Qwen3.8-27B-W4A16-AutoRound-fast")
    ap.add_argument("--ctx", type=int, default=32000)
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--model-len", type=int, default=262144,
                    help="server max_model_len; caps the completion budget")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    key = os.environ.get("VLLM_API_KEY") or ""
    kp = os.path.join(REPO, "api_key.txt")
    if not key and os.path.exists(kp): key = open(kp).read().strip()
    hdr = {"Content-Type": "application/json"}
    if key: hdr["Authorization"] = f"Bearer {key}"
    url = f"http://{a.host}:{a.port}/v1/chat/completions"

    print(f"building haystack (~{a.ctx:,} tokens)...", flush=True)
    hay = build(tok, a.ctx)
    print(f"  {len(tok.encode(hay, add_special_tokens=False)):,} tokens, {len(hay):,} chars", flush=True)

    rng = random.Random(20260822)
    ok_all = True
    for d in [float(x) for x in a.depths.split(",")]:
        code = f"{rng.randint(10,99)}-{rng.choice(['ALPHA','BRAVO','DELTA','ECHO'])}-{rng.randint(1000,9999)}"
        needle = (f"\n\nIMPORTANT RECORD: The Copenhagen server maintenance code is {code}. "
                  f"Remember the Copenhagen server maintenance code.\n\n")
        cut = int(len(hay) * d)
        prompt = (hay[:cut] + needle + hay[cut:] +
                  "\n\nQuestion: What is the Copenhagen server maintenance code? "
                  "Answer with the code only.")
        ntok = len(tok.encode(prompt, add_special_tokens=False))
        # The server rejects prompt+max_tokens over max_model_len with a 400, so
        # the completion budget has to shrink as the haystack approaches it. 512
        # is far more than "answer with the code only" needs.
        budget = max(512, min(10000, a.model_len - ntok - 256))
        t0 = time.time()
        try:
            r = requests.post(url, headers=hdr, timeout=a.timeout, json={
                "model": a.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": budget, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}})
            r.raise_for_status(); j = r.json()
            msg = j["choices"][0]["message"]
            ans = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
            u = j.get("usage", {})
            hit = code.replace("-", "").lower() in ans.replace("-", "").replace(" ", "").lower()
            dt = time.time() - t0
            pt = u.get("prompt_tokens", ntok)
            print(f"  depth {d:>4}  {pt:>8,} prompt tok  {dt:>7.1f}s  "
                  f"prefill~{pt/dt:>7.0f} tok/s  {'PASS' if hit else 'FAIL'}  "
                  f"want={code} got={ans.strip()[:60]!r}", flush=True)
            ok_all &= hit
        except Exception as e:
            print(f"  depth {d:>4}  ERROR after {time.time()-t0:.0f}s: {e}", flush=True); ok_all = False
    print("RESULT:", "all passed" if ok_all else "FAILURES")
    return 0 if ok_all else 1

if __name__ == "__main__":
    sys.exit(main())
