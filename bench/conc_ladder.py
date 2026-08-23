"""Concurrency ladder: per-stream decode tok/s at N=1..8, with the acceptance and
the mean tokens per forward pass the server reports over the same window.

  venv/bin/python bench/conc_ladder.py [--max-n 8] [--out 256] [--reps 2] [--shared]

Per-stream decode rate is measured first-token to last-token, so it excludes TTFT;
the aggregate column does not, and is dominated by prefill once several long prompts
arrive at once. Distinct 4k-token prompts per stream by default; --shared makes them
identical, which is what a prefix-cache-friendly client looks like.

The column to read when per-stream throughput falls off a cliff at some N is tok/pass.
A batched decode step at N streams and k drafts runs N*(k+1) query tokens, so if the
aggregate rate stops rising with N while tok/pass stays flat, the streams are taking
turns instead of batching -- a scheduling problem, not a kernel one. Prefill passes
are in the same mean, so compare it across N rather than against N*(k+1) directly.

Written for issue #16 (a reported per-stream dip at exactly N=2, which did not
reproduce here: 105 / 98 / 86 tok/s at N=1/2/3, SPEC=mtp k=3 PREFIX_CACHE=1).
"""
import json, os, sys, time, urllib.request, threading, re

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
PORT = os.environ.get("PORT", "18020")
API = f"http://127.0.0.1:{PORT}"


def _key(path):  # a key is optional; keyless servers ignore the header
    try:
        return open(path).read().strip()
    except OSError:
        return ""


KEY = os.environ.get("VLLM_API_KEY") or _key(os.path.join(REPO, "api_key.txt"))
MAXN = int(sys.argv[sys.argv.index("--max-n") + 1]) if "--max-n" in sys.argv else 8
NOUT = int(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else 256
REPS = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 2
SHARED = "--shared" in sys.argv

FILLER = ("The RTX 3090 has 24 GB of GDDR6X and 82 streaming multiprocessors. "
          "Memory bandwidth is 936 GB/s, which is what decode is bound by. ")


def make_prompt(i):
    head = "" if SHARED else f"Document {i} (revision {i * 7 + 3}). "
    body = FILLER * 260                      # ~4k tokens
    return (head + body +
            "\n\nWrite a detailed technical explanation of why memory bandwidth, "
            "not compute, limits single-stream decoding on this hardware. Be thorough.")


def metrics():
    req = urllib.request.Request(API + "/metrics", headers={"Authorization": f"Bearer {KEY}"})
    txt = urllib.request.urlopen(req, timeout=20).read().decode()
    out = {}
    for name in ("vllm:spec_decode_num_accepted_tokens_total",
                 "vllm:spec_decode_num_draft_tokens_total",
                 "vllm:spec_decode_num_drafts_total",
                 "vllm:iteration_tokens_total_sum",
                 "vllm:iteration_tokens_total_count"):
        m = re.findall(rf"^{re.escape(name)}\{{[^}}]*}} ([0-9.e+]+)$", txt, re.M)
        out[name.split("vllm:")[-1].replace("spec_decode_", "")] = sum(float(x) for x in m) if m else 0.0
    return out


def stream(i, res):
    body = json.dumps({
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": make_prompt(i)}],
        "max_tokens": NOUT, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(API + "/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; last = None; ntok = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                ntok = d["usage"]["completion_tokens"]
            ch = d.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                now = time.perf_counter()
                if first is None:
                    first = now
                last = now
    res[i] = dict(ttft=(first - t0) if first else None,
                  decode_s=(last - first) if first and last else None, ntok=ntok)


print(f"prompts: {'shared' if SHARED else 'distinct'}, out={NOUT}, reps={REPS}")
print(f"{'N':>2} {'per-stream tok/s':>17} {'aggregate':>10} {'tok/step':>9} {'tok/pass':>9} {'TTFT ms':>9}")
rows = []
for n in range(1, MAXN + 1):
    best = None
    for rep in range(REPS):
        m0 = metrics()
        res = {}
        ts = [threading.Thread(target=stream, args=(i, res)) for i in range(n)]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        wall = time.perf_counter() - t0
        m1 = metrics()
        good = [v for v in res.values() if v["decode_s"] and v["ntok"] > 1]
        if len(good) != n:
            print(f"{n:>2}  incomplete ({len(good)}/{n})"); continue
        per = sum((v["ntok"] - 1) / v["decode_s"] for v in good) / n
        agg = sum(v["ntok"] for v in good) / wall
        # tokens per forward pass: a properly batched decode step at N streams and k
        # drafts runs N*(k+1) query tokens. Half that means the streams are taking
        # turns rather than batching -- scheduling, not kernels.
        di = m1["iteration_tokens_total_count"] - m0["iteration_tokens_total_count"]
        its = (m1["iteration_tokens_total_sum"] - m0["iteration_tokens_total_sum"]) / di if di else float("nan")
        dn = m1["num_drafts_total"] - m0["num_drafts_total"]
        da = m1["num_accepted_tokens_total"] - m0["num_accepted_tokens_total"]
        tps = (da / dn + 1) if dn else float("nan")
        ttft = sum(v["ttft"] for v in good) / n * 1000
        if best is None or per > best[0]:
            best = (per, agg, tps, ttft, its)
    if best:
        rows.append((n,) + best)
        print(f"{n:>2} {best[0]:>17.1f} {best[1]:>10.1f} {best[2]:>9.2f} {best[4]:>9.1f} {best[3]:>9.0f}")
print()
print("| N | " + " | ".join(str(r[0]) for r in rows) + " |")
print("|---|" + "---|" * len(rows))
print("| tok/s | " + " | ".join(f"{r[1]:.0f}" for r in rows) + " |")
print("| tok/step | " + " | ".join(f"{r[3]:.2f}" for r in rows) + " |")
print("| tok/pass | " + " | ".join(f"{r[5]:.1f}" for r in rows) + " |")
