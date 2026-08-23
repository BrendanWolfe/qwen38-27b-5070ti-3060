"""Render the side-by-side demo from two demo_capture.py recordings.

One GPU means the two configurations cannot run simultaneously, so each lane was
recorded separately with per-token arrival times and this replays them together at
their real speed. Nothing is sped up, slowed down or interpolated: at video time t
a lane shows exactly the tokens whose recorded arrival was <= t.

  python3 bench/demo_render.py baseline.json dflash2.json --out docs/media/demo

Writes <out>.gif (for inline README rendering) and <out>.mp4. Needs pillow + ffmpeg.
"""
import json
import os
import shutil
import subprocess
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont

FPS = 15
W, H = 960, 540
PAD = 18
COL_W = (W - PAD * 3) // 2
HOLD_S = 1.6          # freeze at the end of each prompt so the result is readable
TITLE_S = 1.1          # title card between prompts, so the change of prompt is obvious
FINAL_S = 3.2          # dedicated closing beat after the last prompt
BODY_LINES = 12

# -- palette --------------------------------------------------------------
# Self-contained: the frame paints its own background, so it reads the same
# whether the surrounding page is GitHub light or dark. Blue/orange is a
# colourblind-safe pair (Okabe-Ito), unlike the amber/green this replaces.
BG = (10, 11, 13)
CARD = (19, 21, 25)
EDGE = (40, 43, 49)
EDGE_SOFT = (29, 31, 36)
FG = (233, 235, 238)
DIM = (142, 148, 158)
DIM2 = (92, 97, 106)
NEUTRAL = (150, 155, 245)      # violet-blue accent for chrome (pills, titles)
STOCK = (92, 165, 235)          # blue  -- "Stock vLLM"
REPO = (255, 158, 68)           # orange -- "This repo"


def font(path, sz, index=0):
    try:
        return ImageFont.truetype(path, sz, index=index)
    except Exception:
        return ImageFont.load_default()


SANS = "/System/Library/Fonts/HelveticaNeue.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"

F_TITLE = font(SANS, 21, 1)     # header title (bold)
F_SUBTITLE = font(SANS, 12, 7)  # header subtitle (light)
F_PILL = font(SANS, 12, 10)     # prompt pill (medium)
F_LANE = font(SANS, 16, 1)      # lane name (bold)
F_LANESUB = font(MONO, 10, 0)   # lane command, kept mono -- it *is* a command
F_BODY = font(MONO, 12, 0)      # streaming text
F_HERO = font(SANS, 40, 1)      # tok/s hero number (bold)
F_UNIT = font(SANS, 12, 10)     # "tok/s" (medium)
F_STAT = font(SANS, 12, 0)      # token count / elapsed
F_DONE = font(SANS, 11, 10)     # done badge
F_STAMP = font(SANS, 10, 7)     # honesty stamp
F_CARD_TAG = font(SANS, 13, 10)
F_CARD_TITLE = font(SANS, 38, 1)
F_CARD_SUB = font(SANS, 13, 7)
F_FINAL_KICKER = font(SANS, 13, 10)
F_FINAL_HERO = font(SANS, 72, 1)
F_FINAL_SUB = font(SANS, 15, 0)
F_FINAL_ROW = font(SANS, 13, 0)
F_FINAL_ROW_B = font(SANS, 13, 10)

GLOBAL_MAX = 0.0   # set in main(): shared scale so bar length is always comparable


def prepare(p):
    """Annotate each streamed chunk with how many output tokens it carried.

    A chunk is not a token. With a 16-token verify block DFlash2 delivers a whole
    accepted run in one SSE event, so counting chunks would show 29 where 400 tokens
    arrived. The server only reports the total, so split it across chunks by character
    count -- monotone, sums to the real total, and exact for the one-token-per-chunk
    case (the unspeculated lane).
    """
    chunks = p["tokens"]
    total_chars = sum(len(c[1]) for c in chunks) or 1
    cum_c, cum = 0, []
    for t_ms, txt in chunks:
        cum_c += len(txt)
        cum.append((t_ms, cum_c * p["n_out"] / total_chars))
    p["_cum"] = cum
    return p


def visible(p, t_ms):
    """Text, token count and index visible at t_ms, by recorded arrival time."""
    chunks = p["tokens"]
    lo, hi = 0, len(chunks)
    while lo < hi:                      # chunks are in arrival order
        mid = (lo + hi) // 2
        if chunks[mid][0] <= t_ms:
            lo = mid + 1
        else:
            hi = mid
    n_tok = p["_cum"][lo - 1][1] if lo else 0.0
    return "".join(c[1] for c in chunks[:lo]), int(round(n_tok)), lo


def rate_at(p, i, t_ms, window=800.0):
    """Trailing-window tok/s, so the readout moves like a live meter."""
    cum = p["_cum"]
    if i < 2:
        return 0.0
    t0 = max(0.0, t_ms - window)
    first = 0
    for j in range(i - 1, -1, -1):
        if cum[j][0] < t0:
            first = j + 1
            break
    span = (cum[i - 1][0] - cum[first][0]) / 1000.0
    if span <= 0.05:
        return cum[i - 1][1] / max(t_ms / 1000.0, 1e-3)
    return (cum[i - 1][1] - cum[first][1]) / span


def wrap(text, cols):
    out = []
    for para in text.split("\n"):
        out.extend(textwrap.wrap(para, cols) or [""])
    return out


# Menlo has no glyph for these two emoji (renders as a blank box) -- swap in a
# dingbat it does support. Display-only: token counts are computed from the
# original, unmodified text before this ever runs.
GLYPH_FALLBACK = str.maketrans({"✅": "✓", "❌": "✗"})


def rrect(d, box, r, **kw):
    d.rounded_rectangle(box, r, **kw)


def meter(d, x, y, w, h, value, vmax, color, track=EDGE):
    """A horizontal bar on a fixed 0..vmax scale, so two bars drawn at the same
    scale in different columns still show the true ratio between them -- no
    arithmetic needed."""
    rrect(d, [x, y, x + w, y + h], h / 2, fill=track)
    frac = max(0.0, min(1.0, value / vmax)) if vmax else 0.0
    fw = int(w * frac)
    if fw > h:  # avoid a squashed pill shorter than it is tall
        rrect(d, [x, y, x + fw, y + h], h / 2, fill=color)
    elif fw > 0:
        d.ellipse([x, y, x + h, y + h], fill=color)


def dots(d, cx, y, n, idx, active, inactive):
    r, gap = 3, 12
    total = n * gap - (gap - r * 2)
    x = cx - total / 2
    for i in range(n):
        c = active if i == idx else inactive
        d.ellipse([x, y - r, x + r * 2, y + r], fill=c)
        x += gap


def stamp(d):
    d.text((PAD, H - 26), "recorded separately, replayed at true speed",
           font=F_STAMP, fill=DIM2)


def panel(d, x, lane, prompt, t_ms, cols, top, bottom):
    title, sub, accent = lane
    # A lane that has stopped generating must stop its clock too, or it keeps
    # counting while the other lane finishes and the elapsed time is a lie.
    end_ms = prompt["tokens"][-1][0]
    done = t_ms >= end_ms
    t_ms = min(t_ms, end_ms)

    rrect(d, [x, top, x + COL_W, bottom], 10, fill=CARD, outline=EDGE)
    d.ellipse([x + 16, top + 19, x + 24, top + 27], fill=accent)
    d.text((x + 32, top + 14), title, font=F_LANE, fill=FG)
    d.text((x + 16, top + 40), sub, font=F_LANESUB, fill=DIM2)
    d.line([x + 16, top + 60, x + COL_W - 16, top + 60], fill=EDGE_SOFT, width=1)

    text, n, i = visible(prompt, t_ms)
    lines = wrap(text.translate(GLYPH_FALLBACK), cols)[-BODY_LINES:]
    y = top + 72
    for ln in lines:
        d.text((x + 16, y), ln, font=F_BODY, fill=FG)
        y += 17

    rate = prompt["decode_tok_s"] if done else rate_at(prompt, i, t_ms)
    ry = bottom - 78
    big = f"{rate:.0f}"
    bw = d.textlength(big, font=F_HERO)
    d.text((x + 16, ry), big, font=F_HERO, fill=accent)
    d.text((x + 22 + bw, ry + 24), "tok/s", font=F_UNIT, fill=DIM)
    meter(d, x + 16, ry + 46, 130, 6, rate, GLOBAL_MAX, accent)

    stat_x = x + COL_W - 16
    line1 = f"{n:,} / {prompt['n_out']:,} tokens"
    d.text((stat_x - d.textlength(line1, font=F_STAT), ry + 2), line1, font=F_STAT, fill=DIM)
    line2 = f"{t_ms / 1000.0:.2f}s elapsed"
    d.text((stat_x - d.textlength(line2, font=F_STAT), ry + 20), line2, font=F_STAT, fill=DIM2)
    if done:
        badge = f"DONE  ·  avg {prompt['decode_tok_s']:.1f} tok/s"
        bw2 = d.textlength(badge, font=F_DONE)
        by = ry + 46
        rrect(d, [stat_x - bw2 - 16, by - 4, stat_x, by + 16], 9,
              fill=EDGE_SOFT, outline=accent)
        d.text((stat_x - bw2 - 8, by), badge, font=F_DONE, fill=accent)
    return rate


def header(d, idx, n_prompts, pa):
    # Careful with this subtitle and with `kicker` in final_card(): both have
    # twice drifted back into claims that are not true of this picture. It is
    # NOT "same weights" -- the middle lane runs the fast variant's int4-GPTQ
    # lm_head and the left one does not, which is why their answers diverge --
    # and it is not "speculative decoding only", because the middle lane is the
    # whole serving stack and the right lane is a different engine altogether.
    d.text((PAD, 16), "Qwen3.8-27B decode speed", font=F_TITLE, fill=FG)
    d.text((PAD, 42), "one RTX 3090 @ 250 W · same prompts · not run concurrently",
           font=F_SUBTITLE, fill=DIM)

    pill = f"PROMPT {idx + 1}/{n_prompts}  ·  {pa['label'].upper()}"
    pw = d.textlength(pill, font=F_PILL)
    py = 68
    rrect(d, [PAD, py, PAD + pw + 24, py + 26], 13, fill=CARD, outline=NEUTRAL)
    d.text((PAD + 12, py + 6), pill, font=F_PILL, fill=NEUTRAL)
    tok_lbl = f"{pa['prompt_tokens']:,} prompt tokens"
    d.text((PAD + pw + 40, py + 6), tok_lbl, font=F_PILL, fill=DIM)
    dots(d, W - PAD - 40, py + 13, n_prompts, idx, NEUTRAL, EDGE)


def title_card(d, idx, n_prompts, pa):
    tag = f"PROMPT {idx + 1} OF {n_prompts}"
    d.text((W / 2 - d.textlength(tag, font=F_CARD_TAG) / 2, H / 2 - 78), tag,
           font=F_CARD_TAG, fill=NEUTRAL)
    label = pa["label"]
    d.text((W / 2 - d.textlength(label, font=F_CARD_TITLE) / 2, H / 2 - 52),
           label, font=F_CARD_TITLE, fill=FG)
    d.line([W / 2 - 60, H / 2 + 12, W / 2 + 60, H / 2 + 12], fill=EDGE, width=1)
    sub = f"{pa['prompt_tokens']:,} prompt tokens"
    d.text((W / 2 - d.textlength(sub, font=F_CARD_SUB) / 2, H / 2 + 26), sub,
           font=F_CARD_SUB, fill=DIM)
    dots(d, W / 2, H / 2 + 60, n_prompts, idx, NEUTRAL, EDGE)


def arrow(d, x, y, w, color):
    """A small drawn arrow -- HelveticaNeue has no usable glyph for U+2192."""
    yc = y
    d.line([x, yc, x + w - 6, yc], fill=color, width=2)
    d.polygon([(x + w - 10, yc - 5), (x + w, yc), (x + w - 10, yc + 5)], fill=color)


def final_card(d, results, hero, top):
    key, label, ra, rb, ratio = hero
    # See the note in header() before rewording this line.
    kicker = "SAME GPU · SAME MODEL · THIS REPO'S SERVING STACK"
    y = top
    d.text((W / 2 - d.textlength(kicker, font=F_FINAL_KICKER) / 2, y), kicker,
           font=F_FINAL_KICKER, fill=DIM)

    y += 38
    big = f"{ratio:.1f}x faster"
    d.text((W / 2 - d.textlength(big, font=F_FINAL_HERO) / 2, y), big,
           font=F_FINAL_HERO, fill=REPO)

    y += 100
    a_txt, b_txt, unit = f"{ra:.1f}", f"{rb:.1f}", " tok/s"
    parts = [(label + "  ", FG, F_FINAL_SUB), (a_txt, STOCK, F_FINAL_ROW_B)]
    w1 = sum(d.textlength(t, font=fn) for t, _, fn in parts)
    aw = 26
    w2 = d.textlength(b_txt + unit, font=F_FINAL_ROW_B)
    total = w1 + 10 + aw + 10 + w2
    x = W / 2 - total / 2
    for t, c, fn in parts:
        d.text((x, y), t, font=fn, fill=c)
        x += d.textlength(t, font=fn)
    arrow(d, x + 4, y + 9, aw, DIM)
    x += aw + 14
    d.text((x, y), b_txt, font=F_FINAL_ROW_B, fill=REPO)
    x += d.textlength(b_txt, font=F_FINAL_ROW_B)
    d.text((x, y), unit, font=F_FINAL_ROW_B, fill=FG)

    # every result, on the same fixed scale used throughout the video
    rows_w = 480
    rx = W / 2 - rows_w / 2
    y += 56
    for k, lbl, a_rate, b_rate, r in results:
        row_hi = (k == key)
        name_col = FG if row_hi else DIM
        if row_hi:  # this is the number the whole demo is building to -- call it out
            rrect(d, [rx - 14, y - 8, rx + rows_w + 14, y + 46], 8,
                  fill=EDGE_SOFT, outline=REPO)
        d.text((rx, y), lbl, font=F_FINAL_ROW_B if row_hi else F_FINAL_ROW, fill=name_col)
        rr = f"{r:.1f}x"
        d.text((rx + rows_w - d.textlength(rr, font=F_FINAL_ROW_B), y), rr,
               font=F_FINAL_ROW_B, fill=REPO)
        by = y + 22
        bar_w = rows_w - 60
        meter(d, rx, by, bar_w, 7, a_rate, GLOBAL_MAX, STOCK)
        meter(d, rx, by + 12, bar_w, 7, b_rate, GLOBAL_MAX, REPO)
        y += 58

    legend_y = y + 4
    d.ellipse([rx, legend_y, rx + 8, legend_y + 8], fill=STOCK)
    d.text((rx + 14, legend_y - 3), "stock vLLM", font=F_STAMP, fill=DIM)
    d.ellipse([rx + 110, legend_y, rx + 118, legend_y + 8], fill=REPO)
    d.text((rx + 124, legend_y - 3), "this repo", font=F_STAMP, fill=DIM)


def main():
    global GLOBAL_MAX
    a = json.load(open(sys.argv[1]))
    b = json.load(open(sys.argv[2]))
    out = os.path.expanduser(sys.argv[sys.argv.index("--out") + 1]
                             if "--out" in sys.argv else "demo")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frames_dir = out + "_frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir)

    lanes = (("Stock vLLM", "vllm serve  (no speculative decoding)", STOCK),
             ("This repo", "SPEC=dflash2  DFLASH_TOKENS=15  PREFIX_CACHE=1", REPO))

    prompts = list(zip(map(prepare, a["prompts"]), map(prepare, b["prompts"])))
    n_prompts = len(prompts)
    GLOBAL_MAX = max(max(pa["decode_tok_s"], pb["decode_tok_s"]) for pa, pb in prompts) * 1.08

    tmp = Image.new("RGB", (1, 1))
    dtmp = ImageDraw.Draw(tmp)
    char_w = dtmp.textlength("0" * 20, font=F_BODY) / 20
    cols = int((COL_W - 32) / char_w)

    frame = 0

    def save(img):
        nonlocal frame
        img.save(f"{frames_dir}/f{frame:05d}.png")
        frame += 1

    for idx, (pa, pb) in enumerate(prompts):
        # Title card: without a hard break between prompts the panels just refill
        # and it is not obvious a new prompt started.
        for _ in range(int(TITLE_S * FPS)):
            img = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(img)
            title_card(d, idx, n_prompts, pa)
            stamp(d)
            save(img)

        span = max(pa["tokens"][-1][0], pb["tokens"][-1][0])
        total = span / 1000.0 + HOLD_S
        for f in range(int(total * FPS) + 1):
            t_ms = f * 1000.0 / FPS
            img = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(img)

            header(d, idx, n_prompts, pa)
            panel(d, PAD, lanes[0], pa, t_ms, cols, top=104, bottom=H - 40)
            panel(d, PAD * 2 + COL_W, lanes[1], pb, t_ms, cols, top=104, bottom=H - 40)

            # Only once BOTH lanes have finished: a live ratio would divide the
            # finished lane's final average by the running lane's instantaneous dip
            # and read high (11.4x where the honest answer is 8.9x).
            if t_ms >= span:
                note = f"{pb['decode_tok_s'] / pa['decode_tok_s']:.1f}x faster"
                nw = d.textlength(note, font=F_PILL)
                rrect(d, [W / 2 - nw / 2 - 12, 68, W / 2 + nw / 2 + 12, 94], 13,
                      fill=CARD, outline=REPO)
                d.text((W / 2 - nw / 2, 74), note, font=F_PILL, fill=REPO)
            stamp(d)
            save(img)

    # closing beat: land the headline number the whole demo was building to
    results = [(pa["key"], pa["label"], pa["decode_tok_s"], pb["decode_tok_s"],
                pb["decode_tok_s"] / pa["decode_tok_s"]) for pa, pb in prompts]
    hero = next((r for r in results if r[0] == "copy"), max(results, key=lambda r: r[4]))
    for _ in range(int(FINAL_S * FPS)):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        final_card(d, results, hero, top=56)
        stamp(d)
        save(img)

    pat = f"{frames_dir}/f*.png"
    pal = f"{frames_dir}/pal.png"
    # The design is flat UI, not photographic, so a bigger palette with no
    # dithering renders crisper text *and* compresses smaller than a small
    # dithered one: dither noise is what was costing bytes and blurring glyphs.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-pattern_type", "glob", "-i", pat,
                    "-vf", "palettegen=max_colors=192:stats_mode=diff", pal], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-pattern_type", "glob", "-i", pat, "-i", pal,
                    "-lavfi", "paletteuse=dither=none", out + ".gif"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-pattern_type", "glob", "-i", pat, "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-crf", "23",
                    out + ".mp4"], check=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    for f in (out + ".gif", out + ".mp4"):
        print(f"{f}: {os.path.getsize(f) / 1e6:.2f} MB")


main()
