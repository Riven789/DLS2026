#!/usr/bin/env python3
"""Generate a narrated module companion video from a slide spec.

Input: a JSON spec file:
{
  "module": "Q01",
  "title": "From Next-Token Prediction to Capabilities",
  "subtitle": "AI Course 2026 - Module Q01 companion video",
  "slides": [
     {"title": "Slide title", "bullets": ["point one", "point two"],
      "narration": "spoken text for this slide (any length)"},
     ...
  ]
}

Output: <outdir>/<module>_companion.mp4 (1920x1080 H.264 + AAC),
plus per-slide assets. Each slide displays for the duration of its narration
(+0.8s pad). Palette = course ink-navy/terracotta. Requires repo .venv with
pillow + edge-tts, and ffmpeg at tools path or PATH.

Also emits <module>_companion_spec_used.json recording title/duration per slide
so the video can be re-checked for the 10-15 min window without re-rendering.
"""
import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FFMPEG = ""
for cand in ("/home/stevens/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg",
             "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
    if Path(cand).exists():
        FFMPEG = cand
        break
if not FFMPEG:
    sys.exit("ffmpeg not found")

FFPROBE = str(Path(FFMPEG).with_name("ffprobe"))

INK = (27, 42, 74)        # 1B2A4A
ACCENT = (140, 79, 43)    # 8C4F2B
CREAM = (250, 246, 238)   # FAF6EE
SKY = (217, 228, 245)     # D9E4F5
W, H = 1920, 1080

VOICE = "en-US-AndrewNeural"


def fonts():
    try:
        big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
        med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
        return big, med, small
    except Exception:
        d = ImageFont.load_default()
        return d, d, d


def wrap(d, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render_slide(slide, idx, total, title, outdir, big, med, small):
    img = Image.new("RGB", (W, H), INK)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 14], fill=ACCENT)
    d.text((100, 90), title, fill=SKY, font=small)
    y = 240
    for ln in wrap(d, slide["title"], big, W - 200):
        d.text((100, y), ln, fill=CREAM, font=big)
        y += 80
    y += 40
    for b in slide.get("bullets", []):
        first = True
        for ln in wrap(d, b, med, W - 260):
            prefix = "• " if first else "  "
            d.text((140, y), prefix + ln, fill=SKY, font=med)
            y += 56
            first = False
        y += 18
    d.text((100, H - 80), f"{idx + 1} / {total}", fill=SKY, font=small)
    p = outdir / f"slide_{idx:03d}.png"
    img.save(p)
    return p


async def tts(text, out):
    for attempt in range(3):
        try:
            c = __import__("edge_tts").Communicate(text, VOICE)
            await c.save(str(out))
            return True
        except Exception as e:
            print(f"tts retry {attempt + 1}: {e}", file=sys.stderr)
    return False


def dur(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip().split("=")[-1])
    except ValueError:
        return 0.0


async def build(spec_path, outdir):
    spec = json.loads(Path(spec_path).read_text())
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    big, med, small = fonts()
    total = len(spec["slides"])

    seg_files = []
    record = []
    for i, slide in enumerate(spec["slides"]):
        png = render_slide(slide, i, total, spec["title"], outdir, big, med, small)
        wav = outdir / f"narr_{i:03d}.ogg"
        ok = await tts(slide["narration"], wav)
        d = (dur(wav) if ok else 3.0) + 0.8
        # still-image segment with duration = narration
        seg = outdir / f"seg_{i:03d}.mp4"
        cmd = [FFMPEG, "-y", "-loop", "1", "-i", str(png),
               "-i", str(wav) if ok else "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono"]
        if ok:
            cmd = [FFMPEG, "-y", "-loop", "1", "-i", str(png), "-i", str(wav),
                   "-t", f"{d:.2f}", "-c:v", "libx264", "-tune", "stillimage",
                   "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "128k",
                   "-shortest", str(seg)]
        else:
            cmd = [FFMPEG, "-y", "-loop", "1", "-i", str(png),
                   "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                   "-t", f"{d:.2f}", "-c:v", "libx264", "-tune", "stillimage",
                   "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "128k",
                   str(seg)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-500:], file=sys.stderr)
            sys.exit(f"segment {i} failed")
        seg_files.append(seg)
        record.append({"idx": i, "title": slide["title"],
                       "narration_seconds": round(d - 0.8, 2), "segment_seconds": round(d, 2)})
        print(f"slide {i + 1}/{total}: {slide['title'][:50]} ({d:.1f}s)")

    lst = outdir / "concat.txt"
    lst.write_text("".join(f"file '{s.name}'\n" for s in seg_files))
    final = outdir / f"{spec['module']}_companion.mp4"
    r = subprocess.run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", str(final)], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-500:], file=sys.stderr)
        sys.exit("concat failed")

    total_s = dur(final)
    (outdir / f"{spec['module']}_companion_spec_used.json").write_text(json.dumps({
        "module": spec["module"], "title": spec["title"],
        "slides": record, "total_seconds": round(total_s, 2),
        "in_10_15_window": 9.5 * 60 <= total_s <= 16.5 * 60,
    }, indent=1))
    print(f"\nFINAL: {final}  duration={total_s:.1f}s ({total_s / 60:.1f} min)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("outdir")
    a = ap.parse_args()
    asyncio.run(build(a.spec, a.outdir))
