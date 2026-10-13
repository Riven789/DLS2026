#!/usr/bin/env python3
"""Validate a module's companion video: fetch transcript + metadata, score
content alignment against the module's question/topic.

Usage:
  check_video.py <URL>                          # duration/window check only
  check_video.py <URL> --align "topic text"     # + transcript alignment score
  check_video.py <URL> --align-file <path>      # topic text from file (note.tex etc.)

Alignment = keyword coverage: extract salient terms from the topic text and the
transcript, compute what fraction of topic terms appear in the transcript, plus
Jaccard similarity of the two term sets. Exits 0 only if in window AND (when
--align given) coverage >= 0.5 (override with --min-coverage).
Requires: yt-dlp + youtube-transcript-api (repo .venv) — falls back to metadata-
only mode without them.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

STOP = set("""a about above after again against all am an and any are as at be because been
before being below between both but by can cant could did do does doing dont down during each
few for from further had has have having he her here hers herself him himself his how i if in
into is it its itself just like me more most my myself no nor not now of off on once only or
other our ours ourselves out over own same she should so some such than that the their theirs
them themselves then there these they this those through to too under until up very was we
were what when where which while who whom why will with you your yours yourself yourselves
also may might one two three first second thing things really much many lot get got make
makes made way ways use used using let lets okay video today going go goes talk talked talking
right look looking want wanted know known think thought say says said well actually basically
kind sort little yeah hes shes ive theyre dont thats""".split())


def video_id(arg: str) -> str:
    arg = arg.strip()
    if re.fullmatch(r"[\w-]{11}", arg):
        return arg
    m = (re.search(r"(?:v=|/shorts/|youtu\.be/|/embed/|/live/)([\w-]{11})", arg)
         or re.search(r"^([\w-]{11})$", arg))
    if not m:
        raise ValueError(f"cannot parse video id from: {arg!r}")
    return m.group(1)


def terms(text: str) -> set:
    words = re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", text.lower())
    return {w for w in words if w not in STOP and len(w) > 3}


def fetch_transcript(vid: str):
    """Return (segments, full_text) via youtube-transcript-api, or None."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return None
    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(vid, languages=["en", "en-US", "en-GB"])
        segs = [{"start": s.start, "text": s.text} for s in fetched]
        return segs, " ".join(s["text"] for s in segs)
    except Exception as e:
        print(f"transcript fetch failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def fetch_meta(vid: str):
    """Duration + title via a minimal yt-dlp JSON dump (no download)."""
    cmd = [sys.executable, "-m", "yt_dlp", "--skip-download", "--dump-single-json",
           "--no-warnings", "-q", f"https://www.youtube.com/watch?v={vid}"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if out.returncode == 0 and out.stdout.strip():
            return json.loads(out.stdout)
    except Exception as e:
        print(f"yt-dlp failed: {e}", file=sys.stderr)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--align", help="topic text to align against")
    ap.add_argument("--align-file", help="file with topic text (e.g. note.tex)")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    args = ap.parse_args()

    vid = video_id(args.url)
    meta = fetch_meta(vid)
    if not meta:
        print(json.dumps({"video_id": vid, "error": "metadata unavailable"}))
        return 2

    seconds = meta.get("duration") or 0
    mins = seconds / 60
    in_window = 9.5 <= mins <= 16.5
    result = {
        "video_id": vid,
        "url": f"https://www.youtube.com/watch?v={vid}",
        "title": meta.get("title"),
        "channel": meta.get("channel") or meta.get("uploader"),
        "duration": f"{seconds // 60}:{seconds % 60:02d}",
        "duration_seconds": seconds,
        "in_10_15_window": in_window,
    }

    rc = 0 if in_window else 1

    topic = None
    if args.align_file:
        topic = Path(args.align_file).read_text(errors="ignore")
    elif args.align:
        topic = args.align

    if topic:
        tr = fetch_transcript(vid)
        if tr is None:
            result["alignment"] = {"error": "no transcript available"}
            rc = 2 if rc == 0 else rc
        else:
            segs, full = tr
            t_terms = terms(topic)
            v_terms = terms(full)
            covered = t_terms & v_terms
            coverage = len(covered) / max(1, len(t_terms))
            jaccard = len(covered) / max(1, len(t_terms | v_terms))
            missing = sorted(t_terms - v_terms)[:12]
            result["transcript_chars"] = len(full)
            result["transcript_segments"] = len(segs)
            result["alignment"] = {
                "topic_terms": len(t_terms),
                "coverage": round(coverage, 3),
                "jaccard": round(jaccard, 3),
                "covered_examples": sorted(covered)[:10],
                "missing_terms": missing,
            }
            if coverage < args.min_coverage:
                rc = 1

    print(json.dumps(result, ensure_ascii=False, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
