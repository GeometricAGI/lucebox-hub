#!/usr/bin/env python3
"""Step 1 of Spark calibration: extract a text corpus from agent session logs.

Spark calibrates expert placement on *representative traffic*, so the corpus
should be whatever the model actually serves. This reads agent transcripts from
Claude Code (`~/.claude/projects/*/*.jsonl`) and/or Codex
(`~/.codex/sessions/**/rollout-*.jsonl`), pulls the conversational text (user +
assistant, thinking, tool calls and results), and splits train / held-out by
session so validation never sees calibration data.

    python -m spark.extract_sessions --source both --out-dir ./corpus

Produces corpus/train.jsonl (calibration) and corpus/test.jsonl (held-out),
one JSON-encoded text chunk per line.
"""
import argparse
import hashlib
import json
from pathlib import Path


def claude_text_from_session(path):
    """Claude Code transcript (one JSON object per line) -> conversational text."""
    def blocks(content):
        out = []
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "text" and b.get("text"):
                    out.append(b["text"])
                elif t == "thinking" and b.get("thinking"):
                    out.append(b["thinking"])
                elif t == "tool_use" and b.get("input") is not None:
                    out.append(json.dumps(b["input"])[:4000])
                elif t == "tool_result":
                    cc = b.get("content")
                    if isinstance(cc, str):
                        out.append(cc)
                    elif isinstance(cc, list):
                        for x in cc:
                            if isinstance(x, dict) and x.get("type") == "text":
                                out.append(x.get("text", ""))
        return out

    out = []
    try:
        for ln in Path(path).open():
            try:
                o = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if o.get("type") in ("user", "assistant") and isinstance(o.get("message"), dict):
                out += blocks(o["message"].get("content"))
    except OSError:
        return ""
    return "\n".join(s for s in out if s)


def codex_text_from_session(path):
    """Codex rollout (one JSON object per line) -> conversational text.

    Pulls user + assistant `response_item` content blocks. Skips the developer
    role, which is the repeated system/permissions/skills boilerplate and would
    over-weight the calibration toward instructions rather than real traffic."""
    out = []
    try:
        for ln in Path(path).open():
            try:
                o = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if o.get("type") != "response_item":
                continue
            p = o.get("payload", {})
            if p.get("role") not in ("user", "assistant"):
                continue
            c = p.get("content")
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") in ("input_text", "output_text", "text") and b.get("text"):
                        out.append(b["text"])
            elif isinstance(c, str):
                out.append(c)
    except OSError:
        return ""
    return "\n".join(s for s in out if s)


def chunks_from_text(txt, size, min_size):
    txt = txt.strip()
    for i in range(0, len(txt), size):
        c = txt[i:i + size]
        if len(c) >= min_size:
            yield c


def collect_sessions(args):
    """[(session_file, parser_fn, source_tag), ...] from the enabled sources."""
    out = []
    if args.source in ("claude", "both"):
        for f in sorted(Path(args.claude_dir).glob("*/*.jsonl")):
            out.append((f, claude_text_from_session, "claude"))
    if args.source in ("codex", "both"):
        for f in sorted(Path(args.codex_dir).glob("**/rollout-*.jsonl")):
            out.append((f, codex_text_from_session, "codex"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["claude", "codex", "both"], default="both")
    ap.add_argument("--claude-dir", default=str(Path.home() / ".claude" / "projects"),
                    help="dir with <project>/<session>.jsonl transcripts")
    ap.add_argument("--codex-dir", default=str(Path.home() / ".codex" / "sessions"),
                    help="dir with YYYY/MM/DD/rollout-*.jsonl rollouts")
    ap.add_argument("--out-dir", default="./corpus")
    ap.add_argument("--chunk-chars", type=int, default=2000)
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--per-session", type=int, default=4, help="chunks sampled per session")
    ap.add_argument("--test-frac", type=int, default=5, help="1/N sessions held out (5 = 20%%)")
    ap.add_argument("--train-cap", type=int, default=500)
    ap.add_argument("--test-cap", type=int, default=60)
    args = ap.parse_args()

    sessions = collect_sessions(args)
    if not sessions:
        raise SystemExit(f"no sessions found (source={args.source}; "
                         f"claude={args.claude_dir}, codex={args.codex_dir})")

    train, test = [], []
    per_source = {}
    for f, parse, tag in sessions:
        # split by session-path hash so a whole session is train xor test
        bucket = test if int(hashlib.md5(str(f).encode()).hexdigest(), 16) % args.test_frac == 0 else train
        txt = parse(f)
        if not txt:
            continue
        sess = list(chunks_from_text(txt, args.chunk_chars, args.min_chars))
        if not sess:
            continue
        step = max(1, len(sess) // args.per_session)
        picked = sess[::step][:args.per_session]
        bucket.extend(picked)
        per_source[tag] = per_source.get(tag, 0) + len(picked)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def write(name, items, cap):
        items = items[:cap]
        p = out / name
        with p.open("w") as fo:
            for c in items:
                fo.write(json.dumps(c) + "\n")
        nchar = sum(len(c) for c in items)
        print(f"{p}: {len(items)} chunks, ~{nchar // 1000}K chars (~{nchar // 4000}K tok est)")

    write("train.jsonl", train, args.train_cap)
    write("test.jsonl", test, args.test_cap)
    print(f"sources: {dict(per_source)} over {len(sessions)} sessions "
          f"({args.source}), 1/{args.test_frac} held out by session")


if __name__ == "__main__":
    main()
