#!/usr/bin/env python3
"""End-to-end top-k/top-p LM-head calibration driver (GPU).

Single entry point that replaces the run_calib.sh + aggregate.py pair. It can
download the models, tokenize prompts, build the CUDA `test_dflash` binary,
run the calibration per-prompt on a chosen GPU, and aggregate the
`[topk-calib]` coverage/mass tables into a report (stdout + JSON).

Why a *driver* and not a pure-Python re-implementation: the DFlash draft is an
EAGLE-style head that consumes the target's mid-stack hidden states and shares
the target's LM head (see server/src/common/dflash_draft_graph.h). Stock
llama.cpp / llama-cpp-python cannot load or run it, so the only faithful way to
produce the draft candidate set is this repo's C++ `test_dflash`. This script
drives that binary on the GPU (Blackwell sm_100 by default) and does all the
orchestration + analysis in Python.

Typical use (Blackwell B200, GPU 0):

    # one-shot full pipeline (download ~16GB target + draft, prep, build, run)
    python3 calib/calibrate.py --all --prep 200

    # later re-runs once models/bins/binary exist
    python3 calib/calibrate.py --gpu 0 --ngen 256 --json calib/out/calib.json

By default (no stage flags) it only runs+aggregates and tells you which flag to
add if a prerequisite is missing. Stage flags (--download/--prep/--build) opt
in to the heavy steps; --all enables them all.

Deps for --prep: transformers, datasets, jinja2. Deps for --download: the `hf`
CLI (huggingface_hub). The run/aggregate stages are stdlib-only.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEF_TARGET = REPO / "server/models/Qwen3.6-27B-Q4_K_M.gguf"
DEF_DRAFT = REPO / "server/models/draft/dflash-draft-3.6-q4_k_m.gguf"
DEF_BIN = REPO / "server/build/test_dflash"
DEF_PROMPTS = REPO / "calib"
DEF_OUTDIR = REPO / "calib/out"

# HF repos (see README.md): target + DFlash draft.
TARGET_REPO = "unsloth/Qwen3.6-27B-GGUF"
TARGET_FILE = "Qwen3.6-27B-Q4_K_M.gguf"
DRAFT_REPO = "Lucebox/Qwen3.6-27B-DFlash-GGUF"
DRAFT_FILE = "dflash-draft-3.6-q4_k_m.gguf"

SET_METRICS = [
    "greedy (target argmax)",
    "sampling top_k=8",
    "sampling top_k=20 (Qwen def)",
    "sampling top_p=0.95 nucleus",
]


def run(cmd, **kw):
    """Echo + run a command, raising on failure."""
    print("  $ " + " ".join(str(c) for c in cmd), file=sys.stderr)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


# ── stage: download ──────────────────────────────────────────────────────────
def stage_download(target: Path, draft: Path):
    if not shutil.which("hf"):
        sys.exit("error: `hf` CLI not found. `pip install huggingface_hub` (set "
                 "HF_TOKEN to avoid rate limits), or download the GGUFs manually.")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] target -> {target.parent}", file=sys.stderr)
        run(["hf", "download", TARGET_REPO, TARGET_FILE, "--local-dir", target.parent])
    else:
        print(f"[download] target present, skip ({target})", file=sys.stderr)
    if not draft.exists():
        draft.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] draft -> {draft.parent}", file=sys.stderr)
        run(["hf", "download", DRAFT_REPO, DRAFT_FILE, "--local-dir", draft.parent])
    else:
        print(f"[download] draft present, skip ({draft})", file=sys.stderr)


# ── stage: prep prompts ──────────────────────────────────────────────────────
def stage_prep(prompts_dir: Path, n: int, thinking: str):
    print(f"[prep] tokenizing {n} prompts -> {prompts_dir}", file=sys.stderr)
    run([sys.executable, str(REPO / "calib/prep_prompts.py"),
         "--n", n, "--out", prompts_dir, "--thinking", thinking])


# ── stage: build ─────────────────────────────────────────────────────────────
def stage_build(binary: Path, cuda_arch: int):
    build_dir = REPO / "server/build"
    print("[build] ensuring git submodules (deps/llama.cpp, ...)", file=sys.stderr)
    run(["git", "-C", REPO, "submodule", "update", "--init", "--recursive"])
    print(f"[build] configuring test_dflash (sm_{cuda_arch}, NCCL OFF)", file=sys.stderr)
    run(["cmake", "-B", build_dir, "-S", REPO / "server", "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release",
         f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
         "-DGGML_CUDA_NCCL=OFF"])
    run(["cmake", "--build", build_dir, "--target", "test_dflash", "-j"])
    if not binary.exists():
        sys.exit(f"error: build finished but {binary} not found")


# ── stage: run calibration over all prompt bins ──────────────────────────────
def stage_run(args, bins, outdir: Path):
    """Run test_dflash per prompt; return {name: full_log_text} for parsing."""
    outdir.mkdir(parents=True, exist_ok=True)
    logs = {}
    for i, p in enumerate(bins, 1):
        name = p.stem
        out_bin = outdir / f"{name}_out.bin"
        log_path = outdir / f"{name}.log"
        if args.skip_existing and log_path.exists():
            print(f"[run {i}/{len(bins)}] {name}: cached log, skip", file=sys.stderr)
            logs[name] = log_path.read_text()
            continue
        print(f"[run {i}/{len(bins)}] {name} (gpu {args.gpu})", file=sys.stderr)
        # DFLASH27B_PREFILL_UBATCH=1: token-by-token prefill. The batched prefill
        # attention path emits all-NaN logits for some (n_tokens, kv_start) configs,
        # which makes the greedy prefill argmax fall to token 0 and the run abort
        # silently (~76% of prompts, biased toward code). Token-by-token prefill is
        # numerically correct and only marginally slower (prefill ≪ the 256-tok gen).
        env = {"CUDA_VISIBLE_DEVICES": str(args.gpu), "DFLASH_TOPK_CALIB": "1",
               "DFLASH27B_PREFILL_UBATCH": str(args.prefill_ubatch)}
        cmd = [args.bin, args.target, args.draft, p, args.ngen, out_bin,
               "--fast-rollback", f"--max-ctx={args.maxctx}"]
        # Inherit current env, then override the two calib vars.
        full_env = {**os.environ, **env}
        proc = subprocess.run([str(c) for c in cmd], env=full_env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True)
        log_path.write_text(proc.stdout)
        logs[name] = proc.stdout
        if proc.returncode != 0 or "[topk-calib]" not in proc.stdout:
            tail = "\n".join(proc.stdout.splitlines()[-8:])
            print(f"  ! {name} produced no calib table (rc={proc.returncode}). "
                  f"tail:\n{tail}", file=sys.stderr)
    return logs


# ── parse one prompt's [topk-calib] block ────────────────────────────────────
def parse_block(text: str):
    """Return dict for one calib run, or None if no table present.

    {positions, mean_rank, max_rank, vocab,
     set: {metric: {M: pct}}, mass: {M: pct}, mass_bad: {M: (bad, total)}}
    """
    m = re.search(r"\[topk-calib\] positions=(\d+)\s+mean_rank=([\d.]+)\s+"
                  r"max_rank=(\d+)\s+\(vocab=(\d+)\)", text)
    if not m:
        return None
    res = {
        "positions": int(m.group(1)),
        "mean_rank": float(m.group(2)),
        "max_rank": int(m.group(3)),
        "vocab": int(m.group(4)),
        "set": {k: {} for k in SET_METRICS},
        "mass": {},
        "mass_bad": {},
    }
    cur = None
    for line in text.splitlines():
        sc = re.search(r"--- (.+?): set-coverage", line)
        if sc:
            label = sc.group(1)
            cur = label if label in res["set"] else None
            continue
        if "MASS coverage" in line:
            cur = "__mass__"
            continue
        cm = re.search(r"coverage@M=(\d+)\s*:\s*([\d.]+)%", line)
        if cm and cur and cur != "__mass__":
            res["set"][cur][int(cm.group(1))] = float(cm.group(2))
        mm = re.search(r"mass@M=(\d+)\s*:\s*([\d.]+)%.*?<99% covered:\s*(\d+)/(\d+)",
                       line)
        if mm:
            M = int(mm.group(1))
            res["mass"][M] = float(mm.group(2))
            res["mass_bad"][M] = (int(mm.group(3)), int(mm.group(4)))
    return res


# ── aggregate + report ───────────────────────────────────────────────────────
def aggregate(per_prompt: dict):
    """Position-weighted aggregate across prompts."""
    ok = {n: b for n, b in per_prompt.items() if b}
    totpos = sum(b["positions"] for b in ok.values())
    if totpos == 0:
        return None
    Ms = sorted({M for b in ok.values() for M in b["mass"]}
                | {M for b in ok.values() for d in b["set"].values() for M in d})

    # Accumulate sums + per-cell weight so cells no prompt reported stay None
    # (rather than a misleading 0.0). In real output every metric shares the
    # same M-grid, so all cells are populated.
    set_sum = {m: {M: 0.0 for M in Ms} for m in SET_METRICS}
    set_w = {m: {M: 0.0 for M in Ms} for m in SET_METRICS}
    mass_sum = {M: 0.0 for M in Ms}
    mass_w = {M: 0.0 for M in Ms}
    mass_bad = {M: 0 for M in Ms}
    rank_w = 0.0
    max_rank = 0
    for b in ok.values():
        w = b["positions"]
        rank_w += b["mean_rank"] * w
        max_rank = max(max_rank, b["max_rank"])
        for m in SET_METRICS:
            for M, v in b["set"][m].items():
                set_sum[m][M] += v * w
                set_w[m][M] += w
        for M, v in b["mass"].items():
            mass_sum[M] += v * w
            mass_w[M] += w
        for M, (bad, _) in b["mass_bad"].items():
            mass_bad[M] += bad
    set_agg = {m: {M: (set_sum[m][M] / set_w[m][M] if set_w[m][M] else None)
                   for M in Ms} for m in SET_METRICS}
    mass_agg = {M: (mass_sum[M] / mass_w[M] if mass_w[M] else None) for M in Ms}
    return {
        "prompts": len(ok), "failed": len(per_prompt) - len(ok),
        "total_positions": totpos, "mean_rank": rank_w / totpos,
        "max_rank": max_rank, "Ms": Ms,
        "set": set_agg, "mass": mass_agg, "mass_bad": mass_bad,
        "per_prompt": {n: {"positions": b["positions"], "mean_rank": b["mean_rank"],
                           "max_rank": b["max_rank"]} for n, b in ok.items()},
    }


def print_report(agg):
    Ms = agg["Ms"]
    print(f"\n=== top-k head calibration aggregate ===")
    print(f"prompts={agg['prompts']} (failed={agg['failed']})  "
          f"total_positions={agg['total_positions']}  "
          f"mean_rank={agg['mean_rank']:.2f}  max_rank={agg['max_rank']}\n")
    def cell(v, fmt):
        return f"{'-':>10}" if v is None else f"{v:10.{fmt}f}"
    hdr = "metric".ljust(28) + "".join(f"{M:>10}" for M in Ms)
    print("-- set-coverage (entire set within draft top-M, position-weighted %) --")
    print(hdr)
    for m in SET_METRICS:
        row = m[:27].ljust(28) + "".join(cell(agg['set'][m][M], 2) for M in Ms)
        print(row)
    print("\n-- top_p=0.95 nucleus MASS coverage (mean %, + positions <99% covered) --")
    print("mass %".ljust(28) + "".join(cell(agg['mass'][M], 3) for M in Ms))
    print("bad(<99%)".ljust(28) + "".join(f"{agg['mass_bad'][M]:10d}" for M in Ms))
    print("\n-- per-prompt --")
    for n, s in sorted(agg["per_prompt"].items()):
        print(f"  {n:<16} pos={s['positions']:<5} mean_rank={s['mean_rank']:7.2f} "
              f"max_rank={s['max_rank']}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path, default=DEF_TARGET)
    ap.add_argument("--draft", type=Path, default=DEF_DRAFT)
    ap.add_argument("--bin", dest="bin", type=Path, default=DEF_BIN)
    ap.add_argument("--prompts", type=Path, default=DEF_PROMPTS,
                    help="dir holding tokenized *.bin prompts")
    ap.add_argument("--out-dir", type=Path, default=DEF_OUTDIR)
    ap.add_argument("--gpu", type=int, default=0, help="CUDA_VISIBLE_DEVICES index")
    ap.add_argument("--ngen", type=int, default=256)
    ap.add_argument("--maxctx", type=int, default=1024)
    ap.add_argument("--prefill-ubatch", type=int, default=1,
                    help="DFLASH27B_PREFILL_UBATCH. Default 1 (token-by-token) avoids "
                         "the batched-prefill NaN bug that aborts ~76%% of prompts. Set "
                         ">1 to use batched prefill (faster, but reintroduces the bug).")
    ap.add_argument("--cuda-arch", type=int, default=100,
                    help="CMAKE_CUDA_ARCHITECTURES for --build (Blackwell B200=100)")
    ap.add_argument("--json", type=Path, default=None, help="write aggregate JSON here")
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse a prompt's cached .log instead of re-running it")
    # stage opt-ins
    ap.add_argument("--download", action="store_true", help="hf download missing GGUFs")
    ap.add_argument("--prep", type=int, metavar="N", default=0,
                    help="tokenize N prompts via prep_prompts.py before running")
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    ap.add_argument("--build", action="store_true", help="cmake build test_dflash")
    ap.add_argument("--all", action="store_true",
                    help="enable --download + --build (use with --prep N)")
    ap.add_argument("--no-run", action="store_true",
                    help="only run enabled prep/build/download stages, skip calibration")
    args = ap.parse_args()

    if args.all:
        args.download = True
        args.build = True

    if args.download:
        stage_download(args.target, args.draft)
    if args.prep:
        stage_prep(args.prompts, args.prep, args.thinking)
    if args.build:
        stage_build(args.bin, args.cuda_arch)

    if args.no_run:
        return

    # prerequisite checks with actionable hints
    if not args.target.exists():
        sys.exit(f"error: target not found: {args.target}\n  add --download")
    if not args.draft.exists():
        sys.exit(f"error: draft not found: {args.draft}\n  add --download")
    if not args.bin.exists():
        sys.exit(f"error: binary not found: {args.bin}\n  add --build")
    bins = sorted(args.prompts.glob("*.bin"))
    if not bins:
        sys.exit(f"error: no *.bin prompts in {args.prompts}\n  add --prep N")
    print(f"[run] {len(bins)} prompts, ngen={args.ngen}, gpu={args.gpu}", file=sys.stderr)

    logs = stage_run(args, bins, args.out_dir)
    per_prompt = {n: parse_block(t) for n, t in logs.items()}
    agg = aggregate(per_prompt)
    if not agg:
        sys.exit("error: no calibration tables parsed from any prompt run")
    print_report(agg)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(agg, indent=2))
        print(f"\n[json] wrote {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
