#!/usr/bin/env python3
"""Post-process `nsys stats` CSV reports for the DFlash draft forward.

Consumes the cuda_gpu_kern_sum and cuda_gpu_mem_time_sum CSVs emitted by
scripts/prof_draft.sh and prints:
  - per-forward wall / GPU-kernel / memcpy / idle breakdown (idle = launch gaps
    + sync stalls, i.e. where the GPU is starved),
  - the longest-running kernels,
  - a kernel-family rollup (mul_mat / rms_norm / rope / flash_attn / copy / ...)
    to expose fusion candidates,
  - heuristic fusion notes.

All times normalised to per-forward by dividing nsys totals by the number of
forwards profiled (warmup + measured run identical work, so the per-forward
average is stable).
"""
import argparse
import csv
import re
import sys
from collections import defaultdict


def _read_csv(path):
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def _num(row, *keys):
    """First parseable numeric column among keys (nsys column names drift)."""
    for k in keys:
        if k in row and row[k] not in (None, "", "NA"):
            try:
                return float(str(row[k]).replace(",", ""))
            except ValueError:
                pass
    return 0.0


def _name(row):
    for k in ("Name", "Operation", "name"):
        if k in row and row[k]:
            return row[k]
    return "?"


# Map a raw CUDA kernel name to a coarse ggml-op family for fusion rollup.
_FAMILY = [
    ("convert / cast",   re.compile(r"convert", re.I)),  # bf16<->f32 dequant/cast
    ("mul_mat / GEMM",   re.compile(r"mul_mat|gemm|cutlass|cublas|ampere|wmma|s16816|tn_align", re.I)),
    ("flash_attn",       re.compile(r"flash.?attn|fattn", re.I)),
    ("rms_norm",         re.compile(r"rms.?norm", re.I)),
    ("rope",             re.compile(r"rope", re.I)),
    ("silu / unary",     re.compile(r"silu|gelu|unary_op|op_silu|op_gelu", re.I)),
    ("mul / bin_bcast",  re.compile(r"bin_bcast|\bmul_f|op_mul|\bmul\b", re.I)),
    ("add",              re.compile(r"\badd_|op_add|\badd\b", re.I)),
    ("concat",           re.compile(r"concat", re.I)),
    ("copy / cont / dup",re.compile(r"\bcpy|\bcont|\bdup|copy", re.I)),
]


def family(name):
    for label, rx in _FAMILY:
        if rx.search(name):
            return label
    return "other"


def fmt_ms(ns):
    return ns / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kern-sum", required=True, help="cuda_gpu_kern_sum CSV")
    ap.add_argument("--mem-sum", default="", help="cuda_gpu_mem_time_sum CSV")
    ap.add_argument("--forwards", type=int, required=True,
                    help="total forwards profiled (warmup + measured)")
    ap.add_argument("--wall-ms", type=float, default=0.0,
                    help="measured mean wall ms/forward from the harness (for idle calc)")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    kern = _read_csv(args.kern_sum)
    if not kern:
        print(f"!! no kernel rows in {args.kern_sum} — did nsys produce a report?",
              file=sys.stderr)
        sys.exit(1)
    mem = _read_csv(args.mem_sum) if args.mem_sum else []

    F = max(1, args.forwards)

    rows = []
    for r in kern:
        total_ns = _num(r, "Total Time (ns)", "Total Time", "Total")
        inst = _num(r, "Instances", "Count", "Num Calls")
        rows.append((_name(r), total_ns, inst))
    rows.sort(key=lambda x: x[1], reverse=True)

    kern_total_ns = sum(t for _, t, _ in rows)
    mem_total_ns = sum(_num(r, "Total Time (ns)", "Total Time", "Total") for r in mem)

    kern_pf = fmt_ms(kern_total_ns) / F
    mem_pf = fmt_ms(mem_total_ns) / F

    print("=" * 78)
    print(f"DFlash draft forward — per-forward breakdown (over {F} forwards)")
    print("=" * 78)
    print(f"  GPU kernel time : {kern_pf:8.3f} ms/fwd   ({len(rows)} distinct kernels, "
          f"{int(sum(i for *_, i in rows)/F)} launches/fwd)")
    # memcpy/memset run async on copy streams and largely overlap compute, so it
    # is reported alongside — not subtracted from — the wall/idle budget.
    print(f"  memcpy/memset   : {mem_pf:8.3f} ms/fwd   (concurrent; overlaps compute)")
    if args.wall_ms > 0:
        # Idle = wall the GPU spent not executing kernels = launch latency + sync
        # stalls between kernels. Clamp at 0 (kernel total spans warmup+measured,
        # wall is measured-only, so tiny noise can push the diff slightly negative).
        idle = max(0.0, args.wall_ms - kern_pf)
        idle_pct = 100.0 * idle / args.wall_ms
        busy_pct = min(100.0, 100.0 * kern_pf / args.wall_ms)
        print(f"  wall (harness)  : {args.wall_ms:8.3f} ms/fwd")
        print(f"  -> GPU busy     : {busy_pct:6.1f} %  (kernels on the compute stream)")
        print(f"  -> IDLE (gaps)  : {idle:8.3f} ms/fwd  ({idle_pct:5.1f} %)  "
              f"[launch latency + sync stalls; ~0 => kernel-bound, not launch-bound]")

    print("\n" + "-" * 78)
    print(f"Longest kernels (top {args.top}, per-forward)")
    print("-" * 78)
    print(f"  {'ms/fwd':>9}  {'% gpu':>6}  {'launch/fwd':>10}  kernel")
    for name, total_ns, inst in rows[: args.top]:
        pf = fmt_ms(total_ns) / F
        pct = 100.0 * total_ns / kern_total_ns if kern_total_ns else 0.0
        print(f"  {pf:9.3f}  {pct:6.1f}  {inst / F:10.1f}  {name[:60]}")

    print("\n" + "-" * 78)
    print("Kernel-family rollup (fusion candidates)")
    print("-" * 78)
    fam_t = defaultdict(float)
    fam_n = defaultdict(float)
    for name, total_ns, inst in rows:
        fam_t[family(name)] += total_ns
        fam_n[family(name)] += inst
    print(f"  {'ms/fwd':>9}  {'% gpu':>6}  {'launch/fwd':>10}  family")
    for fam in sorted(fam_t, key=lambda k: fam_t[k], reverse=True):
        pf = fmt_ms(fam_t[fam]) / F
        pct = 100.0 * fam_t[fam] / kern_total_ns if kern_total_ns else 0.0
        print(f"  {pf:9.3f}  {pct:6.1f}  {fam_n[fam] / F:10.1f}  {fam}")

    # Heuristic fusion notes keyed off launch counts per forward.
    print("\n" + "-" * 78)
    print("Notes")
    print("-" * 78)
    lpf = {fam: fam_n[fam] / F for fam in fam_n}
    notes = []
    if lpf.get("rms_norm", 0) >= 2 and lpf.get("mul / bin_bcast", 0) >= 2:
        notes.append("rms_norm is followed by a separate elementwise mul (norm-weight "
                     "scale) on Q/K/h paths — a fused rms_norm*weight kernel removes "
                     f"~{lpf['rms_norm']:.0f} mul launches/fwd.")
    if lpf.get("copy / cont / dup", 0) >= 3:
        notes.append(f"{lpf['copy / cont / dup']:.0f} copy/cont launches/fwd — the "
                     "permute+cont before flash_attn_ext materialises Q/K/V; a "
                     "layout-aware attn or fused transpose avoids these.")
    if lpf.get("mul_mat / GEMM", 0) >= 8:
        notes.append(f"{lpf['mul_mat / GEMM']:.0f} GEMM launches/fwd — K/V are computed "
                     "as 4 separate mul_mat (Kctx,Kn,Vctx,Vn); batching wk/wv over "
                     "[ctx||noise] or a fused QKV projection cuts launch count.")
    if args.wall_ms > 0 and (args.wall_ms - kern_pf) / args.wall_ms > 0.30:
        notes.append("Idle >30%% of wall — the forward is launch-bound; enabling CUDA "
                     "graphs (drop GGML_CUDA_DISABLE_GRAPHS=1) or fusing small ops "
                     "will recover most of it.")
    if not notes:
        notes.append("No obvious launch-bound / fusion red flags at this ctx_len.")
    for i, n in enumerate(notes, 1):
        print(f"  {i}. {n}")
    print()


if __name__ == "__main__":
    main()
