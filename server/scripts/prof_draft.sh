#!/usr/bin/env bash
# End-to-end profiler for the DFlash draft forward (src/draft/draft_graph.cpp).
#
# Builds the prof_draft_graph harness, runs it once to get the static op
# histogram + wall latency, then runs it under Nsight Systems and post-processes
# the kernel / idle / memcpy / fusion breakdown.
#
# Usage:
#   scripts/prof_draft.sh [--ctx-len N] [--iters N] [--warmup N]
#                         [--draft PATH] [--graphs] [--outdir DIR] [--no-build]
#
# Defaults: ctx-len 512, iters 50, warmup 10, CUDA graphs DISABLED (so individual
# kernels and the gaps between them are visible — that is how idle time is
# measured). Pass --graphs to profile the production CUDA-graph path instead.
#
# Env overrides: DFLASH_DRAFT (draft model dir or .safetensors), DFLASH_BIN.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CTX=512; ITERS=50; WARMUP=10; GRAPHS=0; BUILD=1
OUTDIR="/tmp/prof_draft"
DRAFT="${DFLASH_DRAFT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ctx-len) CTX="$2"; shift 2;;
    --iters)   ITERS="$2"; shift 2;;
    --warmup)  WARMUP="$2"; shift 2;;
    --draft)   DRAFT="$2"; shift 2;;
    --graphs)  GRAPHS=1; shift;;
    --outdir)  OUTDIR="$2"; shift 2;;
    --no-build) BUILD=0; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

BIN="${DFLASH_BIN:-build/prof_draft_graph}"
mkdir -p "$OUTDIR"

# ── Resolve draft model ──────────────────────────────────────────────────
resolve_draft() {
  local d="$DRAFT"
  if [[ -z "$d" ]]; then
    for c in models/draft35 models/draft; do [[ -e "$c" ]] && { d="$c"; break; }; done
  fi
  [[ -z "$d" ]] && return
  # The harness loads safetensors weights; resolve a directory to the file.
  if [[ -d "$d" ]]; then
    find "$d" -maxdepth 2 -name '*.safetensors' | head -1
  else
    echo "$d"
  fi
}
DRAFT_PATH="$(resolve_draft)"
if [[ -z "${DRAFT_PATH:-}" || ! -e "$DRAFT_PATH" ]]; then
  echo "!! draft model not found. Set DFLASH_DRAFT or --draft <path>." >&2; exit 1
fi
echo "[prof_draft] draft = $DRAFT_PATH"

# ── Build ────────────────────────────────────────────────────────────────
if [[ "$BUILD" == 1 ]]; then
  echo "[prof_draft] building prof_draft_graph ..."
  cmake --build build --target prof_draft_graph -j >/dev/null
fi
[[ -x "$BIN" ]] || { echo "!! $BIN not built" >&2; exit 1; }

GRAPH_ENV=()
if [[ "$GRAPHS" == 0 ]]; then
  GRAPH_ENV=(GGML_CUDA_DISABLE_GRAPHS=1)
  echo "[prof_draft] CUDA graphs DISABLED (per-kernel visibility; idle = real launch gaps)"
else
  echo "[prof_draft] CUDA graphs ENABLED (production path)"
fi

HARNESS_ARGS=("$DRAFT_PATH" --ctx-len "$CTX" --iters "$ITERS" --warmup "$WARMUP")
FORWARDS=$((ITERS + WARMUP))

# ── Pass 1: plain run — static op histogram + wall latency ────────────────
echo
echo "########## pass 1: static graph + wall latency ##########"
PLAIN_LOG="$OUTDIR/plain.log"
env "${GRAPH_ENV[@]}" "$BIN" "${HARNESS_ARGS[@]}" | tee "$PLAIN_LOG"
WALL_MS="$(grep -oE 'median=[0-9.]+ ms' "$PLAIN_LOG" | grep -oE '[0-9.]+' | head -1 || echo 0)"

# ── Pass 2: nsys timeline ─────────────────────────────────────────────────
echo
echo "########## pass 2: nsys profile ##########"
REP="$OUTDIR/draft"
rm -f "$REP".nsys-rep "$REP".sqlite
env "${GRAPH_ENV[@]}" nsys profile \
    --trace=cuda,nvtx \
    --force-overwrite=true \
    --output "$REP" \
    "$BIN" "${HARNESS_ARGS[@]}" >/dev/null
echo "[prof_draft] report: $REP.nsys-rep"

# ── Extract stats to CSV (report names vary slightly across nsys versions) ─
gen_csv() {  # $1=report  $2=outfile
  nsys stats --report "$1" --format csv --force-export=true "$REP.nsys-rep" \
      > "$2" 2>/dev/null || true
}
KERN_CSV="$OUTDIR/kern_sum.csv"; MEM_CSV="$OUTDIR/mem_sum.csv"
gen_csv cuda_gpu_kern_sum "$KERN_CSV"
gen_csv cuda_gpu_mem_time_sum "$MEM_CSV"
# nsys CSV to stdout prepends a comment/blank line before the header; strip to header.
for f in "$KERN_CSV" "$MEM_CSV"; do
  [[ -s "$f" ]] && sed -i -n '/Time (%)\|Total Time/,$p' "$f" 2>/dev/null || true
done

# ── Analyze ────────────────────────────────────────────────────────────────
echo
echo "########## breakdown ##########"
PY="${DFLASH_PY:-python3}"
"$PY" scripts/prof_draft_analyze.py \
    --kern-sum "$KERN_CSV" --mem-sum "$MEM_CSV" \
    --forwards "$FORWARDS" --wall-ms "${WALL_MS:-0}"

echo
echo "[prof_draft] open the full timeline with:  nsys-ui $REP.nsys-rep"
echo "[prof_draft] deep single-kernel metrics (e.g. the dominant GEMM):"
echo "    ncu --set full --launch-count 1 --kernel-name-base demangled \\"
echo "        -k regex:'gemm|cutlass' -o $OUTDIR/draft_ncu \\"
echo "        $BIN $DRAFT_PATH --ctx-len $CTX --iters 1 --warmup 0"
