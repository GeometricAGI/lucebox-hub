#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"
BASE_URL="http://${HOST}:${PORT}"

RUN_ROOT="${RUN_ROOT:-$REPO_DIR/.artifacts/cpp-http-validation}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$RUN_ROOT/$STAMP}"
SERVER_LOG="${SERVER_LOG:-$ARTIFACT_DIR/dflash_server.log}"
PROBE_JSON="${PROBE_JSON:-$ARTIFACT_DIR/probe.json}"
BENCH_JSON="${BENCH_JSON:-$ARTIFACT_DIR/bench.json}"
HARNESS_WORK_DIR="${HARNESS_WORK_DIR:-$REPO_DIR/.harness-work}"

BUILD_DIR="${BUILD_DIR:-$REPO_DIR/dflash/build}"
DFLASH_SERVER_BIN="${DFLASH_SERVER_BIN:-$BUILD_DIR/dflash_server}"
BUILD_SERVER="${BUILD_SERVER:-1}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-}"
CMAKE_ARGS="${CMAKE_ARGS:-}"

TARGET="${TARGET:-$REPO_DIR/dflash/models/Qwen3.6-27B-Q4_K_M.gguf}"
DRAFT="${DRAFT:-$REPO_DIR/dflash/models/draft/dflash-draft-3.6-q8_0.gguf}"
MODEL_ID="${MODEL_ID:-luce-dflash}"
MAX_CTX="${MAX_CTX:-32768}"
MAX_TOKENS="${MAX_TOKENS:-512}"
VERIFY_MODE="${VERIFY_MODE:-ddtree}"
BUDGET="${BUDGET:-22}"
FA_WINDOW="${FA_WINDOW:-2048}"
CACHE_TYPE_K="${CACHE_TYPE_K:-tq3_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-tq3_0}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
START_TIMEOUT="${START_TIMEOUT:-240}"

CLIENTS="${CLIENTS:-all}"
LONG_PROMPT="${LONG_PROMPT:-0}"
PACKAGE_SMOKE="${PACKAGE_SMOKE:-0}"
INSTALL_PACKAGES="${INSTALL_PACKAGES:-0}"
RUN_BENCH="${RUN_BENCH:-1}"
BENCH_SUITE="${BENCH_SUITE:-all}"
BENCH_MODEL="${BENCH_MODEL:-$MODEL_ID}"
BENCH_N_SAMPLE="${BENCH_N_SAMPLE:-}"
BENCH_PROMPTS_DIR="${BENCH_PROMPTS_DIR:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_SERVER="${KEEP_SERVER:-0}"

SERVER_PID=""

log() {
  echo "[cpp-http-validation] $*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

tail_server_log() {
  if [[ -f "$SERVER_LOG" ]]; then
    echo "--- dflash_server.log tail ---" >&2
    tail -n 160 "$SERVER_LOG" >&2 || true
  fi
}

cleanup() {
  if [[ "$KEEP_SERVER" == "1" ]]; then
    return
  fi
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

parse_words() {
  local value="$1"
  if [[ -z "$value" ]]; then
    return
  fi
  # Shell-style splitting is intentional for simple env-provided flag lists.
  read -r -a _PARSED_WORDS <<<"$value"
  printf '%s\n' "${_PARSED_WORDS[@]}"
}

build_server() {
  if [[ "$BUILD_SERVER" != "1" ]]; then
    return
  fi

  local cmake_config=(
    -B "$BUILD_DIR"
    -S "$REPO_DIR/dflash"
    -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"
  )
  if [[ -n "$CMAKE_CUDA_ARCHITECTURES" ]]; then
    cmake_config+=("-DCMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES")
  fi
  while IFS= read -r word; do
    cmake_config+=("$word")
  done < <(parse_words "$CMAKE_ARGS")

  log "configuring dflash_server"
  cmake "${cmake_config[@]}"

  log "building dflash_server"
  cmake --build "$BUILD_DIR" --target dflash_server -j"$BUILD_JOBS"
}

assert_inputs() {
  [[ -x "$DFLASH_SERVER_BIN" ]] || {
    echo "dflash_server not found or not executable: $DFLASH_SERVER_BIN" >&2
    exit 1
  }
  [[ -f "$TARGET" ]] || {
    echo "target model not found: $TARGET" >&2
    exit 1
  }
  [[ -f "$DRAFT" ]] || {
    echo "draft model not found: $DRAFT" >&2
    exit 1
  }
}

start_server() {
  local extra_args=()
  while IFS= read -r word; do
    extra_args+=("$word")
  done < <(parse_words "$EXTRA_SERVER_ARGS")

  local verify_args=()
  if [[ "$VERIFY_MODE" == "ddtree" ]]; then
    verify_args=(--ddtree --ddtree-budget "$BUDGET")
  fi

  local fa_args=()
  if [[ -n "$FA_WINDOW" ]] && [[ "$FA_WINDOW" != "0" ]]; then
    fa_args=(--fa-window "$FA_WINDOW")
  fi

  export DFLASH27B_KV_K="$CACHE_TYPE_K"
  export DFLASH27B_KV_V="$CACHE_TYPE_V"

  log "starting native server on $BASE_URL"
  "$DFLASH_SERVER_BIN" "$TARGET" \
    --draft "$DRAFT" \
    --host "$HOST" \
    --port "$PORT" \
    --max-ctx "$MAX_CTX" \
    --max-tokens "$MAX_TOKENS" \
    --model-name "$MODEL_ID" \
    "${verify_args[@]}" \
    "${fa_args[@]}" \
    "${extra_args[@]}" \
    >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
}

wait_for_health() {
  local deadline=$((SECONDS + START_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
      log "server is healthy"
      return 0
    fi
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server exited before becoming healthy" >&2
      tail_server_log
      return 1
    fi
    sleep 1
  done

  echo "timed out waiting for $BASE_URL/health" >&2
  tail_server_log
  return 1
}

run_probe() {
  local cmd=(
    "$PYTHON_BIN"
    "$REPO_DIR/harness/client_test_runner.py"
    --work-dir "$HARNESS_WORK_DIR"
    probe
    --url "$BASE_URL"
    --clients "$CLIENTS"
    --json-out "$PROBE_JSON"
  )

  if [[ "$LONG_PROMPT" == "1" ]]; then
    cmd+=(--long-prompt)
  fi
  if [[ "$PACKAGE_SMOKE" == "1" ]]; then
    cmd+=(--package-smoke)
  fi
  if [[ "$INSTALL_PACKAGES" == "1" ]]; then
    cmd+=(--install-packages)
  fi

  log "running harness probe"
  "${cmd[@]}"
}

run_bench() {
  if [[ "$RUN_BENCH" != "1" ]]; then
    return
  fi

  local cmd=(
    "$PYTHON_BIN"
    "$REPO_DIR/harness/client_test_runner.py"
    bench
    --url "$BASE_URL"
    --suite "$BENCH_SUITE"
    --model "$BENCH_MODEL"
    --json-out "$BENCH_JSON"
  )

  if [[ -n "$BENCH_N_SAMPLE" ]]; then
    cmd+=(--n-sample "$BENCH_N_SAMPLE")
  fi
  if [[ -n "$BENCH_PROMPTS_DIR" ]]; then
    cmd+=(--prompts-dir "$BENCH_PROMPTS_DIR")
  fi

  log "running harness bench"
  "${cmd[@]}"
}

print_summary() {
  log "artifacts: $ARTIFACT_DIR"
  log "server log: $SERVER_LOG"
  log "probe json: $PROBE_JSON"
  if [[ "$RUN_BENCH" == "1" ]]; then
    log "bench json: $BENCH_JSON"
  fi
}

main() {
  mkdir -p "$ARTIFACT_DIR" "$HARNESS_WORK_DIR"

  log "repo: $REPO_DIR"
  log "binary: $DFLASH_SERVER_BIN"
  log "target: $TARGET"
  log "draft: $DRAFT"
  log "clients: $CLIENTS"
  log "bench suite: $BENCH_SUITE"

  if [[ "$DRY_RUN" == "1" ]]; then
    print_summary
    return 0
  fi

  require_cmd curl
  require_cmd "$PYTHON_BIN"
  if [[ "$BUILD_SERVER" == "1" ]]; then
    require_cmd cmake
  fi

  build_server
  assert_inputs
  start_server
  wait_for_health

  if ! run_probe; then
    tail_server_log
    return 1
  fi
  if ! run_bench; then
    tail_server_log
    return 1
  fi

  print_summary
}

main "$@"
