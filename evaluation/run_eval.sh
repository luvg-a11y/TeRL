#!/usr/bin/env bash
# Sweep: serve each model with SGLang, evaluate on each benchmark with Harbor, tear down.
#
# One model is resident at a time — these do not co-fit on a single node.
#
#   ./run_eval.sh                 # full sweep
#   ./run_eval.sh --list          # show the plan, touch nothing
#   ./run_eval.sh --smoke         # 1 task, 1 attempt, 1 repeat
#   MODELS=qwen3-4b ./run_eval.sh # restrict to one model
set -euo pipefail

# ---------------------------------------------------------------- config ----
JOBS_DIR="${JOBS_DIR:-$PWD/jobs}"   # keep on a path shared with the docker host
LOG_DIR="${LOG_DIR:-$PWD/logs}"
PORT="${PORT:-8000}"
HARBOR_ENV="${HARBOR_ENV:-docker}"        # docker | daytona | modal ...
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}" # SGLang --context-length
MEM_FRACTION="${MEM_FRACTION:-0.90}"      # SGLang --mem-fraction-static (weights + KV pool)
AGENT="${AGENT:-mini-swe-agent}"          # Tmax used a mini-swe-agent variant, not terminus-2
K="${K:-5}"                               # attempts per task (paper: 5 rollouts)
N_CONCURRENT="${N_CONCURRENT:-8}"
REPEATS="${REPEATS:-3}"                   # paper repeats each eval 3x to reduce noise
MAX_RETRIES="${MAX_RETRIES:-3}"           # paper restarts timed-out runs up to 3x
# Paper retried TIMEOUTS, not all exceptions. Bare -r 3 would also retry real failures,
# which inflates results by re-rolling genuine errors. Scope it. Set to "" to retry all.
RETRY_INCLUDE="${RETRY_INCLUDE:-AgentTimeoutError VerifierTimeoutError AgentSetupTimeoutError EnvironmentStartTimeoutError}"
LIMIT="${LIMIT:-}"                        # -l N, empty = all tasks
AGENT_KWARGS="${AGENT_KWARGS:-}"          # e.g. "reasoning_effort=none"
SERVE_TIMEOUT="${SERVE_TIMEOUT:-3600}"    # seconds to wait for SGLang (first run downloads weights)

# name|hf_id|tp_size|extra_sglang_args
ALL_MODELS=(
  "qwen3.5-2b|Qwen/Qwen3.5-2B|1|"
  "qwen3.5-4b|Qwen/Qwen3.5-4B|1|"
  "qwen3.5-9b|Qwen/Qwen3.5-9B|1|"
  "qwen3.6-27b|Qwen/Qwen3.6-27B|1|"
  "qwen3.6-35b-a3b|Qwen/Qwen3.6-35B-A3B|2|"
)

# label|dataset@version
ALL_DATASETS=(
  "tblite|openthoughts-tblite@2.0"
  "tb2|terminal-bench@2.0"
)

# ------------------------------------------------------------------ args ----
LIST_ONLY=0
for a in "$@"; do
  case "$a" in
    --list)  LIST_ONLY=1 ;;
    --smoke) K=1; REPEATS=1; LIMIT=1; N_CONCURRENT=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

# Optional filters: MODELS="qwen3-4b glm-4-9b"  DATASETS="tblite"
select_rows() {  # $1 = filter string, rest = rows
  local filter="$1"; shift
  if [[ -z "$filter" ]]; then printf '%s\n' "$@"; return; fi
  local row key
  for row in "$@"; do
    key="${row%%|*}"
    if [[ " $filter " == *" $key "* ]]; then echo "$row"; fi
  done
}
mapfile -t MODEL_ROWS   < <(select_rows "${MODELS:-}"   "${ALL_MODELS[@]}")
mapfile -t DATASET_ROWS < <(select_rows "${DATASETS:-}" "${ALL_DATASETS[@]}")

[[ ${#MODEL_ROWS[@]}   -gt 0 ]] || { echo "no models matched MODELS='${MODELS:-}'" >&2; exit 2; }
[[ ${#DATASET_ROWS[@]} -gt 0 ]] || { echo "no datasets matched DATASETS='${DATASETS:-}'" >&2; exit 2; }

# ----------------------------------------------------------------- utils ----
log() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

SERVER_PID=""
stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  log "stopping SGLang (pid $SERVER_PID)"
  kill "$SERVER_PID" 2>/dev/null || true
  # SGLang can take a while to release VRAM; wait, then force.
  for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
}
trap 'stop_server' EXIT INT TERM

wait_for_server() {
  local deadline=$(( SECONDS + SERVE_TIMEOUT ))
  while (( SECONDS < deadline )); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then return 0; fi
    # If SGLang died, stop waiting for a server that is never coming.
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then return 1; fi
    sleep 5
  done
  return 1
}

# ------------------------------------------------------------------ plan ----
log "plan"
printf '  agent      %s\n  env        %s\n  k=%s n=%s repeats=%s%s\n' \
  "$AGENT" "$HARBOR_ENV" "$K" "$N_CONCURRENT" "$REPEATS" \
  "${LIMIT:+  limit=$LIMIT}"
printf '  jobs       %s\n' "$JOBS_DIR"
for mrow in "${MODEL_ROWS[@]}"; do
  IFS='|' read -r mname mid mtp _ <<<"$mrow"
  printf '  model      %-14s %-24s tp=%s\n' "$mname" "$mid" "$mtp"
done
for drow in "${DATASET_ROWS[@]}"; do
  IFS='|' read -r dname did <<<"$drow"
  printf '  dataset    %-14s %s\n' "$dname" "$did"
done
total=$(( ${#MODEL_ROWS[@]} * ${#DATASET_ROWS[@]} * REPEATS ))
printf '  => %d harbor runs\n' "$total"
[[ $LIST_ONLY -eq 1 ]] && exit 0

# --------------------------------------------------------------- preflight --
python -c "import sglang" 2>/dev/null || die "sglang not importable (pip install 'sglang[all]')"
command -v harbor >/dev/null || die "harbor not found (uv tool install harbor)"
command -v nvidia-smi >/dev/null || die "no nvidia-smi — this script needs a GPU host"
if [[ "$HARBOR_ENV" == "docker" ]]; then
  docker info >/dev/null 2>&1 || die "docker daemon unreachable. On a Runpod pod use HARBOR_ENV=daytona"
fi
# Harbor bind-mounts JOBS_DIR into task containers; the daemon resolves that path on the
# HOST. Inside slime's image that only works if the path is mounted identically on both
# sides. Mismatch => RewardFileNotFoundError with an empty verifier/.
if [[ -f /.dockerenv && "$HARBOR_ENV" == "docker" ]]; then
  echo "NOTE: running inside a container; $JOBS_DIR must be bind-mounted at the same path on the host" >&2
fi

N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
log "$N_GPU GPU(s) detected"
mkdir -p "$JOBS_DIR" "$LOG_DIR"

# ------------------------------------------------------------------ sweep ----
FAILED=()
for mrow in "${MODEL_ROWS[@]}"; do
  IFS='|' read -r MNAME MID MTP MEXTRA <<<"$mrow"

  if (( MTP > N_GPU )); then
    echo "SKIP $MNAME: needs tensor-parallel $MTP, host has $N_GPU GPU(s)" >&2
    FAILED+=("$MNAME: insufficient GPUs")
    continue
  fi

  log "serving $MNAME  ($MID, tp=$MTP)"
  # shellcheck disable=SC2086
  python -m sglang.launch_server \
    --model-path "$MID" \
    --served-model-name "$MNAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tp-size "$MTP" \
    --context-length "$CONTEXT_LENGTH" \
    --mem-fraction-static "$MEM_FRACTION" \
    $MEXTRA \
    > "$LOG_DIR/sglang.$MNAME.log" 2>&1 &
  SERVER_PID=$!

  if ! wait_for_server; then
    echo "SKIP $MNAME: SGLang did not become ready — see $LOG_DIR/sglang.$MNAME.log" >&2
    tail -20 "$LOG_DIR/sglang.$MNAME.log" >&2 || true
    FAILED+=("$MNAME: serve failed")
    stop_server
    continue
  fi
  log "$MNAME ready"

  export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
  export MSWEA_API_KEY="$OPENAI_API_KEY"   # mini-swe-agent reads its key here

  for drow in "${DATASET_ROWS[@]}"; do
    IFS='|' read -r DNAME DID <<<"$drow"
    for rep in $(seq 1 "$REPEATS"); do
      job="${MNAME}__${DNAME}__rep${rep}"
      out="$JOBS_DIR/$job"
      if [[ -f "$out/result.json" ]]; then
        echo "  skip $job (already has result.json)"
        continue
      fi
      log "run $job"
      set +e
      harbor run \
        -d "$DID" \
        -a "$AGENT" \
        -m "openai/${MNAME}" \
        -e "$HARBOR_ENV" \
        -k "$K" \
        -n "$N_CONCURRENT" \
        -r "$MAX_RETRIES" \
        $(for e in $RETRY_INCLUDE; do printf -- '--retry-include %s ' "$e"; done) \
        --job-name "$job" \
        -o "$JOBS_DIR" \
        ${LIMIT:+-l "$LIMIT"} \
        ${AGENT_KWARGS:+--ak "$AGENT_KWARGS"} \
        2>&1 | tee "$LOG_DIR/harbor.$job.log"
      rc=${PIPESTATUS[0]}
      set -e
      [[ $rc -eq 0 ]] || FAILED+=("$job (exit $rc)")
    done
  done

  stop_server
done

# ---------------------------------------------------------------- summary ----
log "summary"
python3 "$(dirname "$0")/summarize.py" "$JOBS_DIR" 2>/dev/null || echo "(run summarize.py for a table)"

if (( ${#FAILED[@]} )); then
  printf '\n\033[31m%d failure(s):\033[0m\n' "${#FAILED[@]}"
  printf '  - %s\n' "${FAILED[@]}"
  exit 1
fi
log "all runs completed"
