#!/usr/bin/env bash
set -euo pipefail

# Phases 4-7 GPU Orchestration for C-BARR (with persistent logging, phase timers, and experiment ledger)
# Usage:
#   bash scripts/run_gpu_phases_4_to_6.sh                 # Step 0 + 4B + SmolLM3-3B + 0.8B/2B + W8
#   bash scripts/run_gpu_phases_4_to_6.sh --gpu 0         # Pin CUDA_VISIBLE_DEVICES=0 on multi-GPU lab servers
#   bash scripts/run_gpu_phases_4_to_6.sh --with-9b       # Also run Qwen3.5-9B scale sweep
#   bash scripts/run_gpu_phases_4_to_6.sh --with-w7       # Also run optional W7 DeltaNet blend & prefill MLP
#   bash scripts/run_gpu_phases_4_to_6.sh --include-b512  # Also run Contingency R3 B=512 sweep
#   bash scripts/run_gpu_phases_4_to_6.sh --mock          # Full offline CPU orchestration dry-run

export PYTHONUNBUFFERED=1

WITH_9B=0
WITH_W7=0
IS_MOCK=0
SKIP_SMOKE=0
OUT_DIR="outputs"
FIG_DIR="paper/figures"
LOG_DIR=""
EXP_LOG=""
NO_EXP_LOG=0
DTYPE="auto"
GPU_ID=""
EXTRA_FLAGS=()
RAW_CMD="$0 $*"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --with-9b) WITH_9B=1 ;;
    --with-w7) WITH_W7=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    --no-exp-log) NO_EXP_LOG=1 ;;
    --include-b512) EXTRA_FLAGS+=("$1") ;;
    --mock) IS_MOCK=1; EXTRA_FLAGS+=("$1") ;;
    --out-dir)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      OUT_DIR="$1"
      ;;
    --fig-dir)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      FIG_DIR="$1"
      ;;
    --log-dir)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      LOG_DIR="$1"
      ;;
    --exp-log)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      EXP_LOG="$1"
      ;;
    --dtype)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      DTYPE="$1"
      ;;
    --gpu)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      GPU_ID="$1"
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
  shift
done

if [ -n "$GPU_ID" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

if [ -z "$LOG_DIR" ]; then
  LOG_DIR="${OUT_DIR}/logs"
fi
mkdir -p "$OUT_DIR" "$LOG_DIR"

if [ "$NO_EXP_LOG" -eq 0 ] && [ -z "$EXP_LOG" ]; then
  # Only auto-append to the repo-tracked EXPERIMENT_LOG.md on real (non-mock, default out-dir) runs
  if [ "$IS_MOCK" -eq 0 ] && [ "$OUT_DIR" = "outputs" ]; then
    EXP_LOG="EXPERIMENT_LOG.md"
  else
    EXP_LOG="${OUT_DIR}/experiment_log.md"
  fi
fi

PY_BIN="${PYTHON_BIN:-python3}"
if [ -x ".venv/bin/python" ] && [ -z "${PYTHON_BIN:-}" ]; then
  PY_BIN=".venv/bin/python"
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="${LOG_DIR}/cbarr_${RUN_ID}.log"
TIMINGS_FILE="${LOG_DIR}/timings_${RUN_ID}.jsonl"
PIPELINE_START_EPOCH="$(date -u +%s)"
CURRENT_STAGE="Init"
FINALIZED=0

log_banner() {
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[$ts] $*" | tee -a "$RUN_LOG"
}

record_stage_timing() {
  local stage_name="$1"
  local status_str="$2"
  local dur_sec="$3"
  "$PY_BIN" -c "
import json, pathlib, sys
rec = {'stage': sys.argv[1], 'status': sys.argv[2], 'duration_sec': float(sys.argv[3])}
with pathlib.Path(sys.argv[4]).open('a', encoding='utf-8') as f:
    f.write(json.dumps(rec) + '\n')
" "$stage_name" "$status_str" "$dur_sec" "$TIMINGS_FILE"
}

run_stage() {
  local stage_name="$1"
  shift
  CURRENT_STAGE="$stage_name"
  local t0 t1 dur rc
  t0="$(date -u +%s)"
  log_banner "=== [$stage_name] START ==="
  set +e
  "$@" 2>&1 | tee -a "$RUN_LOG"
  rc=$?
  set -e
  t1="$(date -u +%s)"
  dur=$((t1 - t0))
  if [ "$rc" -eq 0 ]; then
    record_stage_timing "$stage_name" "PASS" "$dur"
    log_banner "=== [$stage_name] DONE (${dur}s) ==="
  else
    record_stage_timing "$stage_name" "FAIL(exit=${rc})" "$dur"
    log_banner "=== [$stage_name] FAILED (exit=${rc}, ${dur}s) ==="
    return "$rc"
  fi
}

finalize_run() {
  local rc=$?
  if [ "$FINALIZED" -eq 1 ]; then
    return
  fi
  FINALIZED=1
  local t_end total_sec status_str md_args=()
  t_end="$(date -u +%s)"
  total_sec=$((t_end - PIPELINE_START_EPOCH))
  if [ "$rc" -eq 0 ]; then
    status_str="PASSED"
  else
    status_str="FAILED (exit=${rc} at ${CURRENT_STAGE})"
  fi
  if [ "$NO_EXP_LOG" -eq 0 ] && [ -n "$EXP_LOG" ]; then
    md_args=(--md-log "$EXP_LOG")
  fi
  "$PY_BIN" scripts/log_experiment.py \
    --run-id "$RUN_ID" \
    --status "$status_str" \
    --command "$RAW_CMD" \
    --out-dir "$OUT_DIR" \
    --timings-file "$TIMINGS_FILE" \
    --log-file "$RUN_LOG" \
    --total-sec "$total_sec" \
    ${md_args[@]+"${md_args[@]}"} 2>&1 | tee -a "$RUN_LOG" || true
  log_banner "Run ${RUN_ID} finished with status=${status_str} (total=${total_sec}s). Full log: ${RUN_LOG}"
}
trap finalize_run EXIT

# Log environment snapshot upfront for debugging & reproducibility
{
  echo "========================================================================"
  echo "C-BARR Run ID      : ${RUN_ID}"
  echo "UTC Start          : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Git Revision       : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "Python Binary      : ${PY_BIN}"
  echo "OUT_DIR / FIG_DIR  : ${OUT_DIR} / ${FIG_DIR}"
  echo "DTYPE / CUDA_DEV   : ${DTYPE} / ${CUDA_VISIBLE_DEVICES:-auto}"
  echo "HF_HOME / HF_TOKEN : ${HF_HOME:-default} / $([ -n "${HF_TOKEN:-}" ] && echo set || echo unset)"
  echo "Command            : ${RAW_CMD}"
  echo "--- Disk Space ---"
  df -h . || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- nvidia-smi ---"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader || nvidia-smi || true
  fi
  echo "========================================================================"
} 2>&1 | tee -a "$RUN_LOG"

run_stage "Gate 0: Preflight Tests" "$PY_BIN" -m unittest discover -s tests -v

if [ "$IS_MOCK" -eq 0 ] && [ "$SKIP_SMOKE" -eq 0 ]; then
  run_stage "Step 0a: Hardware & Kernel Probe" "$PY_BIN" -c "
import torch, transformers
print('PyTorch:', torch.__version__, '| CUDA:', getattr(torch.version, 'cuda', None), '| Transformers:', transformers.__version__)
print('CUDA available:', torch.cuda.is_available(), '| Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    sm = torch.cuda.get_device_capability(0)
    print(f'GPU[0]: {props.name} | VRAM={props.total_memory / (1024**3):.1f} GB | SM={sm[0]}.{sm[1]} | BF16={torch.cuda.is_bf16_supported() and sm[0] >= 8}')
for pkg in ('fla', 'causal_conv1d'):
    try:
        mod = __import__(pkg)
        print(f'Optional kernel {pkg}: OK ({getattr(mod, \"__version__\", \"installed\")})')
    except Exception as exc:
        print(f'Optional kernel {pkg}: unavailable ({exc}) -> PyTorch fallback will be used')
"
  run_stage "Step 0a: Live GPU 12-Item Smoke" "$PY_BIN" scripts/run_smoke_test.py --live-gpu
fi

run_stage "Step 0b: Primary 4B Gate-2 Calibration" \
  "$PY_BIN" scripts/run_matrix.py --model Qwen/Qwen3.5-4B --out-dir "$OUT_DIR" --dtype "$DTYPE" --calibrate-only ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}

# Run strict341 first so its cached diag_ehurt/controls entries are reused during diag491
run_stage "Phase 4 / W4: Qwen3.5-4B (strict341)" \
  "$PY_BIN" scripts/run_matrix.py --model Qwen/Qwen3.5-4B --out-dir "$OUT_DIR" --dtype "$DTYPE" --strict-diag ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
run_stage "Phase 4 / W4: Qwen3.5-4B (diag491)" \
  "$PY_BIN" scripts/run_matrix.py --model Qwen/Qwen3.5-4B --out-dir "$OUT_DIR" --dtype "$DTYPE" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}

run_stage "Phase 5 / W5: SmolLM3-3B (strict341)" \
  "$PY_BIN" scripts/run_matrix.py --model HuggingFaceTB/SmolLM3-3B --out-dir "$OUT_DIR" --dtype "$DTYPE" --strict-diag ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
run_stage "Phase 5 / W5: SmolLM3-3B (diag491)" \
  "$PY_BIN" scripts/run_matrix.py --model HuggingFaceTB/SmolLM3-3B --out-dir "$OUT_DIR" --dtype "$DTYPE" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}

run_stage "Phase 6 / W6: Qwen3.5-0.8B (strict341)" \
  "$PY_BIN" scripts/run_matrix.py --model Qwen/Qwen3.5-0.8B --out-dir "$OUT_DIR" --dtype "$DTYPE" --strict-diag ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
run_stage "Phase 6 / W6: Qwen3.5-2B (strict341)" \
  "$PY_BIN" scripts/run_matrix.py --model Qwen/Qwen3.5-2B --out-dir "$OUT_DIR" --dtype "$DTYPE" --strict-diag ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}

if [ "$WITH_9B" -eq 1 ]; then
  run_stage "Phase 6 / W6: Qwen3.5-9B (strict341)" \
    "$PY_BIN" scripts/run_matrix.py --model Qwen/Qwen3.5-9B --out-dir "$OUT_DIR" --dtype "$DTYPE" --strict-diag ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
fi

if [ "$WITH_W7" -eq 1 ]; then
  W7_FLAGS=()
  if [ "$IS_MOCK" -eq 1 ]; then
    W7_FLAGS+=(--mock)
  fi
  run_stage "W7: DeltaNet Blend (Qwen3.5-4B)" \
    "$PY_BIN" scripts/ext_deltanet_blend.py --model Qwen/Qwen3.5-4B --out-dir "$OUT_DIR" ${W7_FLAGS[@]+"${W7_FLAGS[@]}"}
  run_stage "W7: Prefill MLP Probe (Qwen3.5-4B)" \
    "$PY_BIN" scripts/ext_prefill_mlp.py --model Qwen/Qwen3.5-4B --out-dir "$OUT_DIR" ${W7_FLAGS[@]+"${W7_FLAGS[@]}"}
fi

run_stage "Phase 7 / W8: Analysis, Tables & Figures" \
  "$PY_BIN" scripts/analyze_and_plot.py --out-dir "$OUT_DIR" --fig-dir "$FIG_DIR"

run_stage "Submission Audit: Double-Blind Anonymity" \
  bash scripts/check_anonymity.sh
