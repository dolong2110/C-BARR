#!/usr/bin/env bash
set -euo pipefail

# Phases 4-7 GPU Orchestration for C-BARR (all-in-one or 4-part concurrent split, per-step ledger, optional push)
#
# Common usage:
#   CBARR_COLOR=1 bash scripts/run_gpu_phases_4_to_6.sh --gpu 0 --push --with-9b --with-w7
#   bash scripts/run_part1_worker_qwen4b.sh --shards auto --push
#   bash scripts/run_part2_worker_smollm3.sh --shards auto --push
#   bash scripts/run_part3_worker_scale.sh --shards auto --push
#   bash scripts/run_part4_combine_analyze.sh --push --strict-version-match
#   bash scripts/run_gpu_phases_4_to_6.sh --mock --out-dir /tmp/mock_out --exp-log /tmp/mock_log.md
#
# Key flags:
#   --part <all|1|2|3|4>    Select full pipeline or Part 1/2/3 parallel GPU worker / Part 4 combiner (default: all)
#   --shards <N|auto>       Intra-model GPU shard count; 'auto' uses min(GPU_COUNT, 4) when >=2 GPUs visible (default: 1)
#   --gpu <id>              Pin CUDA_VISIBLE_DEVICES (default: auto)
#   --device <dev>          Compute device override: auto, cuda, mps, cpu (default: auto)
#   --dtype <auto|bf16|fp16|fp32>  Model precision override (default: auto -> bf16 on sm>=80 & Apple Silicon MPS BF16)
#   --models "<m1 m2>"      Optional space-separated subset of models (default: part-specific models)
#   --model <id>            Convenience alias for single-model local run (repeatable)
#   --calibrate-only        Run Step 0a + Gate-2 calibration only (seed_60 + dev_60)
#   --strict-only           Skip the second diag491 pass and run strict341 only
#   --offline               Set HF_HUB_OFFLINE=1 and load locally cached checkpoints without Hub network checks
#   --push                  Sync results cleanly (workers: disjoint outputs/<model>/* with --no-ledger; part 4/all: Phase 7 + ledger)
#   --unpack / --no-unpack  Control unpacking repo outputs/*/*_cache.jsonl.gz in --part 4 (default: auto when present)
#   --strict-version-match  Fail Phase 7 analysis if combined model versions differ
#   --checkpoint-every <N>  Mid-eval_450 checkpoint cadence in new items (default: 150 live, 0 mock)
#   --with-9b / --with-w7   Enable Qwen3.5-9B scale sweep / optional W7 extensions
#   --include-b512          Include Contingency R3 B=512 sweep
#   --skip-smoke            Skip Step 0a hardware probe & 12-item smoke gate
#   --skip-preflight        Skip Gate 0 offline unittest discovery
#   --data-dir / --run-id / --out-dir / --fig-dir / --log-dir / --exp-log / --no-exp-log
#   --seed / --lam-utility / --norm-tokens / --reg-c / --tau-min / --tau-max / --tau-steps
#   --mock                  Run full offline CPU orchestration dry-run without overwriting repo ledger

export PYTHONUNBUFFERED=1
export CBARR_COLOR="${CBARR_COLOR:-0}"

PART_MODE="all"
SHARDS_ARG="1"
WITH_9B=0
WITH_W7=0
IS_MOCK=0
SKIP_SMOKE=0
SKIP_PREFLIGHT=0
CALIBRATE_ONLY=0
STRICT_ONLY=0
STRICT_VERSION_MATCH=0
UNPACK_MODE="auto"
OFFLINE_MODE=0
AUTO_PUSH="${CBARR_AUTO_PUSH:-0}"
DATA_DIR="data"
OUT_DIR="outputs"
FIG_DIR="paper/figures"
LOG_DIR=""
EXP_LOG=""
NO_EXP_LOG=0
RUN_ID_ARG=""
DTYPE="auto"
DEVICE_ARG="auto"
GPU_ID=""
MODELS_ARG=()
SEED=42
LAM_UTILITY="0.05"
NORM_TOKENS="250.0"
REG_C="1.0"
TAU_MIN="0.15"
TAU_MAX="0.85"
TAU_STEPS=29
CHECKPOINT_EVERY=""
EXTRA_FLAGS=()
RAW_CMD="$0 $*"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --part)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      PART_MODE="$1"
      ;;
    --shards)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      SHARDS_ARG="$1"
      ;;
    --with-9b) WITH_9B=1 ;;
    --with-w7) WITH_W7=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    --skip-preflight) SKIP_PREFLIGHT=1 ;;
    --calibrate-only) CALIBRATE_ONLY=1 ;;
    --strict-only) STRICT_ONLY=1 ;;
    --strict-version-match) STRICT_VERSION_MATCH=1 ;;
    --unpack) UNPACK_MODE="on" ;;
    --no-unpack) UNPACK_MODE="off" ;;
    --offline) OFFLINE_MODE=1; export HF_HUB_OFFLINE=1; EXTRA_FLAGS+=("--offline") ;;
    --no-exp-log) NO_EXP_LOG=1 ;;
    --push) AUTO_PUSH=1 ;;
    --include-b512) EXTRA_FLAGS+=("$1") ;;
    --mock) IS_MOCK=1; EXTRA_FLAGS+=("$1") ;;
    --data-dir)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      DATA_DIR="$1"
      ;;
    --device)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      DEVICE_ARG="$1"
      ;;
    --model)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      MODELS_ARG+=("$1")
      ;;
    --models)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      read -r -a _tmp_models <<< "$1"
      MODELS_ARG+=("${_tmp_models[@]}")
      ;;
    --run-id)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      RUN_ID_ARG="$1"
      ;;
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
    --seed)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      SEED="$1"
      ;;
    --lam-utility)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      LAM_UTILITY="$1"
      ;;
    --norm-tokens)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      NORM_TOKENS="$1"
      ;;
    --reg-c)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      REG_C="$1"
      ;;
    --tau-min)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      TAU_MIN="$1"
      ;;
    --tau-max)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      TAU_MAX="$1"
      ;;
    --tau-steps)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      TAU_STEPS="$1"
      ;;
    --checkpoint-every)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      CHECKPOINT_EVERY="$1"
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
  shift
done

case "$PART_MODE" in
  all|1|2|3|4) ;;
  *) echo "Unknown --part '$PART_MODE' (expected all, 1, 2, 3, or 4)" >&2; exit 1 ;;
esac

WORKER_MODE=0
if [ "$PART_MODE" = "1" ] || [ "$PART_MODE" = "2" ] || [ "$PART_MODE" = "3" ]; then
  # Parallel workers must never mutate shared root EXPERIMENT_LOG.md during execution
  WORKER_MODE=1
fi

if [ -n "$GPU_ID" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

if [ -z "$LOG_DIR" ]; then
  LOG_DIR="${OUT_DIR}/logs"
fi
mkdir -p "$OUT_DIR" "$LOG_DIR"

PY_BIN="${PYTHON_BIN:-python3}"
if [ -x ".venv/bin/python" ] && [ -z "${PYTHON_BIN:-}" ]; then
  PY_BIN=".venv/bin/python"
fi

if [ "$SHARDS_ARG" = "auto" ]; then
  if [ -n "$GPU_ID" ] && [[ "$GPU_ID" != *","* ]]; then
    N_SHARDS=1
  else
    N_SHARDS="$("$PY_BIN" -c "import torch; print(min(torch.cuda.device_count(), 4) if torch.cuda.is_available() else 1)" 2>/dev/null || echo 1)"
  fi
else
  N_SHARDS="$SHARDS_ARG"
fi
if [ "${N_SHARDS:-1}" -lt 1 ] 2>/dev/null; then
  N_SHARDS=1
fi

if [ -n "$RUN_ID_ARG" ]; then
  RUN_ID="$RUN_ID_ARG"
else
  if [ "$PART_MODE" = "all" ]; then
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
  else
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_p${PART_MODE}"
  fi
fi

if [ "$NO_EXP_LOG" -eq 0 ] && [ -z "$EXP_LOG" ]; then
  if [ "$WORKER_MODE" -eq 1 ]; then
    EXP_LOG="${LOG_DIR}/worker_p${PART_MODE}_${RUN_ID}.md"
  elif [ "$IS_MOCK" -eq 0 ] && [ "$OUT_DIR" = "outputs" ]; then
    # Only auto-append to shared repo-tracked EXPERIMENT_LOG.md on all-in-one or Part 4 combiner runs
    EXP_LOG="EXPERIMENT_LOG.md"
  else
    EXP_LOG="${OUT_DIR}/experiment_log.md"
  fi
fi

RUN_LOG="${LOG_DIR}/cbarr_${RUN_ID}.log"
TIMINGS_FILE="${LOG_DIR}/timings_${RUN_ID}.jsonl"
PARAMS_FILE="${LOG_DIR}/params_${RUN_ID}.json"
PIPELINE_START_EPOCH="$(date -u +%s)"
CURRENT_STAGE="Init"
FINALIZED=0

if [ -z "$CHECKPOINT_EVERY" ]; then
  if [ "$IS_MOCK" -eq 1 ] || [ "$WORKER_MODE" -eq 1 ]; then
    CHECKPOINT_EVERY=0
  else
    CHECKPOINT_EVERY=150
  fi
fi

# Shared hyperparameters and per-step checkpoint flags forwarded into every scripts/run_matrix.py invocation
MATRIX_PARAM_FLAGS=(
  --device "$DEVICE_ARG"
  --seed "$SEED"
  --lam-utility "$LAM_UTILITY"
  --norm-tokens "$NORM_TOKENS"
  --reg-c "$REG_C"
  --tau-min "$TAU_MIN"
  --tau-max "$TAU_MAX"
  --tau-steps "$TAU_STEPS"
  --run-id "$RUN_ID"
  --timings-file "$TIMINGS_FILE"
  --log-file "$RUN_LOG"
  --params-json "@${PARAMS_FILE}"
  --step-checkpoint-every "$CHECKPOINT_EVERY"
)
if [ "$NO_EXP_LOG" -eq 0 ] && [ -n "$EXP_LOG" ]; then
  MATRIX_PARAM_FLAGS+=(--exp-log "$EXP_LOG")
  if [ "$AUTO_PUSH" -eq 1 ] && [ "$IS_MOCK" -eq 0 ] && [ "$WORKER_MODE" -eq 0 ]; then
    MATRIX_PARAM_FLAGS+=(--git-push)
  fi
fi

"$PY_BIN" -c "
import json, pathlib, sys
payload = {
    'code_version': pathlib.Path('VERSION').read_text(encoding='utf-8').strip() if pathlib.Path('VERSION').is_file() else '0.0.0',
    'dtype': sys.argv[1],
    'seed': int(sys.argv[2]),
    'lam_utility': float(sys.argv[3]),
    'norm_tokens': float(sys.argv[4]),
    'logreg_C': float(sys.argv[5]),
    'tau_grid': [float(sys.argv[6]), float(sys.argv[7]), int(sys.argv[8])],
    'with_9b': bool(int(sys.argv[9])),
    'with_w7': bool(int(sys.argv[10])),
    'is_mock': bool(int(sys.argv[11])),
    'auto_push': bool(int(sys.argv[12])),
}
pathlib.Path(sys.argv[13]).write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
" "$DTYPE" "$SEED" "$LAM_UTILITY" "$NORM_TOKENS" "$REG_C" "$TAU_MIN" "$TAU_MAX" "$TAU_STEPS" "$WITH_9B" "$WITH_W7" "$IS_MOCK" "$AUTO_PUSH" "$PARAMS_FILE"

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

checkpoint_exp_log() {
  local loop_label="$1"
  local run_status="$2"
  local now_end total_sec md_args=() push_args=() smoke_args=()
  now_end="$(date -u +%s)"
  total_sec=$((now_end - PIPELINE_START_EPOCH))
  if [ "$NO_EXP_LOG" -eq 0 ] && [ -n "$EXP_LOG" ]; then
    md_args=(--md-log "$EXP_LOG")
  fi
  if [ "$AUTO_PUSH" -eq 1 ] && [ "$IS_MOCK" -eq 0 ] && [ "$WORKER_MODE" -eq 0 ] && [ "$NO_EXP_LOG" -eq 0 ] && [ -n "$EXP_LOG" ]; then
    push_args=(--git-push)
  fi
  if [ "$IS_MOCK" -eq 0 ] && [ "$SKIP_SMOKE" -eq 0 ]; then
    if [ -f "${OUT_DIR}/smoke_test_gate3_report.json" ]; then
      smoke_args=(--smoke-report "${OUT_DIR}/smoke_test_gate3_report.json")
    elif [ -f "outputs/smoke_test_gate3_report.json" ]; then
      smoke_args=(--smoke-report "outputs/smoke_test_gate3_report.json")
    fi
  fi
  "$PY_BIN" scripts/log_experiment.py \
    --run-id "$RUN_ID" \
    --status "$run_status" \
    --loop-name "$loop_label" \
    --command "$RAW_CMD" \
    --out-dir "$OUT_DIR" \
    --timings-file "$TIMINGS_FILE" \
    --log-file "$RUN_LOG" \
    --params-json "@${PARAMS_FILE}" \
    --total-sec "$total_sec" \
    --upsert \
    ${smoke_args[@]+"${smoke_args[@]}"} \
    ${md_args[@]+"${md_args[@]}"} \
    ${push_args[@]+"${push_args[@]}"} 2>&1 | tee -a "$RUN_LOG" || true
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
    checkpoint_exp_log "$stage_name" "RUNNING (after ${stage_name})"
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
  local t_end total_sec status_str
  t_end="$(date -u +%s)"
  total_sec=$((t_end - PIPELINE_START_EPOCH))
  if [ "$rc" -eq 0 ]; then
    status_str="PASSED"
  else
    status_str="FAILED (exit=${rc} at ${CURRENT_STAGE})"
  fi
  checkpoint_exp_log "$CURRENT_STAGE" "$status_str"
  log_banner "Run ${RUN_ID} finished with status=${status_str} (total=${total_sec}s). Full log: ${RUN_LOG}"
}
# Persist final PASSED/FAILED status, hardware diagnostics, and total wall time even on unexpected abort
trap finalize_run EXIT

run_matrix_sharded() {
  local stage_name="$1"
  local model_id="$2"
  shift 2
  local base_args=(
    "$PY_BIN" scripts/run_matrix.py
    --model "$model_id"
    --data-dir "$DATA_DIR"
    --out-dir "$OUT_DIR"
    --dtype "$DTYPE"
    "$@"
    ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
  )
  if [ "$N_SHARDS" -le 1 ]; then
    run_stage "$stage_name" "${base_args[@]}" "${MATRIX_PARAM_FLAGS[@]}"
    return $?
  fi

  local pids=() s_idx slug s_log rc_all=0
  slug="${model_id//\//__}"
  CURRENT_STAGE="$stage_name"
  local t0 t1 dur
  t0="$(date -u +%s)"
  log_banner "=== [$stage_name] START (${N_SHARDS}-GPU sharded) ==="
  for ((s_idx = 0; s_idx < N_SHARDS; s_idx++)); do
    s_log="${LOG_DIR}/cbarr_${RUN_ID}_gpu${s_idx}_${slug}.log"
    CUDA_VISIBLE_DEVICES="$s_idx" "${base_args[@]}" \
      --device "$DEVICE_ARG" \
      --seed "$SEED" \
      --lam-utility "$LAM_UTILITY" \
      --norm-tokens "$NORM_TOKENS" \
      --reg-c "$REG_C" \
      --tau-min "$TAU_MIN" \
      --tau-max "$TAU_MAX" \
      --tau-steps "$TAU_STEPS" \
      --shard-idx "$s_idx" \
      --num-shards "$N_SHARDS" \
      --step-checkpoint-every 0 >"$s_log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      rc_all=1
    fi
  done
  if [ "$rc_all" -ne 0 ]; then
    t1="$(date -u +%s)"
    dur=$((t1 - t0))
    record_stage_timing "$stage_name" "FAIL(shard_error)" "$dur"
    log_banner "=== [$stage_name] FAILED in parallel GPU shard (${dur}s; inspect ${LOG_DIR}/cbarr_${RUN_ID}_gpu*_${slug}.log) ==="
    return 1
  fi
  run_stage "${stage_name} (merge)" "${base_args[@]}" --merge-shards "${MATRIX_PARAM_FLAGS[@]}"
}

run_hw_probe_and_smoke() {
  local run_smoke="${1:-1}"
  shift || true
  if [ "$IS_MOCK" -ne 0 ] || [ "$SKIP_SMOKE" -ne 0 ]; then
    return 0
  fi
  run_stage "Step 0a: Hardware & Kernel Probe" "$PY_BIN" -c "
import platform, torch, transformers
print('PyTorch:', torch.__version__, '| Transformers:', transformers.__version__, '| Arch:', platform.machine())
print('CUDA available:', torch.cuda.is_available(), '| Device count:', torch.cuda.device_count(), '| MPS available:', bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()))
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    sm = torch.cuda.get_device_capability(0)
    print(f'GPU[0]: {props.name} | VRAM={props.total_memory / (1024**3):.1f} GB | SM={sm[0]}.{sm[1]} | BF16={torch.cuda.is_bf16_supported() and sm[0] >= 8}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    mem_gb = torch.mps.recommended_max_memory() / (1024**3) if hasattr(torch, 'mps') and hasattr(torch.mps, 'recommended_max_memory') else 0.0
    bf16_ok = False
    try:
        p = torch.ones((2, 2), device='mps', dtype=torch.bfloat16)
        _ = float((p @ p).sum().item())
        bf16_ok = True
    except Exception:
        bf16_ok = False
    print(f'MPS: Apple Silicon | MaxRecMem={mem_gb:.1f} GB | BF16={bf16_ok}')
for pkg in ('fla', 'causal_conv1d'):
    try:
        mod = __import__(pkg)
        print(f'Optional kernel {pkg}: OK ({getattr(mod, \"__version__\", \"installed\")})')
    except Exception as exc:
        print(f'Optional kernel {pkg}: unavailable ({exc}) -> PyTorch fallback will be used')
"
  if [ "$run_smoke" -eq 1 ]; then
    local smoke_args=()
    if [ "$#" -gt 0 ]; then
      smoke_args=(--models "$@")
    elif [ "${#MODELS_ARG[@]}" -gt 0 ]; then
      smoke_args=(--models "${MODELS_ARG[@]}")
    fi
    run_stage "Step 0a: Live GPU 12-Item Smoke" "$PY_BIN" scripts/run_smoke_test.py --live-gpu --device "$DEVICE_ARG" --dtype "$DTYPE" --out-dir "$OUT_DIR" ${smoke_args[@]+"${smoke_args[@]}"}
  fi
}

run_model_eval_pair() {
  local prefix="$1"
  local model_id="$2"
  run_matrix_sharded "${prefix}: ${model_id} (strict341)" "$model_id" --strict-diag
  if [ "$STRICT_ONLY" -eq 0 ]; then
    run_matrix_sharded "${prefix}: ${model_id} (diag491)" "$model_id"
  fi
}

run_w7_stage() {
  local w7_model="$1"
  if [ "$WITH_W7" -ne 1 ]; then
    return 0
  fi
  local w7_flags=()
  if [ "$IS_MOCK" -eq 1 ]; then
    w7_flags+=(--mock)
  fi
  run_stage "W7: DeltaNet Blend (${w7_model})" \
    "$PY_BIN" scripts/ext_deltanet_blend.py --model "$w7_model" --out-dir "$OUT_DIR" ${w7_flags[@]+"${w7_flags[@]}"}
  run_stage "W7: Prefill MLP Probe (${w7_model})" \
    "$PY_BIN" scripts/ext_prefill_mlp.py --model "$w7_model" --out-dir "$OUT_DIR" ${w7_flags[@]+"${w7_flags[@]}"}
}

run_phase7_stage() {
  local p7_flags=(--out-dir "$OUT_DIR" --fig-dir "$FIG_DIR")
  if [ "$STRICT_VERSION_MATCH" -eq 1 ]; then
    p7_flags+=(--strict-version-match)
  fi
  run_stage "Phase 7 / W8: Analysis, Tables & Figures" \
    "$PY_BIN" scripts/analyze_and_plot.py "${p7_flags[@]}"
}

sync_part_results() {
  local sync_label="$1"
  local include_phase7="$2"
  shift 2
  if [ "$AUTO_PUSH" -ne 1 ] || [ "$IS_MOCK" -ne 0 ]; then
    return 0
  fi
  local sync_flags=(
    --mode push
    --repo-dir .
    --out-dir "$OUT_DIR"
    --label "${sync_label}:${RUN_ID}"
    --git-push
  )
  if [ "$include_phase7" -eq 0 ]; then
    sync_flags+=(--no-phase7)
  fi
  if [ "$WORKER_MODE" -eq 1 ]; then
    sync_flags+=(--no-ledger)
  fi
  if [ "$#" -gt 0 ]; then
    sync_flags+=(--models "$@")
  fi
  run_stage "GitHub Sync (${sync_label})" "$PY_BIN" scripts/sync_results.py "${sync_flags[@]}"
}

# Log environment snapshot upfront for debugging & reproducibility
{
  echo "========================================================================"
  echo "C-BARR Run ID      : ${RUN_ID} (part=${PART_MODE}, shards=${N_SHARDS})"
  echo "UTC Start          : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Code Version       : v$(cat VERSION 2>/dev/null || echo 0.0.0) (git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown))"
  echo "Python Binary      : ${PY_BIN}"
  echo "DATA_DIR / OUT_DIR : ${DATA_DIR} / ${OUT_DIR} -> FIG_DIR=${FIG_DIR}"
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

case "$PART_MODE" in
  1)
    if [ "$SKIP_PREFLIGHT" -eq 0 ]; then
      run_stage "Gate 0: Preflight Tests" "$PY_BIN" -m unittest discover -s tests -v
    fi
    run_hw_probe_and_smoke 1 Qwen/Qwen3.5-4B
    run_matrix_sharded "Step 0b: Primary 4B Gate-2 Calibration" Qwen/Qwen3.5-4B --calibrate-only
    if [ "$CALIBRATE_ONLY" -eq 0 ]; then
      run_model_eval_pair "Phase 4 / W4" Qwen/Qwen3.5-4B
      sync_part_results "script-part1-4b" 0 Qwen/Qwen3.5-4B
    fi
    run_stage "Submission Audit: Double-Blind Anonymity" bash scripts/check_anonymity.sh
    ;;

  2)
    run_hw_probe_and_smoke 0
    if [ "$CALIBRATE_ONLY" -eq 1 ]; then
      run_matrix_sharded "Calibrate: HuggingFaceTB/SmolLM3-3B" HuggingFaceTB/SmolLM3-3B --calibrate-only
    else
      run_model_eval_pair "Phase 5 / W5" HuggingFaceTB/SmolLM3-3B
      sync_part_results "script-part2-smollm3" 0 HuggingFaceTB/SmolLM3-3B
    fi
    run_stage "Submission Audit: Double-Blind Anonymity" bash scripts/check_anonymity.sh
    ;;

  3)
    run_hw_probe_and_smoke 0
    SCALE_MODELS=(Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B)
    if [ "$WITH_9B" -eq 1 ]; then
      SCALE_MODELS+=(Qwen/Qwen3.5-9B)
    fi
    for _sm in "${SCALE_MODELS[@]}"; do
      run_matrix_sharded "Phase 6 / W6: ${_sm} (strict341)" "$_sm" --strict-diag
    done
    sync_part_results "script-part3-scale" 0 "${SCALE_MODELS[@]}"
    run_stage "Submission Audit: Double-Blind Anonymity" bash scripts/check_anonymity.sh
    ;;

  4)
    if [ "$UNPACK_MODE" = "on" ] || { [ "$UNPACK_MODE" = "auto" ] && [ "$IS_MOCK" -eq 0 ] && compgen -G "outputs/*/*_cache.jsonl.gz" >/dev/null; }; then
      run_stage "Unpack Synced Worker Results" "$PY_BIN" scripts/sync_results.py --mode unpack --repo-dir . --out-dir "$OUT_DIR"
    fi
    if [ "$SKIP_PREFLIGHT" -eq 0 ]; then
      run_stage "Gate 0: Preflight Tests" "$PY_BIN" -m unittest discover -s tests -v
    fi
    run_w7_stage Qwen/Qwen3.5-4B
    run_phase7_stage
    run_stage "Submission Audit: Double-Blind Anonymity" bash scripts/check_anonymity.sh
    sync_part_results "script-part4-combined" 1
    ;;

  all)
    if [ "$SKIP_PREFLIGHT" -eq 0 ]; then
      run_stage "Gate 0: Preflight Tests" "$PY_BIN" -m unittest discover -s tests -v
    fi
    run_hw_probe_and_smoke 1
    if [ "${#MODELS_ARG[@]}" -gt 0 ]; then
      for _m in "${MODELS_ARG[@]}"; do
        if [ "$CALIBRATE_ONLY" -eq 1 ]; then
          run_matrix_sharded "Calibrate: ${_m}" "$_m" --calibrate-only
        else
          run_model_eval_pair "Matrix" "$_m"
        fi
      done
      if [ "$CALIBRATE_ONLY" -eq 0 ]; then
        run_w7_stage "${MODELS_ARG[0]}"
        run_phase7_stage
      fi
      run_stage "Submission Audit: Double-Blind Anonymity" bash scripts/check_anonymity.sh
      exit 0
    fi

    run_matrix_sharded "Step 0b: Primary 4B Gate-2 Calibration" Qwen/Qwen3.5-4B --calibrate-only
    if [ "$CALIBRATE_ONLY" -eq 1 ]; then
      run_stage "Submission Audit: Double-Blind Anonymity" bash scripts/check_anonymity.sh
      exit 0
    fi

    # Run strict341 first so its cached diag_ehurt/controls entries are reused during diag491
    run_model_eval_pair "Phase 4 / W4" Qwen/Qwen3.5-4B
    run_model_eval_pair "Phase 5 / W5" HuggingFaceTB/SmolLM3-3B
    run_matrix_sharded "Phase 6 / W6: Qwen3.5-0.8B (strict341)" Qwen/Qwen3.5-0.8B --strict-diag
    run_matrix_sharded "Phase 6 / W6: Qwen3.5-2B (strict341)" Qwen/Qwen3.5-2B --strict-diag
    if [ "$WITH_9B" -eq 1 ]; then
      run_matrix_sharded "Phase 6 / W6: Qwen3.5-9B (strict341)" Qwen/Qwen3.5-9B --strict-diag
    fi
    run_w7_stage Qwen/Qwen3.5-4B
    run_phase7_stage
    run_stage "Submission Audit: Double-Blind Anonymity" bash scripts/check_anonymity.sh
    sync_part_results "script-all-in-one" 1
    ;;
esac
