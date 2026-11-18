#!/usr/bin/env bash
set -euo pipefail

# Phases 1-7 GPU Orchestration for C-BARR (all-in-one A-Z, 4 parallel workers, or Step 5 combiner)
#
# Common usage:
#   GITHUB_TOKEN=... bash scripts/run_all_a_to_z.sh
#   CBARR_COLOR=1 bash scripts/run_gpu_phases_4_to_6.sh --setup auto --shards auto --auto-9b --with-w7 --commit --push
#   bash scripts/run_part1_worker_qwen4b.sh --shards auto --push
#   bash scripts/run_part2_worker_smollm3.sh --shards auto --push
#   bash scripts/run_part3_worker_qwen2b.sh --shards auto --push
#   bash scripts/run_part4_worker_qwen08b.sh --shards auto --push
#   bash scripts/run_part5_combine_analyze.sh --push --strict-version-match
#   bash scripts/run_gpu_phases_4_to_6.sh --mock --out-dir /tmp/mock_out --exp-log /tmp/mock_log.md
#
# Key flags:
#   --part <all|1|2|3|4|5|combine|w7|ifbench|r6|analyze>  Select full A-Z pipeline, Parts 1-4 parallel GPU workers, combiner (5), or independent step (w7/ifbench/r6/analyze) (default: all)
#   --setup / --setup-mode <auto|on|off>  Provision packages, fast DeltaNet kernels, IFEval, NLTK, and frozen splits (default: off)
#   --no-setup              Disable environment provisioning checks before running
#   --shards <N|auto>       Intra-model GPU shard count; 'auto' uses min(GPU_COUNT, --max-shards) when >=2 GPUs visible (default: 1)
#   --max-shards <N>        Maximum GPUs used when --shards auto is active (default: 8)
#   --workers-per-gpu <N|auto>  Concurrent shard workers pinned per GPU ('auto' runs 2/GPU on >=32 GB GPUs for <=4B models, 1 for 9B; default: auto)
#   --wait-gpu-min <N>      Poll up to N minutes if all GPUs are temporarily occupied (default: 60)
#   --stage-retries <N>     Auto-retry shard OOM/partial failures after harvesting completed items (default: 2)
#   --shard-progress-sec <N>  Interval in seconds for multi-GPU progress snapshots (default: 20; 0 disables)
#   --shard-log <mode>      Sharded live stdout mode: auto (GPU0 + all milestones/errors), all, milestones, off (default: auto)
#   --gpu <id[,id...]>      Pin CUDA_VISIBLE_DEVICES (e.g. 0 or 0,1,2,3; default: auto)
#   --device <dev>          Compute device override: auto, cuda, mps, cpu (default: auto)
#   --dtype <auto|bf16|fp16|fp32>  Model precision override (default: auto -> bf16 on sm>=80 & Apple Silicon MPS BF16)
#   --models "<m1 m2>"      Optional space-separated subset of models (default: part-specific models)
#   --model <id>            Convenience alias for single-model local run (repeatable)
#   --calibrate-only        Run Step 0a + Gate-2 calibration only (seed_60 + dev_60)
#   --strict-only           Skip the second diag491 pass and run strict341 only
#   --offline               Set HF_HUB_OFFLINE=1 and load locally cached checkpoints without Hub network checks
#   --hf-home <dir>         Override HF_HOME weight cache directory (helpful if / is nearly full)
#   --prune-hf-cache-below-gb <N>  Auto-prune cached HF model weights between models when free disk < N GB (default: 25; 0=off)
#   --allow-kernel-source-build  Allow compiling causal-conv1d C++/CUDA from source during setup (slow; off by default)
#   --commit / --commit-only  Commit synced caches, tables, figures, and EXPERIMENT_LOG.md locally to git
#   --push / --no-push      Commit and push synced results to GitHub (--push enables --commit)
#   --branch <name>         Target git branch when committing/pushing (default: current branch or master)
#   --github-token <PAT>    Transient GitHub PAT exported into GITHUB_TOKEN (scrubbed from logs)
#   --hf-token <TOKEN>      Transient Hugging Face token exported into HF_TOKEN (scrubbed from logs)
#   --git-name / --git-email  Optional local git author identity override for automated commits
#   --unpack / --no-unpack  Control unpacking repo outputs/*/*_cache.jsonl.gz in --part 5/all (default: auto when present)
#   --strict-version-match  Fail Phase 7 analysis if combined model versions differ
#   --checkpoint-every <N>  Mid-eval_450 checkpoint cadence in new items (default: 150 live, 0 mock)
#   --with-9b / --auto-9b / --no-9b  Enable Qwen3.5-9B always, automatically on >=22 GB GPUs, or disable
#   --with-w7 / --no-w7     Enable/disable optional W7 extensions (DeltaNet conv+KV, Dense KV evict, Prefill MLP, Stage-2 solve_first)
#   --with-r6 / --auto-r6 / --no-r6  Run Contingency R6 3-seed sampled validation (T=0.6, top_p=0.95, pp=1.5)
#   --r6-models "<list>"    Override R6 model list (default: "Qwen/Qwen3.5-4B")
#   --r6-seeds "<list>"     Override R6 seed list (default: "42 43 44" live, "42" mock)
#   --with-ifbench / --auto-ifbench / --no-ifbench  Run Appendix B OOD evaluation on allenai/IFBench_test
#   --ifbench-models "<list>"  Override IFBench OOD model list (default: "Qwen/Qwen3.5-4B")
#   --include-b512 / --auto-b512 / --no-b512  Contingency R3 B=512 sweep (always / auto on trigger / off)
#   --skip-smoke            Skip Step 0a hardware probe & 12-item smoke gate
#   --skip-preflight        Skip Gate 0 offline unittest discovery
#   --full-preflight        Run full E2E suite in Gate 0 instead of fast preflight (~1s)
#   --data-dir / --run-id / --out-dir / --fig-dir / --table-dir / --log-dir / --exp-log / --no-exp-log
#   --seed / --lam-utility / --norm-tokens / --reg-c / --tau-min / --tau-max / --tau-steps
#   --mock                  Run full offline CPU orchestration dry-run without overwriting repo ledger

export PYTHONUNBUFFERED=1
export CBARR_COLOR="${CBARR_COLOR:-0}"
export TRITON_DEFAULT_FP_FUSION="${TRITON_DEFAULT_FP_FUSION:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PART_MODE="all"
SETUP_MODE="off"
SHARDS_ARG="1"
MAX_SHARDS_ARG=8
WORKERS_PER_GPU_ARG="${CBARR_WORKERS_PER_GPU:-auto}"
SHARD_PROGRESS_SEC="${CBARR_SHARD_PROGRESS_SEC:-20}"
SHARD_LOG_MODE="${CBARR_SHARD_LOG:-auto}"
WAIT_GPU_MIN="${CBARR_WAIT_GPU_MIN:-60}"
STAGE_RETRIES="${CBARR_STAGE_RETRIES:-2}"
WITH_9B="0"
WITH_W7=0
WITH_R6="auto"
R6_MODELS_ARG=("Qwen/Qwen3.5-4B")
R6_SEEDS_ARG=()
WITH_IFBENCH="auto"
IFBENCH_MODELS_ARG=("Qwen/Qwen3.5-4B")
AUTO_B512="auto"
INCLUDE_B512=0
IS_MOCK=0
SKIP_SMOKE=0
SKIP_PREFLIGHT=0
FULL_PREFLIGHT=0
RAN_LIVE_SMOKE=0
CALIBRATE_ONLY=0
STRICT_ONLY=0
STRICT_VERSION_MATCH=0
UNPACK_MODE="auto"
OFFLINE_MODE=0
PRUNE_HF_BELOW_GB="25"
AUTO_PUSH="${CBARR_AUTO_PUSH:-0}"
AUTO_COMMIT=0
GIT_BRANCH=""
GIT_USER_NAME=""
GIT_USER_EMAIL=""
DATA_DIR="data"
OUT_DIR="outputs"
FIG_DIR="paper/figures"
TABLE_DIR=""
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

# Mask any transient secret arguments before recording RAW_CMD into logs/ledger
SAFE_ARGS=()
_mask_next=0
for _arg in "$@"; do
  if [ "$_mask_next" -eq 1 ]; then
    SAFE_ARGS+=("***")
    _mask_next=0
    continue
  fi
  case "$_arg" in
    --github-token|--hf-token)
      SAFE_ARGS+=("$_arg")
      _mask_next=1
      ;;
    --github-token=*|--hf-token=*)
      SAFE_ARGS+=("${_arg%%=*}=***")
      ;;
    *)
      SAFE_ARGS+=("$_arg")
      ;;
  esac
done
if [ "${#SAFE_ARGS[@]}" -gt 0 ]; then
  RAW_CMD="$0 ${SAFE_ARGS[*]}"
else
  RAW_CMD="$0"
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --part)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      PART_MODE="$1"
      ;;
    --setup)
      if [ "$#" -ge 2 ] && { [ "$2" = "on" ] || [ "$2" = "off" ] || [ "$2" = "auto" ]; }; then
        shift
        SETUP_MODE="$1"
      else
        SETUP_MODE="on"
      fi
      ;;
    --no-setup) SETUP_MODE="off" ;;
    --setup-mode)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      SETUP_MODE="$1"
      ;;
    --shards)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      SHARDS_ARG="$1"
      ;;
    --max-shards)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      MAX_SHARDS_ARG="$1"
      ;;
    --with-9b) WITH_9B="1" ;;
    --auto-9b) WITH_9B="auto" ;;
    --no-9b) WITH_9B="0" ;;
    --with-w7) WITH_W7=1 ;;
    --no-w7) WITH_W7=0 ;;
    --with-r6) WITH_R6="1" ;;
    --auto-r6) WITH_R6="auto" ;;
    --no-r6) WITH_R6="0" ;;
    --r6-models)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      read -r -a R6_MODELS_ARG <<< "$1"
      ;;
    --r6-seeds)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      read -r -a R6_SEEDS_ARG <<< "$1"
      ;;
    --with-ifbench) WITH_IFBENCH="1" ;;
    --auto-ifbench) WITH_IFBENCH="auto" ;;
    --no-ifbench) WITH_IFBENCH="0" ;;
    --ifbench-models)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      read -r -a IFBENCH_MODELS_ARG <<< "$1"
      ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    --skip-preflight) SKIP_PREFLIGHT=1 ;;
    --full-preflight) FULL_PREFLIGHT=1 ;;
    --calibrate-only) CALIBRATE_ONLY=1 ;;
    --strict-only) STRICT_ONLY=1 ;;
    --strict-version-match) STRICT_VERSION_MATCH=1 ;;
    --unpack) UNPACK_MODE="on" ;;
    --no-unpack) UNPACK_MODE="off" ;;
    --offline) OFFLINE_MODE=1; export HF_HUB_OFFLINE=1; EXTRA_FLAGS+=("--offline") ;;
    --no-exp-log) NO_EXP_LOG=1 ;;
    --commit|--commit-only) AUTO_COMMIT=1 ;;
    --push) AUTO_COMMIT=1; AUTO_PUSH=1 ;;
    --no-push) AUTO_PUSH=0 ;;
    --branch)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      GIT_BRANCH="$1"
      ;;
    --github-token)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      export GITHUB_TOKEN="$1"
      ;;
    --hf-token)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      export HF_TOKEN="$1"
      ;;
    --git-name)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      GIT_USER_NAME="$1"
      ;;
    --git-email)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      GIT_USER_EMAIL="$1"
      ;;
    --include-b512) INCLUDE_B512=1; AUTO_B512="1"; EXTRA_FLAGS+=("$1") ;;
    --auto-b512) AUTO_B512="auto" ;;
    --no-b512) AUTO_B512="0" ;;
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
    --table-dir)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      TABLE_DIR="$1"
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
    --hf-home)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      export HF_HOME="$1"
      ;;
    --prune-hf-cache-below-gb)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      PRUNE_HF_BELOW_GB="$1"
      ;;
    --allow-kernel-source-build)
      export CBARR_ALLOW_KERNEL_SOURCE_BUILD=1
      ;;
    --workers-per-gpu)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      WORKERS_PER_GPU_ARG="$1"
      ;;
    --shard-progress-sec)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      SHARD_PROGRESS_SEC="$1"
      ;;
    --shard-log)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      SHARD_LOG_MODE="$1"
      ;;
    --wait-gpu-min)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      WAIT_GPU_MIN="$1"
      ;;
    --stage-retries)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument" >&2; exit 1; fi
      shift
      STAGE_RETRIES="$1"
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
  shift
done

case "$PART_MODE" in
  all|1|2|3|4|5|combine|w7|ifbench|r6|analyze) ;;
  *) echo "Unknown --part '$PART_MODE' (expected all, 1, 2, 3, 4, 5, combine, w7, ifbench, r6, or analyze)" >&2; exit 1 ;;
esac

WORKER_MODE=0
if [ "$PART_MODE" = "1" ] || [ "$PART_MODE" = "2" ] || [ "$PART_MODE" = "3" ] || [ "$PART_MODE" = "4" ]; then
  # Parallel workers must never mutate shared root EXPERIMENT_LOG.md during execution
  WORKER_MODE=1
fi

case "$SETUP_MODE" in
  on|off|auto) ;;
  *) echo "Unknown --setup-mode '$SETUP_MODE' (expected on, off, or auto)" >&2; exit 1 ;;
esac
case "$WITH_9B" in
  0|1|auto) ;;
  *) echo "Unknown WITH_9B value '$WITH_9B' (expected 0, 1, or auto)" >&2; exit 1 ;;
esac
case "$SHARD_LOG_MODE" in
  auto|all|milestones|off) ;;
  *) echo "Unknown --shard-log '$SHARD_LOG_MODE' (expected auto, all, milestones, or off)" >&2; exit 1 ;;
esac
if [ "$WORKERS_PER_GPU_ARG" != "auto" ]; then
  if ! [[ "$WORKERS_PER_GPU_ARG" =~ ^[1-9][0-9]*$ ]]; then
    echo "Unknown --workers-per-gpu '$WORKERS_PER_GPU_ARG' (expected auto or positive integer)" >&2
    exit 1
  fi
fi

PINNED_GPU_CSV=""
if [ -n "$GPU_ID" ]; then
  PINNED_GPU_CSV="$GPU_ID"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  PINNED_GPU_CSV="$CUDA_VISIBLE_DEVICES"
fi
export CBARR_WAIT_GPU_SEC=$((WAIT_GPU_MIN * 60))
export CBARR_WORKERS_PER_GPU="$WORKERS_PER_GPU_ARG"

VISIBLE_GPUS=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
fi

if [ -z "$LOG_DIR" ]; then
  LOG_DIR="${OUT_DIR}/logs"
fi
mkdir -p "$OUT_DIR" "$LOG_DIR"

# When --hf-home points to a roomy scratch/NFS mount while /tmp is nearly full, route compiler/pip caches there.
if [ -n "${HF_HOME:-}" ]; then
  mkdir -p "${HF_HOME}/.tmp" "${HF_HOME}/.triton" "${HF_HOME}/.torchinductor" "${HF_HOME}/.pip_cache" 2>/dev/null || true
  if [ -d "${HF_HOME}/.triton" ] && [ -z "${TRITON_CACHE_DIR:-}" ]; then
    export TRITON_CACHE_DIR="${HF_HOME}/.triton"
  fi
  if [ -d "${HF_HOME}/.torchinductor" ] && [ -z "${TORCHINDUCTOR_CACHE_DIR:-}" ]; then
    export TORCHINDUCTOR_CACHE_DIR="${HF_HOME}/.torchinductor"
  fi
  if [ -d "${HF_HOME}/.pip_cache" ] && [ -z "${PIP_CACHE_DIR:-}" ]; then
    export PIP_CACHE_DIR="${HF_HOME}/.pip_cache"
  fi
fi

PY_BIN="${PYTHON_BIN:-python3}"
if [ -x ".venv/bin/python" ] && [ -z "${PYTHON_BIN:-}" ]; then
  PY_BIN=".venv/bin/python"
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
    # Only auto-append to shared repo-tracked EXPERIMENT_LOG.md on all-in-one or Part 5 combiner runs
    EXP_LOG="EXPERIMENT_LOG.md"
  else
    EXP_LOG="${OUT_DIR}/experiment_log.md"
  fi
fi

RUN_LOG="${LOG_DIR}/cbarr_${RUN_ID}.log"
TIMINGS_FILE="${LOG_DIR}/timings_${RUN_ID}.jsonl"
PARAMS_FILE="${LOG_DIR}/params_${RUN_ID}.json"
export CBARR_RUN_ID="$RUN_ID"
export CBARR_LOG_DIR="$LOG_DIR"
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

select_gpu_pool_slots() {
  local tmp_err out rc=0
  tmp_err="$(mktemp "${LOG_DIR}/.gpu_pool_${RUN_ID}_XXXXXX.err")"
  set +e
  out="$("$PY_BIN" scripts/gpu_pool.py "$@" 2>"$tmp_err")"
  rc=$?
  set -e
  if [ -s "$tmp_err" ]; then
    cat "$tmp_err" | tee -a "$RUN_LOG" >&2
  fi
  rm -f "$tmp_err"
  printf '%s' "$out"
  return "$rc"
}

check_git_sync_readiness() {
  if [ "$IS_MOCK" -ne 0 ]; then
    return 0
  fi
  if [ "$AUTO_COMMIT" -eq 0 ] && [ "$AUTO_PUSH" -eq 0 ]; then
    return 0
  fi
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: --commit/--push requested, but current directory is not a git repository." >&2
    exit 1
  fi
  if [ -n "$GIT_USER_NAME" ]; then
    git config user.name "$GIT_USER_NAME"
  fi
  if [ -n "$GIT_USER_EMAIL" ]; then
    git config user.email "$GIT_USER_EMAIL"
  fi
  "$PY_BIN" -c "import pathlib; from scripts.log_experiment import _ensure_git_identity; _ensure_git_identity(pathlib.Path('.'))"
  if [ "$AUTO_PUSH" -eq 1 ] && [ "$OFFLINE_MODE" -eq 0 ]; then
    local remote_url
    remote_url="$(git remote get-url origin 2>/dev/null || true)"
    if [ -z "$remote_url" ]; then
      echo "ERROR: --push requested, but git remote 'origin' is not configured." >&2
      exit 1
    fi
    if ! "$PY_BIN" -c "import os, pathlib, sys; from scripts.log_experiment import _has_git_push_auth; sys.exit(0 if _has_git_push_auth(pathlib.Path('.'), sys.argv[1], os.environ.get('GITHUB_TOKEN', '').strip()) else 1)" "$remote_url"; then
      echo "ERROR: --push requested for HTTPS remote '$remote_url', but GITHUB_TOKEN is unset and no git credential.helper was found." >&2
      echo "Set GITHUB_TOKEN=<your_PAT> (or use SSH / credential helper, or pass --no-push / --commit-only)." >&2
      exit 1
    fi
  fi
}

bootstrap_environment_and_data() {
  if [ "$IS_MOCK" -ne 0 ]; then
    return 0
  fi
  local need_setup=0
  if [ "$SETUP_MODE" = "on" ]; then
    need_setup=1
  elif [ "$SETUP_MODE" = "auto" ]; then
    if ! "$PY_BIN" -c "
import importlib, importlib.metadata, importlib.util, os, pathlib, re, nltk, torch
def _vt(p):
    return tuple(int(x) for x in re.findall(r'\d+', importlib.metadata.version(p))[:3])
reqs = ['numpy', 'jinja2', 'torch', 'transformers', 'accelerate', 'datasets', 'matplotlib', 'scipy', 'sklearn', 'nltk', 'immutabledict', 'langdetect', 'absl']
for m in reqs:
    importlib.import_module(m)
assert _vt('torch') >= (2, 6, 0)
assert _vt('transformers') >= (4, 53, 0)
assert _vt('jinja2') >= (3, 1, 6)
if importlib.util.find_spec('httpx2') is not None and importlib.util.find_spec('brotli') is not None:
    import brotli
    brotli.Decompressor().process(b'', output_buffer_limit=1024)
assert pathlib.Path('instruction_following_eval').is_dir()
for res in ('tokenizers/punkt', 'tokenizers/punkt_tab', 'corpora/stopwords', 'taggers/averaged_perceptron_tagger_eng'):
    nltk.data.find(res)
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)[:2]
    if cc >= (7, 5):
        importlib.import_module('fla')
        if os.environ.get('CBARR_ALLOW_KERNEL_SOURCE_BUILD') == '1':
            importlib.import_module('causal_conv1d')
" 2>/dev/null || [ ! -f "${DATA_DIR}/manifest_sha256.json" ]; then
      need_setup=1
    fi
  fi
  if [ "$need_setup" -eq 1 ]; then
    local had_venv=0 setup_t0 setup_t1 setup_dur setup_rc=0
    [ -x ".venv/bin/python" ] && had_venv=1
    setup_t0="$(date -u +%s)"
    log_banner "=== [Step 0: Environment Setup & Kernel Provisioning] ==="
    set +e
    PYTHON_BIN="$PY_BIN" bash scripts/setup_env.sh 2>&1 | tee -a "$RUN_LOG"
    setup_rc=$?
    set -e
    setup_t1="$(date -u +%s)"
    setup_dur=$((setup_t1 - setup_t0))
    if [ "$had_venv" -eq 0 ] && [ -x ".venv/bin/python" ]; then
      PY_BIN=".venv/bin/python"
    fi
    if [ "$setup_rc" -eq 0 ]; then
      record_stage_timing "Step 0: Environment Setup & Kernel Provisioning" "PASS" "$setup_dur"
    else
      record_stage_timing "Step 0: Environment Setup & Kernel Provisioning" "FAIL(exit=${setup_rc})" "$setup_dur"
      return "$setup_rc"
    fi
  fi
  if [ "$DATA_DIR" = "data" ] && [ ! -f "${DATA_DIR}/manifest_sha256.json" ]; then
    local frz_t0 frz_t1 frz_dur frz_rc=0
    frz_t0="$(date -u +%s)"
    log_banner "=== [Step 0: Freezing Deterministic Splits (${DATA_DIR})] ==="
    set +e
    "$PY_BIN" data/freeze_datasets.py 2>&1 | tee -a "$RUN_LOG"
    frz_rc=$?
    set -e
    frz_t1="$(date -u +%s)"
    frz_dur=$((frz_t1 - frz_t0))
    if [ "$frz_rc" -eq 0 ]; then
      record_stage_timing "Step 0: Freeze Deterministic Splits" "PASS" "$frz_dur"
    else
      record_stage_timing "Step 0: Freeze Deterministic Splits" "FAIL(exit=${frz_rc})" "$frz_dur"
      return "$frz_rc"
    fi
  fi
  "$PY_BIN" -c "import pathlib, sys; from scripts.run_matrix import verify_manifest_integrity; m = verify_manifest_integrity(pathlib.Path(sys.argv[1])); print(f'Verified {len(m)} frozen dataset splits in {sys.argv[1]}')" "$DATA_DIR" 2>&1 | tee -a "$RUN_LOG"
}

prune_hf_cache_if_low_disk() {
  local thresh_gb="${1:-$PRUNE_HF_BELOW_GB}"
  if [ "$IS_MOCK" -ne 0 ] || [ "$OFFLINE_MODE" -eq 1 ]; then
    return 0
  fi
  "$PY_BIN" -c "
import os, pathlib, shutil, sys
thresh = float(sys.argv[1])
if thresh <= 0:
    sys.exit(0)
hf_hub = pathlib.Path(os.environ.get('HF_HOME') or (pathlib.Path.home() / '.cache' / 'huggingface')) / 'hub'
if not hf_hub.is_dir():
    sys.exit(0)
free_gb = shutil.disk_usage(str(hf_hub)).free / (1024**3)
if free_gb < thresh:
    print(f'[Disk Guard] Free disk {free_gb:.1f} GB < {thresh:.1f} GB; pruning cached HF weights under {hf_hub}...', flush=True)
    for item in hf_hub.iterdir():
        if item.is_dir() and item.name.startswith('models--'):
            shutil.rmtree(item, ignore_errors=True)
" "$thresh_gb" 2>&1 | tee -a "$RUN_LOG" || true
}

check_workspace_disk_space() {
  local stage_label="${1:-Stage}"
  if [ "$IS_MOCK" -ne 0 ]; then
    return 0
  fi
  "$PY_BIN" -c "
import pathlib, shutil, sys
out_p = pathlib.Path(sys.argv[1])
if not out_p.exists():
    out_p = pathlib.Path('.')
free_gb = shutil.disk_usage(str(out_p)).free / (1024**3)
if free_gb < 1.0:
    print(f'[Disk Guard] ERROR: {sys.argv[2]} aborted — only {free_gb:.2f} GB free on {out_p.resolve()} (< 1.0 GB floor).', flush=True)
    sys.exit(1)
if free_gb < 5.0:
    print(f'[Disk Guard] WARN: Low disk space before {sys.argv[2]}: {free_gb:.1f} GB free on {out_p.resolve()}.', flush=True)
" "$OUT_DIR" "$stage_label" 2>&1 | tee -a "$RUN_LOG"
}

check_git_sync_readiness
bootstrap_environment_and_data

if [ "$SHARDS_ARG" = "auto" ]; then
  if [ -n "$GPU_ID" ] && [[ "$GPU_ID" != *","* ]]; then
    N_SHARDS=1
  else
    N_SHARDS="$("$PY_BIN" -c "import sys, torch; cap = max(1, int(sys.argv[1])); print(min(torch.cuda.device_count(), cap) if torch.cuda.is_available() else 1)" "$MAX_SHARDS_ARG" 2>/dev/null || echo 1)"
  fi
else
  N_SHARDS="$SHARDS_ARG"
fi
if [ "${N_SHARDS:-1}" -lt 1 ] 2>/dev/null; then
  N_SHARDS=1
fi

GPU_MIN_VRAM_GB=0
if [ "$IS_MOCK" -eq 0 ]; then
  GPU_MIN_VRAM_GB="$("$PY_BIN" -c "
import torch
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    mins = min(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count()))
    print(int(mins // (1024**3)))
else:
    print(0)
" 2>/dev/null || echo 0)"
fi

SHARD_VIS_GPUS_CSV=""
if [ "${#VISIBLE_GPUS[@]}" -gt 0 ]; then
  _gpu_subset=("${VISIBLE_GPUS[@]:0:$N_SHARDS}")
  SHARD_VIS_GPUS_CSV="$(IFS=','; echo "${_gpu_subset[*]}")"
else
  _base_gpus=()
  for ((_gi = 0; _gi < N_SHARDS; _gi++)); do
    _base_gpus+=("$_gi")
  done
  SHARD_VIS_GPUS_CSV="$(IFS=','; echo "${_base_gpus[*]}")"
fi

effective_workers_per_gpu() {
  local model_id="$1"
  local stage_kind="${2:-matrix}"
  if [ "$WORKERS_PER_GPU_ARG" != "auto" ]; then
    echo "$WORKERS_PER_GPU_ARG"
    return
  fi
  if [ "$IS_MOCK" -ne 0 ] || [ "$GPU_MIN_VRAM_GB" -lt 14 ]; then
    echo 1
    return
  fi
  if [ "$stage_kind" = "mlp" ] || [ "$stage_kind" = "replay" ] || [ "$stage_kind" = "smoke" ]; then
    echo 1
    return
  fi
  case "$model_id" in
    *9[Bb]*)
      # 9B uses ~21-24 GB VRAM per worker; keep 1 worker/GPU so 46 GB GPUs stay clear of OOM
      echo 1
      ;;
    *0.8[Bb]*|*08[Bb]*|*1[Bb]*)
      if [ "$stage_kind" = "blend" ]; then
        echo 2
      elif [ "$GPU_MIN_VRAM_GB" -ge 32 ]; then
        echo 4
      else
        echo 2
      fi
      ;;
    *2[Bb]*)
      if [ "$stage_kind" = "blend" ]; then
        echo 2
      elif [ "$GPU_MIN_VRAM_GB" -ge 64 ]; then
        echo 4
      elif [ "$GPU_MIN_VRAM_GB" -ge 32 ]; then
        echo 3
      else
        echo 2
      fi
      ;;
    *)
      if [ "$stage_kind" = "blend" ]; then
        if [ "$GPU_MIN_VRAM_GB" -ge 32 ]; then
          echo 2
        else
          echo 1
        fi
      elif [ "$GPU_MIN_VRAM_GB" -ge 64 ]; then
        echo 3
      elif [ "$GPU_MIN_VRAM_GB" -ge 32 ]; then
        echo 2
      else
        echo 1
      fi
      ;;
  esac
}

if [ "$WITH_9B" = "auto" ]; then
  if [ "$IS_MOCK" -ne 0 ]; then
    WITH_9B=0
  else
    WITH_9B="$("$PY_BIN" -c "import torch; ok = torch.cuda.is_available() and (torch.cuda.get_device_properties(0).total_memory / (1024**3) >= 22.0); print(1 if ok else 0)" 2>/dev/null || echo 0)"
  fi
fi

if [ "$WITH_R6" = "auto" ]; then
  if [ "$IS_MOCK" -ne 0 ] || [ "$PART_MODE" != "all" ] || [ "$STRICT_ONLY" -ne 0 ] || [ "$CALIBRATE_ONLY" -ne 0 ] || [ "${#MODELS_ARG[@]}" -gt 0 ]; then
    WITH_R6=0
  else
    WITH_R6=1
  fi
fi

if [ "$WITH_IFBENCH" = "auto" ]; then
  if [ "$IS_MOCK" -ne 0 ] || [ "$PART_MODE" != "all" ] || [ "$STRICT_ONLY" -ne 0 ] || [ "$CALIBRATE_ONLY" -ne 0 ] || [ "${#MODELS_ARG[@]}" -gt 0 ]; then
    WITH_IFBENCH=0
  else
    WITH_IFBENCH=1
  fi
fi

if [ "${#R6_SEEDS_ARG[@]}" -eq 0 ]; then
  if [ "$IS_MOCK" -eq 0 ]; then
    R6_SEEDS_ARG=(42 43 44)
  else
    R6_SEEDS_ARG=(42)
  fi
fi

if [ -z "$CHECKPOINT_EVERY" ]; then
  if [ "$IS_MOCK" -eq 1 ] || [ "$WORKER_MODE" -eq 1 ]; then
    CHECKPOINT_EVERY=0
  else
    CHECKPOINT_EVERY=150
  fi
fi

# Shared hyperparameters and per-step checkpoint flags forwarded into every scripts/run_matrix.py invocation
_rebuild_matrix_param_flags() {
  local eff_seed="${1:-$SEED}"
  MATRIX_PARAM_FLAGS=(
    --device "$DEVICE_ARG"
    --seed "$eff_seed"
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
  MATRIX_CACHED_PARAM_FLAGS=("${MATRIX_PARAM_FLAGS[@]}")
  if [ "$NO_EXP_LOG" -eq 0 ] && [ -n "$EXP_LOG" ]; then
    MATRIX_PARAM_FLAGS+=(--exp-log "$EXP_LOG")
    MATRIX_CACHED_PARAM_FLAGS+=(--exp-log "$EXP_LOG")
    if [ "$AUTO_PUSH" -eq 1 ] && [ "$IS_MOCK" -eq 0 ] && [ "$WORKER_MODE" -eq 0 ]; then
      MATRIX_PARAM_FLAGS+=(--git-push)
      if [ -n "$GIT_BRANCH" ]; then
        MATRIX_PARAM_FLAGS+=(--git-branch "$GIT_BRANCH")
      fi
    fi
  fi
}
_rebuild_matrix_param_flags "$SEED"

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
    'auto_commit': bool(int(sys.argv[12])),
    'auto_push': bool(int(sys.argv[13])),
    'git_branch': sys.argv[14],
    'setup_mode': sys.argv[15],
    'with_r6': bool(int(sys.argv[17])),
    'auto_b512': sys.argv[18],
    'with_ifbench': bool(int(sys.argv[19])),
    'r6_seeds': [int(s) for s in sys.argv[20].split() if s.strip()],
}
pathlib.Path(sys.argv[16]).write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
" "$DTYPE" "$SEED" "$LAM_UTILITY" "$NORM_TOKENS" "$REG_C" "$TAU_MIN" "$TAU_MAX" "$TAU_STEPS" "$WITH_9B" "$WITH_W7" "$IS_MOCK" "$AUTO_COMMIT" "$AUTO_PUSH" "$GIT_BRANCH" "$SETUP_MODE" "$PARAMS_FILE" "$WITH_R6" "$AUTO_B512" "$WITH_IFBENCH" "${R6_SEEDS_ARG[*]}"

checkpoint_exp_log() {
  local loop_label="$1"
  local run_status="$2"
  local allow_push="${3:-1}"
  local run_notes="${4:-}"
  local now_end total_sec md_args=() push_args=() smoke_args=() notes_args=()
  now_end="$(date -u +%s)"
  total_sec=$((now_end - PIPELINE_START_EPOCH))
  if [ "$NO_EXP_LOG" -eq 0 ] && [ -n "$EXP_LOG" ]; then
    md_args=(--md-log "$EXP_LOG")
  fi
  if [ "$allow_push" -eq 1 ] && [ "$AUTO_PUSH" -eq 1 ] && [ "$IS_MOCK" -eq 0 ] && [ "$WORKER_MODE" -eq 0 ] && [ "$NO_EXP_LOG" -eq 0 ] && [ -n "$EXP_LOG" ]; then
    push_args=(--git-push)
    if [ -n "$GIT_BRANCH" ]; then
      push_args+=(--git-branch "$GIT_BRANCH")
    fi
  fi
  if [ "$IS_MOCK" -eq 0 ] && [ "$SKIP_SMOKE" -eq 0 ] && [ "$RAN_LIVE_SMOKE" -eq 1 ]; then
    if [ -f "${OUT_DIR}/smoke_test_gate3_report.json" ]; then
      smoke_args=(--smoke-report "${OUT_DIR}/smoke_test_gate3_report.json")
    elif [ -f "outputs/smoke_test_gate3_report.json" ]; then
      smoke_args=(--smoke-report "outputs/smoke_test_gate3_report.json")
    fi
  fi
  if [ -n "$run_notes" ]; then
    notes_args=(--notes "$run_notes")
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
    ${notes_args[@]+"${notes_args[@]}"} \
    ${smoke_args[@]+"${smoke_args[@]}"} \
    ${md_args[@]+"${md_args[@]}"} \
    ${push_args[@]+"${push_args[@]}"} 2>&1 | tee -a "$RUN_LOG" || true
}

run_stage() {
  local stage_name="$1"
  shift
  CURRENT_STAGE="$stage_name"
  local t0 t1 dur rc stage_push=1
  case "$stage_name" in
    "Gate 0:"*|"Step 0a:"*|"Unpack "*|"Submission Audit"*|"GitHub Sync"*|*" (cached)") stage_push=0 ;;
  esac
  t0="$(date -u +%s)"
  log_banner "=== [$stage_name] START ==="
  set +e
  "$@" 2>&1 | tee -a "$RUN_LOG"
  rc=$?
  set -e
  t1="$(date -u +%s)"
  dur=$((t1 - t0))
  if [ "$rc" -eq 0 ]; then
    local total_run_sec=$((t1 - PIPELINE_START_EPOCH))
    record_stage_timing "$stage_name" "PASS" "$dur"
    log_banner "=== [$stage_name] DONE (${dur}s | total=${total_run_sec}s) ==="
    checkpoint_exp_log "$stage_name" "RUNNING (after ${stage_name})" "$stage_push"
  else
    record_stage_timing "$stage_name" "FAIL(exit=${rc})" "$dur"
    log_banner "=== [$stage_name] FAILED (exit=${rc}, ${dur}s) ==="
    return "$rc"
  fi
}

_extract_failure_notes() {
  "$PY_BIN" - "$RUN_LOG" <<'PY' 2>/dev/null || true
import pathlib, re, sys
log_p = pathlib.Path(sys.argv[1])
if not log_p.is_file():
    sys.exit(0)
lines = log_p.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
cleaned = [re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", ln).strip() for ln in lines]
cleaned = [ln for ln in cleaned if ln and not ln.startswith("[git-push]") and not ln.startswith("[log_experiment]")]
err_pat = re.compile(
    r"OutOfMemoryError|CUDA out of memory|Traceback\s*\(|RuntimeError:|ValueError:|"
    r"FileNotFoundError:|AssertionError:|CalledProcessError:|command not found|\bFAILED\b|\[ERROR\]|ERROR:",
    re.IGNORECASE,
)
hits = [ln for ln in cleaned if err_pat.search(ln)]
picked = hits[-3:] if hits else cleaned[-3:]
summary = " | ".join(picked).replace("`", "'")
print(summary[:320])
PY
}

finalize_run() {
  local rc=$?
  if [ "$FINALIZED" -eq 1 ]; then
    return
  fi
  FINALIZED=1
  local t_end total_sec status_str fail_notes=""
  t_end="$(date -u +%s)"
  total_sec=$((t_end - PIPELINE_START_EPOCH))
  if [ "$rc" -eq 0 ]; then
    status_str="PASSED"
  else
    status_str="FAILED (exit=${rc} at ${CURRENT_STAGE})"
    fail_notes="$(_extract_failure_notes)"
  fi
  checkpoint_exp_log "$CURRENT_STAGE" "$status_str" 1 "$fail_notes"
  log_banner "Run ${RUN_ID} finished with status=${status_str} (total=${total_sec}s). Full log: ${RUN_LOG}"
}
# Persist final PASSED/FAILED status, hardware diagnostics, and total wall time even on unexpected abort
trap finalize_run EXIT

run_matrix_sharded() {
  local stage_name="$1"
  local model_id="$2"
  shift 2
  CURRENT_STAGE="$stage_name"
  local base_args=(
    "$PY_BIN" scripts/run_matrix.py
    --model "$model_id"
    --data-dir "$DATA_DIR"
    --out-dir "$OUT_DIR"
    --dtype "$DTYPE"
    "$@"
    ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
  )
  local eff_wpg eff_shards rc_all=0 rc_stage=0 slug="${model_id//\//__}" stage_vis_csv="$SHARD_VIS_GPUS_CSV"
  local attempt=0 max_attempts=1
  if [ "$IS_MOCK" -eq 0 ]; then
    max_attempts=$((STAGE_RETRIES + 1))
  fi

  while [ "$attempt" -lt "$max_attempts" ]; do
    attempt=$((attempt + 1))
    check_workspace_disk_space "$stage_name" || return 1

    if [ "$IS_MOCK" -eq 0 ] && "${base_args[@]}" --seed "$SEED" --check-fully-cached >/dev/null 2>&1; then
      log_banner "=== [$stage_name] All items already cached — skipping GPU launch ==="
      run_stage "${stage_name} (cached)" "${base_args[@]}" --merge-shards "${MATRIX_CACHED_PARAM_FLAGS[@]}"
      return $?
    fi

    eff_shards=1
    eff_wpg=""
    stage_vis_csv="$SHARD_VIS_GPUS_CSV"
    local picked_single_gpu="" pool_stage="matrix" retry_mult="1.0" retry_wpg="$WORKERS_PER_GPU_ARG"
    for _fl in "$@"; do
      if [ "$_fl" = "--calibrate-only" ]; then
        pool_stage="calibrate"
        break
      fi
    done
    if [ "$pool_stage" = "matrix" ]; then
      if [[ "$stage_name" == *"(Contingency R3 B=512)"* ]]; then
        pool_stage="b512"
      elif [[ "$stage_name" == *"(diag491)"* ]]; then
        pool_stage="diag"
      fi
    fi
    if [ "$attempt" -eq 2 ]; then
      retry_mult="1.25"
      retry_wpg="1"
    elif [ "$attempt" -ge 3 ]; then
      retry_mult="1.50"
      retry_wpg="1"
    fi
    if [ "$IS_MOCK" -eq 0 ] && [ "$SHARDS_ARG" = "auto" ]; then
      local live_slots=""
      if ! live_slots="$(select_gpu_pool_slots \
        --model "$model_id" \
        --candidates "$PINNED_GPU_CSV" \
        --workers-per-gpu "$retry_wpg" \
        --max-gpus "$MAX_SHARDS_ARG" \
        --stage "$pool_stage" \
        --safety-mult "$retry_mult" \
        --wait-sec "$CBARR_WAIT_GPU_SEC")"; then
        record_stage_timing "$stage_name" "FAIL(no_free_gpu)" 0
        log_banner "=== [$stage_name] FAILED: no GPU with sufficient free VRAM became available ==="
        return 1
      fi
      if [ "$live_slots" != "NO_CUDA" ] && [ -n "$live_slots" ]; then
        local slot_arr=()
        IFS=',' read -r -a slot_arr <<< "$live_slots"
        eff_shards="${#slot_arr[@]}"
        stage_vis_csv="$live_slots"
        if [ "$eff_shards" -eq 1 ]; then
          picked_single_gpu="${slot_arr[0]}"
        fi
      else
        if [ "$attempt" -ge 2 ]; then
          eff_wpg=1
        else
          eff_wpg="$(effective_workers_per_gpu "$model_id" "$pool_stage")"
        fi
        eff_shards=$((N_SHARDS * eff_wpg))
      fi
    else
      if [ "$attempt" -ge 2 ]; then
        eff_wpg=1
      else
        eff_wpg="$(effective_workers_per_gpu "$model_id" "$pool_stage")"
      fi
      eff_shards=$((N_SHARDS * eff_wpg))
    fi

    if [ "$eff_shards" -le 1 ]; then
      set +e
      if [ -n "$picked_single_gpu" ]; then
        CUDA_VISIBLE_DEVICES="$picked_single_gpu" run_stage "$stage_name" "${base_args[@]}" "${MATRIX_PARAM_FLAGS[@]}"
      else
        run_stage "$stage_name" "${base_args[@]}" "${MATRIX_PARAM_FLAGS[@]}"
      fi
      rc_stage=$?
      set -e
      if [ "$rc_stage" -eq 0 ]; then
        return 0
      fi
      if [ "$attempt" -lt "$max_attempts" ]; then
        log_banner "=== [$stage_name] Attempt ${attempt}/${max_attempts} failed (rc=${rc_stage}); completed items cached — rescanning live GPUs in 15s ==="
        sleep 15
        continue
      fi
      return "$rc_stage"
    fi

    local t0 t1 dur
    t0="$(date -u +%s)"
    if [ -n "$eff_wpg" ]; then
      log_banner "=== [$stage_name] START (${eff_shards} shards across ${N_SHARDS} GPU(s), ${eff_wpg} workers/GPU | vis=${stage_vis_csv}) ==="
    else
      log_banner "=== [$stage_name] START (${eff_shards} shards on GPU slots [${stage_vis_csv}], attempt ${attempt}/${max_attempts}) ==="
    fi
    set +e
    "$PY_BIN" scripts/run_sharded.py \
      --stage-name "$stage_name" \
      --model "$model_id" \
      --data-dir "$DATA_DIR" \
      --out-dir "$OUT_DIR" \
      --log-dir "$LOG_DIR" \
      --run-id "$RUN_ID" \
      --shards "$eff_shards" \
      --visible-gpus "$stage_vis_csv" \
      --progress-sec "$SHARD_PROGRESS_SEC" \
      --shard-log "$SHARD_LOG_MODE" \
      -- \
      "${base_args[@]}" \
      --device "$DEVICE_ARG" \
      --seed "$SEED" \
      --lam-utility "$LAM_UTILITY" \
      --norm-tokens "$NORM_TOKENS" \
      --reg-c "$REG_C" \
      --tau-min "$TAU_MIN" \
      --tau-max "$TAU_MAX" \
      --tau-steps "$TAU_STEPS" 2>&1 | tee -a "$RUN_LOG"
    rc_all=$?
    set -e
    t1="$(date -u +%s)"
    dur=$((t1 - t0))
    if [ "$rc_all" -eq 0 ]; then
      local total_run_sec=$((t1 - PIPELINE_START_EPOCH))
      record_stage_timing "$stage_name" "PASS" "$dur"
      log_banner "=== [$stage_name] SHARDS DONE (${dur}s | total=${total_run_sec}s) — merging shards ==="
      run_stage "${stage_name} (merge)" "${base_args[@]}" --merge-shards "${MATRIX_PARAM_FLAGS[@]}"
      return $?
    fi

    record_stage_timing "$stage_name" "FAIL(shard_error)" "$dur"
    log_banner "=== [$stage_name] FAILED in parallel GPU shard (attempt ${attempt}/${max_attempts}, ${dur}s; inspect ${LOG_DIR}/cbarr_${RUN_ID}_gpu*_${slug}.log) ==="
    if [ "$attempt" -lt "$max_attempts" ]; then
      log_banner "=== [$stage_name] Partial shards harvested — rescanning live free GPUs in 15s ==="
      sleep 15
    else
      return 1
    fi
  done
  return 1
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
    local smoke_args=() smoke_gpu=""
    if [ "$#" -gt 0 ]; then
      smoke_args=(--models "$@")
    elif [ "${#MODELS_ARG[@]}" -gt 0 ]; then
      smoke_args=(--models "${MODELS_ARG[@]}")
    fi
    if "$PY_BIN" scripts/run_smoke_test.py --check-cached --out-dir "$OUT_DIR" ${smoke_args[@]+"${smoke_args[@]}"} >/dev/null 2>&1; then
      log_banner "=== [Step 0a: Live GPU 12-Item Smoke] All smoke items already cached — skipping GPU launch ==="
      run_stage "Step 0a: Live GPU 12-Item Smoke (cached)" "$PY_BIN" scripts/run_smoke_test.py --live-gpu --resume --device "$DEVICE_ARG" --dtype "$DTYPE" --out-dir "$OUT_DIR" ${smoke_args[@]+"${smoke_args[@]}"}
    else
      if ! smoke_gpu="$(select_gpu_pool_slots --model "Qwen/Qwen3.5-4B" --candidates "$PINNED_GPU_CSV" --stage smoke --single --wait-sec "$CBARR_WAIT_GPU_SEC")"; then
        record_stage_timing "Step 0a: Live GPU 12-Item Smoke" "FAIL(no_free_gpu)" 0
        log_banner "=== [Step 0a: Live GPU 12-Item Smoke] FAILED: no GPU with sufficient free VRAM became available ==="
        return 1
      fi
      if [ -n "$smoke_gpu" ] && [ "$smoke_gpu" != "NO_CUDA" ]; then
        CUDA_VISIBLE_DEVICES="$smoke_gpu" run_stage "Step 0a: Live GPU 12-Item Smoke" "$PY_BIN" scripts/run_smoke_test.py --live-gpu --resume --device "$DEVICE_ARG" --dtype "$DTYPE" --out-dir "$OUT_DIR" ${smoke_args[@]+"${smoke_args[@]}"}
      else
        run_stage "Step 0a: Live GPU 12-Item Smoke" "$PY_BIN" scripts/run_smoke_test.py --live-gpu --resume --device "$DEVICE_ARG" --dtype "$DTYPE" --out-dir "$OUT_DIR" ${smoke_args[@]+"${smoke_args[@]}"}
      fi
    fi
    RAN_LIVE_SMOKE=1
  fi
}

run_gate0_preflight() {
  if [ "$SKIP_PREFLIGHT" -ne 0 ]; then
    return 0
  fi
  if [ "$FULL_PREFLIGHT" -eq 1 ]; then
    run_stage "Gate 0: Preflight Tests" "$PY_BIN" -W ignore::DeprecationWarning -W ignore::ImportWarning -m unittest discover -s tests -v
  else
    run_stage "Gate 0: Preflight Tests" "$PY_BIN" -W ignore::DeprecationWarning -W ignore::ImportWarning -m unittest discover -s tests -p "test_cbarr_preflight.py" -v
  fi
}

_r3_b512_triggered() {
  local model_id="$1" slug="${model_id//\//__}"
  local summary_p="${OUT_DIR}/${slug}/matrix_summary_strict341.json"
  if [ ! -f "$summary_p" ]; then
    summary_p="${OUT_DIR}/${slug}/matrix_summary.json"
  fi
  if [ ! -f "$summary_p" ]; then
    return 1
  fi
  "$PY_BIN" - "$summary_p" <<'PY' 2>/dev/null
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if "M2_B512" in (d.get("auxiliary_systems") or {}):
    sys.exit(1)
sys_d = d.get("systems") or {}
b1b = sys_d.get("B1b") or {}
m2 = sys_d.get("M2") or {}
b1_hard = float((b1b.get("slice_acc") or {}).get("gsm8k_hard", 1.0))
m2_hard = float((m2.get("slice_acc") or {}).get("gsm8k_hard", 1.0))
b1_plan = float((b1b.get("slice_acc") or {}).get("ifeval_planning", 1.0))
triggered = ((b1_hard - m2_hard) > 0.03) or (b1_hard < 0.70) or (b1_plan < 0.70)
sys.exit(0 if triggered else 1)
PY
}

run_model_eval_pair() {
  local prefix="$1"
  local model_id="$2"
  local diag_extra=()
  run_matrix_sharded "${prefix}: ${model_id} (strict341)" "$model_id" --strict-diag
  if [ "$INCLUDE_B512" -eq 0 ] && [ "$AUTO_B512" != "0" ] && [ "$IS_MOCK" -eq 0 ] && [ "$STRICT_ONLY" -eq 0 ]; then
    if _r3_b512_triggered "$model_id"; then
      log_banner "=== [Contingency R3] ${model_id} triggered B=512 sweep — backfilling M2_B512 onto eval_450 ==="
      run_matrix_sharded "${prefix}: ${model_id} (Contingency R3 B=512)" "$model_id" --strict-diag --include-b512
      diag_extra=(--include-b512)
    fi
  fi
  if [ "$STRICT_ONLY" -eq 0 ]; then
    run_matrix_sharded "${prefix}: ${model_id} (diag491)" "$model_id" ${diag_extra[@]+"${diag_extra[@]}"}
  fi
}

_run_single_gpu_ext() {
  local stage_label="$1"
  local ext_model="$2"
  local pool_stage="$3"
  shift 3
  local ext_gpu="NO_CUDA"
  if [ "$IS_MOCK" -eq 0 ]; then
    if ! ext_gpu="$(select_gpu_pool_slots --model "$ext_model" --candidates "$PINNED_GPU_CSV" --stage "$pool_stage" --single --wait-sec "$CBARR_WAIT_GPU_SEC")"; then
      record_stage_timing "$stage_label" "FAIL(no_free_gpu)" 0
      log_banner "=== [${stage_label}] FAILED: no GPU with sufficient free VRAM became available ==="
      return 1
    fi
  fi
  if [ -n "$ext_gpu" ] && [ "$ext_gpu" != "NO_CUDA" ]; then
    CUDA_VISIBLE_DEVICES="$ext_gpu" run_stage "$stage_label" "$@"
  else
    run_stage "$stage_label" "$@"
  fi
}

_run_blend_ext() {
  local stage_label="$1"
  local ext_model="$2"
  shift 2
  if [ "$IS_MOCK" -eq 0 ] && "$PY_BIN" scripts/ext_deltanet_blend.py --model "$ext_model" --out-dir "$OUT_DIR" --check-cached "$@" >/dev/null 2>&1; then
    log_banner "=== [${stage_label}] All pairs already cached — computing summary on CPU ==="
    run_stage "${stage_label} (cached)" "$PY_BIN" scripts/ext_deltanet_blend.py --model "$ext_model" --out-dir "$OUT_DIR" "$@"
    return $?
  fi
  if [ -n "$PINNED_GPU_CSV" ]; then
    CUDA_VISIBLE_DEVICES="$PINNED_GPU_CSV" run_stage "$stage_label" \
      "$PY_BIN" scripts/ext_deltanet_blend.py --model "$ext_model" --out-dir "$OUT_DIR" "$@"
  else
    run_stage "$stage_label" \
      env -u CUDA_VISIBLE_DEVICES "$PY_BIN" scripts/ext_deltanet_blend.py --model "$ext_model" --out-dir "$OUT_DIR" "$@"
  fi
}

run_w7_stage() {
  local w7_model="$1"
  if [ "$WITH_W7" -ne 1 ]; then
    return 0
  fi
  check_workspace_disk_space "W7 (${w7_model})" || return 1
  local w7_slug="${w7_model//\//__}"
  local w7_flags=() blend_flags=() offline_flags=()
  if [ "$IS_MOCK" -eq 1 ]; then
    w7_flags+=(--mock)
  else
    blend_flags=(--max-retries "$STAGE_RETRIES")
  fi
  if [ "$OFFLINE_MODE" -eq 1 ]; then
    offline_flags+=(--offline)
  fi

  case "$w7_model" in
    *[Ss]mol[Ll][Mm]*)
      if [ "$IS_MOCK" -eq 1 ]; then
        _run_blend_ext "W7: Dense KV Eviction (${w7_model})" "$w7_model" --evict-think-kv --out-json "${OUT_DIR}/${w7_slug}/ext_deltanet_blend_evict_kv.json" "${w7_flags[@]}"
      else
        _run_blend_ext "W7: Dense KV Eviction (${w7_model})" "$w7_model" --evict-think-kv --alphas 0.0 --out-json "${OUT_DIR}/${w7_slug}/ext_deltanet_blend_evict_kv.json" ${blend_flags[@]+"${blend_flags[@]}"}
      fi
      ;;
    *)
      _run_blend_ext "W7: DeltaNet Blend (${w7_model})" "$w7_model" ${blend_flags[@]+"${blend_flags[@]}"} ${w7_flags[@]+"${w7_flags[@]}"}
      if [ "$IS_MOCK" -eq 0 ] || [ "$PART_MODE" = "all" ]; then
        _run_blend_ext "W7: DeltaNet Conv+KV Blend (${w7_model})" "$w7_model" --blend-conv-state --evict-think-kv --out-json "${OUT_DIR}/${w7_slug}/ext_deltanet_blend_conv_kv.json" ${blend_flags[@]+"${blend_flags[@]}"} ${w7_flags[@]+"${w7_flags[@]}"}
      fi
      ;;
  esac

  if [ "$IS_MOCK" -eq 0 ] && "$PY_BIN" scripts/ext_prefill_mlp.py --model "$w7_model" --out-dir "$OUT_DIR" --check-cached >/dev/null 2>&1; then
    log_banner "=== [W7: Prefill MLP Probe (${w7_model})] Using cached .npz features — skipping GPU launch ==="
    run_stage "W7: Prefill MLP Probe (${w7_model}) (cached)" \
      "$PY_BIN" scripts/ext_prefill_mlp.py --model "$w7_model" --out-dir "$OUT_DIR"
  else
    _run_single_gpu_ext "W7: Prefill MLP Probe (${w7_model})" "$w7_model" mlp \
      "$PY_BIN" scripts/ext_prefill_mlp.py --model "$w7_model" --out-dir "$OUT_DIR" ${w7_flags[@]+"${w7_flags[@]}"}
  fi

  if [ "$IS_MOCK" -eq 0 ] && "$PY_BIN" scripts/compare_m2_v2.py --model "$w7_model" --out-dir "$OUT_DIR" --check-cached >/dev/null 2>&1; then
    log_banner "=== [W7: Stage-2 SolveFirst Replay (${w7_model})] Already cached — skipping GPU launch ==="
    record_stage_timing "W7: Stage-2 SolveFirst Replay (${w7_model}) (cached)" "PASS" 0
  else
    _run_single_gpu_ext "W7: Stage-2 SolveFirst Replay (${w7_model})" "$w7_model" replay \
      "$PY_BIN" scripts/compare_m2_v2.py --model "$w7_model" --out-dir "$OUT_DIR" --device "$DEVICE_ARG" --dtype "$DTYPE" ${offline_flags[@]+"${offline_flags[@]}"} ${w7_flags[@]+"${w7_flags[@]}"}
  fi
}

run_ifbench_ood_stage() {
  if [ "$WITH_IFBENCH" != "1" ] || [ "$STRICT_ONLY" -ne 0 ] || [ "$CALIBRATE_ONLY" -ne 0 ]; then
    return 0
  fi
  local _ifbm _ifb_slug
  for _ifbm in "${IFBENCH_MODELS_ARG[@]}"; do
    check_workspace_disk_space "IFBench OOD (${_ifbm})" || return 1
    _ifb_slug="${_ifbm//\//__}"
    local ifb_args=(
      --model "$_ifbm"
      --out-dir "$OUT_DIR"
      --dtype "$DTYPE"
    )
    if [ "$OFFLINE_MODE" -eq 1 ]; then
      ifb_args+=(--local-files-only)
    fi
    if [ "$IS_MOCK" -eq 1 ]; then
      ifb_args+=(--mock --allow-default-params --shards 1)
    else
      if [ -f "${OUT_DIR}/${_ifb_slug}/ext_ifbench_ood.json" ] && "$PY_BIN" scripts/ext_ifbench_ood.py "${ifb_args[@]}" --check-cached >/dev/null 2>&1; then
        log_banner "=== [Appendix B: IFBench OOD (${_ifbm})] Already cached — skipping GPU launch ==="
        record_stage_timing "Appendix B: IFBench OOD (${_ifbm}) (cached)" "PASS" 0
        continue
      fi
      ifb_args+=(--shards auto --workers-per-gpu "$WORKERS_PER_GPU_ARG" --wait-gpu-min "$WAIT_GPU_MIN" --max-retries "$STAGE_RETRIES")
      if [ -n "$PINNED_GPU_CSV" ]; then
        ifb_args+=(--gpu-ids "$PINNED_GPU_CSV")
      fi
    fi
    run_stage "Appendix B: IFBench OOD (${_ifbm})" "$PY_BIN" scripts/ext_ifbench_ood.py "${ifb_args[@]}"
    sync_model_checkpoint "$_ifbm"
  done
}

run_r6_sampled_stage() {
  if [ "$WITH_R6" != "1" ] || [ "$STRICT_ONLY" -ne 0 ] || [ "$CALIBRATE_ONLY" -ne 0 ]; then
    return 0
  fi
  local _r6m _r6_seed _r6_slug _orig_seed
  _orig_seed="$SEED"
  for _r6m in "${R6_MODELS_ARG[@]}"; do
    for _r6_seed in "${R6_SEEDS_ARG[@]}"; do
      SEED="$_r6_seed"
      _rebuild_matrix_param_flags "$SEED"
      run_matrix_sharded "Contingency R6 Sampled (T=0.6, seed=${_r6_seed}): ${_r6m}" "$_r6m" \
        --strict-diag --temperature 0.6 --top-p 0.95 --presence-penalty 1.5
      _r6_slug="${_r6m//\//__}__T0.6_tp0.95_pp1.5_s${_r6_seed}"
      sync_model_checkpoint "$_r6_slug"
    done
  done
  SEED="$_orig_seed"
  _rebuild_matrix_param_flags "$SEED"
}

run_phase7_stage() {
  local p7_flags=(--out-dir "$OUT_DIR" --fig-dir "$FIG_DIR")
  if [ -n "$TABLE_DIR" ]; then
    p7_flags+=(--table-dir "$TABLE_DIR")
  fi
  if [ "$STRICT_VERSION_MATCH" -eq 1 ]; then
    p7_flags+=(--strict-version-match)
  fi
  run_stage "Phase 7 / W8: Analysis, Tables & Figures" \
    "$PY_BIN" scripts/analyze_and_plot.py "${p7_flags[@]}"
  if [ "$IS_MOCK" -eq 0 ] && [ "$OUT_DIR" = "outputs" ] && [ -f "paper/main.tex" ] && command -v pdflatex >/dev/null 2>&1; then
    log_banner "=== [Phase 8: Compiling LaTeX Manuscript PDF (paper/main.pdf)] ==="
    if (
      cd paper &&
      pdflatex -interaction=nonstopmode -halt-on-error main.tex &&
      { bibtex main || true; } &&
      pdflatex -interaction=nonstopmode -halt-on-error main.tex &&
      pdflatex -interaction=nonstopmode -halt-on-error main.tex
    ) >>"$RUN_LOG" 2>&1; then
      record_stage_timing "Phase 8: Paper LaTeX Build" "PASS" 0
      log_banner "=== [Phase 8: Compiled paper/main.pdf cleanly] ==="
    else
      record_stage_timing "Phase 8: Paper LaTeX Build" "WARN(pdflatex_failed)" 0
      log_banner "=== [Phase 8: WARN — pdflatex compilation reported an issue; inspect ${RUN_LOG}] ==="
    fi
  fi
}

sync_model_checkpoint() {
  local model_id="$1"
  local slug="${model_id##*/}"
  if [ "$AUTO_COMMIT" -ne 1 ] || [ "$IS_MOCK" -ne 0 ] || [ "$WORKER_MODE" -ne 0 ]; then
    return 0
  fi
  local ckpt_flags=(
    --mode push
    --repo-dir .
    --out-dir "$OUT_DIR"
    --label "checkpoint-${slug}:${RUN_ID}"
    --no-phase7
    --models "$model_id"
  )
  if [ "$AUTO_PUSH" -eq 1 ]; then
    ckpt_flags+=(--git-push)
  else
    ckpt_flags+=(--no-git-push)
  fi
  if [ -n "$GIT_BRANCH" ]; then
    ckpt_flags+=(--git-branch "$GIT_BRANCH")
  fi
  log_banner "=== [Checkpoint Sync] Backing up ${model_id} caches & summaries ==="
  "$PY_BIN" scripts/sync_results.py "${ckpt_flags[@]}" 2>&1 | tee -a "$RUN_LOG" || \
    log_banner "[WARN] Non-fatal incremental checkpoint sync failed for ${model_id} (will retry on next/final sync)."
}

sync_part_results() {
  local sync_label="$1"
  local include_phase7="$2"
  shift 2
  if { [ "$AUTO_COMMIT" -ne 1 ] && [ "$AUTO_PUSH" -ne 1 ]; } || [ "$IS_MOCK" -ne 0 ]; then
    return 0
  fi
  # Mark ledger PASSED locally first so sync_results.py commits & pushes final state atomically
  FINALIZED=1
  checkpoint_exp_log "$CURRENT_STAGE" "PASSED" 0

  local sync_flags=(
    --mode push
    --repo-dir .
    --out-dir "$OUT_DIR"
    --label "${sync_label}:${RUN_ID}"
  )
  if [ "$AUTO_PUSH" -eq 1 ]; then
    sync_flags+=(--git-push --strict)
  else
    sync_flags+=(--no-git-push)
  fi
  if [ -n "$GIT_BRANCH" ]; then
    sync_flags+=(--git-branch "$GIT_BRANCH")
  fi
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
  local t_end total_sec
  t_end="$(date -u +%s)"
  total_sec=$((t_end - PIPELINE_START_EPOCH))
  log_banner "Run ${RUN_ID} finished with status=PASSED (total=${total_sec}s). Full log: ${RUN_LOG}"
}

# Log environment snapshot upfront for debugging & reproducibility
{
  echo "========================================================================"
  echo "C-BARR Run ID      : ${RUN_ID} (part=${PART_MODE}, shards=${N_SHARDS}, with_9b=${WITH_9B}, with_w7=${WITH_W7})"
  echo "UTC Start          : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Code Version       : v$(cat VERSION 2>/dev/null || echo 0.0.0) (git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown))"
  echo "Python Binary      : ${PY_BIN}"
  echo "DATA_DIR / OUT_DIR : ${DATA_DIR} / ${OUT_DIR} -> FIG_DIR=${FIG_DIR}"
  echo "DTYPE / CUDA_DEV   : ${DTYPE} / ${CUDA_VISIBLE_DEVICES:-auto}"
  echo "Shard Monitor      : gpus=${N_SHARDS} (vis=${SHARD_VIS_GPUS_CSV}) | workers_per_gpu=${WORKERS_PER_GPU_ARG} (min_vram=${GPU_MIN_VRAM_GB}G) | retries=${STAGE_RETRIES} | wait=${WAIT_GPU_MIN}m | progress=${SHARD_PROGRESS_SEC}s | shard_log=${SHARD_LOG_MODE}"
  echo "GIT COMMIT / PUSH  : commit=${AUTO_COMMIT} push=${AUTO_PUSH} branch=${GIT_BRANCH:-current}"
  echo "HF_HOME / PRUNE    : ${HF_HOME:-default} (prune below ${PRUNE_HF_BELOW_GB} GB) | HF_TOKEN=$([ -n "${HF_TOKEN:-}" ] && echo set || echo unset)"
  echo "Command            : ${RAW_CMD}"
  echo "--- Disk Space ---"
  df -h . || true
  "$PY_BIN" -c "
import os, pathlib, shutil
hf = pathlib.Path(os.environ.get('HF_HOME') or pathlib.Path.home())
if not hf.exists():
    hf = hf.parent if hf.parent.exists() else pathlib.Path('.')
free_gb = shutil.disk_usage(str(hf)).free / (1024**3)
print(f'HF Cache Free Disk : {free_gb:.1f} GB ({hf})')
if free_gb < 45.0:
    print(f'[Disk Guard] Free space {free_gb:.1f} GB < 45 GB; auto-pruning between models below ${PRUNE_HF_BELOW_GB} GB (use --hf-home <dir> if another mount has more space).')
" 2>/dev/null || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- nvidia-smi ---"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader || nvidia-smi || true
  fi
  echo "========================================================================"
} 2>&1 | tee -a "$RUN_LOG"

case "$PART_MODE" in
  1)
    run_gate0_preflight
    run_hw_probe_and_smoke 1 Qwen/Qwen3.5-4B
    run_matrix_sharded "Step 0b: Primary 4B Gate-2 Calibration" Qwen/Qwen3.5-4B --calibrate-only
    if [ "$CALIBRATE_ONLY" -eq 0 ]; then
      run_model_eval_pair "Phase 4 / W4" Qwen/Qwen3.5-4B
    fi
    run_stage "Submission Audit: Double-Blind Anonymity" bash scripts/check_anonymity.sh
    if [ "$CALIBRATE_ONLY" -eq 0 ]; then
      sync_part_results "script-part1-4b" 0 Qwen/Qwen3.5-4B
    fi
    ;;

  2)
    run_hw_probe_and_smoke 0
    if [ "$CALIBRATE_ONLY" -eq 1 ]; then
      run_matrix_sharded "Calibrate: HuggingFaceTB/SmolLM3-3B" HuggingFaceTB/SmolLM3-3B --calibrate-only
    else
      run_model_eval_pair "Phase 5 / W5" HuggingFaceTB/SmolLM3-3B
    fi
    run_stage "Submission Audit: Double-Blind Anonymity" bash scripts/check_anonymity.sh
    if [ "$CALIBRATE_ONLY" -eq 0 ]; then
      sync_part_results "script-part2-smollm3" 0 HuggingFaceTB/SmolLM3-3B
    fi
    ;;

  3)
    run_hw_probe_and_smoke 0
    if [ "$CALIBRATE_ONLY" -eq 1 ]; then
      run_matrix_sharded "Calibrate: Qwen/Qwen3.5-2B" Qwen/Qwen3.5-2B --calibrate-only
    else
      run_matrix_sharded "Phase 6 / W6: Qwen/Qwen3.5-2B (strict341)" Qwen/Qwen3.5-2B --strict-diag
    fi
    run_stage "Submission Audit: Double-Blind Anonymity" bash scripts/check_anonymity.sh
    if [ "$CALIBRATE_ONLY" -eq 0 ]; then
      sync_part_results "script-part3-2b" 0 Qwen/Qwen3.5-2B
    fi
    ;;

  4)
    run_hw_probe_and_smoke 0
    SCALE4_MODELS=(Qwen/Qwen3.5-0.8B)
    if [ "$WITH_9B" -eq 1 ]; then
      SCALE4_MODELS+=(Qwen/Qwen3.5-9B)
    fi
    if [ "$CALIBRATE_ONLY" -eq 1 ]; then
      for _sm in "${SCALE4_MODELS[@]}"; do
        run_matrix_sharded "Calibrate: ${_sm}" "$_sm" --calibrate-only
      done
    else
      for _sm in "${SCALE4_MODELS[@]}"; do
        run_matrix_sharded "Phase 6 / W6: ${_sm} (strict341)" "$_sm" --strict-diag
      done
    fi
    run_stage "Submission Audit: Double-Blind Anonymity" bash scripts/check_anonymity.sh
    if [ "$CALIBRATE_ONLY" -eq 0 ]; then
      sync_part_results "script-part4-08b" 0 "${SCALE4_MODELS[@]}"
    fi
    ;;

  5|combine)
    if [ "$UNPACK_MODE" = "on" ] || { [ "$UNPACK_MODE" = "auto" ] && [ "$IS_MOCK" -eq 0 ] && compgen -G "outputs/*/*_cache.jsonl.gz" >/dev/null; }; then
      run_stage "Unpack Synced Worker Results" "$PY_BIN" scripts/sync_results.py --mode unpack --repo-dir . --out-dir "$OUT_DIR"
    fi
    run_gate0_preflight
    if [ "${#MODELS_ARG[@]}" -gt 0 ]; then
      for _wm in "${MODELS_ARG[@]}"; do
        run_w7_stage "$_wm"
      done
    else
      run_w7_stage Qwen/Qwen3.5-4B
      if [ -d "${OUT_DIR}/HuggingFaceTB__SmolLM3-3B" ]; then
        run_w7_stage HuggingFaceTB/SmolLM3-3B
      fi
    fi
    run_ifbench_ood_stage
    run_r6_sampled_stage
    run_phase7_stage
    run_stage "Submission Audit: Double-Blind Anonymity" bash scripts/check_anonymity.sh
    sync_part_results "script-part5-combined" 1
    ;;

  w7)
    if [ "$UNPACK_MODE" = "on" ] || { [ "$UNPACK_MODE" = "auto" ] && [ "$IS_MOCK" -eq 0 ] && compgen -G "outputs/*/*_cache.jsonl.gz" >/dev/null; }; then
      run_stage "Unpack Synced Worker Results" "$PY_BIN" scripts/sync_results.py --mode unpack --repo-dir . --out-dir "$OUT_DIR"
    fi
    run_gate0_preflight
    WITH_W7=1
    if [ "${#MODELS_ARG[@]}" -gt 0 ]; then
      for _wm in "${MODELS_ARG[@]}"; do
        run_w7_stage "$_wm"
      done
      _w7_sync_models=("${MODELS_ARG[@]}")
    else
      _w7_sync_models=()
      if [ -d "${OUT_DIR}/Qwen__Qwen3.5-4B" ] || [ "$IS_MOCK" -eq 1 ]; then
        run_w7_stage Qwen/Qwen3.5-4B
        _w7_sync_models+=(Qwen/Qwen3.5-4B)
      fi
      if [ -d "${OUT_DIR}/HuggingFaceTB__SmolLM3-3B" ]; then
        run_w7_stage HuggingFaceTB/SmolLM3-3B
        _w7_sync_models+=(HuggingFaceTB/SmolLM3-3B)
      fi
    fi
    run_phase7_stage
    run_stage "Submission Audit: Double-Blind Anonymity" bash scripts/check_anonymity.sh
    sync_part_results "script-part-w7" 1 ${_w7_sync_models[@]+"${_w7_sync_models[@]}"}
    ;;

  r6)
    if [ "$UNPACK_MODE" = "on" ] || { [ "$UNPACK_MODE" = "auto" ] && [ "$IS_MOCK" -eq 0 ] && compgen -G "outputs/*/*_cache.jsonl.gz" >/dev/null; }; then
      run_stage "Unpack Synced Worker Results" "$PY_BIN" scripts/sync_results.py --mode unpack --repo-dir . --out-dir "$OUT_DIR"
    fi
    run_gate0_preflight
    WITH_R6="1"
    run_r6_sampled_stage
    run_phase7_stage
    run_stage "Submission Audit: Double-Blind Anonymity" bash scripts/check_anonymity.sh
    sync_part_results "script-part-r6" 1
    ;;

  analyze)
    if [ "$UNPACK_MODE" = "on" ] || { [ "$UNPACK_MODE" = "auto" ] && [ "$IS_MOCK" -eq 0 ] && compgen -G "outputs/*/*_cache.jsonl.gz" >/dev/null; }; then
      run_stage "Unpack Synced Worker Results" "$PY_BIN" scripts/sync_results.py --mode unpack --repo-dir . --out-dir "$OUT_DIR"
    fi
    run_gate0_preflight
    run_phase7_stage
    run_stage "Submission Audit: Double-Blind Anonymity" bash scripts/check_anonymity.sh
    sync_part_results "script-part-analyze" 1
    ;;

  ifbench)
    if [ "$UNPACK_MODE" = "on" ] || { [ "$UNPACK_MODE" = "auto" ] && [ "$IS_MOCK" -eq 0 ] && compgen -G "outputs/*/*_cache.jsonl.gz" >/dev/null; }; then
      run_stage "Unpack Synced Worker Results" "$PY_BIN" scripts/sync_results.py --mode unpack --repo-dir . --out-dir "$OUT_DIR"
    fi
    run_gate0_preflight
    WITH_IFBENCH="1"
    run_ifbench_ood_stage
    run_phase7_stage
    run_stage "Submission Audit: Double-Blind Anonymity" bash scripts/check_anonymity.sh
    sync_part_results "script-part-ifbench" 1 "${IFBENCH_MODELS_ARG[@]}"
    ;;

  all)
    if [ "$UNPACK_MODE" = "on" ] || { [ "$UNPACK_MODE" = "auto" ] && [ "$IS_MOCK" -eq 0 ] && compgen -G "outputs/*/*_cache.jsonl.gz" >/dev/null; }; then
      run_stage "Unpack Synced Worker Results" "$PY_BIN" scripts/sync_results.py --mode unpack --repo-dir . --out-dir "$OUT_DIR"
    fi
    run_gate0_preflight
    run_hw_probe_and_smoke 1
    if [ "${#MODELS_ARG[@]}" -gt 0 ]; then
      for _idx in "${!MODELS_ARG[@]}"; do
        _m="${MODELS_ARG[$_idx]}"
        if [ "$_idx" -gt 0 ]; then
          prune_hf_cache_if_low_disk
        fi
        if [ "$CALIBRATE_ONLY" -eq 1 ]; then
          run_matrix_sharded "Calibrate: ${_m}" "$_m" --calibrate-only
        else
          run_model_eval_pair "Matrix" "$_m"
          if [ "$_idx" -eq 0 ]; then
            run_w7_stage "$_m"
          fi
          sync_model_checkpoint "$_m"
        fi
      done
      if [ "$CALIBRATE_ONLY" -eq 0 ]; then
        run_ifbench_ood_stage
        run_r6_sampled_stage
        run_phase7_stage
      fi
      run_stage "Submission Audit: Double-Blind Anonymity" bash scripts/check_anonymity.sh
      if [ "$CALIBRATE_ONLY" -eq 0 ]; then
        sync_part_results "script-all-in-one" 1 "${MODELS_ARG[@]}"
      fi
      exit 0
    fi

    if [ "$CALIBRATE_ONLY" -eq 1 ]; then
      run_matrix_sharded "Step 0b: Primary 4B Gate-2 Calibration" Qwen/Qwen3.5-4B --calibrate-only
      run_stage "Submission Audit: Double-Blind Anonymity" bash scripts/check_anonymity.sh
      exit 0
    fi

    # Run strict341 first so calibration & cached diag_ehurt/controls entries are reused during diag491
    run_model_eval_pair "Phase 4 / W4" Qwen/Qwen3.5-4B
    run_w7_stage Qwen/Qwen3.5-4B
    sync_model_checkpoint Qwen/Qwen3.5-4B
    prune_hf_cache_if_low_disk
    run_model_eval_pair "Phase 5 / W5" HuggingFaceTB/SmolLM3-3B
    run_w7_stage HuggingFaceTB/SmolLM3-3B
    sync_model_checkpoint HuggingFaceTB/SmolLM3-3B
    prune_hf_cache_if_low_disk
    run_matrix_sharded "Phase 6 / W6: Qwen3.5-0.8B (strict341)" Qwen/Qwen3.5-0.8B --strict-diag
    sync_model_checkpoint Qwen/Qwen3.5-0.8B
    run_matrix_sharded "Phase 6 / W6: Qwen3.5-2B (strict341)" Qwen/Qwen3.5-2B --strict-diag
    sync_model_checkpoint Qwen/Qwen3.5-2B
    if [ "$WITH_9B" -eq 1 ]; then
      prune_hf_cache_if_low_disk 28
      run_matrix_sharded "Phase 6 / W6: Qwen3.5-9B (strict341)" Qwen/Qwen3.5-9B --strict-diag
      sync_model_checkpoint Qwen/Qwen3.5-9B
    fi
    run_ifbench_ood_stage
    run_r6_sampled_stage
    run_phase7_stage
    run_stage "Submission Audit: Double-Blind Anonymity" bash scripts/check_anonymity.sh
    sync_part_results "script-all-in-one" 1
    ;;
esac
