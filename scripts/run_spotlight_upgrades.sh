#!/usr/bin/env bash
set -euo pipefail

# Single-Command Non-Overlapping Spotlight Upgrade Suite (U1 + U2 + U3 + Phase 7):
#   - Never overwrites or mutates existing `seed_60`, `dev_60`, `eval_450`, or `diag_*` caches!
#   - Stage U1: End-to-End Zero-Token State Surgery (`M2_ZeroTok`) & Regime-Calibrated Multi-Horizon
#               `M2_Adapt` (`B in {0, 64, 128, 250, 512}`) across all 5 greedy backbones
#   - Stage U2: Dense GQA `HuggingFaceTB/SmolLM3-3B` OOD `IFBench_test` (N=300) + 3-Seed `R6` (T=0.6)
#   - Stage U3: Competition-Level Hard Math `MATH-500` (N=200 stratified Levels 1-5) on `Qwen3.5-4B` & `SmolLM3-3B`
#   - Stage U4: Combined Phase 7 Analysis, Tables/Figures, Anonymity Audit & Atomic GitHub Sync
#
# Usage on GPU Server:
#   CBARR_COLOR=1 bash scripts/run_spotlight_upgrades.sh --gpu-boost --hf-home "$HOME/maplecg_nfs/.cbarr_hf_cache" --push

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="outputs"
SHARDS_ARG="auto"
WORKERS_PER_GPU="auto"
GPU_IDS=""
MATH_PER_LEVEL=40
DO_PUSH=1
IS_MOCK=0
SKIP_U1=0
SKIP_U2=0
SKIP_U3=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --out-dir) shift; OUT_DIR="$1" ;;
    --shards) shift; SHARDS_ARG="$1" ;;
    --gpu) shift; GPU_IDS="$1" ;;
    --workers-per-gpu) shift; WORKERS_PER_GPU="$1" ;;
    --gpu-boost)
      WORKERS_PER_GPU="max"
      export CBARR_GPU_BOOST=1
      export CBARR_WORKERS_PER_GPU="max"
      ;;
    --hf-home) shift; export HF_HOME="$1" ;;
    --github-token) shift; export GITHUB_TOKEN="$1" ;;
    --hf-token) shift; export HF_TOKEN="$1" ;;
    --math-per-level) shift; MATH_PER_LEVEL="$1" ;;
    --push) DO_PUSH=1 ;;
    --no-push) DO_PUSH=0 ;;
    --mock) IS_MOCK=1 ;;
    --skip-u1) SKIP_U1=1 ;;
    --skip-u2) SKIP_U2=1 ;;
    --skip-u3) SKIP_U3=1 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
  shift
done

if [ -x ".venv/bin/python" ]; then
  PY_BIN=".venv/bin/python"
else
  PY_BIN="$(command -v python3 || command -v python)"
fi

export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Auto-relocate HF_HOME onto roomy NFS/scratch mount when default home partition is tight
if [ "$IS_MOCK" -eq 0 ]; then
  RESOLVED_HF_HOME="$("$PY_BIN" -c 'from scripts.gpu_pool import resolve_roomy_hf_home; print(resolve_roomy_hf_home(min_free_gb=20.0))' 2>/dev/null || true)"
  if [ -n "$RESOLVED_HF_HOME" ]; then
    export HF_HOME="$RESOLVED_HF_HOME"
    mkdir -p "$HF_HOME"
    echo "[spotlight] Using HF_HOME=${HF_HOME}"
  fi
fi

checkpoint_model() {
  local m_id="$1"
  local label="$2"
  if [ "$IS_MOCK" -ne 0 ]; then
    return 0
  fi
  local push_flags=(--commit)
  if [ "$DO_PUSH" -eq 1 ]; then
    push_flags+=(--push)
  fi
  "$PY_BIN" scripts/sync_results.py \
    --mode pack \
    --repo-dir . \
    --out-dir "$OUT_DIR" \
    --only-models "$m_id" \
    --no-ledger \
    --label "$label" \
    "${push_flags[@]}" || echo "[WARN] Non-fatal checkpoint sync warning for ${m_id}"
}

# 0. Unpack prior synced compressed caches (*.jsonl.gz -> *.jsonl) without overwriting newer local files
if [ "$IS_MOCK" -eq 0 ] && compgen -G "${OUT_DIR}/*/*_cache.jsonl.gz" >/dev/null; then
  echo "=== [Spotlight Step 0] Unpacking synced compressed caches ==="
  "$PY_BIN" scripts/sync_results.py --mode unpack --repo-dir . --out-dir "$OUT_DIR"
fi

COMMON_PY_FLAGS=(--out-dir "$OUT_DIR" --shards "$SHARDS_ARG" --workers-per-gpu "$WORKERS_PER_GPU")
if [ -n "$GPU_IDS" ]; then
  COMMON_PY_FLAGS+=(--gpu-ids "$GPU_IDS")
fi
if [ "$IS_MOCK" -eq 1 ]; then
  COMMON_PY_FLAGS+=(--mock)
fi

# 1. Stage U1: End-to-End Zero-Token State Surgery (M2_ZeroTok) & Regime-Calibrated M2_Adapt across all 5 greedy models
if [ "$SKIP_U1" -eq 0 ]; then
  U1_MODELS=(
    "Qwen/Qwen3.5-4B"
    "HuggingFaceTB/SmolLM3-3B"
    "Qwen/Qwen3.5-0.8B"
    "Qwen/Qwen3.5-2B"
    "Qwen/Qwen3.5-9B"
  )
  for m_id in "${U1_MODELS[@]}"; do
    m_slug="${m_id//\//__}"
    if [ -d "${OUT_DIR}/${m_slug}" ]; then
      echo "=== [Spotlight U1: State Surgery & M2-Adapt] ${m_id} ==="
      "$PY_BIN" scripts/ext_m2_state_and_adaptive.py --model "$m_id" "${COMMON_PY_FLAGS[@]}"
      checkpoint_model "$m_id" "spotlight-u1-${m_slug}"
    fi
  done
fi

# 2. Stage U2: Dense GQA HuggingFaceTB/SmolLM3-3B OOD IFBench_test (N=300) + 3-Seed Sampled R6 (T=0.6)
if [ "$SKIP_U2" -eq 0 ] && [ -d "${OUT_DIR}/HuggingFaceTB__SmolLM3-3B" ]; then
  echo "=== [Spotlight U2a: Dense OOD IFBench_test] HuggingFaceTB/SmolLM3-3B ==="
  IFB_FLAGS=(--model "HuggingFaceTB/SmolLM3-3B" --out-dir "$OUT_DIR" --shards "$SHARDS_ARG" --workers-per-gpu "$WORKERS_PER_GPU" --resume)
  [ -n "$GPU_IDS" ] && IFB_FLAGS+=(--gpu-ids "$GPU_IDS")
  [ "$IS_MOCK" -eq 1 ] && IFB_FLAGS+=(--mock)
  "$PY_BIN" scripts/ext_ifbench_ood.py "${IFB_FLAGS[@]}"
  checkpoint_model "HuggingFaceTB/SmolLM3-3B" "spotlight-u2a-ifbench-smollm3"

  echo "=== [Spotlight U2b: Dense 3-Seed Sampled R6 (T=0.6, s=42,43,44)] HuggingFaceTB/SmolLM3-3B ==="
  R6_WRAP_FLAGS=(
    --part r6
    --no-unpack
    --skip-preflight
    --out-dir "$OUT_DIR"
    --shards "$SHARDS_ARG"
    --r6-models "HuggingFaceTB/SmolLM3-3B"
    --r6-seeds "42 43 44"
  )
  [ "$WORKERS_PER_GPU" = "max" ] && R6_WRAP_FLAGS+=(--gpu-boost)
  [ -n "$GPU_IDS" ] && R6_WRAP_FLAGS+=(--gpu "$GPU_IDS")
  [ -n "${HF_HOME:-}" ] && R6_WRAP_FLAGS+=(--hf-home "$HF_HOME")
  [ "$IS_MOCK" -eq 1 ] && R6_WRAP_FLAGS+=(--mock)
  if [ "$DO_PUSH" -eq 1 ] && [ "$IS_MOCK" -eq 0 ]; then
    R6_WRAP_FLAGS+=(--commit --push)
  else
    R6_WRAP_FLAGS+=(--no-push)
  fi
  bash scripts/run_gpu_phases_4_to_6.sh "${R6_WRAP_FLAGS[@]}"
fi

# 3. Stage U3: Competition-Level Hard Math MATH-500 (Stratified Levels 1-5) on Qwen3.5-4B & SmolLM3-3B
if [ "$SKIP_U3" -eq 0 ]; then
  for m_id in "Qwen/Qwen3.5-4B" "HuggingFaceTB/SmolLM3-3B"; do
    m_slug="${m_id//\//__}"
    if [ -d "${OUT_DIR}/${m_slug}" ] || [ "$IS_MOCK" -eq 1 ]; then
      echo "=== [Spotlight U3: Stratified MATH-500 Hard Reasoning] ${m_id} ==="
      "$PY_BIN" scripts/ext_math500_eval.py \
        --model "$m_id" \
        --per-level "$MATH_PER_LEVEL" \
        "${COMMON_PY_FLAGS[@]}"
      checkpoint_model "$m_id" "spotlight-u3-math500-${m_slug}"
    fi
  done
fi

# 4. Stage U4: Final Combined Phase 7 Analysis, Tables, Figures, Anonymity Audit & Unified Sync
echo "=== [Spotlight U4: Unified Phase 7 Analysis & Sync] ==="
"$PY_BIN" scripts/analyze_and_plot.py --out-dir "$OUT_DIR"
if [ "$IS_MOCK" -eq 0 ]; then
  bash scripts/check_anonymity.sh
  FINAL_FLAGS=(--mode pack --repo-dir . --out-dir "$OUT_DIR" --label "spotlight-complete" --commit)
  [ "$DO_PUSH" -eq 1 ] && FINAL_FLAGS+=(--push)
  "$PY_BIN" scripts/sync_results.py "${FINAL_FLAGS[@]}"
fi

echo "=== ALL SPOTLIGHT UPGRADES COMPLETED SUCCESSFULLY ==="
