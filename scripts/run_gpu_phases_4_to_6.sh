#!/usr/bin/env bash
set -euo pipefail

# Phases 4-7 GPU Orchestration for C-BARR
# Usage:
#   bash scripts/run_gpu_phases_4_to_6.sh                 # Step 0 + 4B + SmolLM3-3B + 0.8B/2B + W8
#   bash scripts/run_gpu_phases_4_to_6.sh --with-9b       # Also run Qwen3.5-9B scale sweep
#   bash scripts/run_gpu_phases_4_to_6.sh --with-w7       # Also run optional W7 DeltaNet blend & prefill MLP
#   bash scripts/run_gpu_phases_4_to_6.sh --include-b512  # Also run Contingency R3 B=512 sweep
#   bash scripts/run_gpu_phases_4_to_6.sh --mock          # Full offline CPU orchestration dry-run

WITH_9B=0
WITH_W7=0
IS_MOCK=0
SKIP_SMOKE=0
OUT_DIR="outputs"
FIG_DIR="paper/figures"
EXTRA_FLAGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --with-9b) WITH_9B=1 ;;
    --with-w7) WITH_W7=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
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
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
  shift
done

PY_BIN="${PYTHON_BIN:-python3}"
if [ -x ".venv/bin/python" ] && [ -z "${PYTHON_BIN:-}" ]; then
  PY_BIN=".venv/bin/python"
fi

echo "=== [Gate 0] Preflight unit tests & frozen split manifest ==="
"$PY_BIN" -m unittest discover -s tests -v

if [ "$IS_MOCK" -eq 0 ] && [ "$SKIP_SMOKE" -eq 0 ]; then
  echo "=== [Step 0a] Verify CUDA FLA / causal_conv1d kernels & Live Gate 3 Smoke ==="
  "$PY_BIN" -c "
import torch
print('CUDA available:', torch.cuda.is_available(), 'Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    import fla, causal_conv1d
    print('CUDA kernels OK: fla + causal_conv1d')
"
  "$PY_BIN" scripts/run_smoke_test.py --live-gpu
fi

echo "=== [Step 0b] Primary 4B Gate-2 Calibration Audit ==="
"$PY_BIN" scripts/run_matrix.py --model Qwen/Qwen3.5-4B --out-dir "$OUT_DIR" --calibrate-only ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}

echo "=== [Phase 4 / W4] Qwen/Qwen3.5-4B (strict341 -> diag491) ==="
# Run strict341 first so its cached diag_ehurt/controls entries are reused during diag491
"$PY_BIN" scripts/run_matrix.py --model Qwen/Qwen3.5-4B --out-dir "$OUT_DIR" --strict-diag ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
"$PY_BIN" scripts/run_matrix.py --model Qwen/Qwen3.5-4B --out-dir "$OUT_DIR" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}

echo "=== [Phase 5 / W5] HuggingFaceTB/SmolLM3-3B Dense Control (strict341 -> diag491) ==="
"$PY_BIN" scripts/run_matrix.py --model HuggingFaceTB/SmolLM3-3B --out-dir "$OUT_DIR" --strict-diag ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
"$PY_BIN" scripts/run_matrix.py --model HuggingFaceTB/SmolLM3-3B --out-dir "$OUT_DIR" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}

echo "=== [Phase 6 / W6] Qwen3.5 Scale Sweep (0.8B, 2B) ==="
"$PY_BIN" scripts/run_matrix.py --model Qwen/Qwen3.5-0.8B --out-dir "$OUT_DIR" --strict-diag ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
"$PY_BIN" scripts/run_matrix.py --model Qwen/Qwen3.5-2B --out-dir "$OUT_DIR" --strict-diag ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}

if [ "$WITH_9B" -eq 1 ]; then
  echo "=== [Phase 6 / W6] Qwen/Qwen3.5-9B Scale Sweep ==="
  "$PY_BIN" scripts/run_matrix.py --model Qwen/Qwen3.5-9B --out-dir "$OUT_DIR" --strict-diag ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
fi

if [ "$WITH_W7" -eq 1 ]; then
  echo "=== [Workstream W7] Optional Hybrid DeltaNet Blend & Prefill MLP ==="
  W7_FLAGS=()
  if [ "$IS_MOCK" -eq 1 ]; then
    W7_FLAGS+=(--mock)
  fi
  "$PY_BIN" scripts/ext_deltanet_blend.py --model Qwen/Qwen3.5-4B --out-dir "$OUT_DIR" ${W7_FLAGS[@]+"${W7_FLAGS[@]}"}
  "$PY_BIN" scripts/ext_prefill_mlp.py --model Qwen/Qwen3.5-4B --out-dir "$OUT_DIR" ${W7_FLAGS[@]+"${W7_FLAGS[@]}"}
fi

echo "=== [Phase 7 / W8] Statistical Analysis, Paper Tables, and PDF Figures ==="
"$PY_BIN" scripts/analyze_and_plot.py --out-dir "$OUT_DIR" --fig-dir "$FIG_DIR"

echo "=== [Submission Audit] Double-Blind Anonymity Check ==="
bash scripts/check_anonymity.sh

echo "All Phases 4-7 completed cleanly."
