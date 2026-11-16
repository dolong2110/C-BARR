#!/usr/bin/env bash
set -euo pipefail

# Single-Command End-to-End A-Z Runner for strong multi-GPU machines:
#   1. Auto-bootstraps missing Python deps, fast DeltaNet/CUDA kernels, IFEval, NLTK & frozen splits (--setup auto)
#   2. Verifies git commit/push credentials up front so long GPU runs never fail to push at the end
#   3. Unpacks prior synced caches if present (--unpack auto) and runs fast Gate 0 preflight (~1s)
#   4. Runs Step 0a smoke gate, Step 0b calibration, Phase 4 (Qwen3.5-4B), Phase 5 (SmolLM3-3B),
#      Phase 6 scale sweep (0.8B, 2B, +9B automatically when GPU VRAM >= 22 GB), optional W7 extensions,
#      Phase 7 stats/tables/PDF figures, and double-blind anonymity check across all visible GPUs (--shards auto)
#   5. Commits & pushes compact caches, summaries, paper tables/figures, RESULTS_VERSION, and EXPERIMENT_LOG.md
#
# Usage:
#   GITHUB_TOKEN=... bash scripts/run_all_a_to_z.sh
#   bash scripts/run_all_a_to_z.sh --no-push        # commit locally without pushing
#   bash scripts/run_all_a_to_z.sh --gpu 0,1,2,3    # pin specific GPU IDs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_gpu_phases_4_to_6.sh" \
  --part all \
  --setup-mode auto \
  --shards auto \
  --auto-9b \
  --with-w7 \
  --commit \
  --push \
  "$@"
