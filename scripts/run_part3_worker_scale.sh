#!/usr/bin/env bash
set -euo pipefail

# Part 3 (Parallel GPU Worker 3/3): Phase 6 Scale Sweep (Qwen/Qwen3.5-0.8B & 2B, plus 9B with --with-9b).
# Usage: bash scripts/run_part3_worker_scale.sh [--with-9b] [--shards auto] [--push] [run_gpu_phases_4_to_6.sh flags...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_gpu_phases_4_to_6.sh" --part 3 "$@"
