#!/usr/bin/env bash
set -euo pipefail

# Part 4 (Parallel GPU Worker 4/4): Phase 6 Scale Sweep (Qwen/Qwen3.5-0.8B, plus Qwen/Qwen3.5-9B with --with-9b).
# Usage: bash scripts/run_part4_worker_qwen08b.sh [--with-9b] [--shards auto] [--push] [run_gpu_phases_4_to_6.sh flags...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_gpu_phases_4_to_6.sh" --part 4 "$@"
