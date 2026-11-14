#!/usr/bin/env bash
set -euo pipefail

# Part 1 (Parallel GPU Worker 1/3): Gate 0 + Step 0a Smoke + Step 0b Calibration + Qwen/Qwen3.5-4B (strict341 -> diag491).
# Usage: bash scripts/run_part1_worker_qwen4b.sh [--shards auto] [--push] [run_gpu_phases_4_to_6.sh flags...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_gpu_phases_4_to_6.sh" --part 1 "$@"
