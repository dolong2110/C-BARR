#!/usr/bin/env bash
set -euo pipefail

# Part 2 (Parallel GPU Worker 2/3): HuggingFaceTB/SmolLM3-3B (strict341 -> diag491).
# Usage: bash scripts/run_part2_worker_smollm3.sh [--shards auto] [--push] [run_gpu_phases_4_to_6.sh flags...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_gpu_phases_4_to_6.sh" --part 2 "$@"
