#!/usr/bin/env bash
set -euo pipefail

# Part 4 (Combiner after Parts 1-3 finish): Unpack worker caches + Gate 0 + [Optional W7] + Combined Phase 7 + Anonymity + Ledger Sync.
# Usage: bash scripts/run_part4_combine_analyze.sh [--with-w7] [--strict-version-match] [--push] [run_gpu_phases_4_to_6.sh flags...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_gpu_phases_4_to_6.sh" --part 4 "$@"
