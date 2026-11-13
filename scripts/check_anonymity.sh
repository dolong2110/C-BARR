#!/usr/bin/env bash
set -euo pipefail

# Double-blind submission anonymity check for paper/ artifacts.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="${ROOT_DIR}/paper"

if [ ! -d "$PAPER_DIR" ]; then
  echo "[FAIL] Missing paper directory: $PAPER_DIR" >&2
  exit 1
fi

FAIL=0

# 1. Require exact double-blind author placeholder in main.tex
if [ -f "${PAPER_DIR}/main.tex" ]; then
  if ! grep -q '\\author{Anonymous Authors}' "${PAPER_DIR}/main.tex"; then
    echo "[FAIL] paper/main.tex missing '\\author{Anonymous Authors}'" >&2
    FAIL=1
  fi
fi

# 2. Check .tex and .bib files for email addresses, local filesystem paths, or internal doc names
# Using NUL-delimited find ensures filenames with spaces never word-split incorrectly.
TARGETS=()
while IFS= read -r -d '' f; do
  TARGETS+=("$f")
done < <(find "$PAPER_DIR" -type f \( -name '*.tex' -o -name '*.bib' \) -print0)

# Resolve OS username dynamically so the repository never stores local account names literally
CURRENT_USER="$(id -un 2>/dev/null || true)"
FORBIDDEN_REGEX='(/Users/|/home/|@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|drive\.google\.com|docs\.google\.com|NLP Research_ C-BARR|\.docx)'
if [ -n "$CURRENT_USER" ]; then
  FORBIDDEN_REGEX="(/Users/|/home/|@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}|drive\\.google\\.com|docs\\.google\\.com|NLP Research_ C-BARR|\\.docx|${CURRENT_USER})"
fi
for f in "${TARGETS[@]}"; do
  if grep -En "$FORBIDDEN_REGEX" "$f" >/dev/null 2>&1; then
    echo "[FAIL] Potential anonymity leak in $f:" >&2
    grep -En "$FORBIDDEN_REGEX" "$f" >&2
    FAIL=1
  fi
done

# 3. Check compiled PDF metadata if paper/main.pdf exists and pdfinfo is available
if [ -f "${PAPER_DIR}/main.pdf" ] && command -v pdfinfo >/dev/null 2>&1; then
  PDF_REGEX='(/Users/|/home/)'
  if [ -n "$CURRENT_USER" ]; then
    PDF_REGEX="(/Users/|/home/|${CURRENT_USER})"
  fi
  if pdfinfo "${PAPER_DIR}/main.pdf" | grep -Ei 'Author:|Creator:|Title:' | grep -Ei "$PDF_REGEX" >/dev/null 2>&1; then
    echo "[FAIL] Potential metadata leak in paper/main.pdf" >&2
    pdfinfo "${PAPER_DIR}/main.pdf" >&2
    FAIL=1
  fi
fi

if [ "$FAIL" -ne 0 ]; then
  exit 1
fi

echo "Double-blind anonymity audit PASSED for paper/"
