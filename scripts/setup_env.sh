#!/usr/bin/env bash
set -euo pipefail

# Section 6.1 & Phase 1 (W1): Validated Environment Setup (container/CUDA-safe)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

PY_BIN="${PYTHON_BIN:-}"
if [ -z "$PY_BIN" ]; then
    if [ -x ".venv/bin/python" ]; then
        PY_BIN=".venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PY_BIN="python3"
    else
        PY_BIN="python"
    fi
fi

# On PEP 668 externally-managed host Pythons (e.g., macOS Homebrew / Ubuntu 24.04),
# create .venv with --system-site-packages so system CUDA PyTorch stays visible.
PIP_EXTRA_ARGS=()
IS_EXT_MANAGED="$("$PY_BIN" -c 'import pathlib, sys, sysconfig; p = pathlib.Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED"; print(1 if (sys.prefix == sys.base_prefix and p.is_file()) else 0)')"
if [ "$IS_EXT_MANAGED" -eq 1 ]; then
    if [ ! -x ".venv/bin/python" ]; then
        echo "[setup_env] Creating .venv (--system-site-packages) for externally-managed Python..."
        if "$PY_BIN" -m venv --system-site-packages .venv; then
            PY_BIN=".venv/bin/python"
        else
            PIP_EXTRA_ARGS+=(--break-system-packages)
        fi
    else
        PY_BIN=".venv/bin/python"
    fi
fi

# Preserve preinstalled CUDA container PyTorch >= 2.6; install torch only when absent or too old
MISSING_PKGS="$("$PY_BIN" -c '
import importlib.metadata, re
def ver_tuple(pkg):
    try:
        v = importlib.metadata.version(pkg)
        return tuple(int(x) for x in re.findall(r"\d+", v)[:2])
    except importlib.metadata.PackageNotFoundError:
        return None
pkgs = [
    ("numpy", "numpy", None),
    ("transformers", "transformers>=4.53.0", (4, 53)),
    ("accelerate", "accelerate", None),
    ("datasets", "datasets", None),
    ("matplotlib", "matplotlib", None),
    ("scipy", "scipy", None),
    ("scikit-learn", "scikit-learn", None),
    ("nltk", "nltk", None),
    ("immutabledict", "immutabledict", None),
    ("langdetect", "langdetect", None),
    ("absl-py", "absl-py", None),
]
tv = ver_tuple("torch")
need = []
if tv is None or tv < (2, 6):
    need.append("torch>=2.6")
for dist, spec, min_v in pkgs:
    cur = ver_tuple(dist)
    if cur is None or (min_v is not None and cur < min_v):
        need.append(spec)
print(" ".join(need))
')"
if [ -n "$MISSING_PKGS" ]; then
    "$PY_BIN" -m pip install --upgrade pip -q ${PIP_EXTRA_ARGS[@]+"${PIP_EXTRA_ARGS[@]}"}
    read -r -a _PKGS <<< "$MISSING_PKGS"
    "$PY_BIN" -m pip install -q ${PIP_EXTRA_ARGS[@]+"${PIP_EXTRA_ARGS[@]}"} "${_PKGS[@]}"
fi

# Optional fast CUDA/Triton kernels for GPU hosts (Qwen3.5 Gated DeltaNet layers) with live self-test
"$PY_BIN" scripts/setup_fast_kernels.py --mode auto --allow-source-build || true

# Vendor upstream Google Research instruction_following_eval if not already present
if [ ! -d "./instruction_following_eval" ]; then
    rm -rf /tmp/gr
    git clone --depth 1 --filter=blob:none --sparse https://github.com/google-research/google-research.git /tmp/gr
    (cd /tmp/gr && git sparse-checkout set instruction_following_eval)
    cp -r /tmp/gr/instruction_following_eval ./instruction_following_eval
    rm -rf /tmp/gr
fi

# Download NLTK sentence/POS resources required by google-research instruction_following_eval
"$PY_BIN" -m nltk.downloader -q punkt punkt_tab stopwords averaged_perceptron_tagger_eng

# Ensure frozen dataset splits and manifest integrity exist cleanly
if [ ! -f "data/manifest_sha256.json" ]; then
    "$PY_BIN" data/freeze_datasets.py
fi
"$PY_BIN" -c "import pathlib; from scripts.run_matrix import verify_manifest_integrity; verify_manifest_integrity(pathlib.Path('data'))"

# Verify seeded langdetect import (Patch D3) before running any evaluation
"$PY_BIN" -c "from langdetect import DetectorFactory; DetectorFactory.seed = 0; print('Environment setup OK')"
