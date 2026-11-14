#!/usr/bin/env bash
set -euo pipefail

# Section 6.1 & Phase 1 (W1): Validated Environment Setup
pip install --upgrade pip
pip install "torch>=2.6" "transformers>=4.53.0" accelerate datasets matplotlib scipy scikit-learn \
    nltk immutabledict langdetect absl-py

# Optional fast CUDA/Triton kernels for GPU hosts (Qwen3.5 Gated DeltaNet layers) with live self-test
python3 scripts/setup_fast_kernels.py --mode auto --allow-source-build || true

# Vendor upstream Google Research instruction_following_eval if not already present
if [ ! -d "./instruction_following_eval" ]; then
    rm -rf /tmp/gr
    git clone --depth 1 --filter=blob:none --sparse https://github.com/google-research/google-research.git /tmp/gr
    (cd /tmp/gr && git sparse-checkout set instruction_following_eval)
    cp -r /tmp/gr/instruction_following_eval ./instruction_following_eval
    rm -rf /tmp/gr
fi

# Download NLTK sentence/POS resources required by google-research instruction_following_eval
python3 -m nltk.downloader punkt punkt_tab stopwords averaged_perceptron_tagger_eng
# Verify seeded langdetect import (Patch D3) before running any evaluation
python3 -c "from langdetect import DetectorFactory; DetectorFactory.seed = 0; print('Environment setup OK')"
