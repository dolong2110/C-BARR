# C-BARR: Cascaded Budgeted Adaptive Reasoning Routing

- Doc: [C-BARR](https://docs.google.com/document/d/1AGvSbhy1wJ5dkb957VBXWJH0DKigL5tvZquLPeps7KI).

## Overview

**Cascaded Budgeted Adaptive Reasoning Routing (C-BARR)** is a training-free test-time compute routing and constraint-retention framework operating on frozen language model weights across Hybrid Gated DeltaNet (`Qwen/Qwen3.5-0.8B/2B/4B/9B`) and Dense Softmax GQA (`HuggingFaceTB/SmolLM3-3B`) backbones.

It combines:
1. **Gate 1 (`src/engine.py:run_gate1_lexical`)**: Sub-millisecond numeral-masked lexical screening across all 25 canonical IFEval constraint templates (10 Planning + 15 Precision: 5 Inhibition and 10 Structural) and recursive-safe anchor clause extraction.
2. **Gate 2 (`src/engine.py:CBARRDualModeEngine.run_gate2_fast_probe`)**: A $\le 32$-token non-thinking Shannon entropy $(H_{\text{first}}, H_{\text{mean}}, H_{\text{max}})$ and Thinking Inertia $I_{\text{inertia}}(x)$ probe fitted via zero-leakage two-stage calibration (`seed_60` $\rightarrow$ `dev_60`).
3. **Method M2 (`src/engine.py:CBARRDualModeEngine.run_capped_think_and_anchor`)**: Tight tiered thinking termination ($B \in \{64, 128, 250\}$, guaranteeing $T_{\text{delib}} = T_{\text{probe}} + T_{\text{think}} \le 250$ for 100% $\text{OAA}_{250}$ compliance) paired with channel-stripped post-`</think>` constraint re-anchoring (`post_stripped`).

---

## Repository Architecture

| Path | Role | Key Outputs & Responsibilities |
| :--- | :--- | :--- |
| `notebooks/C_BARR_Colab_Kaggle.ipynb` | One-click Google Colab & Kaggle GPU runner | Auto-detects GPU (`T4`/`P100` `float16` vs. `L4`/`A100` `bfloat16`) and VRAM, caches HF weights safely off `/kaggle/working`, clones repo, builds/verifies frozen splits, and runs Phases 4–7. |
| `data/freeze_datasets.py` | Deterministic zero-overlap dataset builder (`seed = 42`) | Materializes `seed_60.jsonl`, `dev_60.jsonl`, `eval_450.jsonl`, `ifeval_diag_491.jsonl`, `ifeval_diag_341_strict.jsonl`, and `manifest_sha256.json`. |
| `src/graders.py` | Seeded channel parser, 4-gram loop & CoT spillover audit, and verifiers | Pins `DetectorFactory.seed = 0`; strips `<think>` and EOS markers; grades GSM8K, CSQA, and IFEval (Inhibition + Structural); logs `has_cot_spillover` (Patch D16). |
| `src/engine.py` | 25-template Gate 1, `CBARRDualModeEngine`, $B \in \{64, 128, 250\}$ termination, Controls A1–B3 | Normalizes `Qwen3.5` and `SmolLM3-3B` chat templates (freezing `Today Date: 01 November 2026`); caps answer budget at 1024 tokens (Patch D14); executes `post_stripped` and `pre_close` re-anchoring. |
| `src/routers.py` | Two-stage `seed_60`/`dev_60` calibrator and B0–M2 policy resolver | Fits logistic Gate-2 weights on `seed_60`, tunes $\tau^*$ on `dev_60`, caps top-tier deliberation at 250 tokens (Patch D13), and resolves `B0`–`B2b`, `M1a`–`M1c`, `M2_B64/128/256`, `M2_Pre`, and `M2`. |
| `src/metrics.py` | Statistical inference, efficiency metrics, and causal attribution | Computes exact & Lancaster mid-$p$ McNemar, Holm-Bonferroni, Wilson 95% CIs (`wilson_ci`), Clean-446 defect filtering (`filter_clean_446`), paired bootstrap CIs, `MacroAcc`, `OAA_250`, `AUC_OAA`, `U_0.05`, `BSR(B)`, `spillover_rate`, `Oracle_UB`, and $\rho_{\text{recency}}$ vs. $\rho_{\text{interference}}$. |
| `scripts/run_matrix.py` | Incremental GPU cache runner and LaTeX row generator | Verifies `manifest_sha256.json`, supports `--calibrate-only` / `--device` / `--dtype` (`auto` selects `bfloat16` on `sm >= 80`, `float16` on `T4`/`P100`/`mps`) / `--mock` / `--include-b512`, incrementally caches primitives and `E_hurt` controls (`diag_controls_cache.jsonl`), and writes non-clobbering `matrix_summary_{strict341,diag491}.json`. |
| `scripts/analyze_and_plot.py` | Phase 7 (W8) analysis, tables, and vector PDF figures | Runs Clean-446 sensitivity, 10k paired bootstrap CIs, Gates 4–6 & Contingency R1–R4 audit, and generates `outputs/stats_summary.json`, `outputs/paper_tables.tex`, and `paper/figures/fig{1,2,3}_*.pdf`. |
| `scripts/run_gpu_phases_4_to_6.sh` | Linux CUDA orchestration (Step 0 + Phases 4–7) | Runs preflight tests, `--calibrate-only`, `Qwen3.5-4B` (`strict341` + `diag491`), `SmolLM3-3B`, `0.8B/2B` (optional `--with-9b`, `--with-w7`, `--include-b512`), `analyze_and_plot.py`, and `check_anonymity.sh`. |
| `scripts/ext_deltanet_blend.py` / `scripts/ext_prefill_mlp.py` | Optional Workstream W7 extensions | Gated DeltaNet `recurrent_states` blending (`Qwen3.5`) on $E_{\text{hurt}}$ and mid-layer prefill hidden-state MLP escalation probe. |
| `scripts/check_anonymity.sh` | Double-blind submission linter | Verifies `\author{Anonymous Authors}` and checks `.tex`, `.bib`, and PDF metadata under `paper/` for PII/path leaks. |
| `scripts/run_smoke_test.py` | Phase 3 (W3) 12-item smoke test runner and Gate 3 verifier | Slices 2 items/slice into `outputs/smoke_12.jsonl` and `/tmp/smoke_12.jsonl` and verifies all Phase 3 invariants. |
| `tests/test_cbarr_preflight.py` | Pre-flight unit test suite (W1–W3 & Patches D1–D16) | Verifies 25/25 IFEval regex recall, `DetectorFactory.seed == 0`, graders, D13–D16 patches, split disjointness, and smoke gate. |
| `paper/main.tex` | Manuscript & Booktabs LaTeX Templates | Integrates Tables 1, 2, and 3 (`paper/references.bib` includes 2025–2026 baselines with anonymized metadata). |

---

### Running on Google Colab or Kaggle GPUs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dolong2110/C-BARR/blob/master/notebooks/C_BARR_Colab_Kaggle.ipynb)

Upload or import [notebooks/C_BARR_Colab_Kaggle.ipynb](notebooks/C_BARR_Colab_Kaggle.ipynb) into **Google Colab** (Runtime -> GPU: `T4`, `L4`, or `A100`) or **Kaggle** (Accelerator: `GPU P100` or `GPU T4 x2`, Internet: **On**).

The notebook automatically:
1. Detects Colab vs. Kaggle, selects `--dtype auto` (`float16` on `T4`/`P100`, `bfloat16` on `L4`/`A100`), and redirects `HF_HOME` to `/kaggle/temp/hf_cache` (or `/tmp/hf_cache` on Colab) so `/kaggle/working` never hits the 20 GB output quota.
2. Clones `dolong2110/C-BARR`, installs missing runtime dependencies without overwriting container PyTorch, and optionally mounts Google Drive on Colab for persistent incremental `outputs/`.
3. Verifies existing SHA-256 frozen JSONL splits (or downloads HuggingFace `gsm8k`, `tau/commonsense_qa`, and `google/IFEval` via `data/freeze_datasets.py` if absent).
4. Runs `run_smoke_test.py --live-gpu`, `--calibrate-only`, primary evaluation (`Qwen3.5-4B`, `SmolLM3-3B`), scale sweep (`0.8B`, `2B`, plus `9B` automatically when VRAM >= 22 GB), optional W7 extensions, `analyze_and_plot.py`, and packages results into `cbarr_results_bundle.tar.gz`.

Set optional secrets `HF_TOKEN` (for faster HuggingFace downloads) and `GITHUB_TOKEN` (if cloning while the repo is private) in Colab Secrets or Kaggle Add-ons -> Secrets.

---

## Execution Playbook (Phases 1–3)

### Phase 1 (W1): Workspace Provisioning, Environment Setup & Pre-Flight Verification
```bash
bash scripts/setup_env.sh
python -m unittest discover -s tests -v
```

### Phase 2 (W2): Deterministic Split Freezing & Integrity Check
```bash
python data/freeze_datasets.py && chmod 444 data/*.jsonl data/manifest_sha256.json
```

### Phase 3 (W3): End-to-End 12-Example Pipeline Smoke Test
```bash
# Deterministic pipeline smoke test + Gate 3 verification
python scripts/run_smoke_test.py

# On GPU host with Qwen/Qwen3.5-4B and HuggingFaceTB/SmolLM3-3B weights:
python scripts/run_smoke_test.py --live-gpu
```

### Phases 4–7 (W4–W8): Calibration, Full Matrix, Analysis & Publication Artifacts
```bash
# Single-command GPU orchestration (calibration -> 4B -> SmolLM3-3B -> 0.8B/2B -> stats & figures)
bash scripts/run_gpu_phases_4_to_6.sh

# Or run stages manually:
python scripts/run_matrix.py --model Qwen/Qwen3.5-4B --calibrate-only
python scripts/run_matrix.py --model Qwen/Qwen3.5-4B --strict-diag
python scripts/run_matrix.py --model Qwen/Qwen3.5-4B
python scripts/run_matrix.py --model HuggingFaceTB/SmolLM3-3B --strict-diag
python scripts/run_matrix.py --model HuggingFaceTB/SmolLM3-3B

for MODEL in Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-9B; do
  python scripts/run_matrix.py --model "${MODEL}"
done

python scripts/analyze_and_plot.py
bash scripts/check_anonymity.sh
```
