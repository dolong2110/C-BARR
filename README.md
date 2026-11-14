# C-BARR: Cascaded Budgeted Adaptive Reasoning Routing

- Doc: [C-BARR](https://docs.google.com/document/d/1AGvSbhy1wJ5dkb957VBXWJH0DKigL5tvZquLPeps7KI).

## Overview

**Cascaded Budgeted Adaptive Reasoning Routing (C-BARR)** is a training-free test-time compute routing and constraint-retention framework operating on frozen language model weights across Hybrid Gated DeltaNet (`Qwen/Qwen3.5-0.8B/2B/4B/9B`) and Dense Softmax GQA (`HuggingFaceTB/SmolLM3-3B`) backbones.

It combines:
1. **Gate 1 (`src/engine.py:run_gate1_lexical`)**: Sub-millisecond numeral-masked lexical screening across all 25 canonical IFEval constraint templates (10 Planning + 15 Precision: 5 Inhibition and 10 Structural) and recursive-safe anchor clause extraction.
2. **Gate 2 (`src/engine.py:CBARRDualModeEngine.run_gate2_fast_probe`)**: A $\le 32$-token non-thinking Shannon entropy $(H_{\text{first}}, H_{\text{mean}}, H_{\text{max}})$ and Thinking Inertia $I_{\text{inertia}}(x)$ probe fitted via zero-leakage two-stage calibration (`seed_60` $\rightarrow$ `dev_60`).
3. **Method M2 (`src/engine.py:CBARRDualModeEngine.run_capped_think_and_anchor`)**: Tight tiered thinking termination ($B \in \{64, 128, 250\}$, guaranteeing $T_{\text{delib}} = T_{\text{probe}} + T_{\text{think}} \le 250$ for 100% $\text{OAA}_{250}$ compliance) paired with channel-stripped post-`</think>` constraint re-anchoring (`post_stripped`) inside a turn-delimited `user` bridge to prevent echo loops.

---

## Repository Architecture

| Path | Role | Key Outputs & Responsibilities |
| :--- | :--- | :--- |
| `notebooks/C_BARR_Colab_Kaggle.ipynb` | 10-cell Google Colab & Kaggle GPU runner | Auto-detects GPU (`T4`/`P100` `float16` vs. `L4`/`A100` `bfloat16`) and VRAM (`9B` gated on $\ge 22$ GB), installs optional DeltaNet kernels, configures training params, tees ANSI-highlighted logs to `outputs/logs/cbarr_<RUN_ID>.log`, updates & pushes `EXPERIMENT_LOG.md` per step/loop via `GITHUB_TOKEN`, and exports `cbarr_results_bundle.zip`. |
| `data/freeze_datasets.py` | Deterministic zero-overlap dataset builder (`seed = 42`) | Materializes `seed_60.jsonl`, `dev_60.jsonl`, `eval_450.jsonl`, `ifeval_diag_491.jsonl`, `ifeval_diag_341_strict.jsonl`, and `manifest_sha256.json`. |
| `src/graders.py` | Seeded channel parser, 4-gram loop & CoT spillover audit, and verifiers | Pins `DetectorFactory.seed = 0`; strips `<think>` and EOS markers; grades GSM8K, CSQA, and IFEval (Inhibition + Structural); logs `has_cot_spillover` (Patch D16). |
| `src/engine.py` | 25-template Gate 1, `CBARRDualModeEngine`, $B \in \{64, 128, 250\}$ termination, Controls A1–B3 | Runs under `@torch.inference_mode()`; normalizes `Qwen3.5` and `SmolLM3-3B` chat templates (`Today Date: 01 November 2026`); caps answer budget at 1024 tokens (Patch D14); reuses deterministic greedy `B1b` Stage-1 prefixes and Stage-2 completions without changing outputs; executes turn-delimited `post_stripped` and `pre_close` re-anchoring. |
| `src/routers.py` | Two-stage `seed_60`/`dev_60` calibrator and B0–M2 policy resolver | Fits balanced Gate-2 LogReg weights on `seed_60`, sweeps $\tau^*$ on `dev_60` using $U_{\lambda}$, computes train/val Precision/Recall/F1/ROC-AUC diagnostics, enforces Patch D13 ($T_{\text{delib}} \le 250$), and resolves `B0`–`B2b`, `M1a`–`M1c`, `M2_B64/128/256`, `M2_Pre`, and `M2`. |
| `src/metrics.py` | Statistical inference, efficiency metrics, and causal attribution | Computes exact & Lancaster mid-$p$ McNemar, Holm-Bonferroni, Wilson 95% CIs (`wilson_ci`), Clean-446 defect filtering (`filter_clean_446`), paired bootstrap CIs, `MacroAcc`, `OAA_250`, `AUC_OAA`, `U_0.05`, `BSR(B)`, `spillover_rate`, `Oracle_UB`, and $\rho_{\text{recency}}$ vs. $\rho_{\text{interference}}$. |
| `src/logfmt.py` | Highlighted terminal & notebook formatter | Emits ANSI SGR color banners/statuses when `CBARR_COLOR=1` or interactive TTY, falling back cleanly to plain text under `NO_COLOR` or file redirection. |
| `scripts/setup_env.sh` | Baseline environment & IFEval setup | Installs runtime Python packages, attempts optional DeltaNet kernel setup, sparse-clones `google-research/instruction_following_eval`, and downloads required NLTK data. |
| `scripts/setup_fast_kernels.py` | Safe GPU kernel installer & self-test | Provisions `flash-linear-attention` (`fla`) and wheel-first `causal-conv1d` on Volta/Turing+ (`sm >= 7.5`) pinned to active `torch`/`transformers`, runs a live CUDA check on Qwen3.5 hooks, and auto-rolls back to PyTorch fallback if incompatible. |
| `scripts/run_matrix.py` | Incremental GPU cache runner and LaTeX row generator | Verifies `manifest_sha256.json`; accepts hyperparameter overrides (`--seed`, `--lam-utility`, `--norm-tokens`, `--reg-c`, `--tau-min/max/steps`) and per-step ledger flags (`--run-id`, `--exp-log`, `--step-checkpoint-every`, `--git-push`); streams running Acc/`IFEval-Prec`/MeanTok/ETA; caches primitives + `E_hurt` controls (`diag_controls_cache.jsonl`); and checkpoints `Calibration`, mid-`eval_450` `progress_checkpoint.json`, `eval450_complete`, each Section 6 control, and final `matrix_summary_{strict341,diag491}.json`. |
| `scripts/run_gpu_phases_4_to_6.sh` | Lab GPU orchestration (Step 0 + Phases 4–7) | Tees highlighted runs to `outputs/logs/cbarr_<RUN_ID>.log`, saves per-stage durations to `outputs/logs/timings_<RUN_ID>.jsonl` and `params_<RUN_ID>.json`, finalizes PASS/FAIL on `EXIT` trap, forwards hyperparameter + `--checkpoint-every` flags, and optionally pushes every step/loop via `--push`. |
| `scripts/log_experiment.py` & `EXPERIMENT_LOG.md` | Per-step & per-loop experiment ledger | Upserts anonymized `<!-- CBARR_RUN_START/END -->` blocks into `EXPERIMENT_LOG.md`, appends records to `outputs/experiment_history.jsonl`, renders training params, hardware/kernel status, live split progress, stage timings, Gate-2 classifier metrics, variant tables, Section 6 attribution, W7, and Phase 7 Gate-4 results, and pushes cleanly via transient `GIT_ASKPASS`. |
| `scripts/analyze_and_plot.py` | Phase 7 (W8) analysis, tables, and vector PDF figures | Runs Clean-446 sensitivity, 10k paired bootstrap CIs, Gates 4–6 & Contingency R1–R4 audit, and generates `outputs/stats_summary.json`, `outputs/paper_tables.tex`, and `paper/figures/fig{1,2,3}_*.pdf`. |
| `scripts/ext_deltanet_blend.py` / `scripts/ext_prefill_mlp.py` | Optional Workstream W7 extensions | Gated DeltaNet `recurrent_states` blending (`Qwen3.5`) on $E_{\text{hurt}}$ and mid-layer prefill hidden-state MLP escalation probe. |
| `scripts/check_anonymity.sh` | Double-blind submission linter | Verifies `\author{Anonymous Authors}` and checks `.tex`, `.bib`, and PDF metadata under `paper/` for PII/path leaks. |
| `scripts/run_smoke_test.py` | Phase 3 (W3) 12-item smoke test & Gate 3 verifier | Slices 2 items/slice into `outputs/smoke_12_<model>.jsonl` (plus `/tmp/smoke_12.jsonl` on mock CPU) and writes `outputs/smoke_test_gate3_report.json`. |
| `tests/test_cbarr_preflight.py` & `tests/test_phases_4_to_7.py` | Offline CPU verification suite (16 tests total) | Covers 25/25 Gate-1 regex templates, seeded `langdetect`, D13–D16 invariants, Clean-446, 10k bootstrap, Stage-1/Stage-2 reuse parity, hardware dtype safety, fast kernel gating, `logfmt`, notebook AST validation, bare-remote `EXPERIMENT_LOG.md` push, per-step `run_matrix.py` checkpoints, and `paper/` anonymity. |
| `paper/main.tex` | Manuscript & Booktabs LaTeX Templates | Integrates Tables 1, 2, and 3 (`paper/references.bib` includes 2025–2026 baselines with anonymized metadata). |

---

## Running on Google Colab or Kaggle GPUs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dolong2110/C-BARR/blob/master/notebooks/C_BARR_Colab_Kaggle.ipynb)

Upload or import [notebooks/C_BARR_Colab_Kaggle.ipynb](notebooks/C_BARR_Colab_Kaggle.ipynb) into **Google Colab** (Runtime -> GPU: `T4`, `L4`, or `A100`) or **Kaggle** (Accelerator: `GPU P100` or `GPU T4 x2`, Internet: **On**).

The 10-cell notebook automatically:
1. **Detects runtime & sets precision (`cbarr-env-detect`)**: Selects `--dtype auto` (`float16` on `T4`/`P100`, `bfloat16` on `L4`/`A100`), optionally mounts Google Drive on Colab so `OUT_DIR` caches survive disconnects, and routes `HF_HOME` to `/kaggle/temp/hf_cache` (or `/tmp/hf_cache`) so `/kaggle/working` stays under quota.
2. **Loads config & secrets (`cbarr-config`)**: Exposes model lists, `DTYPE`, training hyperparameters (`SEED`, `LAM_UTILITY`, `NORM_TOKENS`, `LOGREG_C`, `TAU_MIN`, `TAU_MAX`, `TAU_STEPS`), `CHECKPOINT_EVERY_ITEMS = 150`, `RECORD_EXPERIMENT_LOG = True`, `AUTO_PUSH_RESULTS_TO_GITHUB = True`, plus optional secrets `HF_TOKEN` (higher HF Hub rate limits) and `GITHUB_TOKEN` (fine-grained PAT with **Contents: Read and write** for automatic per-step GitHub pushes).
3. **Safely syncs repo & installs fast kernels (`cbarr-clone-install`)**: Clones or rebases (`git checkout -- outputs` + `git pull --rebase --autostash`), installs missing runtime deps without overwriting container PyTorch, and runs `scripts/setup_fast_kernels.py --mode auto` for Triton/CUDA `fla` + `causal-conv1d` (falling back safely to PyTorch if unavailable or P100 `sm_60`).
4. **Downloads/verifies data (`cbarr-download-data`)**: Verifies frozen JSONL splits against `data/manifest_sha256.json` (or builds deterministically via `data/freeze_datasets.py`) and warms HF tokenizers.
5. **Executes Stages with live highlighted logs & per-step GitHub push (`cbarr-gate0-smoke` through `cbarr-scale-w7`)**: Uses `run_and_tee()` to display ANSI-highlighted cell logs while saving an ANSI-stripped log to `OUT_DIR/logs/cbarr_<RUN_ID>.log`, pushing updated telemetry into `EXPERIMENT_LOG.md` after every Gate 0, Step 0a, Calibration, `CHECKPOINT_EVERY_ITEMS` items in `eval_450`, Eval-450 completion, each Section-6 control, Scale Sweep, and W7 step.
6. **Analyzes & exports bundle (`cbarr-analyze-export`)**: Runs `analyze_and_plot.py` and `check_anonymity.sh`, prints Gate-4 metrics, and packages `EXPERIMENT_LOG.md`, `outputs/logs/*`, `stats_summary.json`, `paper_tables.tex`, per-model summaries, and `paper/figures/*.pdf` into `cbarr_results_bundle.zip`.

---

## Execution Playbook (Local & Lab GPUs)

### Phase 1 (W1): Workspace Provisioning & Pre-Flight Verification
```bash
bash scripts/setup_env.sh
python -m unittest discover -s tests -v
```
*(Optional on modern GPUs if running outside `setup_env.sh`: `python scripts/setup_fast_kernels.py --mode auto`)*

### Phase 2 (W2): Deterministic Split Freezing & Integrity Check
```bash
python data/freeze_datasets.py && chmod 444 data/*.jsonl data/manifest_sha256.json
```

### Phase 3 (W3): End-to-End 12-Example Pipeline Smoke Test
```bash
# Deterministic offline CPU smoke test + Gate 3 verification
python scripts/run_smoke_test.py

# On GPU host with Qwen/Qwen3.5-4B and HuggingFaceTB/SmolLM3-3B weights:
python scripts/run_smoke_test.py --live-gpu
```

### Phases 4–7 (W4–W8): Lab GPU Orchestration & Offline Dry-Run

#### Recommended Single-Command Lab GPU Run
```bash
# Highlights key metrics in terminal, saves raw log to outputs/logs/cbarr_<RUN_ID>.log,
# checkpoints EXPERIMENT_LOG.md every 150 new items + every Calibration/Section-6/Phase step,
# and pushes each checkpoint commit cleanly to GitHub (--push uses SSH agent or GITHUB_TOKEN).
CBARR_COLOR=1 bash scripts/run_gpu_phases_4_to_6.sh --gpu 0 --push --checkpoint-every 150 --with-9b --with-w7
```

Key flags for `scripts/run_gpu_phases_4_to_6.sh`:
- `--gpu <id>` / `--dtype <auto|bf16|fp16|fp32>`: Select `CUDA_VISIBLE_DEVICES` and precision (`auto` picks `bfloat16` on Ampere/Ada/Hopper `sm >= 80`, `float16` on `T4`/`P100`).
- `--push`: Commit & push `EXPERIMENT_LOG.md` at every internal `run_matrix.py` checkpoint and outer loop.
- `--checkpoint-every <N>`: Mid-`eval_450` checkpoint interval in newly evaluated items (default `150` on GPU, `0` on `--mock`).
- `--with-9b`, `--with-w7`, `--include-b512`, `--skip-smoke`: Enable 9B scale sweep, W7 DeltaNet/Prefill MLP probes, Contingency R3 `B=512` sweep, or skip Step 0a.
- `--seed 42 --lam-utility 0.05 --norm-tokens 250.0 --reg-c 1.0 --tau-min 0.15 --tau-max 0.85 --tau-steps 29`: Override calibration hyperparameters (logged into `EXPERIMENT_LOG.md`).
- `--mock`: Run full offline CPU orchestration without touching the repository-tracked `EXPERIMENT_LOG.md` when `--out-dir` is redirected:
  ```bash
  bash scripts/run_gpu_phases_4_to_6.sh --mock --out-dir /tmp/cbarr_mock --fig-dir /tmp/cbarr_figs --exp-log /tmp/cbarr_exp.md
  ```

#### Manual Per-Stage Execution
```bash
python scripts/run_matrix.py --model Qwen/Qwen3.5-4B --calibrate-only
python scripts/run_matrix.py --model Qwen/Qwen3.5-4B --strict-diag --step-checkpoint-every 150 --exp-log EXPERIMENT_LOG.md --git-push
python scripts/run_matrix.py --model Qwen/Qwen3.5-4B --step-checkpoint-every 150 --exp-log EXPERIMENT_LOG.md --git-push
python scripts/run_matrix.py --model HuggingFaceTB/SmolLM3-3B --strict-diag
python scripts/run_matrix.py --model HuggingFaceTB/SmolLM3-3B

for MODEL in Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-9B; do
  python scripts/run_matrix.py --model "${MODEL}" --strict-diag
done

python scripts/analyze_and_plot.py
bash scripts/check_anonymity.sh
```

### Logging & GitHub Sync Summary
- **Tracked in Git**: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) receives sanitized per-step/per-loop tables, hyperparameters, hardware/kernel metadata, timings, and Gate-4 audit summaries.
- **Kept Local / Gitignored (`.gitignore`)**: Per-item caches (`outputs/*/*_cache.jsonl`), full raw console logs (`outputs/logs/cbarr_<RUN_ID>.log`), per-stage timing streams (`outputs/logs/timings_<RUN_ID>.jsonl`), and append-only JSONL run history (`outputs/experiment_history.jsonl`).
