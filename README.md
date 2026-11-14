# C-BARR: Cascaded Budgeted Adaptive Reasoning Routing

- Doc: [C-BARR](https://docs.google.com/document/d/1AGvSbhy1wJ5dkb957VBXWJH0DKigL5tvZquLPeps7KI).

## Overview

**Cascaded Budgeted Adaptive Reasoning Routing (C-BARR)** is a training-free test-time compute routing and constraint-retention framework operating on frozen language model weights across Hybrid Gated DeltaNet (`Qwen/Qwen3.5-0.8B/2B/4B/9B`) and Dense Softmax GQA (`HuggingFaceTB/SmolLM3-3B`) backbones.

It combines:
1. **Gate 1 (`src/engine.py:run_gate1_lexical`)**: Sub-millisecond numeral-masked lexical screening across all 25 canonical IFEval constraint templates (10 Planning + 15 Precision: 5 Inhibition and 10 Structural, disambiguating capital-word frequency rules from planning rules and question-clause math counts from constraint counts) plus CRGC-style conflict-balanced `Structure: ... | Format: ...` clause extraction.
2. **Gate 2 (`src/engine.py:CBARRDualModeEngine.run_gate2_fast_probe`)**: A $\le 32$-token non-thinking Shannon entropy $(H_{\text{first}}, H_{\text{mean}}, H_{\text{max}})$ and Thinking Inertia $I_{\text{inertia}}(x)$ probe (also logging sub-trajectory horizons $K \in \{1, 4, 8, 16, 32\}$ for early-glimpse vs.\ 32-token horizon ablations) fitted via zero-leakage two-stage calibration (`seed_60` $\rightarrow$ `dev_60`).
3. **Method M2 (`src/engine.py:CBARRDualModeEngine.run_capped_think_and_anchor`)**: Tight tiered thinking termination ($B \in \{64, 128, 250\}$, guaranteeing $T_{\text{delib}} = T_{\text{probe}} + T_{\text{think}} \le 250$ for 100% $\text{OAA}_{250}$ compliance) with Stage-1 $\rightarrow$ Stage-2 KV/DeltaNet state continuation paired with channel-stripped post-`</think>` constraint re-anchoring (`post_stripped`) inside a turn-delimited `user` bridge to prevent echo loops.

---

## Repository Architecture

| Path | Role | Key Outputs & Responsibilities |
| :--- | :--- | :--- |
| `notebooks/C_BARR_Colab_Kaggle.ipynb` | Unified 10-cell Colab/Kaggle GPU runner | Auto-detects GPUs (`GPU_COUNT >= 2` runs independent models concurrently across `CUDA_VISIBLE_DEVICES=0,1` on Kaggle `T4 x2` without git lock contention), precision (`float16` on `T4`/`P100`, `bfloat16` on `L4`/`A100`), and VRAM (`9B` gated on $\ge 22$ GB), installs optional DeltaNet kernels, tees logs to `outputs/logs/cbarr_<RUN_ID>.log`, syncs compact caches + `EXPERIMENT_LOG.md`, and exports `cbarr_results_bundle.zip`. |
| `notebooks/C_BARR_Part1_Primary_Colab_Kaggle.ipynb` | Split Session 1 of 2 (Primary Models) | Runs Gate 0, Step 0a smoke, Step 0b calibration, and Phase 4/5 (`strict341 -> diag491` for `Qwen/Qwen3.5-4B` & `HuggingFaceTB/SmolLM3-3B`) concurrently on dual GPUs or sequentially on 1 GPU, then pushes compact summaries + `*.jsonl.gz` caches + `EXPERIMENT_LOG.md` to GitHub via `scripts/sync_results.py` and saves `cbarr_part1_bundle.zip`. |
| `notebooks/C_BARR_Part2_Scale_Analyze_Colab_Kaggle.ipynb` | Split Session 2 of 2 (Scale Sweep + Combined Phase 7) | Pulls & unpacks Part 1 caches via `scripts/sync_results.py --mode unpack`, runs Phase 6 Scale Sweep (`0.8B` & `2B` in parallel on dual GPUs, `+9B` when VRAM $\ge 22$ GB) + optional W7 on `4B`, runs combined Phase 7 (`analyze_and_plot.py` + `check_anonymity.sh`) across all Part 1 + Part 2 models, and pushes the combined artifacts (`stats_summary.json`, `paper_tables.tex`, anonymized `paper/figures/*.pdf`, `outputs/*/*.jsonl.gz`, `EXPERIMENT_LOG.md`) to GitHub. |
| `scripts/build_notebooks.py` | Single-source notebook generator | Emits `C_BARR_Colab_Kaggle.ipynb`, `C_BARR_Part1_Primary_Colab_Kaggle.ipynb`, and `C_BARR_Part2_Scale_Analyze_Colab_Kaggle.ipynb` with compile-time AST and magic-free validation. |
| `scripts/sync_results.py` | Compact cross-session result sync utility | Packs per-model summaries + deterministic `mtime=0` gzipped caches (`*.jsonl.gz`, ~11x smaller than raw `*.jsonl`), unpacks them into `OUT_DIR` for Part 2 (`--mode unpack`), verifies `check_anonymity.sh` before staging PDFs, and commits + pushes cleanly via `GIT_ASKPASS`. |
| `data/freeze_datasets.py` | Deterministic zero-overlap dataset builder (`seed = 42`) | Materializes `seed_60.jsonl`, `dev_60.jsonl`, `eval_450.jsonl`, `ifeval_diag_491.jsonl`, `ifeval_diag_341_strict.jsonl`, and `manifest_sha256.json`. |
| `src/graders.py` | Seeded channel parser, 4-gram loop & CoT spillover audit, and verifiers | Pins `DetectorFactory.seed = 0`; strips `<think>` and EOS markers; grades GSM8K, CSQA, and IFEval (Inhibition + Structural); logs `has_cot_spillover` (Patch D16). |
| `src/engine.py` | 25-template Gate 1, `CBARRDualModeEngine`, $B \in \{64, 128, 250\}$ termination, Controls A1–B3 | Runs under `@torch.inference_mode()`; normalizes `Qwen3.5` and `SmolLM3-3B` chat templates (`Today Date: 01 November 2026`); caps answer budget at 1024 tokens (Patch D14); enforces `use_cache=True`, offline local-cache fallback, and device-safe Qwen3.5 PyTorch fallback on non-CUDA tensors when `fla`/`causal_conv1d` are installed; reuses deterministic greedy Stage-1 prefixes, native probe/`B0` continuations, and Stage-2 completions without changing outputs; executes turn-delimited `post_stripped` and `pre_close` re-anchoring. |
| `src/routers.py` | Two-stage `seed_60`/`dev_60` calibrator and B0–M2 policy resolver | Fits balanced Gate-2 LogReg weights on `seed_60`, sweeps $\tau^*$ on `dev_60` using $U_{\lambda}$, computes train/val Precision/Recall/F1/ROC-AUC diagnostics, enforces Patch D13 ($T_{\text{delib}} \le 250$), and resolves `B0`–`B2b`, `M1a`–`M1c`, `M2_B64/128/256`, `M2_Pre`, and `M2`. |
| `src/metrics.py` | Statistical inference, efficiency metrics, and causal attribution | Computes exact & Lancaster mid-$p$ McNemar, Holm-Bonferroni, Wilson 95% CIs (`wilson_ci`), Clean-446 defect filtering (`filter_clean_446`), paired bootstrap CIs, finite-sample split Conformal Risk Control & Hoeffding bounds (`compute_conformal_risk_bounds`), discrete hindsight `Oracle_UB` efficiency ratios (`compute_oracle_efficiency_audit`), `MacroAcc`, `OAA_250`, `AUC_OAA`, `U_0.05`, `BSR(B)`, `spillover_rate`, and $\rho_{\text{recency}}$ vs. $\rho_{\text{interference}}$. |
| `src/logfmt.py` | Highlighted terminal & notebook formatter | Emits ANSI SGR color banners/statuses when `CBARR_COLOR=1` or interactive TTY, falling back cleanly to plain text under `NO_COLOR` or file redirection. |
| `scripts/setup_env.sh` | Baseline environment & IFEval setup | Installs runtime Python packages, attempts optional DeltaNet kernel setup, sparse-clones `google-research/instruction_following_eval`, and downloads required NLTK data. |
| `scripts/setup_fast_kernels.py` | Safe GPU kernel installer & self-test | Provisions `flash-linear-attention` (`fla`) and wheel-first `causal-conv1d` on Volta/Turing+ (`sm >= 7.5`) pinned to active `torch`/`transformers`, runs a live CUDA check on Qwen3.5 hooks, and auto-rolls back to PyTorch fallback if incompatible. |
| `scripts/run_matrix.py` | Incremental GPU/MPS cache runner and LaTeX row generator | Verifies `manifest_sha256.json`; auto-selects `bfloat16` on supported Apple Silicon MPS and Ampere+ CUDA GPUs; skips unused `seed_60` tiers (`calib_seed_only`), fuses native probe $\to$ `B0`, and deduplicates `B2b_post_raw` from `B2a_post_anchor`; accepts `--offline`, hyperparameter overrides (`--seed`, `--lam-utility`, `--norm-tokens`, `--reg-c`, `--tau-min/max/steps`), and per-step ledger flags (`--run-id`, `--exp-log`, `--step-checkpoint-every`, `--git-push`); streams running Acc/`IFEval-Prec`/MeanTok/ETA; and checkpoints `Calibration`, mid-`eval_450` `progress_checkpoint.json`, `eval450_complete`, each Section 6 control, and final `matrix_summary_{strict341,diag491}.json`. |
| `scripts/run_gpu_phases_4_to_6.sh` | Lab GPU & local MPS orchestration (Step 0 + Phases 4–7) | Probes CUDA & Apple MPS (`BF16`, `MaxRecMem`), tees highlighted runs to `outputs/logs/cbarr_<RUN_ID>.log`, saves per-stage durations to `outputs/logs/timings_<RUN_ID>.jsonl` and `params_<RUN_ID>.json`, finalizes PASS/FAIL on `EXIT` trap, supports `--device`, `--model`/`--models`, `--calibrate-only`, `--strict-only`, `--offline`, hyperparameter + `--checkpoint-every` flags, and optionally pushes every step/loop via `--push`. |
| `scripts/log_experiment.py` & `EXPERIMENT_LOG.md` | Per-step & per-loop experiment ledger | Upserts anonymized `<!-- CBARR_RUN_START/END -->` blocks into `EXPERIMENT_LOG.md`, appends records to `outputs/experiment_history.jsonl`, detects CUDA GPUs and Apple Silicon MPS (`sysctl` chip, `recommended_max_memory`, live `bf16_native`), renders training params, hardware/kernel status, live split progress, stage timings, Gate-2 classifier metrics, variant tables, Section 6 attribution, W7, and Phase 7 Gate-4 results, and pushes cleanly via transient `GIT_ASKPASS`. |
| `scripts/analyze_and_plot.py` | Phase 7 (W8) analysis, tables, and vector PDF figures | Runs Clean-446 sensitivity, 10k paired bootstrap CIs, Conformal Risk & Oracle Efficiency audits, $K \in \{1, 4, 8, 16, 32\}$ probe-horizon ablations, Gates 4–6 & Contingency R1–R4 audit, and generates `outputs/stats_summary.json`, `outputs/paper_tables.tex`, and `paper/figures/fig{1,2,3}_*.pdf`. |
| `scripts/ext_deltanet_blend.py` / `scripts/ext_prefill_mlp.py` | Optional Workstream W7 extensions | Gated DeltaNet `recurrent_states` + optional 1D `conv_states` (`--blend-conv-state`) blending and full-attention `<think>` KV eviction (`--evict-think-kv`) on $E_{\text{hurt}}$; `.npz`-cached multi-head prefill MLP probe (`MultiHeadPrefillProbe`) with `IFEval++`-style paraphrase stress evaluation. |
| `scripts/check_anonymity.sh` | Double-blind submission linter | Verifies `\author{Anonymous Authors}` and checks `.tex`, `.bib`, and PDF metadata under `paper/` for PII/path leaks. |
| `scripts/run_smoke_test.py` | Phase 3 (W3) 12-item smoke test & Gate 3 verifier | Slices 2 items/slice into `outputs/smoke_12_<model>.jsonl` (plus `/tmp/smoke_12.jsonl` on mock CPU) and writes `outputs/smoke_test_gate3_report.json`. |
| `tests/test_cbarr_preflight.py` & `tests/test_phases_4_to_7.py` | Offline CPU verification suite (26 tests total) | Covers 25/25 Gate-1 regex templates, capital-word/numeral disambiguation, CRGC conflict bridges, seeded `langdetect`, D13–D16 invariants, Clean-446, 10k bootstrap, Conformal Risk & Oracle efficiency bounds, Stage-1/Stage-2 KV/DeltaNet continuation parity, native probe $\to$ `B0` fusion and `calib_seed_only`, $K \in \{1, 4, 8, 16, 32\}$ horizon probes, hardware dtype safety (CUDA + MPS BF16), device-safe Qwen3.5 fallback under mocked Triton/`fla`, MPS telemetry, fast kernel gating, `logfmt`, all 3 notebooks AST validation, Part 1 -> Part 2 `sync_results.py` bare-remote combine test, bare-remote `EXPERIMENT_LOG.md` push, per-step `run_matrix.py` checkpoints, `ext_deltanet_blend` `conv_states`/KV-eviction, `ext_prefill_mlp` `.npz` caching + `IFEval++` paraphrase stress, and `paper/` anonymity. |
| `paper/main.tex` | Full NeurIPS-style 8-section manuscript & Booktabs LaTeX templates | Integrates Proposition 1 (Conformal Risk Control), $K \in \{1, 4, 8, 16, 32\}$ probe-horizon & `CRGC` bridge details, Figures 1–3, Tables 1–3, and W7 mechanistic subsections (`paper/references.bib` includes verified 2025–2026 citations with anonymized metadata). |

---

## Running on Google Colab or Kaggle GPUs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dolong2110/C-BARR/blob/master/notebooks/C_BARR_Colab_Kaggle.ipynb)

Choose either the single end-to-end runner or the **2-session Kaggle/Colab split** (recommended for Kaggle's 9–12h GPU session cap):

- **Recommended 2-Session Split** (auto-parallelizes across `GPU 0` + `GPU 1` on Kaggle `GPU T4 x2`, then merges everything cleanly on GitHub):
  1. **Session 1** — [notebooks/C_BARR_Part1_Primary_Colab_Kaggle.ipynb](notebooks/C_BARR_Part1_Primary_Colab_Kaggle.ipynb): Runs Gate 0, Smoke, Calibration, and Primary Models (`Qwen/Qwen3.5-4B` & `HuggingFaceTB/SmolLM3-3B`, `strict341 -> diag491`). Automatically compresses compact caches (`*.jsonl.gz`, ~1.5–2 MB/model) and pushes them + summaries + `EXPERIMENT_LOG.md` to GitHub after Stage A and Stage B.
  2. **Session 2** — [notebooks/C_BARR_Part2_Scale_Analyze_Colab_Kaggle.ipynb](notebooks/C_BARR_Part2_Scale_Analyze_Colab_Kaggle.ipynb): Pulls `dolong2110/C-BARR`, unpacks Part 1's `4B` and `SmolLM3-3B` caches into `OUT_DIR` via `scripts/sync_results.py --mode unpack`, runs the Scale Sweep (`Qwen/Qwen3.5-0.8B` & `Qwen/Qwen3.5-2B`, plus `9B` if `VRAM >= 22 GB`) and optional W7 extensions, runs combined **Phase 7** (`analyze_and_plot.py` + `check_anonymity.sh`) over **all Part 1 + Part 2 models**, and pushes the combined tables, stats, anonymized PDFs, `*.jsonl.gz`, and `EXPERIMENT_LOG.md` back to GitHub.
- **Single-Session Runner** — [notebooks/C_BARR_Colab_Kaggle.ipynb](notebooks/C_BARR_Colab_Kaggle.ipynb): Runs everything in one session (best on Colab Pro `L4`/`A100` or resume-enabled Google Drive).

Upload or import any of the three notebooks into **Google Colab** (Runtime -> GPU: `T4`, `L4`, or `A100`) or **Kaggle** (Accelerator: `GPU P100` or `GPU T4 x2`, Internet: **On**). Make sure `GITHUB_TOKEN` (fine-grained PAT with **Contents: Read and write**) is added as a Kaggle/Colab Secret so `Part 1 -> Part 2` hand-off pushes to GitHub automatically.

Each 10-cell notebook automatically:
1. **Detects runtime, GPU count & precision (`cbarr-env-detect`)**: Selects `--dtype auto` (`float16` on `T4`/`P100`, `bfloat16` on `L4`/`A100`), detects `GPU_COUNT >= 2` for concurrent dual-GPU model dispatch, optionally mounts Google Drive on Colab so `OUT_DIR` caches survive disconnects, and routes `HF_HOME` to `/kaggle/temp/hf_cache` (or `/tmp/hf_cache`) so `/kaggle/working` stays under quota.
2. **Loads config & secrets (`cbarr-config`)**: Exposes model lists, `AUTO_PARALLEL_MULTI_GPU = True`, `SYNC_RESULTS_TO_GITHUB = True`, `DTYPE`, training hyperparameters (`SEED`, `LAM_UTILITY`, `NORM_TOKENS`, `LOGREG_C`, `TAU_MIN`, `TAU_MAX`, `TAU_STEPS`), `CHECKPOINT_EVERY_ITEMS = 150`, `RECORD_EXPERIMENT_LOG = True`, `AUTO_PUSH_RESULTS_TO_GITHUB = True`, plus optional secrets `HF_TOKEN` and `GITHUB_TOKEN`.
3. **Safely syncs repo & installs fast kernels (`cbarr-clone-install`)**: Clones or rebases (`git checkout -- outputs/smoke_test_gate3_report.json` + `git pull --rebase --autostash`), installs missing runtime deps without overwriting container PyTorch, and runs `scripts/setup_fast_kernels.py --mode auto` for Triton/CUDA `fla` + `causal-conv1d` (falling back safely to PyTorch if unavailable or P100 `sm_60`).
4. **Downloads/verifies data & unpacks prior results (`cbarr-download-data`)**: Verifies frozen JSONL splits against `data/manifest_sha256.json`, unpacks any pushed `outputs/*/*.jsonl.gz` into `OUT_DIR` when `UNPACK_SYNCED_RESULTS_ON_START = True`, and warms HF tokenizers.
5. **Executes stages with live highlighted logs & race-free multi-GPU dispatch (`cbarr-gate0-smoke` through `cbarr-scale-w7`)**: Uses thread-safe `run_and_tee()` to display ANSI-highlighted logs while teeing to `OUT_DIR/logs/cbarr_<RUN_ID>.log`, assigns concurrent workers isolated `CUDA_VISIBLE_DEVICES` slots without git contention, and updates/pushes `EXPERIMENT_LOG.md` + compact caches via `scripts/sync_results.py`.
6. **Analyzes, combines & exports bundle (`cbarr-analyze-export`)**: Verifies Part 1 + Part 2 model completeness, runs `analyze_and_plot.py` and `check_anonymity.sh`, prints Gate-4 metrics, pushes unified artifacts to GitHub, and packages `cbarr_results_bundle.zip` (or `cbarr_part1_bundle.zip`).

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

# On GPU/MPS host with cached Qwen/Qwen3.5-4B, SmolLM3-3B, or single-model Qwen/Qwen3.5-0.8B:
python scripts/run_smoke_test.py --live-gpu
python scripts/run_smoke_test.py --live-gpu --models Qwen/Qwen3.5-0.8B --device auto
```

### Phases 4–7 (W4–W8): Lab GPU & Local MPS Orchestration

#### Recommended Single-Command Lab GPU Run
```bash
# Highlights key metrics in terminal, saves raw log to outputs/logs/cbarr_<RUN_ID>.log,
# checkpoints EXPERIMENT_LOG.md every 150 new items + every Calibration/Section-6/Phase step,
# and pushes each checkpoint commit cleanly to GitHub (--push uses SSH agent or GITHUB_TOKEN).
CBARR_COLOR=1 bash scripts/run_gpu_phases_4_to_6.sh --gpu 0 --push --checkpoint-every 150 --with-9b --with-w7
```

#### Local Apple Silicon (MPS) Single-Model Run
```bash
# Run a single cached model locally on Apple Silicon (auto-selects mps + native bfloat16 when supported):
bash scripts/run_gpu_phases_4_to_6.sh --device auto --models Qwen/Qwen3.5-0.8B --strict-only --offline
```
> **Hardware guidance**: On 16 GB Apple Silicon Macs (e.g., M5 with ~11.8 GB recommended MPS working set and PyTorch fallback without CUDA `fla`/`causal-conv1d`), `Qwen/Qwen3.5-0.8B` (~1.5 GB) and `2B` (~4 GB) fit comfortably for smoke tests and single-model runs, whereas full multi-model (`4B` + `3B` + `0.8B` + `2B` + `9B`) sweeps are substantially faster on Colab/Kaggle/cloud CUDA GPUs.

Key flags for `scripts/run_gpu_phases_4_to_6.sh`:
- `--gpu <id>` / `--device <auto|cuda|mps|cpu>` / `--dtype <auto|bf16|fp16|fp32>`: Select CUDA index, backend (`auto` picks CUDA -> MPS -> CPU), and precision (`auto` picks `bfloat16` on Ampere/Ada/Hopper `sm >= 80` and BF16-capable Apple Silicon MPS, or `float16` on `T4`/`P100`).
- `--models "<space-separated list>"` / `--model <id>`: Override the default model list to target a single model or subset (e.g., `--models Qwen/Qwen3.5-0.8B`).
- `--calibrate-only` / `--strict-only` / `--offline`: Run only Phase 4 calibration, run only Phase 5 strict matrix + Phase 7, or force offline cached HuggingFace weight loading.
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
