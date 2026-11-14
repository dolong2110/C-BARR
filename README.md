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
| `notebooks/C_BARR_All_In_One.ipynb` | Combined All-In-One 10-cell notebook (Parts 1 + 2 + 3 + 4) | Executes the full C-BARR pipeline end-to-end (Gate 0 -> Smoke -> Calibrate -> `4B` + `SmolLM3-3B` -> Scale `0.8B`/`2B`/`+9B` -> Phase 7) in one Colab/Kaggle session with automatic `N`-GPU sharding (`MAX_SHARDS_PER_MODEL = 4`). |
| `notebooks/C_BARR_1_Worker_Qwen4B.ipynb` | Parallel GPU Worker 1/3 (`Qwen/Qwen3.5-4B`) | Runs Gate 0 (`unittest`), Step 0a smoke, Step 0b calibration, and `Qwen/Qwen3.5-4B` (`strict341 -> diag491`, auto-sharded across up to `MAX_SHARDS_PER_MODEL = 4` local GPUs), pushing **only** `outputs/Qwen__Qwen3.5-4B/*` + `run_manifest.json` via `--no-ledger` with zero conflicts alongside Workers 2–3. |
| `notebooks/C_BARR_2_Worker_SmolLM3.ipynb` | Parallel GPU Worker 2/3 (`HuggingFaceTB/SmolLM3-3B`) | Dedicated runner for `HuggingFaceTB/SmolLM3-3B` (`strict341 -> diag491`, `N`-GPU sharded), pushing **only** `outputs/HuggingFaceTB__SmolLM3-3B/*` + `run_manifest.json` via `--no-ledger` in parallel with Workers 1 and 3. |
| `notebooks/C_BARR_3_Worker_Scale.ipynb` | Parallel GPU Worker 3/3 (Scale Sweep) | Dedicated runner for Phase 6 (`Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`, `+9B` if VRAM $\ge 22$ GB, `N`-GPU sharded), pushing **only** `outputs/Qwen__Qwen3.5-{0.8B,2B,9B}/*` + `run_manifest.json` via `--no-ledger` in parallel with Workers 1–2. |
| `notebooks/C_BARR_4_Combine_Analyze.ipynb` | Step 4 Combiner (Unpack + Optional W7 + Combined Phase 7) | Pulls & unpacks all worker models via `scripts/sync_results.py --mode unpack`, verifies code versions match across workers, optionally runs W7 on `4B` (CPU-capable in ~2 min when `RUN_W7_EXTENSIONS = False`), runs combined Phase 7, and pushes `stats_summary.json`, `paper_tables.tex`, anonymized PDFs, `RESULTS_VERSION`, and `EXPERIMENT_LOG.md`. |
| `VERSION` & `scripts/versioning.py` & `.githooks/pre-push` | Semantic code version & GitHub update guard | Tracks canonical SemVer code version (`VERSION`), exposes `show`/`bump`/`check`/`install-hooks`, blocks non-result pushes that forget to bump `VERSION` when `.githooks/pre-push` is enabled, and manages `outputs/RESULTS_VERSION` monotonic counters. |
| `scripts/build_notebooks.py` | Single-source notebook generator | Emits the 5 `.ipynb` notebooks (`All-In-One` + 3 parallel GPU workers + 1 combiner) and prunes stale notebooks with compile-time AST and magic-free validation, embedding `NOTEBOOK_CODE_VERSION` and reading live `VERSION` + `git_rev`. |
| `scripts/sync_results.py` | Compact cross-session result sync utility | Packs per-model summaries + deterministic `mtime=0` gzipped caches (`*.jsonl.gz`, ~11x smaller than raw `*.jsonl`), stamps monotonic per-model `run_manifest.json` (`sync_seq`, `result_version = vX.Y.Z+mN`), supports `--no-ledger` so parallel workers push strictly disjoint `outputs/<model>/*` folders without conflicts, increments `outputs/RESULTS_VERSION` + `outputs/results_manifest.json` on shared pushes, prints per-model version audits on `--mode unpack`, verifies `check_anonymity.sh`, and pushes cleanly via `GIT_ASKPASS`. |
| `data/freeze_datasets.py` | Deterministic zero-overlap dataset builder (`seed = 42`) | Materializes `seed_60.jsonl`, `dev_60.jsonl`, `eval_450.jsonl`, `ifeval_diag_491.jsonl`, `ifeval_diag_341_strict.jsonl`, and `manifest_sha256.json`. |
| `src/graders.py` | Seeded channel parser, 4-gram loop & CoT spillover audit, and verifiers | Pins `DetectorFactory.seed = 0`; strips `<think>` and EOS markers; grades GSM8K, CSQA, and IFEval (Inhibition + Structural); logs `has_cot_spillover` (Patch D16). |
| `src/engine.py` | 25-template Gate 1, `CBARRDualModeEngine`, $B \in \{64, 128, 250\}$ termination, Controls A1–B3 | Runs under `@torch.inference_mode()`; normalizes `Qwen3.5` and `SmolLM3-3B` chat templates (`Today Date: 01 November 2026`); caps answer budget at 1024 tokens (Patch D14); enforces `use_cache=True`, offline local-cache fallback, and device-safe Qwen3.5 PyTorch fallback on non-CUDA tensors when `fla`/`causal_conv1d` are installed; reuses deterministic greedy Stage-1 prefixes, native probe/`B0` continuations, and Stage-2 completions without changing outputs; executes turn-delimited `post_stripped` and `pre_close` re-anchoring. |
| `src/routers.py` | Two-stage `seed_60`/`dev_60` calibrator and B0–M2 policy resolver | Fits balanced Gate-2 LogReg weights on `seed_60`, sweeps $\tau^*$ on `dev_60` using $U_{\lambda}$, computes train/val Precision/Recall/F1/ROC-AUC diagnostics, enforces Patch D13 ($T_{\text{delib}} \le 250$), and resolves `B0`–`B2b`, `M1a`–`M1c`, `M2_B64/128/256`, `M2_Pre`, and `M2`. |
| `src/metrics.py` | Statistical inference, efficiency metrics, and causal attribution | Computes exact & Lancaster mid-$p$ McNemar, Holm-Bonferroni, Wilson 95% CIs (`wilson_ci`), Clean-446 defect filtering (`filter_clean_446`), paired bootstrap CIs, finite-sample split Conformal Risk Control & Hoeffding bounds (`compute_conformal_risk_bounds`), discrete hindsight `Oracle_UB` efficiency ratios (`compute_oracle_efficiency_audit`), `MacroAcc`, `OAA_250`, `AUC_OAA`, `U_0.05`, `BSR(B)`, `spillover_rate`, and $\rho_{\text{recency}}$ vs. $\rho_{\text{interference}}$. |
| `src/logfmt.py` | Highlighted terminal & notebook formatter | Emits ANSI SGR color banners/statuses when `CBARR_COLOR=1` or interactive TTY, falling back cleanly to plain text under `NO_COLOR` or file redirection. |
| `scripts/setup_env.sh` | Baseline environment & IFEval setup | Installs runtime Python packages, attempts optional DeltaNet kernel setup, sparse-clones `google-research/instruction_following_eval`, and downloads required NLTK data. |
| `scripts/setup_fast_kernels.py` | Safe GPU kernel installer & self-test | Provisions `flash-linear-attention` (`fla`) and wheel-first `causal-conv1d` on Volta/Turing+ (`sm >= 7.5`) pinned to active `torch`/`transformers`, runs a live CUDA check on Qwen3.5 hooks, and auto-rolls back to PyTorch fallback if incompatible. |
| `scripts/run_matrix.py` | Incremental GPU/MPS cache runner and LaTeX row generator | Verifies `manifest_sha256.json`; auto-selects `bfloat16` on supported Apple Silicon MPS and Ampere+ CUDA GPUs; skips unused `seed_60` tiers (`calib_seed_only`), fuses native probe $\to$ `B0`, and deduplicates `B2b_post_raw` from `B2a_post_anchor`; supports intra-model `N`-GPU data sharding (`--shard-idx`, `--num-shards`, `--merge-shards`, `--keep-shards`) using tokenizer-only merge when 100% cached and sorting into canonical order before fitting/scoring; accepts `--offline`, hyperparameter overrides (`--seed`, `--lam-utility`, `--norm-tokens`, `--reg-c`, `--tau-min/max/steps`), and per-step ledger flags (`--run-id`, `--exp-log`, `--step-checkpoint-every`, `--git-push`); streams running Acc/`IFEval-Prec`/MeanTok/ETA; and stamps `code_version` (`VERSION`), `git_rev`, and `run_id` into checkpoints and `matrix_summary_{strict341,diag491}.json`. |
| `scripts/run_gpu_phases_4_to_6.sh` & `scripts/run_part{1..4}_*.sh` | Lab GPU, multi-machine parallel split & local MPS orchestration | Main runner (`--part all|1|2|3|4`, `--shards <N|auto>`) plus 4 matching wrappers (`run_part1_worker_qwen4b.sh`, `run_part2_worker_smollm3.sh`, `run_part3_worker_scale.sh`, `run_part4_combine_analyze.sh`). Probes CUDA/MPS, records `code_version`, supports `--device`, `--model`/`--models`, `--data-dir`, `--calibrate-only`, `--strict-only`, `--offline`, `--skip-preflight`, `--strict-version-match`, `--unpack`/`--no-unpack`, and `--push` (`--no-ledger` disjoint worker sync in Parts 1–3; unified Phase 7 + ledger in Part 4/all). |
| `scripts/log_experiment.py` & `EXPERIMENT_LOG.md` | Per-step & per-loop experiment ledger | Upserts anonymized `<!-- CBARR_RUN_START/END -->` blocks into `EXPERIMENT_LOG.md`, appends `code_version`/`results_seq` records to `outputs/experiment_history.jsonl`, renders `Code Version`/`Results Release` headers plus `Ver` columns & mismatch alerts in Calibration/Matrix/Phase-7 sections, detects CUDA GPUs and Apple Silicon MPS (`sysctl` chip, `recommended_max_memory`, live `bf16_native`), tracks live progress/timings/Gate-2/variants/W7, and pushes cleanly via transient `GIT_ASKPASS`. |
| `scripts/analyze_and_plot.py` | Phase 7 (W8) analysis, tables, and vector PDF figures | Runs Clean-446 sensitivity, 10k paired bootstrap CIs, Conformal Risk & Oracle Efficiency audits, $K \in \{1, 4, 8, 16, 32\}$ probe-horizon ablations, Gates 4–6 & Contingency R1–R4 audit, writes `version_audit` (with optional `--strict-version-match`) into `outputs/stats_summary.json`, and generates `outputs/paper_tables.tex` and `paper/figures/fig{1,2,3}_*.pdf`. |
| `scripts/ext_deltanet_blend.py` / `scripts/ext_prefill_mlp.py` | Optional Workstream W7 extensions | Gated DeltaNet `recurrent_states` + optional 1D `conv_states` (`--blend-conv-state`) blending and full-attention `<think>` KV eviction (`--evict-think-kv`) on $E_{\text{hurt}}$; `.npz`-cached multi-head prefill MLP probe (`MultiHeadPrefillProbe`) with `IFEval++`-style paraphrase stress evaluation. |
| `scripts/check_anonymity.sh` | Double-blind submission linter | Verifies `\author{Anonymous Authors}` and checks `.tex`, `.bib`, and PDF metadata under `paper/` for PII/path leaks. |
| `scripts/run_smoke_test.py` | Phase 3 (W3) 12-item smoke test & Gate 3 verifier | Slices 2 items/slice into `outputs/smoke_12_<model>.jsonl` (plus `/tmp/smoke_12.jsonl` on mock CPU) and writes version-stamped `outputs/smoke_test_gate3_report.json`. |
| `tests/test_cbarr_preflight.py` & `tests/test_phases_4_to_7.py` | Offline CPU verification suite (30 tests total) | Covers 25/25 Gate-1 regex templates, capital-word/numeral disambiguation, CRGC conflict bridges, seeded `langdetect`, D13–D16 invariants, Clean-446, 10k bootstrap, Conformal Risk & Oracle efficiency bounds, Stage-1/Stage-2 KV/DeltaNet continuation parity, native probe $\to$ `B0` fusion and `calib_seed_only`, $K \in \{1, 4, 8, 16, 32\}$ horizon probes, hardware dtype safety (CUDA + MPS BF16), device-safe Qwen3.5 fallback under mocked Triton/`fla`, MPS telemetry, fast kernel gating, `logfmt`, all 5 notebooks AST validation, parallel 4-part bash script workflow, `N=3` intra-model sharding parity vs 1 GPU, 2-part & 3-worker `--no-ledger` concurrent rebase-merge -> combiner `sync_results.py` integration, SemVer bump/`pre-push` hook + per-model `sync_seq` + Phase 7 `version_audit` mismatch detection, bare-remote `EXPERIMENT_LOG.md` push, per-step `run_matrix.py` checkpoints, `ext_deltanet_blend` `conv_states`/KV-eviction, `ext_prefill_mlp` `.npz` caching + `IFEval++` paraphrase stress, and `paper/` anonymity. |
| `paper/main.tex` | Full NeurIPS-style 8-section manuscript & Booktabs LaTeX templates | Integrates Proposition 1 (Conformal Risk Control), $K \in \{1, 4, 8, 16, 32\}$ probe-horizon & `CRGC` bridge details, Figures 1–3, Tables 1–3, and W7 mechanistic subsections (`paper/references.bib` includes verified 2025–2026 citations with anonymized metadata). |

---

## Running on Google Colab or Kaggle GPUs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dolong2110/C-BARR/blob/master/notebooks/C_BARR_All_In_One.ipynb)

Choose either **Parallel 3+1 Split** (`1`, `2`, `3` together -> `4`) or **All-In-One** (`All_In_One` by itself). Every GPU notebook also automatically shards each model across `N_ACTIVE_SHARDS = min(GPU_COUNT, MAX_SHARDS_PER_MODEL)` local GPUs (`MAX_SHARDS_PER_MODEL = 4`) and merges bit-for-bit before scoring:

- **Option A — Parallel 3 GPU Workers + 1 Combiner**:
  1. **Launch `1`, `2`, `3` in parallel at the same time** (each pushes strictly its own `outputs/<model>/*` folder + `run_manifest.json` with `--no-ledger`, so concurrent pushes auto-rebase with zero `EXPERIMENT_LOG.md` conflicts; finishes in ~1.5–2h on `T4 x2` sessions or ~45–60 min on `4× GPU`):
     - **Worker 1/3** — [notebooks/C_BARR_1_Worker_Qwen4B.ipynb](notebooks/C_BARR_1_Worker_Qwen4B.ipynb): Gate 0, Step 0a smoke, Step 0b calibration, `Qwen/Qwen3.5-4B` (`strict341 -> diag491`).
     - **Worker 2/3** — [notebooks/C_BARR_2_Worker_SmolLM3.ipynb](notebooks/C_BARR_2_Worker_SmolLM3.ipynb): `HuggingFaceTB/SmolLM3-3B` (`strict341 -> diag491`).
     - **Worker 3/3** — [notebooks/C_BARR_3_Worker_Scale.ipynb](notebooks/C_BARR_3_Worker_Scale.ipynb): Phase 6 Scale Sweep (`Qwen/Qwen3.5-0.8B` & `Qwen/Qwen3.5-2B`, plus `9B` if `VRAM >= 22 GB`).
  2. **Once Workers 1–3 finish, run Combiner Step 4** — [notebooks/C_BARR_4_Combine_Analyze.ipynb](notebooks/C_BARR_4_Combine_Analyze.ipynb) (runs on **CPU in ~2 min** unless `RUN_W7_EXTENSIONS = True`): Unpacks all worker caches, verifies code versions match across workers, runs combined **Phase 7** (`analyze_and_plot.py` + `check_anonymity.sh`), and pushes `stats_summary.json`, `paper_tables.tex`, anonymized PDFs, `outputs/RESULTS_VERSION`, and unified `EXPERIMENT_LOG.md`.
- **Option B — Single-Session All-In-One**:
  - [notebooks/C_BARR_All_In_One.ipynb](notebooks/C_BARR_All_In_One.ipynb): Combines Parts `1 + 2 + 3 + 4` end-to-end in one Colab Pro (`L4`/`A100`) or multi-GPU machine session.

Upload or import any notebook into **Google Colab** (Runtime -> GPU: `T4`, `L4`, or `A100`) or **Kaggle** (Accelerator: `GPU P100` or `GPU T4 x2`, Internet: **On**). Add `GITHUB_TOKEN` (fine-grained PAT with **Contents: Read and write**) as a Kaggle/Colab Secret so automated cross-session pushes to GitHub work cleanly.

Each 10-cell notebook automatically:
1. **Detects runtime, GPU count & precision (`cbarr-env-detect`)**: Selects `--dtype auto` (`float16` on `T4`/`P100`, `bfloat16` on `L4`/`A100`), computes `N_ACTIVE_SHARDS = min(GPU_COUNT, MAX_SHARDS_PER_MODEL)`, optionally mounts Google Drive on Colab so `OUT_DIR` caches survive disconnects, and routes `HF_HOME` to `/kaggle/temp/hf_cache` (or `/tmp/hf_cache`) so `/kaggle/working` stays under quota.
2. **Loads config & secrets (`cbarr-config`)**: Exposes model lists, `AUTO_PARALLEL_MULTI_GPU = True`, `MAX_SHARDS_PER_MODEL = 4`, `SYNC_RESULTS_TO_GITHUB = True`, `SYNC_INCLUDE_LEDGER`, `DTYPE`, training hyperparameters (`SEED`, `LAM_UTILITY`, `NORM_TOKENS`, `LOGREG_C`, `TAU_MIN`, `TAU_MAX`, `TAU_STEPS`), `CHECKPOINT_EVERY_ITEMS = 150`, `RECORD_EXPERIMENT_LOG = True`, `AUTO_PUSH_RESULTS_TO_GITHUB = True`, plus optional secrets `HF_TOKEN` and `GITHUB_TOKEN`.
3. **Safely syncs repo & installs fast kernels (`cbarr-clone-install`)**: Clones or rebases (`git checkout -- outputs/smoke_test_gate3_report.json` + `git pull --rebase --autostash`), installs missing runtime deps without overwriting container PyTorch, and runs `scripts/setup_fast_kernels.py --mode auto` for Triton/CUDA `fla` + `causal-conv1d` (falling back safely to PyTorch if unavailable or P100 `sm_60`).
4. **Downloads/verifies data & unpacks prior results (`cbarr-download-data`)**: Verifies frozen JSONL splits against `data/manifest_sha256.json`, unpacks any pushed `outputs/*/*.jsonl.gz` into `OUT_DIR` when `UNPACK_SYNCED_RESULTS_ON_START = True`, and warms HF tokenizers.
5. **Executes `N`-GPU sharded stages with live highlighted logs (`cbarr-gate0-smoke` through `cbarr-scale-w7`)**: Shards each model cleanly across `CUDA_VISIBLE_DEVICES=0..N-1` using `run_and_tee()`, merges deterministically with `--merge-shards`, updates `EXPERIMENT_LOG.md` locally or remotely (`SYNC_INCLUDE_LEDGER`), and pushes compact caches via `scripts/sync_results.py` (`--no-ledger` for concurrent workers).
6. **Analyzes, combines & exports bundle (`cbarr-analyze-export`)**: Verifies all expected worker models are present, runs `analyze_and_plot.py` and `check_anonymity.sh`, prints Gate-4 metrics, pushes unified artifacts + `EXPERIMENT_LOG.md` to GitHub, and packages the session's zip bundle.
7. **Tracks & matches versions end-to-end (`VERSION` + `run_manifest.json` + `RESULTS_VERSION`)**: Every code update increments `VERSION` (checked on push by `.githooks/pre-push` via `python scripts/versioning.py bump`); every model sync increments only its own disjoint `outputs/<model>/run_manifest.json` (`sync_seq`, `result_version = v{code_version}+m{sync_seq}`) while combiner pushes increment `outputs/RESULTS_VERSION`, and unpack/Phase-7 `version_audit` (`--strict-version-match`) flags any code-version mismatch across separately trained workers.

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

#### Parallel 4-Part Split (Multi-Machine or Per-GPU Shell Orchestration)
```bash
# Run Parts 1, 2, 3 in parallel on 3 machines/VMs (with --push) OR pinned to GPUs 0/1/2:
bash scripts/run_part1_worker_qwen4b.sh --shards auto --push     # Worker 1: Gate 0 + Calib + 4B
bash scripts/run_part2_worker_smollm3.sh --shards auto --push    # Worker 2: SmolLM3-3B
bash scripts/run_part3_worker_scale.sh --shards auto --push      # Worker 3: 0.8B + 2B (+ --with-9b)

# After Parts 1–3 finish, run Part 4 on CPU (unpack + version audit + Phase 7 + unified ledger):
bash scripts/run_part4_combine_analyze.sh --strict-version-match --push
```
*(When running all 3 workers in parallel inside the **same local clone**, omit `--push` on Parts 1–3 to avoid local `.git/index.lock` races and pass `--push` on Part 4 only; across separate VMs/clones, `--push` on Parts 1–3 pushes disjoint `outputs/<model>/*` folders safely with `--no-ledger`.)*

#### Local Apple Silicon (MPS) Single-Model Run
```bash
# Run a single cached model locally on Apple Silicon (auto-selects mps + native bfloat16 when supported):
bash scripts/run_gpu_phases_4_to_6.sh --device auto --models Qwen/Qwen3.5-0.8B --strict-only --offline
```
> **Hardware guidance**: On 16 GB Apple Silicon Macs (e.g., M5 with ~11.8 GB recommended MPS working set and PyTorch fallback without CUDA `fla`/`causal-conv1d`), `Qwen/Qwen3.5-0.8B` (~1.5 GB) and `2B` (~4 GB) fit comfortably for smoke tests and single-model runs, whereas full multi-model (`4B` + `3B` + `0.8B` + `2B` + `9B`) sweeps are substantially faster on Colab/Kaggle/cloud CUDA GPUs.

Key flags for `scripts/run_gpu_phases_4_to_6.sh` (and forwarded by `scripts/run_part{1..4}_*.sh`):
- `--part <all|1|2|3|4>` / `--shards <N|auto>`: Select full end-to-end run or one of the 4 parallel-workflow parts, and intra-model GPU sharding (`auto` uses up to `min(GPU_COUNT, 4)` local GPUs).
- `--gpu <id>` / `--device <auto|cuda|mps|cpu>` / `--dtype <auto|bf16|fp16|fp32>`: Select CUDA index, backend (`auto` picks CUDA -> MPS -> CPU), and precision (`auto` picks `bfloat16` on Ampere/Ada/Hopper `sm >= 80` and BF16-capable Apple Silicon MPS, or `float16` on `T4`/`P100`).
- `--models "<space-separated list>"` / `--model <id>` / `--data-dir <dir>`: Override the default model list or frozen data directory (e.g., `--models Qwen/Qwen3.5-0.8B`).
- `--calibrate-only` / `--strict-only` / `--offline` / `--strict-version-match`: Run only Phase 4 calibration, run only strict341, force offline HF cache loading, or fail Part 4/Phase 7 if worker code versions mismatch.
- `--push`: On Parts 1–3, sync disjoint `outputs/<model>/*` + `run_manifest.json` (`--no-ledger`); on Part 4/`all`, push `EXPERIMENT_LOG.md`, `outputs/RESULTS_VERSION`, and unified Phase 7 artifacts.
- `--checkpoint-every <N>`: Mid-`eval_450` checkpoint interval in newly evaluated items (default `150` on GPU, `0` on `--mock` and parallel workers).
- `--with-9b`, `--with-w7`, `--include-b512`, `--skip-smoke`, `--skip-preflight`, `--unpack`/`--no-unpack`: Enable 9B scale sweep, W7 probes, Contingency R3 `B=512`, skip smoke/preflight, or control Part 4 cache unpacking.
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
