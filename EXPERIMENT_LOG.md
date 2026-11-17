# C-BARR Experiment & Runtime Log

Tracks every test, mock dry-run, and live GPU experiment execution (wall-clock duration per stage, hardware configuration, hyperparameters, raw log pointers, live progress, and headline metrics).

- Live runs via `bash scripts/run_gpu_phases_4_to_6.sh` or `notebooks/C_BARR_Colab_Kaggle.ipynb` stream output to `outputs/logs/cbarr_<RUN_ID>.log`, save per-stage JSONL telemetry (`outputs/logs/timings_<RUN_ID>.jsonl` & `outputs/experiment_history.jsonl`), and upsert an anonymized run entry in place (`<!-- CBARR_RUN_START/END -->`) at every internal step (Calibration, every N items in `eval_450`, Eval-450 completion, each Section 6 control mode) and outer loop — including failure status via the `EXIT` trap (`finalize_run`) in `scripts/run_gpu_phases_4_to_6.sh` or exception handlers in `notebooks/C_BARR_Colab_Kaggle.ipynb`.
- Keep entries free of local usernames, email addresses, tokens, or private paths outside the repository.

---

## Run History

### Run `colab_gate3_debug_01` — **FAILED at Step 0a (Fixed in `src/engine.py`)** (2026-11-13 20:05 UTC)
- **Git Commit**: `2dd1d1f` (master)
- **Environment**: Google Colab GPU (`HF_HUB` online; PyTorch fallback for `fla`/`causal_conv1d`)
- **Stage 0 (Preflight Tests)**: 11/11 PASSED in `69.053s`
- **Stage 0a (`run_smoke_test.py --live-gpu`)**: `Qwen/Qwen3.5-4B` downloaded & loaded (`~4s`), passed all 6 GSM8K/CSQA smoke items (`anchor_clause == ""`), then triggered Gate 3 `AssertionError` on `ifeval_3041` `m2_b64`: greedy decoding echoed `[Reminder: strictly follow: ...]` repeatedly when injected inline directly inside the visible assistant buffer after `</think>\n`.
- **Resolution Applied**: Updated `src/engine.py` (`_build_reanchor_block`, `_build_post_think_user_bridge`, `_build_reanchor_for_engine`) so both `M2` (`post_stripped`) and Section 6 (`B2a_post_anchor`, `B2b_post_raw`) deliver the post-`</think>` reminder inside a turn-delimited `user` bridge right before the fresh non-thinking assistant generation prefix (`<think>\n\n</think>\n\n`), preventing self-echo loops and keeping strict format buffers uncontaminated.

### Baseline Verification (CPU / Mock E2E Prior to Lab GPU Execution)

- **Git Commit**: `2dd1d1f` (master) + logging/timing instrumentation
- **Hardware**: Local macOS CPU (`python 3.10.21`) — no CUDA GPU attached
- **Frozen Datasets Verified (`data/manifest_sha256.json`)**: `seed_60.jsonl` (60), `dev_60.jsonl` (60), `eval_450.jsonl` (450), `ifeval_diag_491.jsonl` (491), `ifeval_diag_341_strict.jsonl` (341)
- **Full Unit & Mock E2E Suite (`unittest discover -s tests -v`)**: 11/11 tests PASSED in `47.07s` (includes 25/25 Gate-1 templates, Seed-0 langdetect, Gate 3 12-item smoke gate, mini W4–W7 mock matrix + Clean-446 + 10k bootstrap + 3 PDF figures, Colab/Kaggle notebook AST check, and `paper/` anonymity audit).
- **Live CUDA Lab Runs**: *Primary `Qwen/Qwen3.5-4B` matrix completed; full multi-model + W7 + Phase 7 execution pending.*

### Prior Superseded / Failed Bring-Up Attempts (Archived Summary)

| Run ID | Code Ver | Outcome | Wall Time | Notes / Resolution |
| :--- | :--- | :--- | :--- | :--- |
| `20261114T214425Z` | `v0.5.0` | ORPHANED (Colab timeout at Step 0a) | 14m 41s | Gate 0 PASS (720.7s) on 2x T4; superseded by combiner updates |
| `20261114T223748Z` | `v0.5.0` | FAILED at Phase 7 (Colab) | 12m 47s | Missing worker caches during early combiner test |
| `20261114T231920Z` | `v0.5.2` | FAILED at Phase 7 (Colab) | 13m 04s | Same early combiner test before worker caches were pushed |
| `20261116T155617Z` | `v0.5.5` | FAILED at Step 0a Smoke (8x L40) | 14m 45s | Initial cluster bring-up (`causal_conv1d=False`) |
| `20261116T164629Z` | `v0.5.6` | FAILED at Step 0a Smoke (8x L40) | 20m 41s | Fixed in `v0.5.7` (sharded monitor / kernel flags) |
| `20261116T173652Z` | `v0.5.7` | FAILED at `4B (strict341)` merge | 4h 53m 18s | Pre-`v0.5.8` GSM8K prose-tail false negatives & merge fix |
| `20261117T012331Z` | `v0.5.10` | FAILED at Step 0a Smoke (8x L40) | 37m 45s | Shared-GPU OOM + `causal-conv1d` CUDA version check relaxed in `v0.5.11` |
| `20261117T112446Z` | `v0.5.11` | FAILED at Step 0a Smoke (8x L40) | 10m 44s | Built `causal-conv1d 1.7.0` (~10m); GPU 0 held 41.9 GB externally (`gpu_pool` added in `v0.5.12`) |

<!-- CBARR_RUN_START:20261117T114416Z -->
### Run `20261117T114416Z` — **FAILED (exit=1 at W7: DeltaNet Blend (Qwen/Qwen3.5-4B))** (Updated 2026-11-17 13:25:56 UTC)
- **Code Version**: `v0.5.11` | **Git Commit**: `37fd866-dirty` | **Last Loop**: `W7: DeltaNet Blend (Qwen/Qwen3.5-4B)` | **Wall Time**: `1h 41m 26s` (6086.0s)
- **Hardware**: 4x NVIDIA L40 (44.4 GB, SM 8.9, CUDA 13.2, BF16=True) | PyTorch `2.15.0+cu132` | Transformers `5.23.0` | Kernels: `fla=True, causal_conv1d=True`
- **Command**: `scripts/run_gpu_phases_4_to_6.sh --part all --setup-mode auto --shards auto --auto-9b --with-w7 --commit --push --gpu 3,4,6,7 --hf-home <HOME>/maplecg_nfs/hf_cache`
- **Raw Log**: `outputs/logs/cbarr_20261117T114416Z.log`
- **Notes**: External process allocated 35.85 GiB mid-W7 (`torch.OutOfMemoryError`); self-healing retry added in `v0.5.13`.

#### 1. Training & Experiment Parameters
```json
{
  "auto_commit": true,
  "auto_push": true,
  "b0_max_new_tokens": 1024,
  "b1b_max_new_tokens": 2048,
  "class_weight": "balanced",
  "code_version": "0.5.11",
  "device": "cuda",
  "do_sample": false,
  "dtype": "auto",
  "gate2_probe_tokens": 32,
  "git_branch": "",
  "git_rev": "ceccf7d-dirty",
  "include_b512": false,
  "is_mock": false,
  "lam_utility": 0.05,
  "logreg_C": 1.0,
  "norm_tokens": 250.0,
  "presence_penalty": 0.0,
  "run_id": "20261117T114416Z",
  "seed": 42,
  "setup_mode": "auto",
  "stage2_bridge": "classic",
  "tau_grid": [
    0.15,
    0.85,
    29
  ],
  "temperature": 0.0,
  "tiered_budgets": [
    64,
    128,
    250
  ],
  "top_p": 1.0,
  "with_9b": true,
  "with_w7": true
}
```

#### 2. Stage Timings
| Stage | Status | Wall Time |
| :--- | :--- | :--- |
| Gate 0: Preflight Tests | PASS | 7.0s (7.0s) |
| Step 0a: Hardware & Kernel Probe | PASS | 15.0s (15.0s) |
| Step 0a: Live GPU 12-Item Smoke | PASS | 1m 20.0s (80.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (strict341) (cached) | PASS | 42.0s (42.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (diag491) | PASS | 56m 56.0s (3416.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (diag491) (merge) | PASS | 59.0s (59.0s) |
| W7: DeltaNet Blend (Qwen/Qwen3.5-4B) | FAIL(exit=1) | 38m 51.0s (2331.0s) |

#### 3. Gate-3 Live 12-Item Smoke Check
| Model | Items | D13 <=250 | No Reminder Leak | SmolLM3 Frozen Date |
| :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-4B` | 12 | True | True | N/A |
| `HuggingFaceTB/SmolLM3-3B` | 12 | True | True | True |

#### 4. Calibration & Gate-2 Classifier Results (Per Model)
| Model | Code Ver | tau_prob | tau_h_high | Dev Esc (R1) | G1 Skip | Intercept | Coef (`H1, Hm, Hmax, I, sym`) | Train@0.5 (Acc/Prec/Rec/F1/AUC) | Val-All@tau (Acc/Prec/Rec/F1/AUC) | Val-G2@tau (F1/AUC) | Dev `B1b->M2` Macro (Tok) | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-4B` | `v0.5.11` (`ceccf7d-dirty`) | 0.650 | 2.537 | 20.0% (OK) | 16.7% | -0.64 | [-0.68, 0.60, 0.10, 0.96, -0.43] | 72/19/100/32/0.804 | 87/0/0/0/0.629 | 0/0.410 (n=40) | 38.0%->90.0% (1744->365t) | 0.1s |

#### 5. Evaluation Matrix & Section 6 Diagnostics (Per Model x Pool)

- **`Qwen/Qwen3.5-4B` (`strict341`**, n_diag=341) — ver=`v0.5.11` (`4d03fad-dirty`) | dtype=`torch.bfloat16` | wall=`25.7s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.650, esc=20.0%
  - **4-Regime**: Helped=0.7%, Hurt=50.0%, Flip=50.7%, TiePass=34.7%, TieFail=14.7%
  - **Section 6 Causal (`|E_hurt|=237`)**: rho_rec=96.2%, rho_int=3.8% | A1_strip=100.0%, B2a_post=96.2%, B1_pre=95.8%, B2b_raw=46.4%, A2_swap=73.0%, B3_cut=87.3%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 84.1% | 92.0% | 90.7% | 82.0% | 80.0% | 83.3% | 76.7% | 84.7% | 84.7% | 78.4% | 312 | 1.00 | 7.33e-62 |
| B1a | 80.4% | 93.3% | 92.0% | 84.0% | 66.7% | 76.7% | 70.0% | 36.7% | 71.7% | 74.8% | 395 | 1.00 | 2.08e-59 |
| **B1b** | 30.2% | 56.0% | 44.0% | 41.3% | 11.7% | 21.7% | 6.7% | 0.2% | 18.1% | 0.6% | 1735 | 1.00 | - |
| B2a | 82.4% | 97.3% | 94.7% | 80.7% | 80.0% | 75.0% | 66.7% | 51.6% | 75.3% | 77.8% | 310 | 1.00 | 4.45e-62 |
| B2b | 72.5% | 85.3% | 84.0% | 70.7% | 71.7% | 70.0% | 53.3% | 66.2% | 70.4% | 61.4% | 639 | 1.23 | 1.26e-49 |
| M1a | 51.9% | 66.7% | 49.3% | 82.0% | 16.7% | 76.7% | 20.0% | 43.8% | 52.2% | 40.9% | 979 | 1.00 | 1.93e-34 |
| M1b | 77.8% | 92.0% | 90.7% | 79.3% | 66.7% | 78.3% | 60.0% | 77.1% | 78.7% | 71.6% | 429 | 1.08 | 2.92e-56 |
| M1c | 62.1% | 92.0% | 88.0% | 79.3% | 15.0% | 78.3% | 20.0% | 65.6% | 67.8% | 56.6% | 680 | 1.04 | 2.47e-43 |
| M2_B64 | 84.3% | 92.0% | 90.7% | 81.3% | 80.0% | 81.7% | 80.0% | 84.4% | 83.8% | 77.8% | 335 | 1.04 | 1.27e-61 |
| M2_B128 | 84.6% | 92.0% | 90.7% | 81.3% | 83.3% | 83.3% | 76.7% | 84.9% | 83.6% | 77.8% | 355 | 1.04 | 5.1e-62 |
| M2_B256 | 84.6% | 92.0% | 90.7% | 81.3% | 85.0% | 81.7% | 76.7% | 84.9% | 82.5% | 77.3% | 380 | 1.04 | 5.1e-62 |
| M2_Pre | 82.9% | 92.0% | 89.3% | 81.3% | 83.3% | 81.7% | 70.0% | 84.0% | 82.6% | 76.7% | 364 | 1.04 | 6.1e-60 |
| **M2** | 84.6% | 92.0% | 90.7% | 81.3% | 83.3% | 83.3% | 76.7% | 84.9% | 83.4% | 77.7% | 360 | 1.04 | 5.1e-62 |
| M2_Prec (Aux.) | 85.4% | 92.0% | 90.7% | 81.3% | 83.3% | 91.7% | 73.3% | 85.8% | 83.6% | 78.1% | 384 | 1.04 | 3.05e-64 (raw) |
| **Oracle_UB** | 92.6% | 96.0% | 98.7% | 90.7% | 91.7% | 91.7% | 86.7% | 92.9% | 91.3% | 87.0% | 292 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`diag491`**, n_diag=491) — ver=`v0.5.11` (`ceccf7d-dirty`) | dtype=`torch.bfloat16` | wall=`32.2s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.650, esc=20.0%
  - **4-Regime**: Helped=0.7%, Hurt=50.0%, Flip=50.7%, TiePass=34.7%, TieFail=14.7%
  - **Section 6 Causal (`|E_hurt|=336`)**: rho_rec=96.1%, rho_int=3.9% | A1_strip=100.0%, B2a_post=96.1%, B1_pre=95.5%, B2b_raw=49.4%, A2_swap=70.2%, B3_cut=86.3%
  - *Eval-450 table identical to `strict341` above (omitted duplicate table).*
<!-- CBARR_RUN_END:20261117T114416Z -->

<!-- CBARR_RUN_START:20261117T142919Z -->
### Run `20261117T142919Z` — **FAILED (exit=127 at Phase 4 / W4: Qwen/Qwen3.5-4B (diag491) (cached))** (Updated 2026-11-17 14:32:48 UTC)
- **Code Version**: `v0.5.12` | **Git Commit**: `4be1be4-dirty` | **Last Loop**: `Phase 4 / W4: Qwen/Qwen3.5-4B (diag491) (cached)` | **Wall Time**: `3m 22.0s` (202.0s)
- **Hardware**: 8x NVIDIA L40 (44.4 GB, SM 8.9, CUDA 13.2, BF16=True) | PyTorch `2.15.0+cu132` | Transformers `5.23.0` | Kernels: `fla=True, causal_conv1d=True`
- **Command**: `scripts/run_gpu_phases_4_to_6.sh --part all --setup-mode auto --shards auto --auto-9b --with-w7 --commit --push --hf-home <HOME>/maplecg_nfs/hf_cache`
- **Raw Log**: `outputs/logs/cbarr_20261117T142919Z.log`
- **Notes**: `env -u CUDA_VISIBLE_DEVICES run_stage "W7: ..."` exited 127 (`run_stage` bash function invoked via `env`); fixed in `v0.5.13`.

#### 1. Training & Experiment Parameters
```json
{
  "auto_commit": true,
  "auto_push": true,
  "b0_max_new_tokens": 1024,
  "b1b_max_new_tokens": 2048,
  "class_weight": "balanced",
  "code_version": "0.5.12",
  "device": "cuda",
  "do_sample": false,
  "dtype": "auto",
  "gate2_probe_tokens": 32,
  "git_branch": "",
  "git_rev": "13c1e5a-dirty",
  "include_b512": false,
  "is_mock": false,
  "lam_utility": 0.05,
  "logreg_C": 1.0,
  "norm_tokens": 250.0,
  "presence_penalty": 0.0,
  "run_id": "20261117T142919Z",
  "seed": 42,
  "setup_mode": "auto",
  "stage2_bridge": "classic",
  "tau_grid": [
    0.15,
    0.85,
    29
  ],
  "temperature": 0.0,
  "tiered_budgets": [
    64,
    128,
    250
  ],
  "top_p": 1.0,
  "with_9b": true,
  "with_w7": true
}
```

#### 2. Stage Timings
| Stage | Status | Wall Time |
| :--- | :--- | :--- |
| Gate 0: Preflight Tests | PASS | 6.0s (6.0s) |
| Step 0a: Hardware & Kernel Probe | PASS | 8.0s (8.0s) |
| Step 0a: Live GPU 12-Item Smoke | PASS | 6.0s (6.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (strict341) (cached) | PASS | 42.0s (42.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (diag491) (cached) | PASS | 44.0s (44.0s) |

#### 3. Gate-3 Live 12-Item Smoke Check
| Model | Items | D13 <=250 | No Reminder Leak | SmolLM3 Frozen Date |
| :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-4B` | 12 | True | True | N/A |
| `HuggingFaceTB/SmolLM3-3B` | 12 | True | True | True |

#### 4. Calibration & Gate-2 Classifier Results (Per Model)
| Model | Code Ver | tau_prob | tau_h_high | Dev Esc (R1) | G1 Skip | Intercept | Coef (`H1, Hm, Hmax, I, sym`) | Train@0.5 (Acc/Prec/Rec/F1/AUC) | Val-All@tau (Acc/Prec/Rec/F1/AUC) | Val-G2@tau (F1/AUC) | Dev `B1b->M2` Macro (Tok) | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-4B` | `v0.5.12` (`13c1e5a-dirty`) | 0.650 | 2.537 | 20.0% (OK) | 16.7% | -0.64 | [-0.68, 0.60, 0.10, 0.96, -0.43] | 72/19/100/32/0.804 | 87/0/0/0/0.629 | 0/0.410 (n=40) | 38.0%->90.0% (1744->365t) | 0.1s |

#### 5. Evaluation Matrix & Section 6 Diagnostics (Per Model x Pool)

- **`Qwen/Qwen3.5-4B` (`strict341`**, n_diag=341) — ver=`v0.5.12` (`c9597bd-dirty`) | dtype=`torch.bfloat16` | wall=`26.0s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.650, esc=20.0%
  - **4-Regime**: Helped=0.7%, Hurt=50.0%, Flip=50.7%, TiePass=34.7%, TieFail=14.7%
  - **Section 6 Causal (`|E_hurt|=237`)**: rho_rec=96.2%, rho_int=3.8% | A1_strip=100.0%, B2a_post=96.2%, B1_pre=95.8%, B2b_raw=46.4%, A2_swap=73.0%, B3_cut=87.3%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 84.1% | 92.0% | 90.7% | 82.0% | 80.0% | 83.3% | 76.7% | 84.7% | 84.7% | 78.4% | 312 | 1.00 | 7.33e-62 |
| B1a | 80.4% | 93.3% | 92.0% | 84.0% | 66.7% | 76.7% | 70.0% | 36.7% | 71.7% | 74.8% | 395 | 1.00 | 2.08e-59 |
| **B1b** | 30.2% | 56.0% | 44.0% | 41.3% | 11.7% | 21.7% | 6.7% | 0.2% | 18.1% | 0.6% | 1735 | 1.00 | - |
| B2a | 82.4% | 97.3% | 94.7% | 80.7% | 80.0% | 75.0% | 66.7% | 51.6% | 75.3% | 77.8% | 310 | 1.00 | 4.45e-62 |
| B2b | 72.5% | 85.3% | 84.0% | 70.7% | 71.7% | 70.0% | 53.3% | 66.2% | 70.4% | 61.4% | 639 | 1.23 | 1.26e-49 |
| M1a | 51.9% | 66.7% | 49.3% | 82.0% | 16.7% | 76.7% | 20.0% | 43.8% | 52.2% | 40.9% | 979 | 1.00 | 1.93e-34 |
| M1b | 77.8% | 92.0% | 90.7% | 79.3% | 66.7% | 78.3% | 60.0% | 77.1% | 78.7% | 71.6% | 429 | 1.08 | 2.92e-56 |
| M1c | 62.1% | 92.0% | 88.0% | 79.3% | 15.0% | 78.3% | 20.0% | 65.6% | 67.8% | 56.6% | 680 | 1.04 | 2.47e-43 |
| M2_B64 | 84.3% | 92.0% | 90.7% | 81.3% | 80.0% | 81.7% | 80.0% | 84.4% | 83.8% | 77.8% | 335 | 1.04 | 1.27e-61 |
| M2_B128 | 84.6% | 92.0% | 90.7% | 81.3% | 83.3% | 83.3% | 76.7% | 84.9% | 83.6% | 77.8% | 355 | 1.04 | 5.1e-62 |
| M2_B256 | 84.6% | 92.0% | 90.7% | 81.3% | 85.0% | 81.7% | 76.7% | 84.9% | 82.5% | 77.3% | 380 | 1.04 | 5.1e-62 |
| M2_Pre | 82.9% | 92.0% | 89.3% | 81.3% | 83.3% | 81.7% | 70.0% | 84.0% | 82.6% | 76.7% | 364 | 1.04 | 6.1e-60 |
| **M2** | 84.6% | 92.0% | 90.7% | 81.3% | 83.3% | 83.3% | 76.7% | 84.9% | 83.4% | 77.7% | 360 | 1.04 | 5.1e-62 |
| M2_Prec (Aux.) | 85.4% | 92.0% | 90.7% | 81.3% | 83.3% | 91.7% | 73.3% | 85.8% | 83.6% | 78.1% | 384 | 1.04 | 3.05e-64 (raw) |
| **Oracle_UB** | 92.6% | 96.0% | 98.7% | 90.7% | 91.7% | 91.7% | 86.7% | 92.9% | 91.3% | 87.0% | 292 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`diag491`**, n_diag=491) — ver=`v0.5.12` (`13c1e5a-dirty`) | dtype=`torch.bfloat16` | wall=`26.7s` (calib=0.1s, eval=0.4s, diag=0.1s) | tau=0.650, esc=20.0%
  - **4-Regime**: Helped=0.7%, Hurt=50.0%, Flip=50.7%, TiePass=34.7%, TieFail=14.7%
  - **Section 6 Causal (`|E_hurt|=336`)**: rho_rec=96.1%, rho_int=3.9% | A1_strip=100.0%, B2a_post=96.1%, B1_pre=95.5%, B2b_raw=49.4%, A2_swap=70.2%, B3_cut=86.3%
  - *Eval-450 table identical to `strict341` above (omitted duplicate table).*
<!-- CBARR_RUN_END:20261117T142919Z -->

<!-- CBARR_RUN_START:20261117T174419Z -->
### Run `20261117T174419Z` — **RUNNING (after W7: DeltaNet Blend (Qwen/Qwen3.5-4B))** (Updated 2026-11-17 18:32:15 UTC)
- **Code Version**: `v0.5.14` | **Git Commit**: `cdfd4a9` | **Last Loop**: `W7: DeltaNet Blend (Qwen/Qwen3.5-4B)` | **Wall Time**: `47m 41.0s` (2861.0s)
- **Hardware**: 4x NVIDIA L40 (44.4 GB, SM 8.9, CUDA 13.2, BF16=True) | PyTorch `2.15.0+cu132` | Transformers `5.23.0` | Kernels: `fla=True, causal_conv1d=True`
- **Command**: `scripts/run_gpu_phases_4_to_6.sh --part all --setup-mode auto --shards auto --auto-9b --with-w7 --commit --push --hf-home <HOME>/maplecg_nfs/hf_cache --gpu 3,4,6,7`
- **Raw Log**: `outputs/logs/cbarr_20261117T174419Z.log`

#### 1. Training & Experiment Parameters
```json
{
  "auto_commit": true,
  "auto_push": true,
  "b0_max_new_tokens": 1024,
  "b1b_max_new_tokens": 2048,
  "class_weight": "balanced",
  "code_version": "0.5.14",
  "device": "cuda",
  "do_sample": false,
  "dtype": "auto",
  "gate2_probe_tokens": 32,
  "git_branch": "",
  "git_rev": "cdfd4a9",
  "include_b512": false,
  "is_mock": false,
  "lam_utility": 0.05,
  "logreg_C": 1.0,
  "norm_tokens": 250.0,
  "presence_penalty": 0.0,
  "run_id": "20261117T174419Z",
  "seed": 42,
  "setup_mode": "auto",
  "stage2_bridge": "classic",
  "tau_grid": [
    0.15,
    0.85,
    29
  ],
  "temperature": 0.0,
  "tiered_budgets": [
    64,
    128,
    250
  ],
  "top_p": 1.0,
  "with_9b": true,
  "with_w7": true
}
```

#### 2. Stage Timings
| Stage | Status | Wall Time |
| :--- | :--- | :--- |
| Gate 0: Preflight Tests | PASS | 6.0s (6.0s) |
| Step 0a: Hardware & Kernel Probe | PASS | 8.0s (8.0s) |
| Step 0a: Live GPU 12-Item Smoke (cached) | PASS | 5.0s (5.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (strict341) (cached) | PASS | 35.0s (35.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (diag491) (cached) | PASS | 35.0s (35.0s) |
| W7: DeltaNet Blend (Qwen/Qwen3.5-4B) | PASS | 44m 36.0s (2676.0s) |

#### 3. Gate-3 Live 12-Item Smoke Check
| Model | Items | D13 <=250 | No Reminder Leak | SmolLM3 Frozen Date |
| :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-4B` | 12 | True | True | N/A |
| `HuggingFaceTB/SmolLM3-3B` | 12 | True | True | True |

#### 4. Calibration & Gate-2 Classifier Results (Per Model)
| Model | Code Ver | tau_prob | tau_h_high | Dev Esc (R1) | G1 Skip | Intercept | Coef (`H1, Hm, Hmax, I, sym`) | Train@0.5 (Acc/Prec/Rec/F1/AUC) | Val-All@tau (Acc/Prec/Rec/F1/AUC) | Val-G2@tau (F1/AUC) | Dev `B1b->M2` Macro (Tok) | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-4B` | `v0.5.14` (`cdfd4a9-dirty`) | 0.650 | 2.537 | 20.0% (OK) | 16.7% | -0.64 | [-0.68, 0.60, 0.10, 0.96, -0.43] | 72/19/100/32/0.804 | 87/0/0/0/0.629 | 0/0.410 (n=40) | 38.0%->90.0% (1744->365t) | 0.1s |

#### 5. Evaluation Matrix & Section 6 Diagnostics (Per Model x Pool)

- **`Qwen/Qwen3.5-4B` (`strict341`**, n_diag=341) — ver=`v0.5.14` (`cdfd4a9-dirty`) | dtype=`torch.bfloat16` | wall=`20.9s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.650, esc=20.0%
  - **4-Regime**: Helped=0.7%, Hurt=50.0%, Flip=50.7%, TiePass=34.7%, TieFail=14.7%
  - **Section 6 Causal (`|E_hurt|=237`)**: rho_rec=96.2%, rho_int=3.8% | A1_strip=100.0%, B2a_post=96.2%, B1_pre=95.8%, B2b_raw=46.4%, A2_swap=73.0%, B3_cut=87.3%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 84.1% | 92.0% | 90.7% | 82.0% | 80.0% | 83.3% | 76.7% | 84.7% | 84.7% | 78.4% | 312 | 1.00 | 7.33e-62 |
| B1a | 80.4% | 93.3% | 92.0% | 84.0% | 66.7% | 76.7% | 70.0% | 36.7% | 71.7% | 74.8% | 395 | 1.00 | 2.08e-59 |
| **B1b** | 30.2% | 56.0% | 44.0% | 41.3% | 11.7% | 21.7% | 6.7% | 0.2% | 18.1% | 0.6% | 1735 | 1.00 | - |
| B2a | 82.4% | 97.3% | 94.7% | 80.7% | 80.0% | 75.0% | 66.7% | 51.6% | 75.3% | 77.8% | 310 | 1.00 | 4.45e-62 |
| B2b | 72.5% | 85.3% | 84.0% | 70.7% | 71.7% | 70.0% | 53.3% | 66.2% | 70.4% | 61.4% | 639 | 1.23 | 1.26e-49 |
| M1a | 51.9% | 66.7% | 49.3% | 82.0% | 16.7% | 76.7% | 20.0% | 43.8% | 52.2% | 40.9% | 979 | 1.00 | 1.93e-34 |
| M1b | 77.8% | 92.0% | 90.7% | 79.3% | 66.7% | 78.3% | 60.0% | 77.1% | 78.7% | 71.6% | 429 | 1.08 | 2.92e-56 |
| M1c | 62.1% | 92.0% | 88.0% | 79.3% | 15.0% | 78.3% | 20.0% | 65.6% | 67.8% | 56.6% | 680 | 1.04 | 2.47e-43 |
| M2_B64 | 84.3% | 92.0% | 90.7% | 81.3% | 80.0% | 81.7% | 80.0% | 84.4% | 83.8% | 77.8% | 335 | 1.04 | 1.27e-61 |
| M2_B128 | 84.6% | 92.0% | 90.7% | 81.3% | 83.3% | 83.3% | 76.7% | 84.9% | 83.6% | 77.8% | 355 | 1.04 | 5.1e-62 |
| M2_B256 | 84.6% | 92.0% | 90.7% | 81.3% | 85.0% | 81.7% | 76.7% | 84.9% | 82.5% | 77.3% | 380 | 1.04 | 5.1e-62 |
| M2_Pre | 82.9% | 92.0% | 89.3% | 81.3% | 83.3% | 81.7% | 70.0% | 84.0% | 82.6% | 76.7% | 364 | 1.04 | 6.1e-60 |
| **M2** | 84.6% | 92.0% | 90.7% | 81.3% | 83.3% | 83.3% | 76.7% | 84.9% | 83.4% | 77.7% | 360 | 1.04 | 5.1e-62 |
| M2_Prec (Aux.) | 85.4% | 92.0% | 90.7% | 81.3% | 83.3% | 91.7% | 73.3% | 85.8% | 83.6% | 78.1% | 384 | 1.04 | 3.05e-64 (raw) |
| **Oracle_UB** | 92.6% | 96.0% | 98.7% | 90.7% | 91.7% | 91.7% | 86.7% | 92.9% | 91.3% | 87.0% | 292 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`diag491`**, n_diag=491) — ver=`v0.5.14` (`cdfd4a9-dirty`) | dtype=`torch.bfloat16` | wall=`21.6s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.650, esc=20.0%
  - **4-Regime**: Helped=0.7%, Hurt=50.0%, Flip=50.7%, TiePass=34.7%, TieFail=14.7%
  - **Section 6 Causal (`|E_hurt|=336`)**: rho_rec=96.1%, rho_int=3.9% | A1_strip=100.0%, B2a_post=96.1%, B1_pre=95.5%, B2b_raw=49.4%, A2_swap=70.2%, B3_cut=86.3%
  - *Eval-450 table identical to `strict341` above (omitted duplicate table).*

#### 6. Optional W7 Extensions
- **`Qwen__Qwen3.5-4B`**: DeltaNetBlend(n=336: a=0.00:93.2%, a=0.25:95.2%, a=0.50:95.2%, a=0.75:94.0%, a=1.00:91.7%)
<!-- CBARR_RUN_END:20261117T174419Z -->
