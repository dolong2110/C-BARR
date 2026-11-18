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
### Run `20261117T174419Z` — **RUNNING (Section 6 & Matrix Complete: Qwen/Qwen3.5-0.8B (strict341))** (Updated 2026-11-18 01:40:01 UTC)
- **Code Version**: `v0.5.14` | **Git Commit**: `b5935f4` | **Results Release**: `v0.5.14-res.2` | **Last Loop**: `Section 6 & Matrix Complete: Qwen/Qwen3.5-0.8B (strict341)` | **Wall Time**: `28.4s` (28.4s)
- **Hardware**: 4x NVIDIA L40 (44.4 GB, SM 8.9, CUDA 13.2, BF16=True) | PyTorch `2.15.0+cu132` | Transformers `5.23.0` | Kernels: `fla=True, causal_conv1d=True`
- **Command**: `scripts/run_matrix.py`
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
  "git_rev": "b5935f4",
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
| W7: Prefill MLP Probe (Qwen/Qwen3.5-4B) | PASS | 1m 23.0s (83.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (strict341) | PASS | 2h 28m 54s (8934.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (strict341) (merge) | PASS | 54.0s (54.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (diag491) | PASS | 16m 46.0s (1006.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (diag491) (merge) | PASS | 49.0s (49.0s) |
| Phase 6 / W6: Qwen3.5-0.8B (strict341) | PASS | 4h 17m 00s (15420.0s) |

#### 4. Calibration & Gate-2 Classifier Results (Per Model)
| Model | Code Ver | tau_prob | tau_h_high | Dev Esc (R1) | G1 Skip | Intercept | Coef (`H1, Hm, Hmax, I, sym`) | Train@0.5 (Acc/Prec/Rec/F1/AUC) | Val-All@tau (Acc/Prec/Rec/F1/AUC) | Val-G2@tau (F1/AUC) | Dev `B1b->M2` Macro (Tok) | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HuggingFaceTB/SmolLM3-3B` | `v0.5.14` (`3214526-dirty`) | 0.400 | 3.144 | 46.7% (OK) | 16.7% | -0.37 | [0.26, 0.53, -0.09, -0.43, -1.45] | 75/28/71/40/0.806 | 48/16/100/28/0.698 | 36/0.806 (n=40) | 76.0%->74.0% (960->407t) | 0.1s |
| `Qwen/Qwen3.5-0.8B` | `v0.5.14` (`003493a`) | 0.525 | 3.719 | 31.7% (OK) | 16.7% | -0.79 | [-0.06, 0.14, 0.15, -0.21, 0.70] | 57/26/73/38/0.651 | 70/19/38/25/0.618 | 40/0.750 (n=40) | 4.0%->61.0% (2000->467t) | 0.1s |
| `Qwen/Qwen3.5-4B` | `v0.5.14` (`cdfd4a9-dirty`) | 0.650 | 2.537 | 20.0% (OK) | 16.7% | -0.64 | [-0.68, 0.60, 0.10, 0.96, -0.43] | 72/19/100/32/0.804 | 87/0/0/0/0.629 | 0/0.410 (n=40) | 38.0%->90.0% (1744->365t) | 0.1s |

#### 5. Evaluation Matrix & Section 6 Diagnostics (Per Model x Pool)

- **`HuggingFaceTB/SmolLM3-3B` (`strict341`**, n_diag=341) — ver=`v0.5.14` (`421cf4a`) | dtype=`torch.bfloat16` | wall=`29.9s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.400, esc=46.7%
  - **4-Regime**: Helped=13.6%, Hurt=9.6%, Flip=23.1%, TiePass=60.0%, TieFail=16.9%
  - **Section 6 Causal (`|E_hurt|=58`)**: rho_rec=63.8%, rho_int=36.2% | A1_strip=100.0%, B2a_post=63.8%, B1_pre=62.1%, B2b_raw=34.5%, A2_swap=55.2%, B3_cut=60.3%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 67.2% | 78.7% | 72.0% | 67.3% | 65.0% | 80.0% | 40.0% | 69.6% | 69.6% | 64.6% | 249 | 1.00 | 0.855 |
| B1a | 66.6% | 85.3% | 77.3% | 66.7% | 35.0% | 75.0% | 60.0% | 62.2% | 65.3% | 62.5% | 274 | 1.00 | 0.376 |
| **B1b** | 69.4% | 90.7% | 89.3% | 70.0% | 65.0% | 71.7% | 30.0% | 3.6% | 52.3% | 54.0% | 977 | 1.00 | - |
| B2a | 65.6% | 86.7% | 76.0% | 72.7% | 46.7% | 71.7% | 40.0% | 64.2% | 66.3% | 64.5% | 266 | 1.00 | 1 |
| B2b | 67.9% | 86.7% | 76.0% | 70.0% | 63.3% | 78.3% | 33.3% | 38.9% | 61.3% | 59.5% | 601 | 1.46 | 1 |
| M1a | 69.5% | 86.7% | 88.0% | 67.3% | 65.0% | 80.0% | 30.0% | 37.6% | 61.7% | 61.0% | 596 | 1.00 | 1 |
| M1b | 66.1% | 78.7% | 72.0% | 70.7% | 65.0% | 76.7% | 33.3% | 32.0% | 58.8% | 55.5% | 712 | 1.59 | 0.0916 |
| M1c | 66.6% | 80.0% | 72.0% | 70.7% | 65.0% | 81.7% | 30.0% | 38.7% | 61.4% | 57.9% | 627 | 1.31 | 0.541 |
| M2_B64 | 67.7% | 80.0% | 72.0% | 74.0% | 68.3% | 81.7% | 30.0% | 72.0% | 70.6% | 65.1% | 345 | 1.31 | 1 |
| M2_B128 | 70.0% | 80.0% | 72.0% | 74.7% | 65.0% | 81.7% | 46.7% | 72.9% | 70.3% | 65.7% | 357 | 1.31 | 1 |
| M2_B256 | 69.8% | 80.0% | 72.0% | 75.3% | 71.7% | 80.0% | 40.0% | 73.3% | 69.0% | 65.5% | 392 | 1.31 | 1 |
| M2_Pre | 69.6% | 80.0% | 72.0% | 74.0% | 63.3% | 81.7% | 46.7% | 72.4% | 70.3% | 65.3% | 357 | 1.31 | 1 |
| **M2** | 68.5% | 80.0% | 72.0% | 74.0% | 63.3% | 81.7% | 40.0% | 72.0% | 69.9% | 64.6% | 371 | 1.31 | 1 |
| M2_Prec (Aux.) | 69.9% | 80.0% | 72.0% | 74.0% | 63.3% | 83.3% | 46.7% | 72.7% | 70.0% | 64.7% | 400 | 1.31 | 0.749 (raw) |
| **Oracle_UB** | 86.8% | 97.3% | 96.0% | 89.3% | 83.3% | 95.0% | 60.0% | 88.4% | 88.0% | 84.1% | 283 | 1.00 | - |

- **`HuggingFaceTB/SmolLM3-3B` (`diag491`**, n_diag=491) — ver=`v0.5.14` (`3214526-dirty`) | dtype=`torch.bfloat16` | wall=`27.7s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.400, esc=46.7%
  - **4-Regime**: Helped=13.6%, Hurt=9.6%, Flip=23.1%, TiePass=60.0%, TieFail=16.9%
  - **Section 6 Causal (`|E_hurt|=80`)**: rho_rec=62.5%, rho_int=37.5% | A1_strip=100.0%, B2a_post=62.5%, B1_pre=58.8%, B2b_raw=33.8%, A2_swap=60.0%, B3_cut=62.5%
  - *Eval-450 table identical to `strict341` above (omitted duplicate table).*

- **`Qwen/Qwen3.5-0.8B` (`strict341`**, n_diag=341) — ver=`v0.5.14` (`003493a`) | dtype=`torch.bfloat16` | wall=`28.4s` (calib=0.1s, eval=0.4s, diag=0.1s) | tau=0.525, esc=31.7%
  - **4-Regime**: Helped=0.4%, Hurt=52.7%, Flip=53.1%, TiePass=3.6%, TieFail=43.3%
  - **Section 6 Causal (`|E_hurt|=176`)**: rho_rec=76.1%, rho_int=23.9% | A1_strip=100.0%, B2a_post=76.1%, B1_pre=75.6%, B2b_raw=47.2%, A2_swap=40.9%, B3_cut=66.5%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 52.8% | 66.7% | 48.0% | 60.7% | 55.0% | 56.7% | 30.0% | 56.2% | 56.2% | 49.3% | 346 | 1.00 | 7.79e-67 |
| B1a | 50.9% | 64.0% | 38.7% | 59.3% | 55.0% | 55.0% | 33.3% | 30.2% | 48.1% | 42.1% | 584 | 1.00 | 3.37e-65 |
| **B1b** | 2.7% | 4.0% | 2.7% | 8.0% | 0.0% | 1.7% | 0.0% | 0.0% | 2.8% | -35.9% | 1997 | 1.00 | - |
| B2a | 38.4% | 40.0% | 20.0% | 57.3% | 45.0% | 45.0% | 23.3% | 37.3% | 40.0% | 36.7% | 297 | 1.00 | 5.07e-48 |
| B2b | 36.9% | 49.3% | 30.7% | 43.3% | 45.0% | 33.3% | 20.0% | 38.4% | 39.2% | 21.1% | 922 | 1.34 | 1.67e-46 |
| M1a | 23.7% | 22.7% | 5.3% | 60.7% | 0.0% | 50.0% | 3.3% | 30.9% | 31.4% | 9.4% | 1118 | 1.00 | 1.5e-36 |
| M1b | 37.3% | 49.3% | 30.7% | 60.7% | 26.7% | 40.0% | 16.7% | 43.6% | 43.6% | 28.1% | 773 | 1.26 | 2.72e-50 |
| M1c | 31.9% | 48.0% | 29.3% | 60.7% | 0.0% | 50.0% | 3.3% | 40.0% | 40.0% | 22.7% | 867 | 1.13 | 5.93e-46 |
| M2_B64 | 51.1% | 60.0% | 42.7% | 60.7% | 56.7% | 56.7% | 30.0% | 54.4% | 53.9% | 46.2% | 414 | 1.13 | 9.32e-65 |
| M2_B128 | 53.6% | 65.3% | 44.0% | 60.7% | 58.3% | 56.7% | 36.7% | 56.2% | 55.1% | 48.1% | 408 | 1.13 | 7.79e-67 |
| M2_B256 | 52.5% | 66.7% | 42.7% | 60.7% | 51.7% | 56.7% | 36.7% | 55.3% | 53.5% | 46.2% | 458 | 1.13 | 9.04e-66 |
| M2_Pre | 51.7% | 66.7% | 42.7% | 60.7% | 50.0% | 53.3% | 36.7% | 54.7% | 53.3% | 45.1% | 480 | 1.13 | 5.48e-65 |
| **M2** | 53.6% | 66.7% | 42.7% | 60.7% | 58.3% | 56.7% | 36.7% | 56.2% | 54.7% | 47.5% | 436 | 1.13 | 7.79e-67 |
| M2_Prec (Aux.) | 56.1% | 66.7% | 42.7% | 60.7% | 58.3% | 71.7% | 36.7% | 58.2% | 56.2% | 49.2% | 450 | 1.13 | 1.37e-70 (raw) |
| **Oracle_UB** | 73.4% | 82.7% | 70.7% | 75.3% | 75.0% | 80.0% | 56.7% | 75.1% | 73.4% | 68.4% | 337 | 1.00 | - |

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
- **`Qwen__Qwen3.5-4B`**: DeltaNetBlend(n=336: a=0.00:93.2%, a=0.25:95.2%, a=0.50:95.2%, a=0.75:94.0%, a=1.00:91.7%) | PrefillMLP(layer_-1, tau=0.525, Macro=84.1%, OAA250=84.9%, U0.05=77.8%, Tok=357)
<!-- CBARR_RUN_END:20261117T174419Z -->
