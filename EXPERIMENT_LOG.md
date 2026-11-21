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
### Run `20261117T174419Z` — **RUNNING (after Phase 6 / W6: Qwen3.5-2B (strict341) (merge))** (Updated 2026-11-18 05:48:56 UTC)
- **Code Version**: `v0.5.14` | **Git Commit**: `9ed30a8` | **Results Release**: `v0.5.14-res.3` | **Last Loop**: `Phase 6 / W6: Qwen3.5-2B (strict341) (merge)` | **Wall Time**: `12h 04m 28s` (43468.0s)
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
  "git_rev": "9ed30a8",
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
| Phase 6 / W6: Qwen3.5-0.8B (strict341) (merge) | PASS | 50.0s (50.0s) |
| Phase 6 / W6: Qwen3.5-2B (strict341) | PASS | 4h 07m 23s (14843.0s) |
| Phase 6 / W6: Qwen3.5-2B (strict341) (merge) | PASS | 57.0s (57.0s) |

#### 3. Gate-3 Live 12-Item Smoke Check
| Model | Items | D13 <=250 | No Reminder Leak | SmolLM3 Frozen Date |
| :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-4B` | 12 | True | True | N/A |
| `HuggingFaceTB/SmolLM3-3B` | 12 | True | True | True |

#### 4. Calibration & Gate-2 Classifier Results (Per Model)
| Model | Code Ver | tau_prob | tau_h_high | Dev Esc (R1) | G1 Skip | Intercept | Coef (`H1, Hm, Hmax, I, sym`) | Train@0.5 (Acc/Prec/Rec/F1/AUC) | Val-All@tau (Acc/Prec/Rec/F1/AUC) | Val-G2@tau (F1/AUC) | Dev `B1b->M2` Macro (Tok) | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HuggingFaceTB/SmolLM3-3B` | `v0.5.14` (`3214526-dirty`) | 0.400 | 3.144 | 46.7% (OK) | 16.7% | -0.37 | [0.26, 0.53, -0.09, -0.43, -1.45] | 75/28/71/40/0.806 | 48/16/100/28/0.698 | 36/0.806 (n=40) | 76.0%->74.0% (960->407t) | 0.1s |
| `Qwen/Qwen3.5-0.8B` | `v0.5.14` (`003493a`) | 0.525 | 3.719 | 31.7% (OK) | 16.7% | -0.79 | [-0.06, 0.14, 0.15, -0.21, 0.70] | 57/26/73/38/0.651 | 70/19/38/25/0.618 | 40/0.750 (n=40) | 4.0%->61.0% (2000->467t) | 0.1s |
| `Qwen/Qwen3.5-2B` | `v0.5.14` (`464c22d`) | 0.850 | 3.157 | 16.7% (WARN) | 16.7% | -2.95 | [-0.34, 0.06, 0.43, 2.19, 0.39] | 65/9/100/16/0.888 | 88/0/0/0/0.369 | 0/0.684 (n=40) | 13.0%->78.0% (1951->376t) | 0.1s |
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

- **`Qwen/Qwen3.5-2B` (`strict341`**, n_diag=341) — ver=`v0.5.14` (`464c22d`) | dtype=`torch.bfloat16` | wall=`31.7s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.850, esc=16.7%
  - **4-Regime**: Helped=1.1%, Hurt=52.7%, Flip=53.8%, TiePass=17.6%, TieFail=28.7%
  - **Section 6 Causal (`|E_hurt|=215`)**: rho_rec=90.2%, rho_int=9.8% | A1_strip=100.0%, B2a_post=90.2%, B1_pre=87.0%, B2b_raw=50.2%, A2_swap=59.5%, B3_cut=81.9%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 67.7% | 81.3% | 68.0% | 70.0% | 65.0% | 78.3% | 43.3% | 70.2% | 70.2% | 63.6% | 333 | 1.00 | 1.34e-62 |
| B1a | 70.6% | 85.3% | 74.7% | 70.0% | 70.0% | 73.3% | 50.0% | 35.3% | 63.5% | 62.4% | 501 | 1.00 | 2.14e-66 |
| **B1b** | 13.7% | 24.0% | 14.7% | 33.3% | 3.3% | 3.3% | 3.3% | 0.0% | 9.5% | -18.9% | 1879 | 1.00 | - |
| B2a | 61.0% | 86.7% | 72.0% | 70.7% | 43.3% | 60.0% | 33.3% | 46.4% | 59.7% | 58.4% | 382 | 1.00 | 2.15e-55 |
| B2b | 57.2% | 69.3% | 57.3% | 63.3% | 55.0% | 61.7% | 36.7% | 57.6% | 59.1% | 47.2% | 651 | 1.20 | 7.13e-51 |
| M1a | 34.1% | 36.0% | 18.7% | 70.0% | 5.0% | 71.7% | 3.3% | 37.3% | 40.2% | 21.3% | 1078 | 1.00 | 8.63e-32 |
| M1b | 67.4% | 81.3% | 68.0% | 70.0% | 65.0% | 76.7% | 43.3% | 69.8% | 69.8% | 63.2% | 341 | 1.01 | 2.25e-62 |
| M1c | 49.8% | 80.0% | 65.3% | 70.0% | 5.0% | 75.0% | 3.3% | 57.6% | 57.9% | 45.4% | 655 | 1.00 | 9.98e-48 |
| M2_B64 | 67.4% | 81.3% | 68.0% | 70.0% | 65.0% | 76.7% | 43.3% | 70.0% | 69.6% | 62.8% | 359 | 1.00 | 1.95e-61 |
| M2_B128 | 70.2% | 81.3% | 68.0% | 70.0% | 70.0% | 78.3% | 53.3% | 71.6% | 70.7% | 64.0% | 377 | 1.00 | 3.39e-64 |
| M2_B256 | 70.2% | 81.3% | 68.0% | 70.0% | 75.0% | 76.7% | 50.0% | 71.8% | 70.0% | 63.8% | 399 | 1.00 | 1.9e-64 |
| M2_Pre | 68.8% | 81.3% | 68.0% | 70.0% | 63.3% | 80.0% | 50.0% | 70.7% | 69.6% | 62.7% | 399 | 1.00 | 4e-63 |
| **M2** | 69.6% | 81.3% | 68.0% | 70.0% | 70.0% | 78.3% | 50.0% | 71.3% | 70.2% | 63.7% | 382 | 1.00 | 5.98e-64 |
| M2_Prec (Aux.) | 69.6% | 81.3% | 68.0% | 70.0% | 70.0% | 75.0% | 53.3% | 71.1% | 69.4% | 62.9% | 413 | 1.00 | 1.3e-64 (raw) |
| **Oracle_UB** | 87.2% | 94.7% | 92.0% | 86.7% | 86.7% | 90.0% | 73.3% | 88.4% | 86.6% | 82.3% | 308 | 1.00 | - |

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

<!-- CBARR_RUN_START:20261118T214055Z -->
### Run `20261118T214055Z` — **RUNNING (after Step 0a: Hardware & Kernel Probe)** (Updated 2026-11-18 21:42:09 UTC)
- **Code Version**: `v0.5.19` | **Git Commit**: `ba01e3b` | **Results Release**: `v0.5.19-res.4` | **Last Loop**: `Step 0a: Hardware & Kernel Probe` | **Wall Time**: `1m 06.0s` (66.0s)
- **Hardware**: 8x NVIDIA L40 (44.4 GB, SM 8.9, CUDA 13.2, BF16=True) | PyTorch `2.15.0+cu132` | Transformers `5.23.0` | Kernels: `fla=True, causal_conv1d=True`
- **Command**: `scripts/run_gpu_phases_4_to_6.sh --part all --setup-mode auto --shards auto --auto-9b --with-w7 --auto-b512 --auto-r6 --commit --push`
- **Raw Log**: `outputs/logs/cbarr_20261118T214055Z.log`

#### 1. Training & Experiment Parameters
```json
{
  "auto_b512": "auto",
  "auto_commit": true,
  "auto_push": true,
  "b0_max_new_tokens": 1024,
  "b1b_max_new_tokens": 2048,
  "class_weight": "balanced",
  "code_version": "0.5.19",
  "device": "cuda",
  "do_sample": false,
  "dtype": "auto",
  "gate2_probe_tokens": 32,
  "git_branch": "",
  "git_rev": "ba01e3b",
  "include_b512": false,
  "is_mock": false,
  "lam_utility": 0.05,
  "logreg_C": 1.0,
  "norm_tokens": 250.0,
  "presence_penalty": 0.0,
  "r6_seeds": [
    42,
    43,
    44
  ],
  "run_id": "20261118T214055Z",
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
  "with_ifbench": true,
  "with_r6": true,
  "with_w7": true
}
```

#### 2. Stage Timings
| Stage | Status | Wall Time |
| :--- | :--- | :--- |
| Unpack Synced Worker Results | PASS | 1.0s (1.0s) |
| Gate 0: Preflight Tests | PASS | 6.0s (6.0s) |
| Step 0a: Hardware & Kernel Probe | PASS | 9.0s (9.0s) |

#### 4. Calibration & Gate-2 Classifier Results (Per Model)
| Model | Code Ver | tau_prob | tau_h_high | Dev Esc (R1) | G1 Skip | Intercept | Coef (`H1, Hm, Hmax, I, sym`) | Train@0.5 (Acc/Prec/Rec/F1/AUC) | Val-All@tau (Acc/Prec/Rec/F1/AUC) | Val-G2@tau (F1/AUC) | Dev `B1b->M2` Macro (Tok) | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HuggingFaceTB/SmolLM3-3B` | `v0.5.14` (`3214526-dirty`) | 0.400 | 3.144 | 46.7% (OK) | 16.7% | -0.37 | [0.26, 0.53, -0.09, -0.43, -1.45] | 75/28/71/40/0.806 | 48/16/100/28/0.698 | 36/0.806 (n=40) | 76.0%->74.0% (960->407t) | 0.1s |
| `Qwen/Qwen3.5-0.8B` | `v0.5.14` (`003493a`) | 0.525 | 3.719 | 31.7% (OK) | 16.7% | -0.79 | [-0.06, 0.14, 0.15, -0.21, 0.70] | 57/26/73/38/0.651 | 70/19/38/25/0.618 | 40/0.750 (n=40) | 4.0%->61.0% (2000->467t) | 0.1s |
| `Qwen/Qwen3.5-2B` | `v0.5.14` (`464c22d`) | 0.850 | 3.157 | 16.7% (WARN) | 16.7% | -2.95 | [-0.34, 0.06, 0.43, 2.19, 0.39] | 65/9/100/16/0.888 | 88/0/0/0/0.369 | 0/0.684 (n=40) | 13.0%->78.0% (1951->376t) | 0.1s |
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

- **`Qwen/Qwen3.5-2B` (`strict341`**, n_diag=341) — ver=`v0.5.14` (`464c22d`) | dtype=`torch.bfloat16` | wall=`31.7s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.850, esc=16.7%
  - **4-Regime**: Helped=1.1%, Hurt=52.7%, Flip=53.8%, TiePass=17.6%, TieFail=28.7%
  - **Section 6 Causal (`|E_hurt|=215`)**: rho_rec=90.2%, rho_int=9.8% | A1_strip=100.0%, B2a_post=90.2%, B1_pre=87.0%, B2b_raw=50.2%, A2_swap=59.5%, B3_cut=81.9%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 67.7% | 81.3% | 68.0% | 70.0% | 65.0% | 78.3% | 43.3% | 70.2% | 70.2% | 63.6% | 333 | 1.00 | 1.34e-62 |
| B1a | 70.6% | 85.3% | 74.7% | 70.0% | 70.0% | 73.3% | 50.0% | 35.3% | 63.5% | 62.4% | 501 | 1.00 | 2.14e-66 |
| **B1b** | 13.7% | 24.0% | 14.7% | 33.3% | 3.3% | 3.3% | 3.3% | 0.0% | 9.5% | -18.9% | 1879 | 1.00 | - |
| B2a | 61.0% | 86.7% | 72.0% | 70.7% | 43.3% | 60.0% | 33.3% | 46.4% | 59.7% | 58.4% | 382 | 1.00 | 2.15e-55 |
| B2b | 57.2% | 69.3% | 57.3% | 63.3% | 55.0% | 61.7% | 36.7% | 57.6% | 59.1% | 47.2% | 651 | 1.20 | 7.13e-51 |
| M1a | 34.1% | 36.0% | 18.7% | 70.0% | 5.0% | 71.7% | 3.3% | 37.3% | 40.2% | 21.3% | 1078 | 1.00 | 8.63e-32 |
| M1b | 67.4% | 81.3% | 68.0% | 70.0% | 65.0% | 76.7% | 43.3% | 69.8% | 69.8% | 63.2% | 341 | 1.01 | 2.25e-62 |
| M1c | 49.8% | 80.0% | 65.3% | 70.0% | 5.0% | 75.0% | 3.3% | 57.6% | 57.9% | 45.4% | 655 | 1.00 | 9.98e-48 |
| M2_B64 | 67.4% | 81.3% | 68.0% | 70.0% | 65.0% | 76.7% | 43.3% | 70.0% | 69.6% | 62.8% | 359 | 1.00 | 1.95e-61 |
| M2_B128 | 70.2% | 81.3% | 68.0% | 70.0% | 70.0% | 78.3% | 53.3% | 71.6% | 70.7% | 64.0% | 377 | 1.00 | 3.39e-64 |
| M2_B256 | 70.2% | 81.3% | 68.0% | 70.0% | 75.0% | 76.7% | 50.0% | 71.8% | 70.0% | 63.8% | 399 | 1.00 | 1.9e-64 |
| M2_Pre | 68.8% | 81.3% | 68.0% | 70.0% | 63.3% | 80.0% | 50.0% | 70.7% | 69.6% | 62.7% | 399 | 1.00 | 4e-63 |
| **M2** | 69.6% | 81.3% | 68.0% | 70.0% | 70.0% | 78.3% | 50.0% | 71.3% | 70.2% | 63.7% | 382 | 1.00 | 5.98e-64 |
| M2_Prec (Aux.) | 69.6% | 81.3% | 68.0% | 70.0% | 70.0% | 75.0% | 53.3% | 71.1% | 69.4% | 62.9% | 413 | 1.00 | 1.3e-64 (raw) |
| **Oracle_UB** | 87.2% | 94.7% | 92.0% | 86.7% | 86.7% | 90.0% | 73.3% | 88.4% | 86.6% | 82.3% | 308 | 1.00 | - |

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
- **`Qwen__Qwen3.5-4B`**: DeltaNetBlend(n=336: a=0.00:93.2%, a=0.25:95.2%, a=0.50:95.2%, a=0.75:94.0%, a=1.00:91.7%) | PrefillMLP(layer_-1, tau=0.525, Macro=84.1%, OAA250=84.9%, U0.05=77.8%, Tok=357) | Stage2SolveFirst(replayed=17, Macro=84.6%->84.3%, OAA250=84.9%->84.4%)
<!-- CBARR_RUN_END:20261118T214055Z -->

<!-- CBARR_RUN_START:20261118T215019Z -->
### Run `20261118T215019Z` — **FAILED (exit=1 at Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (Contingency R3 B=512))** (Updated 2026-11-18 22:42:59 UTC)
- **Code Version**: `v0.5.19` | **Git Commit**: `0aa291e` | **Results Release**: `v0.5.19-res.5` | **Last Loop**: `Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (Contingency R3 B=512)` | **Wall Time**: `52m 31.0s` (3151.0s)
- **Hardware**: 8x NVIDIA L40 (44.4 GB, SM 8.9, CUDA 13.2, BF16=True) | PyTorch `2.15.0+cu132` | Transformers `5.23.0` | Kernels: `fla=True, causal_conv1d=True`
- **Command**: `scripts/run_gpu_phases_4_to_6.sh --part all --setup-mode auto --shards auto --auto-9b --with-w7 --auto-b512 --auto-r6 --commit --push --shard-progress-sec 10`
- **Raw Log**: `outputs/logs/cbarr_20261118T215019Z.log`
- **Notes**: [GPU7!] OSError: HuggingFaceTB/SmolLM3-3B does not appear to have a file named pytorch_model.bin or model.safetensors. | --- End failed shard log --- | [2026-11-18T22:42:50Z] === [Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (Contingency R3 B=512)] FAILED in parallel GPU shard (attempt 3/3, 28s; inspect outputs/logs/cbarr_20

#### 1. Training & Experiment Parameters
```json
{
  "auto_b512": "auto",
  "auto_commit": true,
  "auto_push": true,
  "b0_max_new_tokens": 1024,
  "b1b_max_new_tokens": 2048,
  "class_weight": "balanced",
  "code_version": "0.5.19",
  "device": "cuda",
  "do_sample": false,
  "dtype": "auto",
  "gate2_probe_tokens": 32,
  "git_branch": "",
  "git_rev": "0aa291e",
  "include_b512": false,
  "is_mock": false,
  "lam_utility": 0.05,
  "logreg_C": 1.0,
  "norm_tokens": 250.0,
  "presence_penalty": 0.0,
  "r6_seeds": [
    42,
    43,
    44
  ],
  "run_id": "20261118T215019Z",
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
  "with_ifbench": true,
  "with_r6": true,
  "with_w7": true
}
```

#### 2. Stage Timings
| Stage | Status | Wall Time |
| :--- | :--- | :--- |
| Unpack Synced Worker Results | PASS | 0.0s (0.0s) |
| Gate 0: Preflight Tests | PASS | 5.0s (5.0s) |
| Step 0a: Hardware & Kernel Probe | PASS | 8.0s (8.0s) |
| Step 0a: Live GPU 12-Item Smoke (cached) | PASS | 5.0s (5.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (strict341) (cached) | PASS | 35.0s (35.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (Contingency R3 B=512) | PASS | 17m 15.0s (1035.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (Contingency R3 B=512) (merge) | PASS | 51.0s (51.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (diag491) (cached) | PASS | 37.0s (37.0s) |
| W7: DeltaNet Blend (Qwen/Qwen3.5-4B) (cached) | PASS | 5.0s (5.0s) |
| W7: DeltaNet Conv+KV Blend (Qwen/Qwen3.5-4B) | PASS | 26m 02.0s (1562.0s) |
| W7: Prefill MLP Probe (Qwen/Qwen3.5-4B) (cached) | PASS | 12.0s (12.0s) |
| W7: Stage-2 SolveFirst Replay (Qwen/Qwen3.5-4B) (cached) | PASS | 0.0s (0.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (strict341) (cached) | PASS | 40.0s (40.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (Contingency R3 B=512) | FAIL(shard_error) | 52.0s (52.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (Contingency R3 B=512) | FAIL(shard_error) | 28.0s (28.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (Contingency R3 B=512) | FAIL(shard_error) | 28.0s (28.0s) |

#### 3. Gate-3 Live 12-Item Smoke Check
| Model | Items | D13 <=250 | No Reminder Leak | SmolLM3 Frozen Date |
| :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-4B` | 12 | True | True | N/A |
| `HuggingFaceTB/SmolLM3-3B` | 12 | True | True | True |

#### 4. Calibration & Gate-2 Classifier Results (Per Model)
| Model | Code Ver | tau_prob | tau_h_high | Dev Esc (R1) | G1 Skip | Intercept | Coef (`H1, Hm, Hmax, I, sym`) | Train@0.5 (Acc/Prec/Rec/F1/AUC) | Val-All@tau (Acc/Prec/Rec/F1/AUC) | Val-G2@tau (F1/AUC) | Dev `B1b->M2` Macro (Tok) | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HuggingFaceTB/SmolLM3-3B` | `v0.5.19` (`0aa291e`) | 0.400 | 3.144 | 46.7% (OK) | 16.7% | -0.37 | [0.26, 0.53, -0.09, -0.43, -1.45] | 75/28/71/40/0.806 | 48/16/100/28/0.698 | 36/0.806 (n=40) | 76.0%->74.0% (960->407t) | 0.2s |
| `Qwen/Qwen3.5-0.8B` | `v0.5.14` (`003493a`) | 0.525 | 3.719 | 31.7% (OK) | 16.7% | -0.79 | [-0.06, 0.14, 0.15, -0.21, 0.70] | 57/26/73/38/0.651 | 70/19/38/25/0.618 | 40/0.750 (n=40) | 4.0%->61.0% (2000->467t) | 0.1s |
| `Qwen/Qwen3.5-2B` | `v0.5.14` (`464c22d`) | 0.850 | 3.157 | 16.7% (WARN) | 16.7% | -2.95 | [-0.34, 0.06, 0.43, 2.19, 0.39] | 65/9/100/16/0.888 | 88/0/0/0/0.369 | 0/0.684 (n=40) | 13.0%->78.0% (1951->376t) | 0.1s |
| `Qwen/Qwen3.5-4B` | `v0.5.19` (`d0c0963-dirty`) | 0.650 | 2.537 | 20.0% (OK) | 16.7% | -0.64 | [-0.68, 0.60, 0.10, 0.96, -0.43] | 72/19/100/32/0.804 | 87/0/0/0/0.629 | 0/0.410 (n=40) | 38.0%->90.0% (1744->365t) | 0.1s |

#### 5. Evaluation Matrix & Section 6 Diagnostics (Per Model x Pool)

- **`HuggingFaceTB/SmolLM3-3B` (`strict341`**, n_diag=341) — ver=`v0.5.19` (`0aa291e`) | dtype=`torch.bfloat16` | wall=`26.5s` (calib=0.2s, eval=0.3s, diag=0.1s) | tau=0.400, esc=46.7%
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

- **`Qwen/Qwen3.5-2B` (`strict341`**, n_diag=341) — ver=`v0.5.14` (`464c22d`) | dtype=`torch.bfloat16` | wall=`31.7s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.850, esc=16.7%
  - **4-Regime**: Helped=1.1%, Hurt=52.7%, Flip=53.8%, TiePass=17.6%, TieFail=28.7%
  - **Section 6 Causal (`|E_hurt|=215`)**: rho_rec=90.2%, rho_int=9.8% | A1_strip=100.0%, B2a_post=90.2%, B1_pre=87.0%, B2b_raw=50.2%, A2_swap=59.5%, B3_cut=81.9%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 67.7% | 81.3% | 68.0% | 70.0% | 65.0% | 78.3% | 43.3% | 70.2% | 70.2% | 63.6% | 333 | 1.00 | 1.34e-62 |
| B1a | 70.6% | 85.3% | 74.7% | 70.0% | 70.0% | 73.3% | 50.0% | 35.3% | 63.5% | 62.4% | 501 | 1.00 | 2.14e-66 |
| **B1b** | 13.7% | 24.0% | 14.7% | 33.3% | 3.3% | 3.3% | 3.3% | 0.0% | 9.5% | -18.9% | 1879 | 1.00 | - |
| B2a | 61.0% | 86.7% | 72.0% | 70.7% | 43.3% | 60.0% | 33.3% | 46.4% | 59.7% | 58.4% | 382 | 1.00 | 2.15e-55 |
| B2b | 57.2% | 69.3% | 57.3% | 63.3% | 55.0% | 61.7% | 36.7% | 57.6% | 59.1% | 47.2% | 651 | 1.20 | 7.13e-51 |
| M1a | 34.1% | 36.0% | 18.7% | 70.0% | 5.0% | 71.7% | 3.3% | 37.3% | 40.2% | 21.3% | 1078 | 1.00 | 8.63e-32 |
| M1b | 67.4% | 81.3% | 68.0% | 70.0% | 65.0% | 76.7% | 43.3% | 69.8% | 69.8% | 63.2% | 341 | 1.01 | 2.25e-62 |
| M1c | 49.8% | 80.0% | 65.3% | 70.0% | 5.0% | 75.0% | 3.3% | 57.6% | 57.9% | 45.4% | 655 | 1.00 | 9.98e-48 |
| M2_B64 | 67.4% | 81.3% | 68.0% | 70.0% | 65.0% | 76.7% | 43.3% | 70.0% | 69.6% | 62.8% | 359 | 1.00 | 1.95e-61 |
| M2_B128 | 70.2% | 81.3% | 68.0% | 70.0% | 70.0% | 78.3% | 53.3% | 71.6% | 70.7% | 64.0% | 377 | 1.00 | 3.39e-64 |
| M2_B256 | 70.2% | 81.3% | 68.0% | 70.0% | 75.0% | 76.7% | 50.0% | 71.8% | 70.0% | 63.8% | 399 | 1.00 | 1.9e-64 |
| M2_Pre | 68.8% | 81.3% | 68.0% | 70.0% | 63.3% | 80.0% | 50.0% | 70.7% | 69.6% | 62.7% | 399 | 1.00 | 4e-63 |
| **M2** | 69.6% | 81.3% | 68.0% | 70.0% | 70.0% | 78.3% | 50.0% | 71.3% | 70.2% | 63.7% | 382 | 1.00 | 5.98e-64 |
| M2_Prec (Aux.) | 69.6% | 81.3% | 68.0% | 70.0% | 70.0% | 75.0% | 53.3% | 71.1% | 69.4% | 62.9% | 413 | 1.00 | 1.3e-64 (raw) |
| **Oracle_UB** | 87.2% | 94.7% | 92.0% | 86.7% | 86.7% | 90.0% | 73.3% | 88.4% | 86.6% | 82.3% | 308 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`strict341`**, n_diag=341) — ver=`v0.5.19` (`ba01e3b-dirty`) | dtype=`torch.bfloat16` | wall=`28.6s` (calib=0.1s, eval=0.4s, diag=0.1s) | tau=0.650, esc=20.0%
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
| M2_B512 | 86.1% | 92.0% | 90.7% | 82.0% | 86.7% | 81.7% | 83.3% | 65.6% | 80.7% | 76.9% | 445 | 1.04 | 3.05e-64 (raw) |
| **Oracle_UB** | 94.2% | 97.3% | 98.7% | 92.7% | 93.3% | 93.3% | 90.0% | 92.2% | 92.3% | 88.4% | 300 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`diag491`**, n_diag=491) — ver=`v0.5.19` (`d0c0963-dirty`) | dtype=`torch.bfloat16` | wall=`22.4s` (calib=0.1s, eval=0.4s, diag=0.1s) | tau=0.650, esc=20.0%
  - **4-Regime**: Helped=0.7%, Hurt=50.0%, Flip=50.7%, TiePass=34.7%, TieFail=14.7%
  - **Section 6 Causal (`|E_hurt|=336`)**: rho_rec=96.1%, rho_int=3.9% | A1_strip=100.0%, B2a_post=96.1%, B1_pre=95.5%, B2b_raw=49.4%, A2_swap=70.2%, B3_cut=86.3%
  - *Eval-450 table identical to `strict341` above (omitted duplicate table).*

#### 6. Optional W7 Extensions
- **`Qwen__Qwen3.5-4B`**: DeltaNetBlend(n=336: a=0.00:93.2%, a=0.25:95.2%, a=0.50:95.2%, a=0.75:94.0%, a=1.00:91.7%) | DeltaNetConvKV(n=336: a=0.00:85.7%, a=0.25:88.1%, a=0.50:89.9%, a=0.75:91.7%, a=1.00:95.8%) | PrefillMLP(layer_-1, tau=0.525, Macro=84.1%, OAA250=84.9%, U0.05=77.8%, Tok=357) | Stage2SolveFirst(replayed=17, Macro=84.6%->84.3%, OAA250=84.9%->84.4%)
<!-- CBARR_RUN_END:20261118T215019Z -->

<!-- CBARR_RUN_START:20261119T000849Z -->
### Run `20261119T000849Z` — **FAILED (exit=1 at Phase 6 / W6: Qwen3.5-9B (strict341) (merge))** (Updated 2026-11-19 05:54:04 UTC)
- **Code Version**: `v0.5.22` | **Git Commit**: `5edaa78` | **Results Release**: `v0.5.22-res.10` | **Last Loop**: `Phase 6 / W6: Qwen3.5-9B (strict341) (merge)` | **Wall Time**: `5h 45m 06s` (20706.0s)
- **Hardware**: 8x NVIDIA L40 (44.4 GB, SM 8.9, CUDA 13.2, BF16=True) | PyTorch `2.15.0+cu132` | Transformers `5.23.0` | Kernels: `fla=True, causal_conv1d=True`
- **Command**: `scripts/run_gpu_phases_4_to_6.sh --part all --setup-mode auto --shards auto --auto-9b --with-w7 --auto-b512 --auto-r6 --commit --push --color --no-setup --shard-progress-sec 10`
- **Raw Log**: `outputs/logs/cbarr_20261119T000849Z.log`
- **Notes**: [Disk Guard] ERROR: IFBench OOD (Qwen/Qwen3.5-4B) aborted — only 0.30 GB free on outputs (< 1.0 GB floor).

#### 1. Training & Experiment Parameters
```json
{
  "auto_b512": "auto",
  "auto_commit": true,
  "auto_push": true,
  "b0_max_new_tokens": 1024,
  "b1b_max_new_tokens": 2048,
  "class_weight": "balanced",
  "code_version": "0.5.22",
  "device": "cuda",
  "do_sample": false,
  "dtype": "auto",
  "gate2_probe_tokens": 32,
  "git_branch": "",
  "git_rev": "5edaa78",
  "include_b512": true,
  "is_mock": false,
  "lam_utility": 0.05,
  "logreg_C": 1.0,
  "norm_tokens": 250.0,
  "presence_penalty": 0.0,
  "r6_seeds": [
    42,
    43,
    44
  ],
  "run_id": "20261119T000849Z",
  "seed": 42,
  "setup_mode": "off",
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
    250,
    512
  ],
  "top_p": 1.0,
  "with_9b": true,
  "with_ifbench": true,
  "with_r6": true,
  "with_w7": true
}
```

#### 2. Stage Timings
| Stage | Status | Wall Time |
| :--- | :--- | :--- |
| Unpack Synced Worker Results | PASS | 1.0s (1.0s) |
| Gate 0: Preflight Tests | PASS | 6.0s (6.0s) |
| Step 0a: Hardware & Kernel Probe | PASS | 9.0s (9.0s) |
| Step 0a: Live GPU 12-Item Smoke (cached) | PASS | 4.0s (4.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (strict341) (cached) | PASS | 38.0s (38.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (Contingency R3 B=512) (cached) | PASS | 35.0s (35.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (diag491) (cached) | PASS | 36.0s (36.0s) |
| W7: DeltaNet Blend (Qwen/Qwen3.5-4B) (cached) | PASS | 5.0s (5.0s) |
| W7: DeltaNet Conv+KV Blend (Qwen/Qwen3.5-4B) (cached) | PASS | 5.0s (5.0s) |
| W7: Prefill MLP Probe (Qwen/Qwen3.5-4B) (cached) | PASS | 12.0s (12.0s) |
| W7: Stage-2 SolveFirst Replay (Qwen/Qwen3.5-4B) (cached) | PASS | 0.0s (0.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (strict341) (cached) | PASS | 35.0s (35.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (Contingency R3 B=512) | PASS | 12m 25.0s (745.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (Contingency R3 B=512) (merge) | PASS | 49.0s (49.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (diag491) (cached) | PASS | 36.0s (36.0s) |
| W7: Dense KV Eviction (HuggingFaceTB/SmolLM3-3B) | PASS | 2m 18.0s (138.0s) |
| W7: Prefill MLP Probe (HuggingFaceTB/SmolLM3-3B) | PASS | 44.0s (44.0s) |
| W7: Stage-2 SolveFirst Replay (HuggingFaceTB/SmolLM3-3B) | PASS | 21m 53.0s (1313.0s) |
| Phase 6 / W6: Qwen3.5-0.8B (strict341) (cached) | PASS | 36.0s (36.0s) |
| Phase 6 / W6: Qwen3.5-2B (strict341) (cached) | PASS | 36.0s (36.0s) |
| Phase 6 / W6: Qwen3.5-9B (strict341) | PASS | 4h 55m 36s (17736.0s) |
| Phase 6 / W6: Qwen3.5-9B (strict341) (merge) | PASS | 57.0s (57.0s) |

#### 3. Gate-3 Live 12-Item Smoke Check
| Model | Items | D13 <=250 | No Reminder Leak | SmolLM3 Frozen Date |
| :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-4B` | 12 | True | True | N/A |
| `HuggingFaceTB/SmolLM3-3B` | 12 | True | True | True |

#### 4. Calibration & Gate-2 Classifier Results (Per Model)
| Model | Code Ver | tau_prob | tau_h_high | Dev Esc (R1) | G1 Skip | Intercept | Coef (`H1, Hm, Hmax, I, sym`) | Train@0.5 (Acc/Prec/Rec/F1/AUC) | Val-All@tau (Acc/Prec/Rec/F1/AUC) | Val-G2@tau (F1/AUC) | Dev `B1b->M2` Macro (Tok) | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HuggingFaceTB/SmolLM3-3B` | `v0.5.21` (`aa095c6-dirty`) | 0.400 | 3.144 | 46.7% (OK) | 16.7% | -0.37 | [0.26, 0.53, -0.09, -0.43, -1.45] | 75/28/71/40/0.806 | 48/16/100/28/0.698 | 36/0.806 (n=40) | 76.0%->74.0% (960->407t) | 0.1s |
| `Qwen/Qwen3.5-0.8B` | `v0.5.21` (`a3d4fa6`) | 0.525 | 3.719 | 31.7% (OK) | 16.7% | -0.79 | [-0.06, 0.14, 0.15, -0.21, 0.70] | 57/26/73/38/0.651 | 70/19/38/25/0.618 | 40/0.750 (n=40) | 4.0%->61.0% (2000->467t) | 0.1s |
| `Qwen/Qwen3.5-2B` | `v0.5.21` (`4ff6f76`) | 0.850 | 3.157 | 16.7% (WARN) | 16.7% | -2.95 | [-0.34, 0.06, 0.43, 2.19, 0.39] | 65/9/100/16/0.888 | 88/0/0/0/0.369 | 0/0.684 (n=40) | 13.0%->78.0% (1951->376t) | 0.1s |
| `Qwen/Qwen3.5-4B` | `v0.5.21` (`73c7891-dirty`) | 0.650 | 2.537 | 20.0% (OK) | 16.7% | -0.64 | [-0.68, 0.60, 0.10, 0.96, -0.43] | 72/19/100/32/0.804 | 87/0/0/0/0.629 | 0/0.410 (n=40) | 38.0%->90.0% (1744->365t) | 0.1s |
| `Qwen/Qwen3.5-9B` | `v0.5.21` (`c766901`) | 0.475 | 2.662 | 43.3% (OK) | 16.7% | -0.31 | [0.30, 0.32, -0.02, 0.00, -1.64] | 57/7/100/13/0.750 | 53/10/75/18/0.656 | 21/0.640 (n=40) | 40.0%->83.0% (1662->351t) | 0.1s |

#### 5. Evaluation Matrix & Section 6 Diagnostics (Per Model x Pool)

- **`HuggingFaceTB/SmolLM3-3B` (`strict341`**, n_diag=341) — ver=`v0.5.21` (`4bcf295-dirty`) | dtype=`torch.bfloat16` | wall=`27.6s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.400, esc=46.7%
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
| M2_B512 | 67.9% | 80.0% | 72.0% | 72.0% | 65.0% | 81.7% | 36.7% | 38.9% | 64.1% | 61.1% | 511 | 1.31 | 0.295 (raw) |
| **Oracle_UB** | 88.2% | 97.3% | 97.3% | 89.3% | 83.3% | 95.0% | 66.7% | 88.0% | 88.5% | 84.7% | 287 | 1.00 | - |

- **`HuggingFaceTB/SmolLM3-3B` (`diag491`**, n_diag=491) — ver=`v0.5.21` (`aa095c6-dirty`) | dtype=`torch.bfloat16` | wall=`22.0s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.400, esc=46.7%
  - **4-Regime**: Helped=13.6%, Hurt=9.6%, Flip=23.1%, TiePass=60.0%, TieFail=16.9%
  - **Section 6 Causal (`|E_hurt|=80`)**: rho_rec=62.5%, rho_int=37.5% | A1_strip=100.0%, B2a_post=62.5%, B1_pre=58.8%, B2b_raw=33.8%, A2_swap=60.0%, B3_cut=62.5%
  - *Eval-450 table identical to `strict341` above (omitted duplicate table).*

- **`Qwen/Qwen3.5-0.8B` (`strict341`**, n_diag=341) — ver=`v0.5.21` (`a3d4fa6`) | dtype=`torch.bfloat16` | wall=`22.8s` (calib=0.1s, eval=0.4s, diag=0.1s) | tau=0.525, esc=31.7%
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

- **`Qwen/Qwen3.5-2B` (`strict341`**, n_diag=341) — ver=`v0.5.21` (`4ff6f76`) | dtype=`torch.bfloat16` | wall=`22.5s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.850, esc=16.7%
  - **4-Regime**: Helped=1.1%, Hurt=52.7%, Flip=53.8%, TiePass=17.6%, TieFail=28.7%
  - **Section 6 Causal (`|E_hurt|=215`)**: rho_rec=90.2%, rho_int=9.8% | A1_strip=100.0%, B2a_post=90.2%, B1_pre=87.0%, B2b_raw=50.2%, A2_swap=59.5%, B3_cut=81.9%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 67.7% | 81.3% | 68.0% | 70.0% | 65.0% | 78.3% | 43.3% | 70.2% | 70.2% | 63.6% | 333 | 1.00 | 1.34e-62 |
| B1a | 70.6% | 85.3% | 74.7% | 70.0% | 70.0% | 73.3% | 50.0% | 35.3% | 63.5% | 62.4% | 501 | 1.00 | 2.14e-66 |
| **B1b** | 13.7% | 24.0% | 14.7% | 33.3% | 3.3% | 3.3% | 3.3% | 0.0% | 9.5% | -18.9% | 1879 | 1.00 | - |
| B2a | 61.0% | 86.7% | 72.0% | 70.7% | 43.3% | 60.0% | 33.3% | 46.4% | 59.7% | 58.4% | 382 | 1.00 | 2.15e-55 |
| B2b | 57.2% | 69.3% | 57.3% | 63.3% | 55.0% | 61.7% | 36.7% | 57.6% | 59.1% | 47.2% | 651 | 1.20 | 7.13e-51 |
| M1a | 34.1% | 36.0% | 18.7% | 70.0% | 5.0% | 71.7% | 3.3% | 37.3% | 40.2% | 21.3% | 1078 | 1.00 | 8.63e-32 |
| M1b | 67.4% | 81.3% | 68.0% | 70.0% | 65.0% | 76.7% | 43.3% | 69.8% | 69.8% | 63.2% | 341 | 1.01 | 2.25e-62 |
| M1c | 49.8% | 80.0% | 65.3% | 70.0% | 5.0% | 75.0% | 3.3% | 57.6% | 57.9% | 45.4% | 655 | 1.00 | 9.98e-48 |
| M2_B64 | 67.4% | 81.3% | 68.0% | 70.0% | 65.0% | 76.7% | 43.3% | 70.0% | 69.6% | 62.8% | 359 | 1.00 | 1.95e-61 |
| M2_B128 | 70.2% | 81.3% | 68.0% | 70.0% | 70.0% | 78.3% | 53.3% | 71.6% | 70.7% | 64.0% | 377 | 1.00 | 3.39e-64 |
| M2_B256 | 70.2% | 81.3% | 68.0% | 70.0% | 75.0% | 76.7% | 50.0% | 71.8% | 70.0% | 63.8% | 399 | 1.00 | 1.9e-64 |
| M2_Pre | 68.8% | 81.3% | 68.0% | 70.0% | 63.3% | 80.0% | 50.0% | 70.7% | 69.6% | 62.7% | 399 | 1.00 | 4e-63 |
| **M2** | 69.6% | 81.3% | 68.0% | 70.0% | 70.0% | 78.3% | 50.0% | 71.3% | 70.2% | 63.7% | 382 | 1.00 | 5.98e-64 |
| M2_Prec (Aux.) | 69.6% | 81.3% | 68.0% | 70.0% | 70.0% | 75.0% | 53.3% | 71.1% | 69.4% | 62.9% | 413 | 1.00 | 1.3e-64 (raw) |
| **Oracle_UB** | 87.2% | 94.7% | 92.0% | 86.7% | 86.7% | 90.0% | 73.3% | 88.4% | 86.6% | 82.3% | 308 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`strict341`**, n_diag=341) — ver=`v0.5.21` (`73c7891-dirty`) | dtype=`torch.bfloat16` | wall=`22.2s` (calib=0.1s, eval=0.4s, diag=0.1s) | tau=0.650, esc=20.0%
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
| M2_B512 | 86.1% | 92.0% | 90.7% | 82.0% | 86.7% | 81.7% | 83.3% | 65.6% | 80.7% | 76.9% | 445 | 1.04 | 3.05e-64 (raw) |
| **Oracle_UB** | 94.2% | 97.3% | 98.7% | 92.7% | 93.3% | 93.3% | 90.0% | 92.2% | 92.3% | 88.4% | 300 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`diag491`**, n_diag=491) — ver=`v0.5.21` (`73c7891-dirty`) | dtype=`torch.bfloat16` | wall=`21.5s` (calib=0.1s, eval=0.5s, diag=0.1s) | tau=0.650, esc=20.0%
  - **4-Regime**: Helped=0.7%, Hurt=50.0%, Flip=50.7%, TiePass=34.7%, TieFail=14.7%
  - **Section 6 Causal (`|E_hurt|=336`)**: rho_rec=96.1%, rho_int=3.9% | A1_strip=100.0%, B2a_post=96.1%, B1_pre=95.5%, B2b_raw=49.4%, A2_swap=70.2%, B3_cut=86.3%
  - *Eval-450 table identical to `strict341` above (omitted duplicate table).*

- **`Qwen/Qwen3.5-9B` (`strict341`**, n_diag=341) — ver=`v0.5.21` (`c766901`) | dtype=`torch.bfloat16` | wall=`33.5s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.475, esc=43.3%
  - **4-Regime**: Helped=0.9%, Hurt=43.8%, Flip=44.7%, TiePass=43.1%, TieFail=12.2%
  - **Section 6 Causal (`|E_hurt|=226`)**: rho_rec=95.1%, rho_int=4.9% | A1_strip=100.0%, B2a_post=95.1%, B1_pre=93.4%, B2b_raw=47.8%, A2_swap=80.1%, B3_cut=92.5%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 87.5% | 92.0% | 93.3% | 81.3% | 88.3% | 86.7% | 83.3% | 86.9% | 86.9% | 80.7% | 310 | 1.00 | 5.03e-52 |
| B1a | 84.3% | 96.0% | 93.3% | 81.3% | 71.7% | 80.0% | 83.3% | 33.6% | 73.0% | 75.5% | 447 | 1.00 | 4.11e-49 |
| **B1b** | 38.3% | 62.7% | 52.0% | 52.0% | 23.3% | 26.7% | 13.3% | 0.0% | 22.2% | 10.6% | 1668 | 1.00 | - |
| B2a | 78.4% | 85.3% | 70.7% | 82.7% | 80.0% | 81.7% | 70.0% | 63.3% | 74.0% | 74.7% | 256 | 1.00 | 3.53e-32 |
| B2b | 70.1% | 81.3% | 76.0% | 66.7% | 68.3% | 65.0% | 63.3% | 51.6% | 60.6% | 52.5% | 895 | 1.42 | 7.17e-33 |
| M1a | 56.8% | 72.0% | 57.3% | 81.3% | 30.0% | 80.0% | 20.0% | 43.8% | 54.2% | 45.6% | 952 | 1.00 | 1.45e-26 |
| M1b | 54.4% | 86.7% | 89.3% | 57.3% | 33.3% | 40.0% | 20.0% | 40.4% | 49.3% | 36.4% | 1158 | 1.56 | 2.94e-19 |
| M1c | 58.8% | 85.3% | 86.7% | 57.3% | 25.0% | 81.7% | 16.7% | 44.2% | 53.2% | 41.9% | 1062 | 1.32 | 1.32e-23 |
| M2_B64 | 86.5% | 93.3% | 93.3% | 80.7% | 85.0% | 86.7% | 80.0% | 86.2% | 84.5% | 79.2% | 353 | 1.32 | 1.97e-49 |
| M2_B128 | 84.6% | 93.3% | 93.3% | 79.3% | 85.0% | 86.7% | 70.0% | 85.1% | 82.2% | 77.6% | 378 | 1.32 | 7.29e-50 |
| M2_B256 | 85.9% | 93.3% | 92.0% | 83.3% | 85.0% | 85.0% | 76.7% | 86.4% | 81.3% | 78.0% | 424 | 1.32 | 1.77e-51 |
| M2_Pre | 86.8% | 93.3% | 92.0% | 80.7% | 91.7% | 86.7% | 76.7% | 86.7% | 83.9% | 79.0% | 382 | 1.32 | 8.51e-51 |
| **M2** | 86.0% | 93.3% | 92.0% | 80.7% | 86.7% | 86.7% | 76.7% | 86.0% | 83.3% | 78.4% | 380 | 1.32 | 5.68e-50 |
| M2_Prec (Aux.) | 86.3% | 93.3% | 92.0% | 80.7% | 86.7% | 88.3% | 76.7% | 86.2% | 82.8% | 78.2% | 401 | 1.32 | 3.24e-51 (raw) |
| **Oracle_UB** | 93.9% | 98.7% | 96.0% | 90.7% | 96.7% | 91.7% | 90.0% | 93.8% | 92.5% | 87.9% | 292 | 1.00 | - |

#### 6. Optional W7 Extensions
- **`HuggingFaceTB__SmolLM3-3B`**: DenseKVEvict(n=80: a=0.00:86.2%) | PrefillMLP(layer_-1, tau=0.850, Macro=66.8%, OAA250=69.6%, U0.05=63.4%, Tok=308) | Stage2SolveFirst(replayed=139, Macro=68.5%->67.0%, OAA250=72.0%->69.6%)
- **`Qwen__Qwen3.5-4B`**: DeltaNetBlend(n=336: a=0.00:93.2%, a=0.25:95.2%, a=0.50:95.2%, a=0.75:94.0%, a=1.00:91.7%) | DeltaNetConvKV(n=336: a=0.00:85.7%, a=0.25:88.1%, a=0.50:89.9%, a=0.75:91.7%, a=1.00:95.8%) | PrefillMLP(layer_-1, tau=0.525, Macro=84.1%, OAA250=84.9%, U0.05=77.8%, Tok=357) | Stage2SolveFirst(replayed=17, Macro=84.6%->84.3%, OAA250=84.9%->84.4%)
<!-- CBARR_RUN_END:20261119T000849Z -->

<!-- CBARR_RUN_START:20261119T105923Z -->
### Run `20261119T105923Z` — **RUNNING (after Phase 6 / W6: Qwen3.5-9B (strict341) (cached))** (Updated 2026-11-19 11:11:53 UTC)
- **Code Version**: `v0.5.26` | **Git Commit**: `ea80916` | **Results Release**: `v0.5.26-res.14` | **Last Loop**: `Phase 6 / W6: Qwen3.5-9B (strict341) (cached)` | **Wall Time**: `12m 22.0s` (742.0s)
- **Hardware**: 8x NVIDIA L40 (44.4 GB, SM 8.9, CUDA 13.2, BF16=True) | PyTorch `2.15.0+cu132` | Transformers `5.23.0` | Kernels: `fla=True, causal_conv1d=True`
- **Command**: `scripts/run_gpu_phases_4_to_6.sh --part all --setup-mode auto --shards auto --auto-9b --with-w7 --auto-b512 --auto-r6 --commit --push --gpu-boost`
- **Raw Log**: `outputs/logs/cbarr_20261119T105923Z.log`

#### 1. Training & Experiment Parameters
```json
{
  "auto_b512": "auto",
  "auto_commit": true,
  "auto_push": true,
  "b0_max_new_tokens": 1024,
  "b1b_max_new_tokens": 2048,
  "class_weight": "balanced",
  "code_version": "0.5.26",
  "device": "cuda",
  "do_sample": false,
  "dtype": "auto",
  "gate2_probe_tokens": 32,
  "git_branch": "",
  "git_rev": "ea80916",
  "include_b512": true,
  "is_mock": false,
  "lam_utility": 0.05,
  "logreg_C": 1.0,
  "norm_tokens": 250.0,
  "presence_penalty": 0.0,
  "r6_seeds": [
    42,
    43,
    44
  ],
  "run_id": "20261119T105923Z",
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
    250,
    512
  ],
  "top_p": 1.0,
  "with_9b": true,
  "with_ifbench": true,
  "with_r6": true,
  "with_w7": true
}
```

#### 2. Stage Timings
| Stage | Status | Wall Time |
| :--- | :--- | :--- |
| Unpack Synced Worker Results | PASS | 1.0s (1.0s) |
| Gate 0: Preflight Tests | PASS | 6.0s (6.0s) |
| Step 0a: Hardware & Kernel Probe | PASS | 9.0s (9.0s) |
| Step 0a: Live GPU 12-Item Smoke (cached) | PASS | 5.0s (5.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (strict341) (cached) | PASS | 40.0s (40.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (Contingency R3 B=512) (cached) | PASS | 41.0s (41.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (diag491) (cached) | PASS | 41.0s (41.0s) |
| W7: DeltaNet Blend (Qwen/Qwen3.5-4B) (cached) | PASS | 5.0s (5.0s) |
| W7: DeltaNet Conv+KV Blend (Qwen/Qwen3.5-4B) (cached) | PASS | 5.0s (5.0s) |
| W7: Prefill MLP Probe (Qwen/Qwen3.5-4B) (cached) | PASS | 16.0s (16.0s) |
| W7: Stage-2 SolveFirst Replay (Qwen/Qwen3.5-4B) (cached) | PASS | 0.0s (0.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (strict341) (cached) | PASS | 41.0s (41.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (Contingency R3 B=512) (cached) | PASS | 41.0s (41.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (diag491) (cached) | PASS | 40.0s (40.0s) |
| W7: Dense KV Eviction (HuggingFaceTB/SmolLM3-3B) (cached) | PASS | 6.0s (6.0s) |
| W7: Prefill MLP Probe (HuggingFaceTB/SmolLM3-3B) (cached) | PASS | 12.0s (12.0s) |
| W7: Stage-2 SolveFirst Replay (HuggingFaceTB/SmolLM3-3B) (cached) | PASS | 0.0s (0.0s) |
| Phase 6 / W6: Qwen3.5-0.8B (strict341) (cached) | PASS | 41.0s (41.0s) |
| Phase 6 / W6: Qwen3.5-2B (strict341) (cached) | PASS | 41.0s (41.0s) |
| Phase 6 / W6: Qwen3.5-9B (strict341) (cached) | PASS | 41.0s (41.0s) |

#### 3. Gate-3 Live 12-Item Smoke Check
| Model | Items | D13 <=250 | No Reminder Leak | SmolLM3 Frozen Date |
| :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-4B` | 12 | True | True | N/A |
| `HuggingFaceTB/SmolLM3-3B` | 12 | True | True | True |

#### 4. Calibration & Gate-2 Classifier Results (Per Model)
| Model | Code Ver | tau_prob | tau_h_high | Dev Esc (R1) | G1 Skip | Intercept | Coef (`H1, Hm, Hmax, I, sym`) | Train@0.5 (Acc/Prec/Rec/F1/AUC) | Val-All@tau (Acc/Prec/Rec/F1/AUC) | Val-G2@tau (F1/AUC) | Dev `B1b->M2` Macro (Tok) | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HuggingFaceTB/SmolLM3-3B` | `v0.5.26` (`2127e69-dirty`) | 0.400 | 3.144 | 46.7% (OK) | 16.7% | -0.37 | [0.26, 0.53, -0.09, -0.43, -1.45] | 75/28/71/40/0.806 | 48/16/100/28/0.698 | 36/0.806 (n=40) | 76.0%->74.0% (960->407t) | 0.1s |
| `Qwen/Qwen3.5-0.8B` | `v0.5.26` (`98e0901`) | 0.525 | 3.719 | 31.7% (OK) | 16.7% | -0.79 | [-0.06, 0.14, 0.15, -0.21, 0.70] | 57/26/73/38/0.651 | 70/19/38/25/0.618 | 40/0.750 (n=40) | 4.0%->61.0% (2000->467t) | 0.1s |
| `Qwen/Qwen3.5-2B` | `v0.5.26` (`e5636f6`) | 0.850 | 3.157 | 16.7% (WARN) | 16.7% | -2.95 | [-0.34, 0.06, 0.43, 2.19, 0.39] | 65/9/100/16/0.888 | 88/0/0/0/0.369 | 0/0.684 (n=40) | 13.0%->78.0% (1951->376t) | 0.1s |
| `Qwen/Qwen3.5-4B` | `v0.5.26` (`95f4d40-dirty`) | 0.650 | 2.537 | 20.0% (OK) | 16.7% | -0.64 | [-0.68, 0.60, 0.10, 0.96, -0.43] | 72/19/100/32/0.804 | 87/0/0/0/0.629 | 0/0.410 (n=40) | 38.0%->90.0% (1744->365t) | 0.1s |
| `Qwen/Qwen3.5-9B` | `v0.5.26` (`ea80916`) | 0.475 | 2.662 | 43.3% (OK) | 16.7% | -0.31 | [0.30, 0.32, -0.02, 0.00, -1.64] | 57/7/100/13/0.750 | 53/10/75/18/0.656 | 21/0.640 (n=40) | 40.0%->83.0% (1662->351t) | 0.1s |

#### 5. Evaluation Matrix & Section 6 Diagnostics (Per Model x Pool)

- **`HuggingFaceTB/SmolLM3-3B` (`strict341`**, n_diag=341) — ver=`v0.5.26` (`2127e69-dirty`) | dtype=`torch.bfloat16` | wall=`26.5s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.400, esc=46.7%
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
| M2_B512 | 67.9% | 80.0% | 72.0% | 72.0% | 65.0% | 81.7% | 36.7% | 38.9% | 64.1% | 61.1% | 511 | 1.31 | 0.295 (raw) |
| **Oracle_UB** | 88.2% | 97.3% | 97.3% | 89.3% | 83.3% | 95.0% | 66.7% | 88.0% | 88.5% | 84.7% | 287 | 1.00 | - |

- **`HuggingFaceTB/SmolLM3-3B` (`diag491`**, n_diag=491) — ver=`v0.5.26` (`2127e69-dirty`) | dtype=`torch.bfloat16` | wall=`25.3s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.400, esc=46.7%
  - **4-Regime**: Helped=13.6%, Hurt=9.6%, Flip=23.1%, TiePass=60.0%, TieFail=16.9%
  - **Section 6 Causal (`|E_hurt|=80`)**: rho_rec=62.5%, rho_int=37.5% | A1_strip=100.0%, B2a_post=62.5%, B1_pre=58.8%, B2b_raw=33.8%, A2_swap=60.0%, B3_cut=62.5%
  - *Eval-450 table identical to `strict341` above (omitted duplicate table).*

- **`Qwen/Qwen3.5-0.8B` (`strict341`**, n_diag=341) — ver=`v0.5.26` (`98e0901`) | dtype=`torch.bfloat16` | wall=`26.9s` (calib=0.1s, eval=0.4s, diag=0.1s) | tau=0.525, esc=31.7%
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

- **`Qwen/Qwen3.5-2B` (`strict341`**, n_diag=341) — ver=`v0.5.26` (`e5636f6`) | dtype=`torch.bfloat16` | wall=`26.8s` (calib=0.1s, eval=0.4s, diag=0.1s) | tau=0.850, esc=16.7%
  - **4-Regime**: Helped=1.1%, Hurt=52.7%, Flip=53.8%, TiePass=17.6%, TieFail=28.7%
  - **Section 6 Causal (`|E_hurt|=215`)**: rho_rec=90.2%, rho_int=9.8% | A1_strip=100.0%, B2a_post=90.2%, B1_pre=87.0%, B2b_raw=50.2%, A2_swap=59.5%, B3_cut=81.9%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 67.7% | 81.3% | 68.0% | 70.0% | 65.0% | 78.3% | 43.3% | 70.2% | 70.2% | 63.6% | 333 | 1.00 | 1.34e-62 |
| B1a | 70.6% | 85.3% | 74.7% | 70.0% | 70.0% | 73.3% | 50.0% | 35.3% | 63.5% | 62.4% | 501 | 1.00 | 2.14e-66 |
| **B1b** | 13.7% | 24.0% | 14.7% | 33.3% | 3.3% | 3.3% | 3.3% | 0.0% | 9.5% | -18.9% | 1879 | 1.00 | - |
| B2a | 61.0% | 86.7% | 72.0% | 70.7% | 43.3% | 60.0% | 33.3% | 46.4% | 59.7% | 58.4% | 382 | 1.00 | 2.15e-55 |
| B2b | 57.2% | 69.3% | 57.3% | 63.3% | 55.0% | 61.7% | 36.7% | 57.6% | 59.1% | 47.2% | 651 | 1.20 | 7.13e-51 |
| M1a | 34.1% | 36.0% | 18.7% | 70.0% | 5.0% | 71.7% | 3.3% | 37.3% | 40.2% | 21.3% | 1078 | 1.00 | 8.63e-32 |
| M1b | 67.4% | 81.3% | 68.0% | 70.0% | 65.0% | 76.7% | 43.3% | 69.8% | 69.8% | 63.2% | 341 | 1.01 | 2.25e-62 |
| M1c | 49.8% | 80.0% | 65.3% | 70.0% | 5.0% | 75.0% | 3.3% | 57.6% | 57.9% | 45.4% | 655 | 1.00 | 9.98e-48 |
| M2_B64 | 67.4% | 81.3% | 68.0% | 70.0% | 65.0% | 76.7% | 43.3% | 70.0% | 69.6% | 62.8% | 359 | 1.00 | 1.95e-61 |
| M2_B128 | 70.2% | 81.3% | 68.0% | 70.0% | 70.0% | 78.3% | 53.3% | 71.6% | 70.7% | 64.0% | 377 | 1.00 | 3.39e-64 |
| M2_B256 | 70.2% | 81.3% | 68.0% | 70.0% | 75.0% | 76.7% | 50.0% | 71.8% | 70.0% | 63.8% | 399 | 1.00 | 1.9e-64 |
| M2_Pre | 68.8% | 81.3% | 68.0% | 70.0% | 63.3% | 80.0% | 50.0% | 70.7% | 69.6% | 62.7% | 399 | 1.00 | 4e-63 |
| **M2** | 69.6% | 81.3% | 68.0% | 70.0% | 70.0% | 78.3% | 50.0% | 71.3% | 70.2% | 63.7% | 382 | 1.00 | 5.98e-64 |
| M2_Prec (Aux.) | 69.6% | 81.3% | 68.0% | 70.0% | 70.0% | 75.0% | 53.3% | 71.1% | 69.4% | 62.9% | 413 | 1.00 | 1.3e-64 (raw) |
| **Oracle_UB** | 87.2% | 94.7% | 92.0% | 86.7% | 86.7% | 90.0% | 73.3% | 88.4% | 86.6% | 82.3% | 308 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`strict341`**, n_diag=341) — ver=`v0.5.26` (`95f4d40-dirty`) | dtype=`torch.bfloat16` | wall=`26.3s` (calib=0.1s, eval=0.5s, diag=0.1s) | tau=0.650, esc=20.0%
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
| M2_B512 | 86.1% | 92.0% | 90.7% | 82.0% | 86.7% | 81.7% | 83.3% | 65.6% | 80.7% | 76.9% | 445 | 1.04 | 3.05e-64 (raw) |
| **Oracle_UB** | 94.2% | 97.3% | 98.7% | 92.7% | 93.3% | 93.3% | 90.0% | 92.2% | 92.3% | 88.4% | 300 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`diag491`**, n_diag=491) — ver=`v0.5.26` (`95f4d40-dirty`) | dtype=`torch.bfloat16` | wall=`26.7s` (calib=0.1s, eval=0.5s, diag=0.1s) | tau=0.650, esc=20.0%
  - **4-Regime**: Helped=0.7%, Hurt=50.0%, Flip=50.7%, TiePass=34.7%, TieFail=14.7%
  - **Section 6 Causal (`|E_hurt|=336`)**: rho_rec=96.1%, rho_int=3.9% | A1_strip=100.0%, B2a_post=96.1%, B1_pre=95.5%, B2b_raw=49.4%, A2_swap=70.2%, B3_cut=86.3%
  - *Eval-450 table identical to `strict341` above (omitted duplicate table).*

- **`Qwen/Qwen3.5-9B` (`strict341`**, n_diag=341) — ver=`v0.5.26` (`ea80916`) | dtype=`torch.bfloat16` | wall=`25.2s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.475, esc=43.3%
  - **4-Regime**: Helped=0.9%, Hurt=43.8%, Flip=44.7%, TiePass=43.1%, TieFail=12.2%
  - **Section 6 Causal (`|E_hurt|=226`)**: rho_rec=95.1%, rho_int=4.9% | A1_strip=100.0%, B2a_post=95.1%, B1_pre=93.4%, B2b_raw=47.8%, A2_swap=80.1%, B3_cut=92.5%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 87.5% | 92.0% | 93.3% | 81.3% | 88.3% | 86.7% | 83.3% | 86.9% | 86.9% | 80.7% | 310 | 1.00 | 5.03e-52 |
| B1a | 84.3% | 96.0% | 93.3% | 81.3% | 71.7% | 80.0% | 83.3% | 33.6% | 73.0% | 75.5% | 447 | 1.00 | 4.11e-49 |
| **B1b** | 38.3% | 62.7% | 52.0% | 52.0% | 23.3% | 26.7% | 13.3% | 0.0% | 22.2% | 10.6% | 1668 | 1.00 | - |
| B2a | 78.4% | 85.3% | 70.7% | 82.7% | 80.0% | 81.7% | 70.0% | 63.3% | 74.0% | 74.7% | 256 | 1.00 | 3.53e-32 |
| B2b | 70.1% | 81.3% | 76.0% | 66.7% | 68.3% | 65.0% | 63.3% | 51.6% | 60.6% | 52.5% | 895 | 1.42 | 7.17e-33 |
| M1a | 56.8% | 72.0% | 57.3% | 81.3% | 30.0% | 80.0% | 20.0% | 43.8% | 54.2% | 45.6% | 952 | 1.00 | 1.45e-26 |
| M1b | 54.4% | 86.7% | 89.3% | 57.3% | 33.3% | 40.0% | 20.0% | 40.4% | 49.3% | 36.4% | 1158 | 1.56 | 2.94e-19 |
| M1c | 58.8% | 85.3% | 86.7% | 57.3% | 25.0% | 81.7% | 16.7% | 44.2% | 53.2% | 41.9% | 1062 | 1.32 | 1.32e-23 |
| M2_B64 | 86.5% | 93.3% | 93.3% | 80.7% | 85.0% | 86.7% | 80.0% | 86.2% | 84.5% | 79.2% | 353 | 1.32 | 1.97e-49 |
| M2_B128 | 84.6% | 93.3% | 93.3% | 79.3% | 85.0% | 86.7% | 70.0% | 85.1% | 82.2% | 77.6% | 378 | 1.32 | 7.29e-50 |
| M2_B256 | 85.9% | 93.3% | 92.0% | 83.3% | 85.0% | 85.0% | 76.7% | 86.4% | 81.3% | 78.0% | 424 | 1.32 | 1.77e-51 |
| M2_Pre | 86.8% | 93.3% | 92.0% | 80.7% | 91.7% | 86.7% | 76.7% | 86.7% | 83.9% | 79.0% | 382 | 1.32 | 8.51e-51 |
| **M2** | 86.0% | 93.3% | 92.0% | 80.7% | 86.7% | 86.7% | 76.7% | 86.0% | 83.3% | 78.4% | 380 | 1.32 | 5.68e-50 |
| M2_Prec (Aux.) | 86.3% | 93.3% | 92.0% | 80.7% | 86.7% | 88.3% | 76.7% | 86.2% | 82.8% | 78.2% | 401 | 1.32 | 3.24e-51 (raw) |
| **Oracle_UB** | 93.9% | 98.7% | 96.0% | 90.7% | 96.7% | 91.7% | 90.0% | 93.8% | 92.5% | 87.9% | 292 | 1.00 | - |

#### 6. Optional W7 Extensions
- **`HuggingFaceTB__SmolLM3-3B`**: DenseKVEvict(n=80: a=0.00:86.2%) | PrefillMLP(layer_-1, tau=0.850, Macro=66.8%, OAA250=69.6%, U0.05=63.4%, Tok=308, OrigPrecF1=86.0%->81.1%, ParaPrecF1=78.9%->81.7%) | Stage2SolveFirst(replayed=139, Macro=68.5%->67.0%, OAA250=72.0%->69.6%)
- **`Qwen__Qwen3.5-4B`**: DeltaNetBlend(n=336: a=0.00:93.2%, a=0.25:95.2%, a=0.50:95.2%, a=0.75:94.0%, a=1.00:91.7%) | DeltaNetConvKV(n=336: a=0.00:85.7%, a=0.25:88.1%, a=0.50:89.9%, a=0.75:91.7%, a=1.00:95.8%) | PrefillMLP(layer_-1, tau=0.525, Macro=84.1%, OAA250=84.9%, U0.05=77.8%, Tok=357, OrigPrecF1=86.0%->81.2%, ParaPrecF1=78.9%->82.6%) | Stage2SolveFirst(replayed=17, Macro=84.6%->84.3%, OAA250=84.9%->84.4%)
<!-- CBARR_RUN_END:20261119T105923Z -->

<!-- CBARR_RUN_START:20261119T114300Z -->
### Run `20261119T114300Z` — **FAILED (exit=1 at Contingency R6 Sampled (T=0.6, seed=42): Qwen/Qwen3.5-4B (merge))** (Updated 2026-11-19 15:08:57 UTC)
- **Code Version**: `v0.5.27` | **Git Commit**: `14be3cd` | **Results Release**: `v0.5.27-res.21` | **Last Loop**: `Contingency R6 Sampled (T=0.6, seed=42): Qwen/Qwen3.5-4B (merge)` | **Wall Time**: `3h 25m 46s` (12346.0s)
- **Hardware**: 8x NVIDIA L40 (44.4 GB, SM 8.9, CUDA 13.2, BF16=True) | PyTorch `2.15.0+cu132` | Transformers `5.23.0` | Kernels: `fla=True, causal_conv1d=True`
- **Command**: `scripts/run_gpu_phases_4_to_6.sh --part all --setup-mode auto --shards auto --auto-9b --with-w7 --auto-b512 --auto-r6 --commit --push --gpu-boost --hf-home <HOME>/maplecg_nfs/.cbarr_hf_cache`
- **Raw Log**: `outputs/logs/cbarr_20261119T114300Z.log`
- **Notes**: Traceback (most recent call last): | ValueError: Input X contains NaN. | [2026-11-19T15:08:46Z] === [Contingency R6 Sampled (T=0.6, seed=42): Qwen/Qwen3.5-4B (merge)] FAILED (exit=1, 16s) ===

#### 1. Training & Experiment Parameters
```json
{
  "auto_b512": "auto",
  "auto_commit": true,
  "auto_push": true,
  "b0_max_new_tokens": 1024,
  "b1b_max_new_tokens": 2048,
  "class_weight": "balanced",
  "code_version": "0.5.27",
  "device": "cuda",
  "do_sample": false,
  "dtype": "auto",
  "gate2_probe_tokens": 32,
  "git_branch": "",
  "git_rev": "14be3cd",
  "include_b512": true,
  "is_mock": false,
  "lam_utility": 0.05,
  "logreg_C": 1.0,
  "norm_tokens": 250.0,
  "presence_penalty": 0.0,
  "r6_seeds": [
    42,
    43,
    44
  ],
  "run_id": "20261119T114300Z",
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
    250,
    512
  ],
  "top_p": 1.0,
  "with_9b": true,
  "with_ifbench": true,
  "with_r6": true,
  "with_w7": true
}
```

#### 2. Stage Timings
| Stage | Status | Wall Time |
| :--- | :--- | :--- |
| Unpack Synced Worker Results | PASS | 1.0s (1.0s) |
| Gate 0: Preflight Tests | PASS | 6.0s (6.0s) |
| Step 0a: Hardware & Kernel Probe | PASS | 9.0s (9.0s) |
| Step 0a: Live GPU 12-Item Smoke (cached) | PASS | 5.0s (5.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (strict341) (cached) | PASS | 42.0s (42.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (Contingency R3 B=512) (cached) | PASS | 40.0s (40.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (diag491) (cached) | PASS | 42.0s (42.0s) |
| W7: DeltaNet Blend (Qwen/Qwen3.5-4B) (cached) | PASS | 5.0s (5.0s) |
| W7: DeltaNet Conv+KV Blend (Qwen/Qwen3.5-4B) (cached) | PASS | 5.0s (5.0s) |
| W7: Prefill MLP Probe (Qwen/Qwen3.5-4B) (cached) | PASS | 13.0s (13.0s) |
| W7: Stage-2 SolveFirst Replay (Qwen/Qwen3.5-4B) (cached) | PASS | 0.0s (0.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (strict341) (cached) | PASS | 41.0s (41.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (Contingency R3 B=512) (cached) | PASS | 41.0s (41.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (diag491) (cached) | PASS | 42.0s (42.0s) |
| W7: Dense KV Eviction (HuggingFaceTB/SmolLM3-3B) (cached) | PASS | 7.0s (7.0s) |
| W7: Prefill MLP Probe (HuggingFaceTB/SmolLM3-3B) (cached) | PASS | 13.0s (13.0s) |
| W7: Stage-2 SolveFirst Replay (HuggingFaceTB/SmolLM3-3B) (cached) | PASS | 0.0s (0.0s) |
| Phase 6 / W6: Qwen3.5-0.8B (strict341) (cached) | PASS | 41.0s (41.0s) |
| Phase 6 / W6: Qwen3.5-2B (strict341) (cached) | PASS | 41.0s (41.0s) |
| Phase 6 / W6: Qwen3.5-9B (strict341) (cached) | PASS | 41.0s (41.0s) |
| Appendix B: IFBench OOD (Qwen/Qwen3.5-4B) | PASS | 43m 01.0s (2581.0s) |
| Contingency R6 Sampled (T=0.6, seed=42): Qwen/Qwen3.5-4B | PASS | 2h 29m 13s (8953.0s) |
| Contingency R6 Sampled (T=0.6, seed=42): Qwen/Qwen3.5-4B (merge) | FAIL(exit=1) | 16.0s (16.0s) |

#### 3. Gate-3 Live 12-Item Smoke Check
| Model | Items | D13 <=250 | No Reminder Leak | SmolLM3 Frozen Date |
| :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-4B` | 12 | True | True | N/A |
| `HuggingFaceTB/SmolLM3-3B` | 12 | True | True | True |

#### 4. Calibration & Gate-2 Classifier Results (Per Model)
| Model | Code Ver | tau_prob | tau_h_high | Dev Esc (R1) | G1 Skip | Intercept | Coef (`H1, Hm, Hmax, I, sym`) | Train@0.5 (Acc/Prec/Rec/F1/AUC) | Val-All@tau (Acc/Prec/Rec/F1/AUC) | Val-G2@tau (F1/AUC) | Dev `B1b->M2` Macro (Tok) | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HuggingFaceTB/SmolLM3-3B` | `v0.5.27` (`1f57685-dirty`) | 0.400 | 3.144 | 46.7% (OK) | 16.7% | -0.37 | [0.26, 0.53, -0.09, -0.43, -1.45] | 75/28/71/40/0.806 | 48/16/100/28/0.698 | 36/0.806 (n=40) | 76.0%->74.0% (960->407t) | 0.1s |
| `Qwen/Qwen3.5-0.8B` | `v0.5.27` (`39b29e3`) | 0.525 | 3.719 | 31.7% (OK) | 16.7% | -0.79 | [-0.06, 0.14, 0.15, -0.21, 0.70] | 57/26/73/38/0.651 | 70/19/38/25/0.618 | 40/0.750 (n=40) | 4.0%->61.0% (2000->467t) | 0.1s |
| `Qwen/Qwen3.5-2B` | `v0.5.27` (`e7f34a0`) | 0.850 | 3.157 | 16.7% (WARN) | 16.7% | -2.95 | [-0.34, 0.06, 0.43, 2.19, 0.39] | 65/9/100/16/0.888 | 88/0/0/0/0.369 | 0/0.684 (n=40) | 13.0%->78.0% (1951->376t) | 0.1s |
| `Qwen/Qwen3.5-4B` | `v0.5.27` (`1f3f0e5-dirty`) | 0.650 | 2.537 | 20.0% (OK) | 16.7% | -0.64 | [-0.68, 0.60, 0.10, 0.96, -0.43] | 72/19/100/32/0.804 | 87/0/0/0/0.629 | 0/0.410 (n=40) | 38.0%->90.0% (1744->365t) | 0.1s |
| `Qwen/Qwen3.5-9B` | `v0.5.27` (`bd2f8d2`) | 0.475 | 2.662 | 43.3% (OK) | 16.7% | -0.31 | [0.30, 0.32, -0.02, 0.00, -1.64] | 57/7/100/13/0.750 | 53/10/75/18/0.656 | 21/0.640 (n=40) | 40.0%->83.0% (1662->351t) | 0.1s |

#### 5. Evaluation Matrix & Section 6 Diagnostics (Per Model x Pool)

- **`HuggingFaceTB/SmolLM3-3B` (`strict341`**, n_diag=341) — ver=`v0.5.27` (`1f57685-dirty`) | dtype=`torch.bfloat16` | wall=`26.9s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.400, esc=46.7%
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
| M2_B512 | 67.9% | 80.0% | 72.0% | 72.0% | 65.0% | 81.7% | 36.7% | 38.9% | 64.1% | 61.1% | 511 | 1.31 | 0.295 (raw) |
| **Oracle_UB** | 88.2% | 97.3% | 97.3% | 89.3% | 83.3% | 95.0% | 66.7% | 88.0% | 88.5% | 84.7% | 287 | 1.00 | - |

- **`HuggingFaceTB/SmolLM3-3B` (`diag491`**, n_diag=491) — ver=`v0.5.27` (`1f57685-dirty`) | dtype=`torch.bfloat16` | wall=`26.2s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.400, esc=46.7%
  - **4-Regime**: Helped=13.6%, Hurt=9.6%, Flip=23.1%, TiePass=60.0%, TieFail=16.9%
  - **Section 6 Causal (`|E_hurt|=80`)**: rho_rec=62.5%, rho_int=37.5% | A1_strip=100.0%, B2a_post=62.5%, B1_pre=58.8%, B2b_raw=33.8%, A2_swap=60.0%, B3_cut=62.5%
  - *Eval-450 table identical to `strict341` above (omitted duplicate table).*

- **`Qwen/Qwen3.5-0.8B` (`strict341`**, n_diag=341) — ver=`v0.5.27` (`39b29e3`) | dtype=`torch.bfloat16` | wall=`26.2s` (calib=0.1s, eval=0.4s, diag=0.1s) | tau=0.525, esc=31.7%
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

- **`Qwen/Qwen3.5-2B` (`strict341`**, n_diag=341) — ver=`v0.5.27` (`e7f34a0`) | dtype=`torch.bfloat16` | wall=`26.2s` (calib=0.1s, eval=0.4s, diag=0.1s) | tau=0.850, esc=16.7%
  - **4-Regime**: Helped=1.1%, Hurt=52.7%, Flip=53.8%, TiePass=17.6%, TieFail=28.7%
  - **Section 6 Causal (`|E_hurt|=215`)**: rho_rec=90.2%, rho_int=9.8% | A1_strip=100.0%, B2a_post=90.2%, B1_pre=87.0%, B2b_raw=50.2%, A2_swap=59.5%, B3_cut=81.9%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 67.7% | 81.3% | 68.0% | 70.0% | 65.0% | 78.3% | 43.3% | 70.2% | 70.2% | 63.6% | 333 | 1.00 | 1.34e-62 |
| B1a | 70.6% | 85.3% | 74.7% | 70.0% | 70.0% | 73.3% | 50.0% | 35.3% | 63.5% | 62.4% | 501 | 1.00 | 2.14e-66 |
| **B1b** | 13.7% | 24.0% | 14.7% | 33.3% | 3.3% | 3.3% | 3.3% | 0.0% | 9.5% | -18.9% | 1879 | 1.00 | - |
| B2a | 61.0% | 86.7% | 72.0% | 70.7% | 43.3% | 60.0% | 33.3% | 46.4% | 59.7% | 58.4% | 382 | 1.00 | 2.15e-55 |
| B2b | 57.2% | 69.3% | 57.3% | 63.3% | 55.0% | 61.7% | 36.7% | 57.6% | 59.1% | 47.2% | 651 | 1.20 | 7.13e-51 |
| M1a | 34.1% | 36.0% | 18.7% | 70.0% | 5.0% | 71.7% | 3.3% | 37.3% | 40.2% | 21.3% | 1078 | 1.00 | 8.63e-32 |
| M1b | 67.4% | 81.3% | 68.0% | 70.0% | 65.0% | 76.7% | 43.3% | 69.8% | 69.8% | 63.2% | 341 | 1.01 | 2.25e-62 |
| M1c | 49.8% | 80.0% | 65.3% | 70.0% | 5.0% | 75.0% | 3.3% | 57.6% | 57.9% | 45.4% | 655 | 1.00 | 9.98e-48 |
| M2_B64 | 67.4% | 81.3% | 68.0% | 70.0% | 65.0% | 76.7% | 43.3% | 70.0% | 69.6% | 62.8% | 359 | 1.00 | 1.95e-61 |
| M2_B128 | 70.2% | 81.3% | 68.0% | 70.0% | 70.0% | 78.3% | 53.3% | 71.6% | 70.7% | 64.0% | 377 | 1.00 | 3.39e-64 |
| M2_B256 | 70.2% | 81.3% | 68.0% | 70.0% | 75.0% | 76.7% | 50.0% | 71.8% | 70.0% | 63.8% | 399 | 1.00 | 1.9e-64 |
| M2_Pre | 68.8% | 81.3% | 68.0% | 70.0% | 63.3% | 80.0% | 50.0% | 70.7% | 69.6% | 62.7% | 399 | 1.00 | 4e-63 |
| **M2** | 69.6% | 81.3% | 68.0% | 70.0% | 70.0% | 78.3% | 50.0% | 71.3% | 70.2% | 63.7% | 382 | 1.00 | 5.98e-64 |
| M2_Prec (Aux.) | 69.6% | 81.3% | 68.0% | 70.0% | 70.0% | 75.0% | 53.3% | 71.1% | 69.4% | 62.9% | 413 | 1.00 | 1.3e-64 (raw) |
| **Oracle_UB** | 87.2% | 94.7% | 92.0% | 86.7% | 86.7% | 90.0% | 73.3% | 88.4% | 86.6% | 82.3% | 308 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`strict341`**, n_diag=341) — ver=`v0.5.27` (`1f3f0e5-dirty`) | dtype=`torch.bfloat16` | wall=`26.3s` (calib=0.1s, eval=0.5s, diag=0.1s) | tau=0.650, esc=20.0%
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
| M2_B512 | 86.1% | 92.0% | 90.7% | 82.0% | 86.7% | 81.7% | 83.3% | 65.6% | 80.7% | 76.9% | 445 | 1.04 | 3.05e-64 (raw) |
| **Oracle_UB** | 94.2% | 97.3% | 98.7% | 92.7% | 93.3% | 93.3% | 90.0% | 92.2% | 92.3% | 88.4% | 300 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`diag491`**, n_diag=491) — ver=`v0.5.27` (`1f3f0e5-dirty`) | dtype=`torch.bfloat16` | wall=`26.8s` (calib=0.1s, eval=0.5s, diag=0.1s) | tau=0.650, esc=20.0%
  - **4-Regime**: Helped=0.7%, Hurt=50.0%, Flip=50.7%, TiePass=34.7%, TieFail=14.7%
  - **Section 6 Causal (`|E_hurt|=336`)**: rho_rec=96.1%, rho_int=3.9% | A1_strip=100.0%, B2a_post=96.1%, B1_pre=95.5%, B2b_raw=49.4%, A2_swap=70.2%, B3_cut=86.3%
  - *Eval-450 table identical to `strict341` above (omitted duplicate table).*

- **`Qwen/Qwen3.5-9B` (`strict341`**, n_diag=341) — ver=`v0.5.27` (`bd2f8d2`) | dtype=`torch.bfloat16` | wall=`26.0s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.475, esc=43.3%
  - **4-Regime**: Helped=0.9%, Hurt=43.8%, Flip=44.7%, TiePass=43.1%, TieFail=12.2%
  - **Section 6 Causal (`|E_hurt|=226`)**: rho_rec=95.1%, rho_int=4.9% | A1_strip=100.0%, B2a_post=95.1%, B1_pre=93.4%, B2b_raw=47.8%, A2_swap=80.1%, B3_cut=92.5%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 87.5% | 92.0% | 93.3% | 81.3% | 88.3% | 86.7% | 83.3% | 86.9% | 86.9% | 80.7% | 310 | 1.00 | 5.03e-52 |
| B1a | 84.3% | 96.0% | 93.3% | 81.3% | 71.7% | 80.0% | 83.3% | 33.6% | 73.0% | 75.5% | 447 | 1.00 | 4.11e-49 |
| **B1b** | 38.3% | 62.7% | 52.0% | 52.0% | 23.3% | 26.7% | 13.3% | 0.0% | 22.2% | 10.6% | 1668 | 1.00 | - |
| B2a | 78.4% | 85.3% | 70.7% | 82.7% | 80.0% | 81.7% | 70.0% | 63.3% | 74.0% | 74.7% | 256 | 1.00 | 3.53e-32 |
| B2b | 70.1% | 81.3% | 76.0% | 66.7% | 68.3% | 65.0% | 63.3% | 51.6% | 60.6% | 52.5% | 895 | 1.42 | 7.17e-33 |
| M1a | 56.8% | 72.0% | 57.3% | 81.3% | 30.0% | 80.0% | 20.0% | 43.8% | 54.2% | 45.6% | 952 | 1.00 | 1.45e-26 |
| M1b | 54.4% | 86.7% | 89.3% | 57.3% | 33.3% | 40.0% | 20.0% | 40.4% | 49.3% | 36.4% | 1158 | 1.56 | 2.94e-19 |
| M1c | 58.8% | 85.3% | 86.7% | 57.3% | 25.0% | 81.7% | 16.7% | 44.2% | 53.2% | 41.9% | 1062 | 1.32 | 1.32e-23 |
| M2_B64 | 86.5% | 93.3% | 93.3% | 80.7% | 85.0% | 86.7% | 80.0% | 86.2% | 84.5% | 79.2% | 353 | 1.32 | 1.97e-49 |
| M2_B128 | 84.6% | 93.3% | 93.3% | 79.3% | 85.0% | 86.7% | 70.0% | 85.1% | 82.2% | 77.6% | 378 | 1.32 | 7.29e-50 |
| M2_B256 | 85.9% | 93.3% | 92.0% | 83.3% | 85.0% | 85.0% | 76.7% | 86.4% | 81.3% | 78.0% | 424 | 1.32 | 1.77e-51 |
| M2_Pre | 86.8% | 93.3% | 92.0% | 80.7% | 91.7% | 86.7% | 76.7% | 86.7% | 83.9% | 79.0% | 382 | 1.32 | 8.51e-51 |
| **M2** | 86.0% | 93.3% | 92.0% | 80.7% | 86.7% | 86.7% | 76.7% | 86.0% | 83.3% | 78.4% | 380 | 1.32 | 5.68e-50 |
| M2_Prec (Aux.) | 86.3% | 93.3% | 92.0% | 80.7% | 86.7% | 88.3% | 76.7% | 86.2% | 82.8% | 78.2% | 401 | 1.32 | 3.24e-51 (raw) |
| **Oracle_UB** | 93.9% | 98.7% | 96.0% | 90.7% | 96.7% | 91.7% | 90.0% | 93.8% | 92.5% | 87.9% | 292 | 1.00 | - |

#### 6. Optional W7 Extensions
- **`HuggingFaceTB__SmolLM3-3B`**: DenseKVEvict(n=80: a=0.00:86.2%) | PrefillMLP(layer_-1, tau=0.850, Macro=66.8%, OAA250=69.6%, U0.05=63.4%, Tok=308, OrigPrecF1=86.0%->81.1%, ParaPrecF1=78.9%->81.7%) | Stage2SolveFirst(replayed=139, Macro=68.5%->67.0%, OAA250=72.0%->69.6%)
- **`Qwen__Qwen3.5-4B`**: DeltaNetBlend(n=336: a=0.00:93.2%, a=0.25:95.2%, a=0.50:95.2%, a=0.75:94.0%, a=1.00:91.7%) | DeltaNetConvKV(n=336: a=0.00:85.7%, a=0.25:88.1%, a=0.50:89.9%, a=0.75:91.7%, a=1.00:95.8%) | PrefillMLP(layer_-1, tau=0.525, Macro=84.1%, OAA250=84.9%, U0.05=77.8%, Tok=357, OrigPrecF1=86.0%->81.2%, ParaPrecF1=78.9%->82.6%) | Stage2SolveFirst(replayed=17, Macro=84.6%->84.3%, OAA250=84.9%->84.4%) | IFBenchOOD(N=300: B0=28.7%/B1b=2.0%/M2=31.3%, OAA250=31.3%, Tok=443, p_vs_B1b=1.73e-24)
<!-- CBARR_RUN_END:20261119T114300Z -->

<!-- CBARR_RUN_START:20261119T222033Z_pr6 -->
### Run `20261119T222033Z_pr6` — **RUNNING (after GitHub Sync (script-part-r6))** (Updated 2026-11-20 03:27:19 UTC)
- **Code Version**: `v0.5.30` | **Git Commit**: `fc3235a` | **Results Release**: `v0.5.30-res.25` | **Last Loop**: `GitHub Sync (script-part-r6)` | **Wall Time**: `5h 06m 39s` (18399.0s)
- **Hardware**: 8x NVIDIA L40 (44.4 GB, SM 8.9, CUDA 13.2, BF16=True) | PyTorch `2.15.0+cu132` | Transformers `5.23.0` | Kernels: `fla=True, causal_conv1d=True`
- **Command**: `scripts/run_gpu_phases_4_to_6.sh --part r6 --shards auto --gpu-boost --hf-home <HOME>/maplecg_nfs/.cbarr_hf_cache --commit --push`
- **Raw Log**: `outputs/logs/cbarr_20261119T222033Z_pr6.log`

#### 1. Training & Experiment Parameters
```json
{
  "auto_b512": "auto",
  "auto_commit": true,
  "auto_push": true,
  "b0_max_new_tokens": 1024,
  "b1b_max_new_tokens": 2048,
  "class_weight": "balanced",
  "code_version": "0.5.30",
  "device": "cuda",
  "do_sample": false,
  "dtype": "auto",
  "gate2_probe_tokens": 32,
  "git_branch": "",
  "git_rev": "fc3235a",
  "include_b512": true,
  "is_mock": false,
  "lam_utility": 0.05,
  "logreg_C": 1.0,
  "norm_tokens": 250.0,
  "presence_penalty": 0.0,
  "r6_seeds": [
    42,
    43,
    44
  ],
  "run_id": "20261119T222033Z_pr6",
  "seed": 42,
  "setup_mode": "off",
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
    250,
    512
  ],
  "top_p": 1.0,
  "with_9b": false,
  "with_ifbench": false,
  "with_r6": false,
  "with_w7": false
}
```

#### 2. Stage Timings
| Stage | Status | Wall Time |
| :--- | :--- | :--- |
| Unpack Synced Worker Results | PASS | 1.0s (1.0s) |
| Gate 0: Preflight Tests | PASS | 6.0s (6.0s) |
| Contingency R6 Sampled (T=0.6, seed=42): Qwen/Qwen3.5-4B (cached) | PASS | 40.0s (40.0s) |
| Contingency R6 Sampled (T=0.6, seed=43): Qwen/Qwen3.5-4B | PASS | 2h 31m 53s (9113.0s) |
| Contingency R6 Sampled (T=0.6, seed=43): Qwen/Qwen3.5-4B (merge) | PASS | 51.0s (51.0s) |
| Contingency R6 Sampled (T=0.6, seed=44): Qwen/Qwen3.5-4B | PASS | 2h 28m 22s (8902.0s) |
| Contingency R6 Sampled (T=0.6, seed=44): Qwen/Qwen3.5-4B (merge) | PASS | 52.0s (52.0s) |
| Phase 7 / W8: Analysis, Tables & Figures | PASS | 1m 16.0s (76.0s) |
| Submission Audit: Double-Blind Anonymity | PASS | 0.0s (0.0s) |
| GitHub Sync (script-part-r6) | PASS | 9.0s (9.0s) |

#### 4. Calibration & Gate-2 Classifier Results (Per Model)
| Model | Code Ver | tau_prob | tau_h_high | Dev Esc (R1) | G1 Skip | Intercept | Coef (`H1, Hm, Hmax, I, sym`) | Train@0.5 (Acc/Prec/Rec/F1/AUC) | Val-All@tau (Acc/Prec/Rec/F1/AUC) | Val-G2@tau (F1/AUC) | Dev `B1b->M2` Macro (Tok) | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HuggingFaceTB/SmolLM3-3B` | `v0.5.27` (`1f57685-dirty`) | 0.400 | 3.144 | 46.7% (OK) | 16.7% | -0.37 | [0.26, 0.53, -0.09, -0.43, -1.45] | 75/28/71/40/0.806 | 48/16/100/28/0.698 | 36/0.806 (n=40) | 76.0%->74.0% (960->407t) | 0.1s |
| `Qwen/Qwen3.5-0.8B` | `v0.5.27` (`39b29e3`) | 0.525 | 3.719 | 31.7% (OK) | 16.7% | -0.79 | [-0.06, 0.14, 0.15, -0.21, 0.70] | 57/26/73/38/0.651 | 70/19/38/25/0.618 | 40/0.750 (n=40) | 4.0%->61.0% (2000->467t) | 0.1s |
| `Qwen/Qwen3.5-2B` | `v0.5.27` (`e7f34a0`) | 0.850 | 3.157 | 16.7% (WARN) | 16.7% | -2.95 | [-0.34, 0.06, 0.43, 2.19, 0.39] | 65/9/100/16/0.888 | 88/0/0/0/0.369 | 0/0.684 (n=40) | 13.0%->78.0% (1951->376t) | 0.1s |
| `Qwen/Qwen3.5-4B` | `v0.5.27` (`1f3f0e5-dirty`) | 0.650 | 2.537 | 20.0% (OK) | 16.7% | -0.64 | [-0.68, 0.60, 0.10, 0.96, -0.43] | 72/19/100/32/0.804 | 87/0/0/0/0.629 | 0/0.410 (n=40) | 38.0%->90.0% (1744->365t) | 0.1s |
| `Qwen/Qwen3.5-4B` | `v0.5.30` (`af6a598-dirty`) | 0.625 | 0.850 | 50.0% (OK) | 16.7% | -0.29 | [0.00, 0.00, 0.00, 0.86, -2.01] | 70/15/75/25/0.761 | 63/17/67/27/0.710 | 33/0.778 (n=40) | 50.0%->84.0% (1552->360t) | 0.1s |
| `Qwen/Qwen3.5-4B` | `v0.5.30` (`8881e83`) | 0.425 | 0.850 | 83.3% (OK) | 16.7% | 0.26 | [0.00, 0.00, 0.00, -0.49, 0.06] | 63/9/50/15/0.580 | 12/12/100/21/0.592 | 10/0.237 (n=40) | 45.0%->89.0% (1701->439t) | 0.1s |
| `Qwen/Qwen3.5-4B` | `v0.5.30` (`779927c`) | 0.525 | 0.850 | 50.0% (OK) | 16.7% | 0.04 | [0.00, 0.00, 0.00, 0.11, -0.42] | 33/9/80/17/0.553 | 63/13/60/21/0.624 | 25/0.639 (n=40) | 42.0%->82.0% (1619->400t) | 0.1s |
| `Qwen/Qwen3.5-9B` | `v0.5.27` (`bd2f8d2`) | 0.475 | 2.662 | 43.3% (OK) | 16.7% | -0.31 | [0.30, 0.32, -0.02, 0.00, -1.64] | 57/7/100/13/0.750 | 53/10/75/18/0.656 | 21/0.640 (n=40) | 40.0%->83.0% (1662->351t) | 0.1s |

#### 5. Evaluation Matrix & Section 6 Diagnostics (Per Model x Pool)

- **`HuggingFaceTB/SmolLM3-3B` (`strict341`**, n_diag=341) — ver=`v0.5.27` (`1f57685-dirty`) | dtype=`torch.bfloat16` | wall=`26.9s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.400, esc=46.7%
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
| M2_B512 | 67.9% | 80.0% | 72.0% | 72.0% | 65.0% | 81.7% | 36.7% | 38.9% | 64.1% | 61.1% | 511 | 1.31 | 0.295 (raw) |
| **Oracle_UB** | 88.2% | 97.3% | 97.3% | 89.3% | 83.3% | 95.0% | 66.7% | 88.0% | 88.5% | 84.7% | 287 | 1.00 | - |

- **`HuggingFaceTB/SmolLM3-3B` (`diag491`**, n_diag=491) — ver=`v0.5.27` (`1f57685-dirty`) | dtype=`torch.bfloat16` | wall=`26.2s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.400, esc=46.7%
  - **4-Regime**: Helped=13.6%, Hurt=9.6%, Flip=23.1%, TiePass=60.0%, TieFail=16.9%
  - **Section 6 Causal (`|E_hurt|=80`)**: rho_rec=62.5%, rho_int=37.5% | A1_strip=100.0%, B2a_post=62.5%, B1_pre=58.8%, B2b_raw=33.8%, A2_swap=60.0%, B3_cut=62.5%
  - *Eval-450 table identical to `strict341` above (omitted duplicate table).*

- **`Qwen/Qwen3.5-0.8B` (`strict341`**, n_diag=341) — ver=`v0.5.27` (`39b29e3`) | dtype=`torch.bfloat16` | wall=`26.2s` (calib=0.1s, eval=0.4s, diag=0.1s) | tau=0.525, esc=31.7%
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

- **`Qwen/Qwen3.5-2B` (`strict341`**, n_diag=341) — ver=`v0.5.27` (`e7f34a0`) | dtype=`torch.bfloat16` | wall=`26.2s` (calib=0.1s, eval=0.4s, diag=0.1s) | tau=0.850, esc=16.7%
  - **4-Regime**: Helped=1.1%, Hurt=52.7%, Flip=53.8%, TiePass=17.6%, TieFail=28.7%
  - **Section 6 Causal (`|E_hurt|=215`)**: rho_rec=90.2%, rho_int=9.8% | A1_strip=100.0%, B2a_post=90.2%, B1_pre=87.0%, B2b_raw=50.2%, A2_swap=59.5%, B3_cut=81.9%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 67.7% | 81.3% | 68.0% | 70.0% | 65.0% | 78.3% | 43.3% | 70.2% | 70.2% | 63.6% | 333 | 1.00 | 1.34e-62 |
| B1a | 70.6% | 85.3% | 74.7% | 70.0% | 70.0% | 73.3% | 50.0% | 35.3% | 63.5% | 62.4% | 501 | 1.00 | 2.14e-66 |
| **B1b** | 13.7% | 24.0% | 14.7% | 33.3% | 3.3% | 3.3% | 3.3% | 0.0% | 9.5% | -18.9% | 1879 | 1.00 | - |
| B2a | 61.0% | 86.7% | 72.0% | 70.7% | 43.3% | 60.0% | 33.3% | 46.4% | 59.7% | 58.4% | 382 | 1.00 | 2.15e-55 |
| B2b | 57.2% | 69.3% | 57.3% | 63.3% | 55.0% | 61.7% | 36.7% | 57.6% | 59.1% | 47.2% | 651 | 1.20 | 7.13e-51 |
| M1a | 34.1% | 36.0% | 18.7% | 70.0% | 5.0% | 71.7% | 3.3% | 37.3% | 40.2% | 21.3% | 1078 | 1.00 | 8.63e-32 |
| M1b | 67.4% | 81.3% | 68.0% | 70.0% | 65.0% | 76.7% | 43.3% | 69.8% | 69.8% | 63.2% | 341 | 1.01 | 2.25e-62 |
| M1c | 49.8% | 80.0% | 65.3% | 70.0% | 5.0% | 75.0% | 3.3% | 57.6% | 57.9% | 45.4% | 655 | 1.00 | 9.98e-48 |
| M2_B64 | 67.4% | 81.3% | 68.0% | 70.0% | 65.0% | 76.7% | 43.3% | 70.0% | 69.6% | 62.8% | 359 | 1.00 | 1.95e-61 |
| M2_B128 | 70.2% | 81.3% | 68.0% | 70.0% | 70.0% | 78.3% | 53.3% | 71.6% | 70.7% | 64.0% | 377 | 1.00 | 3.39e-64 |
| M2_B256 | 70.2% | 81.3% | 68.0% | 70.0% | 75.0% | 76.7% | 50.0% | 71.8% | 70.0% | 63.8% | 399 | 1.00 | 1.9e-64 |
| M2_Pre | 68.8% | 81.3% | 68.0% | 70.0% | 63.3% | 80.0% | 50.0% | 70.7% | 69.6% | 62.7% | 399 | 1.00 | 4e-63 |
| **M2** | 69.6% | 81.3% | 68.0% | 70.0% | 70.0% | 78.3% | 50.0% | 71.3% | 70.2% | 63.7% | 382 | 1.00 | 5.98e-64 |
| M2_Prec (Aux.) | 69.6% | 81.3% | 68.0% | 70.0% | 70.0% | 75.0% | 53.3% | 71.1% | 69.4% | 62.9% | 413 | 1.00 | 1.3e-64 (raw) |
| **Oracle_UB** | 87.2% | 94.7% | 92.0% | 86.7% | 86.7% | 90.0% | 73.3% | 88.4% | 86.6% | 82.3% | 308 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`strict341`**, n_diag=341) — ver=`v0.5.27` (`1f3f0e5-dirty`) | dtype=`torch.bfloat16` | wall=`26.3s` (calib=0.1s, eval=0.5s, diag=0.1s) | tau=0.650, esc=20.0%
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
| M2_B512 | 86.1% | 92.0% | 90.7% | 82.0% | 86.7% | 81.7% | 83.3% | 65.6% | 80.7% | 76.9% | 445 | 1.04 | 3.05e-64 (raw) |
| **Oracle_UB** | 94.2% | 97.3% | 98.7% | 92.7% | 93.3% | 93.3% | 90.0% | 92.2% | 92.3% | 88.4% | 300 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`diag491`**, n_diag=491) — ver=`v0.5.27` (`1f3f0e5-dirty`) | dtype=`torch.bfloat16` | wall=`26.8s` (calib=0.1s, eval=0.5s, diag=0.1s) | tau=0.650, esc=20.0%
  - **4-Regime**: Helped=0.7%, Hurt=50.0%, Flip=50.7%, TiePass=34.7%, TieFail=14.7%
  - **Section 6 Causal (`|E_hurt|=336`)**: rho_rec=96.1%, rho_int=3.9% | A1_strip=100.0%, B2a_post=96.1%, B1_pre=95.5%, B2b_raw=49.4%, A2_swap=70.2%, B3_cut=86.3%
  - *Eval-450 table identical to `strict341` above (omitted duplicate table).*

- **`Qwen/Qwen3.5-4B` (`strict341`**, n_diag=341) — ver=`v0.5.30` (`af6a598-dirty`) | dtype=`torch.bfloat16` | sample(T=0.6, top_p=0.95, pp=1.5, seed=42) | wall=`25.1s` (calib=0.1s, eval=0.4s, diag=0.1s) | tau=0.625, esc=50.0%
  - **4-Regime**: Helped=0.7%, Hurt=40.4%, Flip=41.1%, TiePass=44.4%, TieFail=14.4%
  - **Section 6 Causal (`|E_hurt|=216`)**: rho_rec=90.7%, rho_int=7.9% | A1_strip=100.0%, B2a_post=90.7%, B1_pre=92.1%, B2b_raw=45.4%, A2_swap=60.6%, B3_cut=86.1%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 84.4% | 94.7% | 90.7% | 81.3% | 81.7% | 81.7% | 76.7% | 84.9% | 84.9% | 78.7% | 308 | 1.00 | 5.17e-49 |
| B1a | 81.7% | 93.3% | 93.3% | 82.0% | 66.7% | 78.3% | 76.7% | 39.1% | 72.2% | 75.3% | 378 | 1.00 | 6.43e-43 |
| **B1b** | 38.5% | 66.7% | 48.0% | 54.7% | 25.0% | 30.0% | 6.7% | 0.0% | 22.0% | 11.6% | 1675 | 1.00 | - |
| B2a | 82.7% | 96.0% | 92.0% | 81.3% | 76.7% | 76.7% | 73.3% | 49.6% | 75.0% | 77.4% | 317 | 1.00 | 1.26e-45 |
| B2b | 64.3% | 78.7% | 65.3% | 68.7% | 58.3% | 55.0% | 60.0% | 44.4% | 55.0% | 46.0% | 999 | 1.50 | 3.03e-28 |
| M1a | 52.9% | 74.7% | 48.0% | 81.3% | 30.0% | 76.7% | 6.7% | 43.1% | 53.1% | 43.2% | 950 | 1.00 | 5.5e-21 |
| M1b | 72.6% | 86.7% | 90.7% | 54.7% | 71.7% | 71.7% | 60.0% | 48.0% | 58.5% | 52.5% | 921 | 1.45 | 2.86e-33 |
| M1c | 57.4% | 86.7% | 88.0% | 54.7% | 30.0% | 78.3% | 6.7% | 36.7% | 48.4% | 39.4% | 1142 | 1.39 | 5.5e-21 |
| M2_B64 | 83.2% | 93.3% | 89.3% | 83.3% | 85.0% | 81.7% | 66.7% | 84.9% | 82.9% | 77.8% | 352 | 1.39 | 4.1e-46 |
| M2_B128 | 80.3% | 93.3% | 89.3% | 76.0% | 76.7% | 83.3% | 63.3% | 81.3% | 78.1% | 73.4% | 397 | 1.39 | 7.92e-42 |
| M2_B256 | 83.4% | 93.3% | 90.7% | 81.3% | 85.0% | 83.3% | 66.7% | 84.7% | 78.8% | 75.6% | 454 | 1.39 | 1.11e-47 |
| M2_Pre | 80.2% | 93.3% | 89.3% | 83.3% | 70.0% | 81.7% | 63.3% | 82.7% | 79.7% | 75.2% | 375 | 1.39 | 2.72e-44 |
| **M2** | 82.9% | 93.3% | 90.7% | 83.3% | 76.7% | 83.3% | 70.0% | 84.4% | 81.3% | 77.0% | 371 | 1.39 | 1.26e-45 |
| M2_Prec (Aux.) | 84.0% | 93.3% | 90.7% | 83.3% | 76.7% | 86.7% | 73.3% | 85.1% | 81.3% | 77.3% | 391 | 1.39 | 2.12e-47 (raw) |
| **Oracle_UB** | 94.5% | 97.3% | 97.3% | 90.7% | 93.3% | 91.7% | 96.7% | 93.8% | 92.0% | 88.1% | 283 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`strict341`**, n_diag=341) — ver=`v0.5.30` (`8881e83`) | dtype=`torch.bfloat16` | sample(T=0.6, top_p=0.95, pp=1.5, seed=43) | wall=`29.9s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.425, esc=83.3%
  - **4-Regime**: Helped=0.9%, Hurt=36.9%, Flip=37.8%, TiePass=46.9%, TieFail=15.3%
  - **Section 6 Causal (`|E_hurt|=195`)**: rho_rec=92.8%, rho_int=7.2% | A1_strip=100.0%, B2a_post=92.8%, B1_pre=91.8%, B2b_raw=47.2%, A2_swap=67.7%, B3_cut=90.3%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 81.8% | 92.0% | 92.0% | 82.0% | 76.7% | 85.0% | 63.3% | 83.8% | 83.8% | 77.5% | 313 | 1.00 | 5.52e-43 |
| B1a | 78.6% | 93.3% | 93.3% | 80.0% | 65.0% | 76.7% | 63.3% | 37.1% | 70.1% | 72.9% | 401 | 1.00 | 9.1e-35 |
| **B1b** | 42.9% | 73.3% | 56.0% | 53.3% | 21.7% | 30.0% | 23.3% | 0.0% | 23.3% | 14.5% | 1662 | 1.00 | - |
| B2a | 81.1% | 96.0% | 94.7% | 80.7% | 73.3% | 68.3% | 73.3% | 46.2% | 73.4% | 76.0% | 322 | 1.00 | 5.22e-36 |
| B2b | 48.4% | 77.3% | 61.3% | 58.7% | 26.7% | 40.0% | 26.7% | 13.1% | 32.2% | 24.3% | 1451 | 1.83 | 3.25e-06 |
| M1a | 58.8% | 78.7% | 58.7% | 82.0% | 26.7% | 80.0% | 26.7% | 43.8% | 54.6% | 47.2% | 951 | 1.00 | 1.78e-23 |
| M1b | 42.9% | 73.3% | 56.0% | 53.3% | 21.7% | 30.0% | 23.3% | 0.0% | 22.6% | 13.9% | 1693 | 2.00 | 1 |
| M1c | 51.3% | 73.3% | 56.0% | 53.3% | 21.7% | 80.0% | 23.3% | 10.0% | 31.3% | 24.7% | 1486 | 1.68 | 5.59e-09 |
| M2_B64 | 79.5% | 84.0% | 85.3% | 76.0% | 76.7% | 85.0% | 70.0% | 79.8% | 76.7% | 71.9% | 396 | 1.68 | 2.42e-31 |
| M2_B128 | 81.0% | 92.0% | 86.7% | 80.7% | 75.0% | 85.0% | 66.7% | 82.4% | 77.0% | 73.8% | 435 | 1.68 | 1.34e-36 |
| M2_B256 | 83.1% | 88.0% | 89.3% | 84.7% | 78.3% | 85.0% | 73.3% | 84.4% | 75.4% | 74.3% | 510 | 1.68 | 1.84e-40 |
| M2_Pre | 79.2% | 88.0% | 89.3% | 76.0% | 73.3% | 85.0% | 63.3% | 80.2% | 74.2% | 71.2% | 451 | 1.68 | 2.96e-32 |
| **M2** | 79.4% | 88.0% | 89.3% | 76.0% | 75.0% | 85.0% | 63.3% | 80.4% | 74.4% | 71.4% | 454 | 1.68 | 1.5e-31 |
| M2_Prec (Aux.) | 80.0% | 88.0% | 89.3% | 76.0% | 75.0% | 88.3% | 63.3% | 80.9% | 74.2% | 71.4% | 475 | 1.68 | 7.45e-33 (raw) |
| **Oracle_UB** | 93.3% | 98.7% | 96.0% | 88.7% | 95.0% | 91.7% | 90.0% | 92.9% | 91.0% | 87.1% | 288 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`strict341`**, n_diag=341) — ver=`v0.5.30` (`779927c`) | dtype=`torch.bfloat16` | sample(T=0.6, top_p=0.95, pp=1.5, seed=44) | wall=`29.4s` (calib=0.1s, eval=0.5s, diag=0.1s) | tau=0.525, esc=50.0%
  - **4-Regime**: Helped=2.4%, Hurt=37.8%, Flip=40.2%, TiePass=45.1%, TieFail=14.7%
  - **Section 6 Causal (`|E_hurt|=205`)**: rho_rec=92.7%, rho_int=7.3% | A1_strip=100.0%, B2a_post=92.7%, B1_pre=92.2%, B2b_raw=46.8%, A2_swap=62.9%, B3_cut=87.3%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 82.2% | 93.3% | 88.0% | 78.7% | 81.7% | 81.7% | 70.0% | 82.9% | 82.9% | 76.5% | 321 | 1.00 | 5.25e-37 |
| B1a | 79.8% | 96.0% | 96.0% | 80.0% | 65.0% | 78.3% | 63.3% | 39.1% | 72.0% | 74.4% | 381 | 1.00 | 1.73e-38 |
| **B1b** | 41.0% | 69.3% | 60.0% | 53.3% | 31.7% | 28.3% | 3.3% | 0.2% | 23.0% | 14.4% | 1660 | 1.00 | - |
| B2a | 81.8% | 94.7% | 96.0% | 80.0% | 68.3% | 78.3% | 73.3% | 49.8% | 73.6% | 76.6% | 315 | 1.00 | 3.37e-38 |
| B2b | 58.4% | 77.3% | 66.7% | 63.3% | 58.3% | 55.0% | 30.0% | 37.6% | 49.3% | 40.8% | 1071 | 1.54 | 2.58e-16 |
| M1a | 57.1% | 77.3% | 60.0% | 78.7% | 38.3% | 75.0% | 13.3% | 42.4% | 53.2% | 46.0% | 956 | 1.00 | 1.73e-18 |
| M1b | 69.4% | 85.3% | 88.0% | 53.3% | 73.3% | 73.3% | 43.3% | 46.9% | 57.9% | 50.9% | 909 | 1.45 | 1.12e-22 |
| M1c | 58.4% | 85.3% | 86.7% | 53.3% | 38.3% | 76.7% | 10.0% | 36.2% | 48.7% | 39.9% | 1129 | 1.39 | 2.8e-16 |
| M2_B64 | 81.3% | 92.0% | 88.0% | 78.0% | 81.7% | 81.7% | 66.7% | 82.2% | 80.3% | 74.4% | 391 | 1.39 | 3.96e-38 |
| M2_B128 | 84.1% | 92.0% | 88.0% | 81.3% | 91.7% | 81.7% | 70.0% | 84.9% | 81.4% | 76.5% | 417 | 1.39 | 4.76e-42 |
| M2_B256 | 84.9% | 92.0% | 88.0% | 79.3% | 88.3% | 81.7% | 80.0% | 84.4% | 78.6% | 75.3% | 455 | 1.39 | 8.73e-41 |
| M2_Pre | 80.2% | 92.0% | 88.0% | 78.0% | 76.7% | 80.0% | 66.7% | 81.3% | 78.4% | 73.1% | 412 | 1.39 | 1.86e-36 |
| **M2** | 84.7% | 92.0% | 88.0% | 78.0% | 91.7% | 81.7% | 76.7% | 84.2% | 81.0% | 76.0% | 412 | 1.39 | 3.1e-41 |
| M2_Prec (Aux.) | 86.1% | 92.0% | 88.0% | 78.0% | 91.7% | 90.0% | 76.7% | 85.3% | 81.5% | 76.7% | 431 | 1.39 | 2e-45 (raw) |
| **Oracle_UB** | 92.7% | 97.3% | 96.0% | 89.3% | 95.0% | 91.7% | 86.7% | 92.7% | 90.8% | 86.7% | 296 | 1.00 | - |

- **`Qwen/Qwen3.5-9B` (`strict341`**, n_diag=341) — ver=`v0.5.27` (`bd2f8d2`) | dtype=`torch.bfloat16` | wall=`26.0s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.475, esc=43.3%
  - **4-Regime**: Helped=0.9%, Hurt=43.8%, Flip=44.7%, TiePass=43.1%, TieFail=12.2%
  - **Section 6 Causal (`|E_hurt|=226`)**: rho_rec=95.1%, rho_int=4.9% | A1_strip=100.0%, B2a_post=95.1%, B1_pre=93.4%, B2b_raw=47.8%, A2_swap=80.1%, B3_cut=92.5%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 87.5% | 92.0% | 93.3% | 81.3% | 88.3% | 86.7% | 83.3% | 86.9% | 86.9% | 80.7% | 310 | 1.00 | 5.03e-52 |
| B1a | 84.3% | 96.0% | 93.3% | 81.3% | 71.7% | 80.0% | 83.3% | 33.6% | 73.0% | 75.5% | 447 | 1.00 | 4.11e-49 |
| **B1b** | 38.3% | 62.7% | 52.0% | 52.0% | 23.3% | 26.7% | 13.3% | 0.0% | 22.2% | 10.6% | 1668 | 1.00 | - |
| B2a | 78.4% | 85.3% | 70.7% | 82.7% | 80.0% | 81.7% | 70.0% | 63.3% | 74.0% | 74.7% | 256 | 1.00 | 3.53e-32 |
| B2b | 70.1% | 81.3% | 76.0% | 66.7% | 68.3% | 65.0% | 63.3% | 51.6% | 60.6% | 52.5% | 895 | 1.42 | 7.17e-33 |
| M1a | 56.8% | 72.0% | 57.3% | 81.3% | 30.0% | 80.0% | 20.0% | 43.8% | 54.2% | 45.6% | 952 | 1.00 | 1.45e-26 |
| M1b | 54.4% | 86.7% | 89.3% | 57.3% | 33.3% | 40.0% | 20.0% | 40.4% | 49.3% | 36.4% | 1158 | 1.56 | 2.94e-19 |
| M1c | 58.8% | 85.3% | 86.7% | 57.3% | 25.0% | 81.7% | 16.7% | 44.2% | 53.2% | 41.9% | 1062 | 1.32 | 1.32e-23 |
| M2_B64 | 86.5% | 93.3% | 93.3% | 80.7% | 85.0% | 86.7% | 80.0% | 86.2% | 84.5% | 79.2% | 353 | 1.32 | 1.97e-49 |
| M2_B128 | 84.6% | 93.3% | 93.3% | 79.3% | 85.0% | 86.7% | 70.0% | 85.1% | 82.2% | 77.6% | 378 | 1.32 | 7.29e-50 |
| M2_B256 | 85.9% | 93.3% | 92.0% | 83.3% | 85.0% | 85.0% | 76.7% | 86.4% | 81.3% | 78.0% | 424 | 1.32 | 1.77e-51 |
| M2_Pre | 86.8% | 93.3% | 92.0% | 80.7% | 91.7% | 86.7% | 76.7% | 86.7% | 83.9% | 79.0% | 382 | 1.32 | 8.51e-51 |
| **M2** | 86.0% | 93.3% | 92.0% | 80.7% | 86.7% | 86.7% | 76.7% | 86.0% | 83.3% | 78.4% | 380 | 1.32 | 5.68e-50 |
| M2_Prec (Aux.) | 86.3% | 93.3% | 92.0% | 80.7% | 86.7% | 88.3% | 76.7% | 86.2% | 82.8% | 78.2% | 401 | 1.32 | 3.24e-51 (raw) |
| **Oracle_UB** | 93.9% | 98.7% | 96.0% | 90.7% | 96.7% | 91.7% | 90.0% | 93.8% | 92.5% | 87.9% | 292 | 1.00 | - |

#### 6. Optional W7 Extensions
- **`HuggingFaceTB__SmolLM3-3B`**: DenseKVEvict(n=80: a=0.00:86.2%) | PrefillMLP(layer_-1, tau=0.850, Macro=66.8%, OAA250=69.6%, U0.05=63.4%, Tok=308, OrigPrecF1=86.0%->81.1%, ParaPrecF1=78.9%->81.7%) | Stage2SolveFirst(replayed=139, Macro=68.5%->67.0%, OAA250=72.0%->69.6%)
- **`Qwen__Qwen3.5-4B`**: DeltaNetBlend(n=336: a=0.00:93.2%, a=0.25:95.2%, a=0.50:95.2%, a=0.75:94.0%, a=1.00:91.7%) | DeltaNetConvKV(n=336: a=0.00:85.7%, a=0.25:88.1%, a=0.50:89.9%, a=0.75:91.7%, a=1.00:95.8%) | PrefillMLP(layer_-1, tau=0.525, Macro=84.1%, OAA250=84.9%, U0.05=77.8%, Tok=357, OrigPrecF1=86.0%->81.2%, ParaPrecF1=78.9%->82.6%) | Stage2SolveFirst(replayed=17, Macro=84.6%->84.3%, OAA250=84.9%->84.4%) | IFBenchOOD(N=300: B0=28.7%/B1b=2.0%/M2=31.3%, OAA250=31.3%, Tok=443, p_vs_B1b=1.73e-24)

#### 7. Phase 7 Statistical Audit & Gate-4 Summary
- **Version Mismatch Alert**: analyzer=`v0.5.30` | versions seen across models=`['0.5.27']` | mismatched=`['HuggingFaceTB/SmolLM3-3B', 'Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-2B', 'Qwen/Qwen3.5-4B', 'Qwen/Qwen3.5-9B']`
| Model | Version | Clean-446 (max delta) | M2 Macro | B1b Macro | M2 OAA_250 | Delta OAA_250 [95% CI] | Tok Red | p_holm vs B1b | p(M2 vs B0) | p(M2 vs Pre) | Gate 4 | R2 (rec / int) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HuggingFaceTB/SmolLM3-3B` | `v0.5.27+m4` | 448 (0.5 pp) | 68.5% | 69.4% | 72.0% | +68.4 [64.0, 72.7] pp | 62.0% | 1 | 0.1524 | 0.7539 | CHECK | 62.5% / 37.5% |
| `Qwen/Qwen3.5-0.8B` | `v0.5.27+m4` | 448 (0.2 pp) | 53.6% | 2.7% | 56.2% | +56.2 [51.6, 60.9] pp | 78.2% | 7.792e-67 | 1 | 0.2295 | PASS | 76.1% / 23.9% |
| `Qwen/Qwen3.5-2B` | `v0.5.27+m4` | 448 (0.1 pp) | 69.6% | 13.7% | 71.3% | +71.3 [67.1, 75.6] pp | 79.7% | 5.977e-64 | 0.3593 | 0.6072 | PASS | 90.2% / 9.8% |
| `Qwen/Qwen3.5-4B` | `v0.5.27+m6` | 448 (0.4 pp) | 84.6% | 30.2% | 84.9% | +84.7 [81.3, 88.0] pp | 79.2% | 5.104e-62 | 1 | 0.424 | PASS | 96.1% / 3.9% |
| `Qwen/Qwen3.5-9B` | `v0.5.27+m3` | 448 (0.4 pp) | 86.0% | 38.3% | 86.0% | +86.0 [82.7, 89.1] pp | 77.2% | 5.684e-50 | 0.6177 | 0.5488 | PASS | 95.1% / 4.9% |

#### 8. Confirmatory Audit & Appendix B Summary
- **`HuggingFaceTB/SmolLM3-3B` (Confirmatory 4-pair Holm)**: M2_vs_B1b: p_holm4=0.961 (midp=0.817), M2_vs_B0: p_holm4=0.61 (midp=0.476), M2_vs_B2a: p_holm4=0.961 (midp=0.817), M2_vs_M1c: p_holm4=0.961 (midp=0.817)
- **`HuggingFaceTB/SmolLM3-3B` (Conformal & Oracle Audit)**: DevDeg=11.7%, CRC<=13.1%, Hoef<=34.7%, EvalDeg(vs B1b)=10.44% (47/450), EvalDeg(vs B0)=4.22% (19/450), OracleRet(Macro/U0.05)=0.0%/0.0%
- **`HuggingFaceTB/SmolLM3-3B` (Probe Horizons)**: K=1:68.4%(366t), K=4:68.3%(358t), K=8:67.2%(335t), K=16:67.2%(331t), K=32:68.5%(371t)
- **`HuggingFaceTB/SmolLM3-3B` (Gate-2 L2 Sensitivity)**: C=0.1:68.5%(esc=50%), C=0.25:68.5%(esc=47%), C=0.5:68.7%(esc=47%), C=1:68.5%(esc=48%)
- **`Qwen/Qwen3.5-0.8B` (Confirmatory 4-pair Holm)**: M2_vs_B1b: p_holm4=2.6e-67 (midp=1.31e-67), M2_vs_B0: p_holm4=1 (midp=1), M2_vs_B2a: p_holm4=4.01e-06 (midp=2.89e-06), M2_vs_M1c: p_holm4=6.35e-22 (midp=3.18e-22)
- **`Qwen/Qwen3.5-0.8B` (Conformal & Oracle Audit)**: DevDeg=0.0%, CRC<=1.6%, Hoef<=23.0%, EvalDeg(vs B1b)=0.44% (2/450), EvalDeg(vs B0)=5.11% (23/450), OracleRet(Macro/U0.05)=0.0%/0.0%
- **`Qwen/Qwen3.5-0.8B` (Probe Horizons)**: K=1:55.4%(429t), K=4:54.1%(439t), K=8:54.9%(435t), K=16:55.4%(428t), K=32:53.6%(436t)
- **`Qwen/Qwen3.5-0.8B` (Gate-2 L2 Sensitivity)**: C=0.1:54.3%(esc=17%), C=0.25:53.6%(esc=32%), C=0.5:53.6%(esc=25%), C=1:53.6%(esc=28%)
- **`Qwen/Qwen3.5-2B` (Confirmatory 4-pair Holm)**: M2_vs_B1b: p_holm4=2.66e-64 (midp=1.36e-64), M2_vs_B0: p_holm4=0.359 (midp=0.263), M2_vs_B2a: p_holm4=0.05 (midp=0.0398), M2_vs_M1c: p_holm4=2.08e-17 (midp=1.04e-17)
- **`Qwen/Qwen3.5-2B` (Conformal & Oracle Audit)**: DevDeg=0.0%, CRC<=1.6%, Hoef<=23.0%, EvalDeg(vs B1b)=1.11% (5/450), EvalDeg(vs B0)=1.56% (7/450), OracleRet(Macro/U0.05)=0.0%/0.0%
- **`Qwen/Qwen3.5-2B` (Probe Horizons)**: K=1:69.8%(410t), K=4:69.6%(381t), K=8:69.6%(381t), K=16:70.3%(403t), K=32:69.6%(382t)
- **`Qwen/Qwen3.5-2B` (Gate-2 L2 Sensitivity)**: C=0.1:70.5%(esc=28%), C=0.25:69.6%(esc=17%), C=0.5:70.9%(esc=32%), C=1:71.3%(esc=43%)
- **`Qwen/Qwen3.5-4B` (Confirmatory 4-pair Holm)**: M2_vs_B1b: p_holm4=1.86e-62 (midp=9.4e-63), M2_vs_B0: p_holm4=1 (midp=1), M2_vs_B2a: p_holm4=1 (midp=1), M2_vs_M1c: p_holm4=8.13e-20 (midp=4.07e-20)
- **`Qwen/Qwen3.5-4B` (Conformal & Oracle Audit)**: DevDeg=1.7%, CRC<=3.3%, Hoef<=24.7%, EvalDeg(vs B1b)=0.67% (3/450), EvalDeg(vs B0)=1.11% (5/450), OracleRet(Macro/U0.05)=0.0%/0.0%
- **`Qwen/Qwen3.5-4B` (Probe Horizons)**: K=1:84.7%(356t), K=4:84.4%(369t), K=8:84.7%(356t), K=16:84.5%(358t), K=32:84.6%(360t)
- **`Qwen/Qwen3.5-4B` (Gate-2 L2 Sensitivity)**: C=0.1:84.8%(esc=20%), C=0.25:84.6%(esc=20%), C=0.5:84.6%(esc=20%), C=1:84.6%(esc=20%)
- **`Qwen/Qwen3.5-9B` (Confirmatory 4-pair Holm)**: M2_vs_B1b: p_holm4=2.53e-50 (midp=1.3e-50), M2_vs_B0: p_holm4=0.618 (midp=0.511), M2_vs_B2a: p_holm4=0.00352 (midp=0.00253), M2_vs_M1c: p_holm4=2e-27 (midp=1.03e-27)
- **`Qwen/Qwen3.5-9B` (Conformal & Oracle Audit)**: DevDeg=1.7%, CRC<=3.3%, Hoef<=24.7%, EvalDeg(vs B1b)=1.11% (5/450), EvalDeg(vs B0)=4.44% (20/450), OracleRet(Macro/U0.05)=0.0%/0.0%
- **`Qwen/Qwen3.5-9B` (Probe Horizons)**: K=1:85.2%(382t), K=4:85.8%(385t), K=8:85.3%(379t), K=16:85.7%(382t), K=32:86.0%(380t)
- **`Qwen/Qwen3.5-9B` (Gate-2 L2 Sensitivity)**: C=0.1:85.9%(esc=50%), C=0.25:86.0%(esc=43%), C=0.5:85.9%(esc=42%), C=1:86.0%(esc=33%)
- **Pooled E_hurt (diag_341_strict)**: A1=100.0% (912/912), B2a=88.6% (808/912), p_midp(B2a vs A1)=4.93e-32
- **Pooled E_hurt (diag_491)**: A1=100.0% (416/416), B2a=89.7% (373/416)
- **R6 Multi-Seed (`Qwen/Qwen3.5-4B`, k=3 [42, 43, 44], T=0.6)**: M2 Macro=82.3+/-2.7%, Tok=412+/-42, Delta_B1b=+41.5 pp
<!-- CBARR_RUN_END:20261119T222033Z_pr6 -->

<!-- CBARR_RUN_START:20261120T123558Z -->
### Run `20261120T123558Z` — **RUNNING (after GitHub Sync (script-all-in-one))** (Updated 2026-11-20 12:54:49 UTC)
- **Code Version**: `v0.5.31` | **Git Commit**: `5a84aa6` | **Results Release**: `v0.5.31-res.34` | **Last Loop**: `GitHub Sync (script-all-in-one)` | **Wall Time**: `18m 43.0s` (1123.0s)
- **Hardware**: 8x NVIDIA L40 (44.4 GB, SM 8.9, CUDA 13.2, BF16=True) | PyTorch `2.15.0+cu132` | Transformers `5.23.0` | Kernels: `fla=True, causal_conv1d=True`
- **Command**: `scripts/run_gpu_phases_4_to_6.sh --part all --setup-mode auto --shards auto --auto-9b --with-w7 --auto-b512 --auto-r6 --commit --push --gpu-boost --with-ifbench --with-r6 --hf-home <HOME>/maplecg_nfs/.cbarr_hf_cache`
- **Raw Log**: `outputs/logs/cbarr_20261120T123558Z.log`

#### 1. Training & Experiment Parameters
```json
{
  "auto_b512": "auto",
  "auto_commit": true,
  "auto_push": true,
  "b0_max_new_tokens": 1024,
  "b1b_max_new_tokens": 2048,
  "class_weight": "balanced",
  "code_version": "0.5.31",
  "device": "cuda",
  "do_sample": false,
  "dtype": "auto",
  "gate2_probe_tokens": 32,
  "git_branch": "",
  "git_rev": "5a84aa6",
  "include_b512": true,
  "is_mock": false,
  "lam_utility": 0.05,
  "logreg_C": 1.0,
  "norm_tokens": 250.0,
  "presence_penalty": 0.0,
  "r6_seeds": [
    42,
    43,
    44
  ],
  "run_id": "20261120T123558Z",
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
    250,
    512
  ],
  "top_p": 1.0,
  "with_9b": true,
  "with_ifbench": true,
  "with_r6": true,
  "with_w7": true
}
```

#### 2. Stage Timings
| Stage | Status | Wall Time |
| :--- | :--- | :--- |
| Unpack Synced Worker Results | PASS | 1.0s (1.0s) |
| Gate 0: Preflight Tests | PASS | 10.0s (10.0s) |
| Step 0a: Hardware & Kernel Probe | PASS | 10.0s (10.0s) |
| Step 0a: Live GPU 12-Item Smoke (cached) | PASS | 5.0s (5.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (strict341) (cached) | PASS | 52.0s (52.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (Contingency R3 B=512) (cached) | PASS | 41.0s (41.0s) |
| Phase 4 / W4: Qwen/Qwen3.5-4B (diag491) (cached) | PASS | 41.0s (41.0s) |
| W7: DeltaNet Blend (Qwen/Qwen3.5-4B) (cached) | PASS | 6.0s (6.0s) |
| W7: DeltaNet Conv+KV Blend (Qwen/Qwen3.5-4B) (cached) | PASS | 4.0s (4.0s) |
| W7: Prefill MLP Probe (Qwen/Qwen3.5-4B) (cached) | PASS | 13.0s (13.0s) |
| W7: Stage-2 SolveFirst Replay (Qwen/Qwen3.5-4B) (cached) | PASS | 0.0s (0.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (strict341) (cached) | PASS | 40.0s (40.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (Contingency R3 B=512) (cached) | PASS | 41.0s (41.0s) |
| Phase 5 / W5: HuggingFaceTB/SmolLM3-3B (diag491) (cached) | PASS | 41.0s (41.0s) |
| W7: Dense KV Eviction (HuggingFaceTB/SmolLM3-3B) (cached) | PASS | 5.0s (5.0s) |
| W7: Prefill MLP Probe (HuggingFaceTB/SmolLM3-3B) (cached) | PASS | 12.0s (12.0s) |
| W7: Stage-2 SolveFirst Replay (HuggingFaceTB/SmolLM3-3B) (cached) | PASS | 0.0s (0.0s) |
| Phase 6 / W6: Qwen3.5-0.8B (strict341) (cached) | PASS | 42.0s (42.0s) |
| Phase 6 / W6: Qwen3.5-2B (strict341) (cached) | PASS | 41.0s (41.0s) |
| Phase 6 / W6: Qwen3.5-9B (strict341) (cached) | PASS | 40.0s (40.0s) |
| Appendix B: IFBench OOD (Qwen/Qwen3.5-4B) (cached) | PASS | 0.0s (0.0s) |
| Contingency R6 Sampled (T=0.6, seed=42): Qwen/Qwen3.5-4B (cached) | PASS | 40.0s (40.0s) |
| Contingency R6 Sampled (T=0.6, seed=43): Qwen/Qwen3.5-4B (cached) | PASS | 39.0s (39.0s) |
| Contingency R6 Sampled (T=0.6, seed=44): Qwen/Qwen3.5-4B (cached) | PASS | 40.0s (40.0s) |
| Phase 7 / W8: Analysis, Tables & Figures | PASS | 1m 25.0s (85.0s) |
| Submission Audit: Double-Blind Anonymity | PASS | 0.0s (0.0s) |
| GitHub Sync (script-all-in-one) | PASS | 10.0s (10.0s) |

#### 3. Gate-3 Live 12-Item Smoke Check
| Model | Items | D13 <=250 | No Reminder Leak | SmolLM3 Frozen Date |
| :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-4B` | 12 | True | True | N/A |
| `HuggingFaceTB/SmolLM3-3B` | 12 | True | True | True |

#### 4. Calibration & Gate-2 Classifier Results (Per Model)
| Model | Code Ver | tau_prob | tau_h_high | Dev Esc (R1) | G1 Skip | Intercept | Coef (`H1, Hm, Hmax, I, sym`) | Train@0.5 (Acc/Prec/Rec/F1/AUC) | Val-All@tau (Acc/Prec/Rec/F1/AUC) | Val-G2@tau (F1/AUC) | Dev `B1b->M2` Macro (Tok) | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HuggingFaceTB/SmolLM3-3B` | `v0.5.31` (`f8d621b-dirty`) | 0.400 | 3.144 | 46.7% (OK) | 16.7% | -0.37 | [0.26, 0.53, -0.09, -0.43, -1.45] | 75/28/71/40/0.806 | 48/16/100/28/0.698 | 36/0.806 (n=40) | 76.0%->74.0% (960->407t) | 0.1s |
| `Qwen/Qwen3.5-0.8B` | `v0.5.31` (`6cf16ab`) | 0.525 | 3.719 | 31.7% (OK) | 16.7% | -0.79 | [-0.06, 0.14, 0.15, -0.21, 0.70] | 57/26/73/38/0.651 | 70/19/38/25/0.618 | 40/0.750 (n=40) | 4.0%->61.0% (2000->467t) | 0.2s |
| `Qwen/Qwen3.5-2B` | `v0.5.31` (`8accece`) | 0.850 | 3.157 | 16.7% (WARN) | 16.7% | -2.95 | [-0.34, 0.06, 0.43, 2.19, 0.39] | 65/9/100/16/0.888 | 88/0/0/0/0.369 | 0/0.684 (n=40) | 13.0%->78.0% (1951->376t) | 0.1s |
| `Qwen/Qwen3.5-4B` | `v0.5.31` (`f7a0ebf-dirty`) | 0.650 | 2.537 | 20.0% (OK) | 16.7% | -0.64 | [-0.68, 0.60, 0.10, 0.96, -0.43] | 72/19/100/32/0.804 | 87/0/0/0/0.629 | 0/0.410 (n=40) | 38.0%->90.0% (1744->365t) | 0.1s |
| `Qwen/Qwen3.5-4B` | `v0.5.31` (`bc8c734`) | 0.625 | 0.850 | 50.0% (OK) | 16.7% | -0.29 | [0.00, 0.00, 0.00, 0.86, -2.01] | 70/15/75/25/0.761 | 63/17/67/27/0.710 | 33/0.778 (n=40) | 50.0%->84.0% (1552->360t) | 0.2s |
| `Qwen/Qwen3.5-4B` | `v0.5.31` (`91bca89`) | 0.425 | 0.850 | 83.3% (OK) | 16.7% | 0.26 | [0.00, 0.00, 0.00, -0.49, 0.06] | 63/9/50/15/0.580 | 12/12/100/21/0.592 | 10/0.237 (n=40) | 45.0%->89.0% (1701->439t) | 0.1s |
| `Qwen/Qwen3.5-4B` | `v0.5.31` (`73e20d2`) | 0.525 | 0.850 | 50.0% (OK) | 16.7% | 0.04 | [0.00, 0.00, 0.00, 0.11, -0.42] | 33/9/80/17/0.553 | 63/13/60/21/0.624 | 25/0.639 (n=40) | 42.0%->82.0% (1619->400t) | 0.1s |
| `Qwen/Qwen3.5-9B` | `v0.5.31` (`d4bccf3`) | 0.475 | 2.662 | 43.3% (OK) | 16.7% | -0.31 | [0.30, 0.32, -0.02, 0.00, -1.64] | 57/7/100/13/0.750 | 53/10/75/18/0.656 | 21/0.640 (n=40) | 40.0%->83.0% (1662->351t) | 0.1s |

#### 5. Evaluation Matrix & Section 6 Diagnostics (Per Model x Pool)

- **`HuggingFaceTB/SmolLM3-3B` (`strict341`**, n_diag=341) — ver=`v0.5.31` (`f8d621b-dirty`) | dtype=`torch.bfloat16` | wall=`26.6s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.400, esc=46.7%
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
| M2_B512 | 67.9% | 80.0% | 72.0% | 72.0% | 65.0% | 81.7% | 36.7% | 38.9% | 64.1% | 61.1% | 511 | 1.31 | 0.295 (raw) |
| **Oracle_UB** | 88.2% | 97.3% | 97.3% | 89.3% | 83.3% | 95.0% | 66.7% | 88.0% | 88.5% | 84.7% | 287 | 1.00 | - |

- **`HuggingFaceTB/SmolLM3-3B` (`diag491`**, n_diag=491) — ver=`v0.5.31` (`f8d621b-dirty`) | dtype=`torch.bfloat16` | wall=`25.4s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.400, esc=46.7%
  - **4-Regime**: Helped=13.6%, Hurt=9.6%, Flip=23.1%, TiePass=60.0%, TieFail=16.9%
  - **Section 6 Causal (`|E_hurt|=80`)**: rho_rec=62.5%, rho_int=37.5% | A1_strip=100.0%, B2a_post=62.5%, B1_pre=58.8%, B2b_raw=33.8%, A2_swap=60.0%, B3_cut=62.5%
  - *Eval-450 table identical to `strict341` above (omitted duplicate table).*

- **`Qwen/Qwen3.5-0.8B` (`strict341`**, n_diag=341) — ver=`v0.5.31` (`6cf16ab`) | dtype=`torch.bfloat16` | wall=`27.8s` (calib=0.2s, eval=0.4s, diag=0.1s) | tau=0.525, esc=31.7%
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

- **`Qwen/Qwen3.5-2B` (`strict341`**, n_diag=341) — ver=`v0.5.31` (`8accece`) | dtype=`torch.bfloat16` | wall=`26.9s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.850, esc=16.7%
  - **4-Regime**: Helped=1.1%, Hurt=52.7%, Flip=53.8%, TiePass=17.6%, TieFail=28.7%
  - **Section 6 Causal (`|E_hurt|=215`)**: rho_rec=90.2%, rho_int=9.8% | A1_strip=100.0%, B2a_post=90.2%, B1_pre=87.0%, B2b_raw=50.2%, A2_swap=59.5%, B3_cut=81.9%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 67.7% | 81.3% | 68.0% | 70.0% | 65.0% | 78.3% | 43.3% | 70.2% | 70.2% | 63.6% | 333 | 1.00 | 1.34e-62 |
| B1a | 70.6% | 85.3% | 74.7% | 70.0% | 70.0% | 73.3% | 50.0% | 35.3% | 63.5% | 62.4% | 501 | 1.00 | 2.14e-66 |
| **B1b** | 13.7% | 24.0% | 14.7% | 33.3% | 3.3% | 3.3% | 3.3% | 0.0% | 9.5% | -18.9% | 1879 | 1.00 | - |
| B2a | 61.0% | 86.7% | 72.0% | 70.7% | 43.3% | 60.0% | 33.3% | 46.4% | 59.7% | 58.4% | 382 | 1.00 | 2.15e-55 |
| B2b | 57.2% | 69.3% | 57.3% | 63.3% | 55.0% | 61.7% | 36.7% | 57.6% | 59.1% | 47.2% | 651 | 1.20 | 7.13e-51 |
| M1a | 34.1% | 36.0% | 18.7% | 70.0% | 5.0% | 71.7% | 3.3% | 37.3% | 40.2% | 21.3% | 1078 | 1.00 | 8.63e-32 |
| M1b | 67.4% | 81.3% | 68.0% | 70.0% | 65.0% | 76.7% | 43.3% | 69.8% | 69.8% | 63.2% | 341 | 1.01 | 2.25e-62 |
| M1c | 49.8% | 80.0% | 65.3% | 70.0% | 5.0% | 75.0% | 3.3% | 57.6% | 57.9% | 45.4% | 655 | 1.00 | 9.98e-48 |
| M2_B64 | 67.4% | 81.3% | 68.0% | 70.0% | 65.0% | 76.7% | 43.3% | 70.0% | 69.6% | 62.8% | 359 | 1.00 | 1.95e-61 |
| M2_B128 | 70.2% | 81.3% | 68.0% | 70.0% | 70.0% | 78.3% | 53.3% | 71.6% | 70.7% | 64.0% | 377 | 1.00 | 3.39e-64 |
| M2_B256 | 70.2% | 81.3% | 68.0% | 70.0% | 75.0% | 76.7% | 50.0% | 71.8% | 70.0% | 63.8% | 399 | 1.00 | 1.9e-64 |
| M2_Pre | 68.8% | 81.3% | 68.0% | 70.0% | 63.3% | 80.0% | 50.0% | 70.7% | 69.6% | 62.7% | 399 | 1.00 | 4e-63 |
| **M2** | 69.6% | 81.3% | 68.0% | 70.0% | 70.0% | 78.3% | 50.0% | 71.3% | 70.2% | 63.7% | 382 | 1.00 | 5.98e-64 |
| M2_Prec (Aux.) | 69.6% | 81.3% | 68.0% | 70.0% | 70.0% | 75.0% | 53.3% | 71.1% | 69.4% | 62.9% | 413 | 1.00 | 1.3e-64 (raw) |
| **Oracle_UB** | 87.2% | 94.7% | 92.0% | 86.7% | 86.7% | 90.0% | 73.3% | 88.4% | 86.6% | 82.3% | 308 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`strict341`**, n_diag=341) — ver=`v0.5.31` (`f7a0ebf-dirty`) | dtype=`torch.bfloat16` | wall=`27.0s` (calib=0.1s, eval=0.5s, diag=0.1s) | tau=0.650, esc=20.0%
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
| M2_B512 | 86.1% | 92.0% | 90.7% | 82.0% | 86.7% | 81.7% | 83.3% | 65.6% | 80.7% | 76.9% | 445 | 1.04 | 3.05e-64 (raw) |
| **Oracle_UB** | 94.2% | 97.3% | 98.7% | 92.7% | 93.3% | 93.3% | 90.0% | 92.2% | 92.3% | 88.4% | 300 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`diag491`**, n_diag=491) — ver=`v0.5.31` (`f7a0ebf-dirty`) | dtype=`torch.bfloat16` | wall=`26.6s` (calib=0.1s, eval=0.5s, diag=0.1s) | tau=0.650, esc=20.0%
  - **4-Regime**: Helped=0.7%, Hurt=50.0%, Flip=50.7%, TiePass=34.7%, TieFail=14.7%
  - **Section 6 Causal (`|E_hurt|=336`)**: rho_rec=96.1%, rho_int=3.9% | A1_strip=100.0%, B2a_post=96.1%, B1_pre=95.5%, B2b_raw=49.4%, A2_swap=70.2%, B3_cut=86.3%
  - *Eval-450 table identical to `strict341` above (omitted duplicate table).*

- **`Qwen/Qwen3.5-4B` (`strict341`**, n_diag=341) — ver=`v0.5.31` (`bc8c734`) | dtype=`torch.bfloat16` | sample(T=0.6, top_p=0.95, pp=1.5, seed=42) | wall=`25.4s` (calib=0.2s, eval=0.4s, diag=0.1s) | tau=0.625, esc=50.0%
  - **4-Regime**: Helped=0.7%, Hurt=40.4%, Flip=41.1%, TiePass=44.4%, TieFail=14.4%
  - **Section 6 Causal (`|E_hurt|=216`)**: rho_rec=90.7%, rho_int=7.9% | A1_strip=100.0%, B2a_post=90.7%, B1_pre=92.1%, B2b_raw=45.4%, A2_swap=60.6%, B3_cut=86.1%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 84.4% | 94.7% | 90.7% | 81.3% | 81.7% | 81.7% | 76.7% | 84.9% | 84.9% | 78.7% | 308 | 1.00 | 5.17e-49 |
| B1a | 81.7% | 93.3% | 93.3% | 82.0% | 66.7% | 78.3% | 76.7% | 39.1% | 72.2% | 75.3% | 378 | 1.00 | 6.43e-43 |
| **B1b** | 38.5% | 66.7% | 48.0% | 54.7% | 25.0% | 30.0% | 6.7% | 0.0% | 22.0% | 11.6% | 1675 | 1.00 | - |
| B2a | 82.7% | 96.0% | 92.0% | 81.3% | 76.7% | 76.7% | 73.3% | 49.6% | 75.0% | 77.4% | 317 | 1.00 | 1.26e-45 |
| B2b | 64.3% | 78.7% | 65.3% | 68.7% | 58.3% | 55.0% | 60.0% | 44.4% | 55.0% | 46.0% | 999 | 1.50 | 3.03e-28 |
| M1a | 52.9% | 74.7% | 48.0% | 81.3% | 30.0% | 76.7% | 6.7% | 43.1% | 53.1% | 43.2% | 950 | 1.00 | 5.5e-21 |
| M1b | 72.6% | 86.7% | 90.7% | 54.7% | 71.7% | 71.7% | 60.0% | 48.0% | 58.5% | 52.5% | 921 | 1.45 | 2.86e-33 |
| M1c | 57.4% | 86.7% | 88.0% | 54.7% | 30.0% | 78.3% | 6.7% | 36.7% | 48.4% | 39.4% | 1142 | 1.39 | 5.5e-21 |
| M2_B64 | 83.2% | 93.3% | 89.3% | 83.3% | 85.0% | 81.7% | 66.7% | 84.9% | 82.9% | 77.8% | 352 | 1.39 | 4.1e-46 |
| M2_B128 | 80.3% | 93.3% | 89.3% | 76.0% | 76.7% | 83.3% | 63.3% | 81.3% | 78.1% | 73.4% | 397 | 1.39 | 7.92e-42 |
| M2_B256 | 83.4% | 93.3% | 90.7% | 81.3% | 85.0% | 83.3% | 66.7% | 84.7% | 78.8% | 75.6% | 454 | 1.39 | 1.11e-47 |
| M2_Pre | 80.2% | 93.3% | 89.3% | 83.3% | 70.0% | 81.7% | 63.3% | 82.7% | 79.7% | 75.2% | 375 | 1.39 | 2.72e-44 |
| **M2** | 82.9% | 93.3% | 90.7% | 83.3% | 76.7% | 83.3% | 70.0% | 84.4% | 81.3% | 77.0% | 371 | 1.39 | 1.26e-45 |
| M2_Prec (Aux.) | 84.0% | 93.3% | 90.7% | 83.3% | 76.7% | 86.7% | 73.3% | 85.1% | 81.3% | 77.3% | 391 | 1.39 | 2.12e-47 (raw) |
| **Oracle_UB** | 94.5% | 97.3% | 97.3% | 90.7% | 93.3% | 91.7% | 96.7% | 93.8% | 92.0% | 88.1% | 283 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`strict341`**, n_diag=341) — ver=`v0.5.31` (`91bca89`) | dtype=`torch.bfloat16` | sample(T=0.6, top_p=0.95, pp=1.5, seed=43) | wall=`25.1s` (calib=0.1s, eval=0.4s, diag=0.1s) | tau=0.425, esc=83.3%
  - **4-Regime**: Helped=0.9%, Hurt=36.9%, Flip=37.8%, TiePass=46.9%, TieFail=15.3%
  - **Section 6 Causal (`|E_hurt|=195`)**: rho_rec=92.8%, rho_int=7.2% | A1_strip=100.0%, B2a_post=92.8%, B1_pre=91.8%, B2b_raw=47.2%, A2_swap=67.7%, B3_cut=90.3%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 81.8% | 92.0% | 92.0% | 82.0% | 76.7% | 85.0% | 63.3% | 83.8% | 83.8% | 77.5% | 313 | 1.00 | 5.52e-43 |
| B1a | 78.6% | 93.3% | 93.3% | 80.0% | 65.0% | 76.7% | 63.3% | 37.1% | 70.1% | 72.9% | 401 | 1.00 | 9.1e-35 |
| **B1b** | 42.9% | 73.3% | 56.0% | 53.3% | 21.7% | 30.0% | 23.3% | 0.0% | 23.3% | 14.5% | 1662 | 1.00 | - |
| B2a | 81.1% | 96.0% | 94.7% | 80.7% | 73.3% | 68.3% | 73.3% | 46.2% | 73.4% | 76.0% | 322 | 1.00 | 5.22e-36 |
| B2b | 48.4% | 77.3% | 61.3% | 58.7% | 26.7% | 40.0% | 26.7% | 13.1% | 32.2% | 24.3% | 1451 | 1.83 | 3.25e-06 |
| M1a | 58.8% | 78.7% | 58.7% | 82.0% | 26.7% | 80.0% | 26.7% | 43.8% | 54.6% | 47.2% | 951 | 1.00 | 1.78e-23 |
| M1b | 42.9% | 73.3% | 56.0% | 53.3% | 21.7% | 30.0% | 23.3% | 0.0% | 22.6% | 13.9% | 1693 | 2.00 | 1 |
| M1c | 51.3% | 73.3% | 56.0% | 53.3% | 21.7% | 80.0% | 23.3% | 10.0% | 31.3% | 24.7% | 1486 | 1.68 | 5.59e-09 |
| M2_B64 | 79.5% | 84.0% | 85.3% | 76.0% | 76.7% | 85.0% | 70.0% | 79.8% | 76.7% | 71.9% | 396 | 1.68 | 2.42e-31 |
| M2_B128 | 81.0% | 92.0% | 86.7% | 80.7% | 75.0% | 85.0% | 66.7% | 82.4% | 77.0% | 73.8% | 435 | 1.68 | 1.34e-36 |
| M2_B256 | 83.1% | 88.0% | 89.3% | 84.7% | 78.3% | 85.0% | 73.3% | 84.4% | 75.4% | 74.3% | 510 | 1.68 | 1.84e-40 |
| M2_Pre | 79.2% | 88.0% | 89.3% | 76.0% | 73.3% | 85.0% | 63.3% | 80.2% | 74.2% | 71.2% | 451 | 1.68 | 2.96e-32 |
| **M2** | 79.4% | 88.0% | 89.3% | 76.0% | 75.0% | 85.0% | 63.3% | 80.4% | 74.4% | 71.4% | 454 | 1.68 | 1.5e-31 |
| M2_Prec (Aux.) | 80.0% | 88.0% | 89.3% | 76.0% | 75.0% | 88.3% | 63.3% | 80.9% | 74.2% | 71.4% | 475 | 1.68 | 7.45e-33 (raw) |
| **Oracle_UB** | 93.3% | 98.7% | 96.0% | 88.7% | 95.0% | 91.7% | 90.0% | 92.9% | 91.0% | 87.1% | 288 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`strict341`**, n_diag=341) — ver=`v0.5.31` (`73e20d2`) | dtype=`torch.bfloat16` | sample(T=0.6, top_p=0.95, pp=1.5, seed=44) | wall=`25.5s` (calib=0.1s, eval=0.6s, diag=0.1s) | tau=0.525, esc=50.0%
  - **4-Regime**: Helped=2.4%, Hurt=37.8%, Flip=40.2%, TiePass=45.1%, TieFail=14.7%
  - **Section 6 Causal (`|E_hurt|=205`)**: rho_rec=92.7%, rho_int=7.3% | A1_strip=100.0%, B2a_post=92.7%, B1_pre=92.2%, B2b_raw=46.8%, A2_swap=62.9%, B3_cut=87.3%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 82.2% | 93.3% | 88.0% | 78.7% | 81.7% | 81.7% | 70.0% | 82.9% | 82.9% | 76.5% | 321 | 1.00 | 5.25e-37 |
| B1a | 79.8% | 96.0% | 96.0% | 80.0% | 65.0% | 78.3% | 63.3% | 39.1% | 72.0% | 74.4% | 381 | 1.00 | 1.73e-38 |
| **B1b** | 41.0% | 69.3% | 60.0% | 53.3% | 31.7% | 28.3% | 3.3% | 0.2% | 23.0% | 14.4% | 1660 | 1.00 | - |
| B2a | 81.8% | 94.7% | 96.0% | 80.0% | 68.3% | 78.3% | 73.3% | 49.8% | 73.6% | 76.6% | 315 | 1.00 | 3.37e-38 |
| B2b | 58.4% | 77.3% | 66.7% | 63.3% | 58.3% | 55.0% | 30.0% | 37.6% | 49.3% | 40.8% | 1071 | 1.54 | 2.58e-16 |
| M1a | 57.1% | 77.3% | 60.0% | 78.7% | 38.3% | 75.0% | 13.3% | 42.4% | 53.2% | 46.0% | 956 | 1.00 | 1.73e-18 |
| M1b | 69.4% | 85.3% | 88.0% | 53.3% | 73.3% | 73.3% | 43.3% | 46.9% | 57.9% | 50.9% | 909 | 1.45 | 1.12e-22 |
| M1c | 58.4% | 85.3% | 86.7% | 53.3% | 38.3% | 76.7% | 10.0% | 36.2% | 48.7% | 39.9% | 1129 | 1.39 | 2.8e-16 |
| M2_B64 | 81.3% | 92.0% | 88.0% | 78.0% | 81.7% | 81.7% | 66.7% | 82.2% | 80.3% | 74.4% | 391 | 1.39 | 3.96e-38 |
| M2_B128 | 84.1% | 92.0% | 88.0% | 81.3% | 91.7% | 81.7% | 70.0% | 84.9% | 81.4% | 76.5% | 417 | 1.39 | 4.76e-42 |
| M2_B256 | 84.9% | 92.0% | 88.0% | 79.3% | 88.3% | 81.7% | 80.0% | 84.4% | 78.6% | 75.3% | 455 | 1.39 | 8.73e-41 |
| M2_Pre | 80.2% | 92.0% | 88.0% | 78.0% | 76.7% | 80.0% | 66.7% | 81.3% | 78.4% | 73.1% | 412 | 1.39 | 1.86e-36 |
| **M2** | 84.7% | 92.0% | 88.0% | 78.0% | 91.7% | 81.7% | 76.7% | 84.2% | 81.0% | 76.0% | 412 | 1.39 | 3.1e-41 |
| M2_Prec (Aux.) | 86.1% | 92.0% | 88.0% | 78.0% | 91.7% | 90.0% | 76.7% | 85.3% | 81.5% | 76.7% | 431 | 1.39 | 2e-45 (raw) |
| **Oracle_UB** | 92.7% | 97.3% | 96.0% | 89.3% | 95.0% | 91.7% | 86.7% | 92.7% | 90.8% | 86.7% | 296 | 1.00 | - |

- **`Qwen/Qwen3.5-9B` (`strict341`**, n_diag=341) — ver=`v0.5.31` (`d4bccf3`) | dtype=`torch.bfloat16` | wall=`25.3s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.475, esc=43.3%
  - **4-Regime**: Helped=0.9%, Hurt=43.8%, Flip=44.7%, TiePass=43.1%, TieFail=12.2%
  - **Section 6 Causal (`|E_hurt|=226`)**: rho_rec=95.1%, rho_int=4.9% | A1_strip=100.0%, B2a_post=95.1%, B1_pre=93.4%, B2b_raw=47.8%, A2_swap=80.1%, B3_cut=92.5%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 87.5% | 92.0% | 93.3% | 81.3% | 88.3% | 86.7% | 83.3% | 86.9% | 86.9% | 80.7% | 310 | 1.00 | 5.03e-52 |
| B1a | 84.3% | 96.0% | 93.3% | 81.3% | 71.7% | 80.0% | 83.3% | 33.6% | 73.0% | 75.5% | 447 | 1.00 | 4.11e-49 |
| **B1b** | 38.3% | 62.7% | 52.0% | 52.0% | 23.3% | 26.7% | 13.3% | 0.0% | 22.2% | 10.6% | 1668 | 1.00 | - |
| B2a | 78.4% | 85.3% | 70.7% | 82.7% | 80.0% | 81.7% | 70.0% | 63.3% | 74.0% | 74.7% | 256 | 1.00 | 3.53e-32 |
| B2b | 70.1% | 81.3% | 76.0% | 66.7% | 68.3% | 65.0% | 63.3% | 51.6% | 60.6% | 52.5% | 895 | 1.42 | 7.17e-33 |
| M1a | 56.8% | 72.0% | 57.3% | 81.3% | 30.0% | 80.0% | 20.0% | 43.8% | 54.2% | 45.6% | 952 | 1.00 | 1.45e-26 |
| M1b | 54.4% | 86.7% | 89.3% | 57.3% | 33.3% | 40.0% | 20.0% | 40.4% | 49.3% | 36.4% | 1158 | 1.56 | 2.94e-19 |
| M1c | 58.8% | 85.3% | 86.7% | 57.3% | 25.0% | 81.7% | 16.7% | 44.2% | 53.2% | 41.9% | 1062 | 1.32 | 1.32e-23 |
| M2_B64 | 86.5% | 93.3% | 93.3% | 80.7% | 85.0% | 86.7% | 80.0% | 86.2% | 84.5% | 79.2% | 353 | 1.32 | 1.97e-49 |
| M2_B128 | 84.6% | 93.3% | 93.3% | 79.3% | 85.0% | 86.7% | 70.0% | 85.1% | 82.2% | 77.6% | 378 | 1.32 | 7.29e-50 |
| M2_B256 | 85.9% | 93.3% | 92.0% | 83.3% | 85.0% | 85.0% | 76.7% | 86.4% | 81.3% | 78.0% | 424 | 1.32 | 1.77e-51 |
| M2_Pre | 86.8% | 93.3% | 92.0% | 80.7% | 91.7% | 86.7% | 76.7% | 86.7% | 83.9% | 79.0% | 382 | 1.32 | 8.51e-51 |
| **M2** | 86.0% | 93.3% | 92.0% | 80.7% | 86.7% | 86.7% | 76.7% | 86.0% | 83.3% | 78.4% | 380 | 1.32 | 5.68e-50 |
| M2_Prec (Aux.) | 86.3% | 93.3% | 92.0% | 80.7% | 86.7% | 88.3% | 76.7% | 86.2% | 82.8% | 78.2% | 401 | 1.32 | 3.24e-51 (raw) |
| **Oracle_UB** | 93.9% | 98.7% | 96.0% | 90.7% | 96.7% | 91.7% | 90.0% | 93.8% | 92.5% | 87.9% | 292 | 1.00 | - |

#### 6. Optional W7 Extensions
- **`HuggingFaceTB__SmolLM3-3B`**: DenseKVEvict(n=80: a=0.00:86.2%) | PrefillMLP(layer_-1, tau=0.850, Macro=66.8%, OAA250=69.6%, U0.05=63.4%, Tok=308, OrigPrecF1=86.0%->81.1%, ParaPrecF1=78.9%->81.7%) | Stage2SolveFirst(replayed=139, Macro=68.5%->67.0%, OAA250=72.0%->69.6%)
- **`Qwen__Qwen3.5-4B`**: DeltaNetBlend(n=336: a=0.00:93.2%, a=0.25:95.2%, a=0.50:95.2%, a=0.75:94.0%, a=1.00:91.7%) | DeltaNetConvKV(n=336: a=0.00:85.7%, a=0.25:88.1%, a=0.50:89.9%, a=0.75:91.7%, a=1.00:95.8%) | PrefillMLP(layer_-1, tau=0.525, Macro=84.1%, OAA250=84.9%, U0.05=77.8%, Tok=357, OrigPrecF1=86.0%->81.2%, ParaPrecF1=78.9%->82.6%) | Stage2SolveFirst(replayed=17, Macro=84.6%->84.3%, OAA250=84.9%->84.4%) | IFBenchOOD(N=300: B0=28.7%/B1b=2.0%/M2=31.3%, OAA250=31.3%, Tok=443, p_vs_B1b=1.73e-24)

#### 7. Phase 7 Statistical Audit & Gate-4 Summary
| Model | Version | Clean-446 (max delta) | M2 Macro | B1b Macro | M2 OAA_250 | Delta OAA_250 [95% CI] | Tok Red | p_holm vs B1b | p(M2 vs B0) | p(M2 vs Pre) | Gate 4 | R2 (rec / int) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HuggingFaceTB/SmolLM3-3B` | `v0.5.31+m5` | 448 (0.5 pp) | 68.5% | 69.4% | 72.0% | +68.4 [64.0, 72.7] pp | 62.0% | 1 | 0.1524 | 0.7539 | CHECK | 62.5% / 37.5% |
| `Qwen/Qwen3.5-0.8B` | `v0.5.31+m5` | 448 (0.2 pp) | 53.6% | 2.7% | 56.2% | +56.2 [51.6, 60.9] pp | 78.2% | 7.792e-67 | 1 | 0.2295 | PASS | 76.1% / 23.9% |
| `Qwen/Qwen3.5-2B` | `v0.5.31+m5` | 448 (0.1 pp) | 69.6% | 13.7% | 71.3% | +71.3 [67.1, 75.6] pp | 79.7% | 5.977e-64 | 0.3593 | 0.6072 | PASS | 90.2% / 9.8% |
| `Qwen/Qwen3.5-4B` | `v0.5.31+m7` | 448 (0.4 pp) | 84.6% | 30.2% | 84.9% | +84.7 [81.3, 88.0] pp | 79.2% | 5.104e-62 | 1 | 0.424 | PASS | 96.1% / 3.9% |
| `Qwen/Qwen3.5-9B` | `v0.5.31+m4` | 448 (0.4 pp) | 86.0% | 38.3% | 86.0% | +86.0 [82.7, 89.1] pp | 77.2% | 5.684e-50 | 0.6177 | 0.5488 | PASS | 95.1% / 4.9% |

#### 8. Confirmatory Audit & Appendix B Summary
- **`HuggingFaceTB/SmolLM3-3B` (Confirmatory 4-pair Holm)**: M2_vs_B1b: p_holm4=0.961 (midp=0.817), M2_vs_B0: p_holm4=0.61 (midp=0.476), M2_vs_B2a: p_holm4=0.961 (midp=0.817), M2_vs_M1c: p_holm4=0.961 (midp=0.817)
- **`HuggingFaceTB/SmolLM3-3B` (Conformal & Oracle Audit)**: DevDeg=11.7%, CRC<=13.1%, Hoef<=34.7%, EvalDeg(vs B1b)=10.44% (47/450), EvalDeg(vs B0)=4.22% (19/450), OracleRet(Macro/U0.05)=0.0%/0.0%
- **`HuggingFaceTB/SmolLM3-3B` (Probe Horizons)**: K=1:68.4%(366t), K=4:68.3%(358t), K=8:67.2%(335t), K=16:67.2%(331t), K=32:68.5%(371t)
- **`HuggingFaceTB/SmolLM3-3B` (Gate-2 L2 Sensitivity)**: C=0.1:68.5%(esc=50%), C=0.25:68.5%(esc=47%), C=0.5:68.7%(esc=47%), C=1:68.5%(esc=48%)
- **`Qwen/Qwen3.5-0.8B` (Confirmatory 4-pair Holm)**: M2_vs_B1b: p_holm4=2.6e-67 (midp=1.31e-67), M2_vs_B0: p_holm4=1 (midp=1), M2_vs_B2a: p_holm4=4.01e-06 (midp=2.89e-06), M2_vs_M1c: p_holm4=6.35e-22 (midp=3.18e-22)
- **`Qwen/Qwen3.5-0.8B` (Conformal & Oracle Audit)**: DevDeg=0.0%, CRC<=1.6%, Hoef<=23.0%, EvalDeg(vs B1b)=0.44% (2/450), EvalDeg(vs B0)=5.11% (23/450), OracleRet(Macro/U0.05)=0.0%/0.0%
- **`Qwen/Qwen3.5-0.8B` (Probe Horizons)**: K=1:55.4%(429t), K=4:54.1%(439t), K=8:54.9%(435t), K=16:55.4%(428t), K=32:53.6%(436t)
- **`Qwen/Qwen3.5-0.8B` (Gate-2 L2 Sensitivity)**: C=0.1:54.3%(esc=17%), C=0.25:53.6%(esc=32%), C=0.5:53.6%(esc=25%), C=1:53.6%(esc=28%)
- **`Qwen/Qwen3.5-2B` (Confirmatory 4-pair Holm)**: M2_vs_B1b: p_holm4=2.66e-64 (midp=1.36e-64), M2_vs_B0: p_holm4=0.359 (midp=0.263), M2_vs_B2a: p_holm4=0.05 (midp=0.0398), M2_vs_M1c: p_holm4=2.08e-17 (midp=1.04e-17)
- **`Qwen/Qwen3.5-2B` (Conformal & Oracle Audit)**: DevDeg=0.0%, CRC<=1.6%, Hoef<=23.0%, EvalDeg(vs B1b)=1.11% (5/450), EvalDeg(vs B0)=1.56% (7/450), OracleRet(Macro/U0.05)=0.0%/0.0%
- **`Qwen/Qwen3.5-2B` (Probe Horizons)**: K=1:69.8%(410t), K=4:69.6%(381t), K=8:69.6%(381t), K=16:70.3%(403t), K=32:69.6%(382t)
- **`Qwen/Qwen3.5-2B` (Gate-2 L2 Sensitivity)**: C=0.1:70.5%(esc=28%), C=0.25:69.6%(esc=17%), C=0.5:70.9%(esc=32%), C=1:71.3%(esc=43%)
- **`Qwen/Qwen3.5-4B` (Confirmatory 4-pair Holm)**: M2_vs_B1b: p_holm4=1.86e-62 (midp=9.4e-63), M2_vs_B0: p_holm4=1 (midp=1), M2_vs_B2a: p_holm4=1 (midp=1), M2_vs_M1c: p_holm4=8.13e-20 (midp=4.07e-20)
- **`Qwen/Qwen3.5-4B` (Conformal & Oracle Audit)**: DevDeg=1.7%, CRC<=3.3%, Hoef<=24.7%, EvalDeg(vs B1b)=0.67% (3/450), EvalDeg(vs B0)=1.11% (5/450), OracleRet(Macro/U0.05)=0.0%/0.0%
- **`Qwen/Qwen3.5-4B` (Probe Horizons)**: K=1:84.7%(356t), K=4:84.4%(369t), K=8:84.7%(356t), K=16:84.5%(358t), K=32:84.6%(360t)
- **`Qwen/Qwen3.5-4B` (Gate-2 L2 Sensitivity)**: C=0.1:84.8%(esc=20%), C=0.25:84.6%(esc=20%), C=0.5:84.6%(esc=20%), C=1:84.6%(esc=20%)
- **`Qwen/Qwen3.5-9B` (Confirmatory 4-pair Holm)**: M2_vs_B1b: p_holm4=2.53e-50 (midp=1.3e-50), M2_vs_B0: p_holm4=0.618 (midp=0.511), M2_vs_B2a: p_holm4=0.00352 (midp=0.00253), M2_vs_M1c: p_holm4=2e-27 (midp=1.03e-27)
- **`Qwen/Qwen3.5-9B` (Conformal & Oracle Audit)**: DevDeg=1.7%, CRC<=3.3%, Hoef<=24.7%, EvalDeg(vs B1b)=1.11% (5/450), EvalDeg(vs B0)=4.44% (20/450), OracleRet(Macro/U0.05)=0.0%/0.0%
- **`Qwen/Qwen3.5-9B` (Probe Horizons)**: K=1:85.2%(382t), K=4:85.8%(385t), K=8:85.3%(379t), K=16:85.7%(382t), K=32:86.0%(380t)
- **`Qwen/Qwen3.5-9B` (Gate-2 L2 Sensitivity)**: C=0.1:85.9%(esc=50%), C=0.25:86.0%(esc=43%), C=0.5:85.9%(esc=42%), C=1:86.0%(esc=33%)
- **Pooled E_hurt (diag_341_strict)**: A1=100.0% (912/912), B2a=88.6% (808/912), p_midp(B2a vs A1)=4.93e-32
- **Pooled E_hurt (diag_491)**: A1=100.0% (416/416), B2a=89.7% (373/416)
- **R6 Multi-Seed (`Qwen/Qwen3.5-4B`, k=3 [42, 43, 44], T=0.6)**: M2 Macro=82.3+/-2.7%, Tok=412+/-42, Delta_B1b=+41.5 pp
<!-- CBARR_RUN_END:20261120T123558Z -->

<!-- CBARR_RUN_START:20261120T200335Z_pr6 -->
### Run `20261120T200335Z_pr6` — **RUNNING (Eval-450: HuggingFaceTB/SmolLM3-3B (strict341))** (Updated 2026-11-21 06:30:34 UTC)
- **Code Version**: `v0.6.1` | **Git Commit**: `efea1b2` | **Results Release**: `v0.6.1-res.35` | **Last Loop**: `Eval-450: HuggingFaceTB/SmolLM3-3B (strict341)` | **Wall Time**: `20.2s` (20.2s)
- **Hardware**: 8x NVIDIA L40 (44.4 GB, SM 8.9, CUDA 13.2, BF16=True) | PyTorch `2.15.0+cu132` | Transformers `5.23.0` | Kernels: `fla=True, causal_conv1d=True`
- **Command**: `scripts/run_matrix.py`
- **Raw Log**: `outputs/logs/cbarr_20261120T200335Z_pr6.log`

#### 1. Training & Experiment Parameters
```json
{
  "auto_b512": "auto",
  "auto_commit": true,
  "auto_push": true,
  "b0_max_new_tokens": 1024,
  "b1b_max_new_tokens": 2048,
  "class_weight": "balanced",
  "code_version": "0.6.1",
  "device": "cuda",
  "do_sample": false,
  "dtype": "auto",
  "gate2_probe_tokens": 32,
  "git_branch": "",
  "git_rev": "efea1b2",
  "include_b512": true,
  "is_mock": false,
  "lam_utility": 0.05,
  "logreg_C": 1.0,
  "norm_tokens": 250.0,
  "presence_penalty": 0.0,
  "r6_seeds": [
    42,
    43,
    44
  ],
  "run_id": "20261120T200335Z_pr6",
  "seed": 42,
  "setup_mode": "off",
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
    250,
    512
  ],
  "top_p": 1.0,
  "with_9b": false,
  "with_ifbench": false,
  "with_r6": false,
  "with_w7": false
}
```

#### 2. Stage Timings
| Stage | Status | Wall Time |
| :--- | :--- | :--- |
| Contingency R6 Sampled (T=0.6, seed=42): HuggingFaceTB/SmolLM3-3B | PASS | 3h 10m 30s (11430.0s) |
| Contingency R6 Sampled (T=0.6, seed=42): HuggingFaceTB/SmolLM3-3B (merge) | PASS | 5h 50m 07s (21007.0s) |
| Contingency R6 Sampled (T=0.6, seed=43): HuggingFaceTB/SmolLM3-3B | PASS | 1h 25m 03s (5103.0s) |

#### 4. Calibration & Gate-2 Classifier Results (Per Model)
| Model | Code Ver | tau_prob | tau_h_high | Dev Esc (R1) | G1 Skip | Intercept | Coef (`H1, Hm, Hmax, I, sym`) | Train@0.5 (Acc/Prec/Rec/F1/AUC) | Val-All@tau (Acc/Prec/Rec/F1/AUC) | Val-G2@tau (F1/AUC) | Dev `B1b->M2` Macro (Tok) | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HuggingFaceTB/SmolLM3-3B` | `v0.5.31` (`f8d621b-dirty`) | 0.400 | 3.144 | 46.7% (OK) | 16.7% | -0.37 | [0.26, 0.53, -0.09, -0.43, -1.45] | 75/28/71/40/0.806 | 48/16/100/28/0.698 | 36/0.806 (n=40) | 76.0%->74.0% (960->407t) | 0.1s |
| `HuggingFaceTB/SmolLM3-3B` | `v0.6.0` (`6ea0a74-dirty`) | 0.425 | 0.850 | 50.0% (OK) | 16.7% | 0.50 | [0.00, 0.00, 0.00, -0.74, -1.60] | 60/12/75/20/0.708 | 42/18/78/29/0.508 | 37/0.630 (n=40) | 80.0%->71.0% (872->371t) | 0.1s |
| `HuggingFaceTB/SmolLM3-3B` | `v0.6.1` (`5bfd729-dirty`) | 0.150 | 0.850 | 83.3% (OK) | 16.7% | 0.15 | [0.00, 0.00, 0.00, 0.30, -2.05] | 45/23/100/38/0.725 | 17/17/100/29/0.488 | 33/0.500 (n=40) | 79.0%->90.0% (922->458t) | 0.1s |
| `Qwen/Qwen3.5-0.8B` | `v0.5.31` (`6cf16ab`) | 0.525 | 3.719 | 31.7% (OK) | 16.7% | -0.79 | [-0.06, 0.14, 0.15, -0.21, 0.70] | 57/26/73/38/0.651 | 70/19/38/25/0.618 | 40/0.750 (n=40) | 4.0%->61.0% (2000->467t) | 0.2s |
| `Qwen/Qwen3.5-2B` | `v0.5.31` (`8accece`) | 0.850 | 3.157 | 16.7% (WARN) | 16.7% | -2.95 | [-0.34, 0.06, 0.43, 2.19, 0.39] | 65/9/100/16/0.888 | 88/0/0/0/0.369 | 0/0.684 (n=40) | 13.0%->78.0% (1951->376t) | 0.1s |
| `Qwen/Qwen3.5-4B` | `v0.5.31` (`f7a0ebf-dirty`) | 0.650 | 2.537 | 20.0% (OK) | 16.7% | -0.64 | [-0.68, 0.60, 0.10, 0.96, -0.43] | 72/19/100/32/0.804 | 87/0/0/0/0.629 | 0/0.410 (n=40) | 38.0%->90.0% (1744->365t) | 0.1s |
| `Qwen/Qwen3.5-4B` | `v0.5.31` (`bc8c734`) | 0.625 | 0.850 | 50.0% (OK) | 16.7% | -0.29 | [0.00, 0.00, 0.00, 0.86, -2.01] | 70/15/75/25/0.761 | 63/17/67/27/0.710 | 33/0.778 (n=40) | 50.0%->84.0% (1552->360t) | 0.2s |
| `Qwen/Qwen3.5-4B` | `v0.5.31` (`91bca89`) | 0.425 | 0.850 | 83.3% (OK) | 16.7% | 0.26 | [0.00, 0.00, 0.00, -0.49, 0.06] | 63/9/50/15/0.580 | 12/12/100/21/0.592 | 10/0.237 (n=40) | 45.0%->89.0% (1701->439t) | 0.1s |
| `Qwen/Qwen3.5-4B` | `v0.5.31` (`73e20d2`) | 0.525 | 0.850 | 50.0% (OK) | 16.7% | 0.04 | [0.00, 0.00, 0.00, 0.11, -0.42] | 33/9/80/17/0.553 | 63/13/60/21/0.624 | 25/0.639 (n=40) | 42.0%->82.0% (1619->400t) | 0.1s |
| `Qwen/Qwen3.5-9B` | `v0.5.31` (`d4bccf3`) | 0.475 | 2.662 | 43.3% (OK) | 16.7% | -0.31 | [0.30, 0.32, -0.02, 0.00, -1.64] | 57/7/100/13/0.750 | 53/10/75/18/0.656 | 21/0.640 (n=40) | 40.0%->83.0% (1662->351t) | 0.1s |

#### 5. Evaluation Matrix & Section 6 Diagnostics (Per Model x Pool)

- **`HuggingFaceTB/SmolLM3-3B` (`strict341`**, n_diag=341) — ver=`v0.5.31` (`f8d621b-dirty`) | dtype=`torch.bfloat16` | wall=`26.6s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.400, esc=46.7%
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
| M2_B512 | 67.9% | 80.0% | 72.0% | 72.0% | 65.0% | 81.7% | 36.7% | 38.9% | 64.1% | 61.1% | 511 | 1.31 | 0.295 (raw) |
| **Oracle_UB** | 88.2% | 97.3% | 97.3% | 89.3% | 83.3% | 95.0% | 66.7% | 88.0% | 88.5% | 84.7% | 287 | 1.00 | - |

- **`HuggingFaceTB/SmolLM3-3B` (`diag491`**, n_diag=491) — ver=`v0.5.31` (`f8d621b-dirty`) | dtype=`torch.bfloat16` | wall=`25.4s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.400, esc=46.7%
  - **4-Regime**: Helped=13.6%, Hurt=9.6%, Flip=23.1%, TiePass=60.0%, TieFail=16.9%
  - **Section 6 Causal (`|E_hurt|=80`)**: rho_rec=62.5%, rho_int=37.5% | A1_strip=100.0%, B2a_post=62.5%, B1_pre=58.8%, B2b_raw=33.8%, A2_swap=60.0%, B3_cut=62.5%
  - *Eval-450 table identical to `strict341` above (omitted duplicate table).*

- **`HuggingFaceTB/SmolLM3-3B` (`strict341`**, n_diag=341) — ver=`v0.6.0` (`6ea0a74-dirty`) | dtype=`torch.bfloat16` | sample(T=0.6, top_p=0.95, pp=1.5, seed=42) | wall=`20975.8s` (calib=0.1s, eval=20924.2s, diag=0.1s) | tau=0.425, esc=50.0%
  - **4-Regime**: Helped=18.4%, Hurt=7.8%, Flip=26.2%, TiePass=58.9%, TieFail=14.9%
  - **Section 6 Causal (`|E_hurt|=48`)**: rho_rec=52.1%, rho_int=47.9% | A1_strip=100.0%, B2a_post=52.1%, B1_pre=50.0%, B2b_raw=22.9%, A2_swap=52.1%, B3_cut=58.3%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 64.8% | 76.0% | 65.3% | 64.0% | 66.7% | 76.7% | 40.0% | 66.7% | 66.7% | 61.9% | 240 | 1.00 | 0.00014 |
| B1a | 65.7% | 88.0% | 77.3% | 70.7% | 45.0% | 70.0% | 43.3% | 64.2% | 66.2% | 65.1% | 210 | 1.00 | 0.002 |
| **B1b** | 73.3% | 92.0% | 89.3% | 76.7% | 68.3% | 73.3% | 40.0% | 3.3% | 55.1% | 59.8% | 879 | 1.00 | - |
| B2a | 65.7% | 89.3% | 74.7% | 68.7% | 41.7% | 70.0% | 50.0% | 61.8% | 64.6% | 63.9% | 226 | 1.00 | 0.000657 |
| B2b | 71.1% | 86.7% | 78.7% | 72.7% | 68.3% | 73.3% | 46.7% | 36.4% | 61.8% | 62.2% | 577 | 1.50 | 0.044 |
| M1a | 70.5% | 88.0% | 89.3% | 64.0% | 70.0% | 75.0% | 36.7% | 36.0% | 60.6% | 61.3% | 569 | 1.00 | 0.0223 |
| M1b | 67.6% | 80.0% | 65.3% | 76.7% | 68.3% | 75.0% | 40.0% | 21.6% | 57.9% | 58.0% | 677 | 1.71 | 0.000268 |
| M1c | 67.6% | 80.0% | 65.3% | 76.7% | 68.3% | 78.3% | 36.7% | 29.8% | 60.4% | 59.9% | 594 | 1.40 | 0.00813 |
| M2_B64 | 67.7% | 81.3% | 65.3% | 72.7% | 68.3% | 78.3% | 40.0% | 70.9% | 69.1% | 64.0% | 342 | 1.40 | 0.0139 |
| M2_B128 | 66.1% | 81.3% | 65.3% | 74.7% | 63.3% | 78.3% | 33.3% | 70.4% | 67.3% | 63.5% | 346 | 1.40 | 0.0115 |
| M2_B256 | 67.4% | 81.3% | 65.3% | 71.3% | 71.7% | 78.3% | 36.7% | 70.7% | 65.5% | 62.6% | 401 | 1.40 | 0.0139 |
| M2_Pre | 67.7% | 81.3% | 65.3% | 72.7% | 68.3% | 78.3% | 40.0% | 70.9% | 68.1% | 63.6% | 364 | 1.40 | 0.0139 |
| **M2** | 66.8% | 81.3% | 65.3% | 72.7% | 63.3% | 78.3% | 40.0% | 70.2% | 67.5% | 62.9% | 368 | 1.40 | 0.00883 |
| M2_Prec (Aux.) | 68.2% | 81.3% | 65.3% | 72.7% | 63.3% | 83.3% | 43.3% | 71.1% | 67.8% | 63.2% | 394 | 1.40 | 0.00375 (raw) |
| **Oracle_UB** | 89.2% | 97.3% | 97.3% | 88.7% | 78.3% | 96.7% | 76.7% | 88.0% | 88.1% | 84.8% | 282 | 1.00 | - |

- **`HuggingFaceTB/SmolLM3-3B` (`strict341`**, n_diag=341) — ver=`v0.6.1` (`5bfd729-dirty`) | dtype=`torch.bfloat16` | sample(T=0.6, top_p=0.95, pp=1.5, seed=43) | status=`eval450_complete` | wall=`20.2s` (calib=0.1s, eval=0.3s, diag=0.0s) | tau=0.150, esc=83.3%
  - **4-Regime**: Helped=15.3%, Hurt=7.1%, Flip=22.4%, TiePass=63.3%, TieFail=14.2%
  - **Section 6 Causal (`|E_hurt|=0`)**: rho_rec=n/a, rho_int=n/a | A1_strip=0.0%, B2a_post=0.0%, B1_pre=0.0%, B2b_raw=0.0%, A2_swap=0.0%, B3_cut=0.0%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 69.3% | 84.0% | 74.7% | 65.3% | 61.7% | 80.0% | 50.0% | 70.4% | 70.4% | 65.5% | 247 | 1.00 | 0.00296 |
| B1a | 63.9% | 85.3% | 81.3% | 70.0% | 33.3% | 63.3% | 50.0% | 61.3% | 64.5% | 63.0% | 219 | 1.00 | 4.47e-06 |
| **B1b** | 76.1% | 94.7% | 85.3% | 76.7% | 68.3% | 78.3% | 53.3% | 4.2% | 55.8% | 60.8% | 893 | 1.00 | - |
| B2a | 64.9% | 90.7% | 72.0% | 72.0% | 38.3% | 76.7% | 40.0% | 63.8% | 65.9% | 64.5% | 230 | 1.00 | 0.00036 |
| B2b | 74.7% | 94.7% | 81.3% | 75.3% | 66.7% | 80.0% | 50.0% | 14.0% | 57.6% | 61.2% | 809 | 1.83 | 0.584 |
| M1a | 74.3% | 92.0% | 85.3% | 65.3% | 71.7% | 78.3% | 53.3% | 38.2% | 62.7% | 63.3% | 581 | 1.00 | 0.18 |
| M1b | 76.1% | 94.7% | 85.3% | 76.7% | 68.3% | 78.3% | 53.3% | 2.0% | 54.8% | 60.3% | 918 | 1.99 | 1 |
| M1c | 76.4% | 94.7% | 85.3% | 76.7% | 68.3% | 80.0% | 53.3% | 11.6% | 57.5% | 62.2% | 832 | 1.68 | 1 |
| M2_B64 | 69.3% | 86.7% | 78.7% | 70.7% | 63.3% | 80.0% | 36.7% | 72.7% | 69.9% | 64.6% | 402 | 1.68 | 0.0499 |
| M2_B128 | 70.1% | 89.3% | 86.7% | 68.0% | 63.3% | 80.0% | 33.3% | 73.3% | 68.6% | 65.2% | 407 | 1.68 | 0.0851 |
| M2_B256 | 71.8% | 89.3% | 84.0% | 74.0% | 68.3% | 81.7% | 33.3% | 75.8% | 67.8% | 66.5% | 465 | 1.68 | 0.584 |
| M2_Pre | 73.5% | 88.0% | 84.0% | 70.7% | 68.3% | 80.0% | 50.0% | 75.3% | 69.7% | 66.6% | 434 | 1.68 | 0.573 |
| **M2** | 69.3% | 88.0% | 84.0% | 70.7% | 63.3% | 80.0% | 30.0% | 73.3% | 67.9% | 64.5% | 442 | 1.68 | 0.0963 |
| M2_Prec (Aux.) | 68.8% | 88.0% | 84.0% | 70.7% | 63.3% | 76.7% | 30.0% | 72.9% | 66.9% | 63.6% | 466 | 1.68 | 0.00734 (raw) |
| **Oracle_UB** | 87.1% | 96.0% | 97.3% | 89.3% | 76.7% | 96.7% | 66.7% | 87.1% | 87.1% | 83.8% | 290 | 1.00 | - |

- **`Qwen/Qwen3.5-0.8B` (`strict341`**, n_diag=341) — ver=`v0.5.31` (`6cf16ab`) | dtype=`torch.bfloat16` | wall=`27.8s` (calib=0.2s, eval=0.4s, diag=0.1s) | tau=0.525, esc=31.7%
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

- **`Qwen/Qwen3.5-2B` (`strict341`**, n_diag=341) — ver=`v0.5.31` (`8accece`) | dtype=`torch.bfloat16` | wall=`26.9s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.850, esc=16.7%
  - **4-Regime**: Helped=1.1%, Hurt=52.7%, Flip=53.8%, TiePass=17.6%, TieFail=28.7%
  - **Section 6 Causal (`|E_hurt|=215`)**: rho_rec=90.2%, rho_int=9.8% | A1_strip=100.0%, B2a_post=90.2%, B1_pre=87.0%, B2b_raw=50.2%, A2_swap=59.5%, B3_cut=81.9%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 67.7% | 81.3% | 68.0% | 70.0% | 65.0% | 78.3% | 43.3% | 70.2% | 70.2% | 63.6% | 333 | 1.00 | 1.34e-62 |
| B1a | 70.6% | 85.3% | 74.7% | 70.0% | 70.0% | 73.3% | 50.0% | 35.3% | 63.5% | 62.4% | 501 | 1.00 | 2.14e-66 |
| **B1b** | 13.7% | 24.0% | 14.7% | 33.3% | 3.3% | 3.3% | 3.3% | 0.0% | 9.5% | -18.9% | 1879 | 1.00 | - |
| B2a | 61.0% | 86.7% | 72.0% | 70.7% | 43.3% | 60.0% | 33.3% | 46.4% | 59.7% | 58.4% | 382 | 1.00 | 2.15e-55 |
| B2b | 57.2% | 69.3% | 57.3% | 63.3% | 55.0% | 61.7% | 36.7% | 57.6% | 59.1% | 47.2% | 651 | 1.20 | 7.13e-51 |
| M1a | 34.1% | 36.0% | 18.7% | 70.0% | 5.0% | 71.7% | 3.3% | 37.3% | 40.2% | 21.3% | 1078 | 1.00 | 8.63e-32 |
| M1b | 67.4% | 81.3% | 68.0% | 70.0% | 65.0% | 76.7% | 43.3% | 69.8% | 69.8% | 63.2% | 341 | 1.01 | 2.25e-62 |
| M1c | 49.8% | 80.0% | 65.3% | 70.0% | 5.0% | 75.0% | 3.3% | 57.6% | 57.9% | 45.4% | 655 | 1.00 | 9.98e-48 |
| M2_B64 | 67.4% | 81.3% | 68.0% | 70.0% | 65.0% | 76.7% | 43.3% | 70.0% | 69.6% | 62.8% | 359 | 1.00 | 1.95e-61 |
| M2_B128 | 70.2% | 81.3% | 68.0% | 70.0% | 70.0% | 78.3% | 53.3% | 71.6% | 70.7% | 64.0% | 377 | 1.00 | 3.39e-64 |
| M2_B256 | 70.2% | 81.3% | 68.0% | 70.0% | 75.0% | 76.7% | 50.0% | 71.8% | 70.0% | 63.8% | 399 | 1.00 | 1.9e-64 |
| M2_Pre | 68.8% | 81.3% | 68.0% | 70.0% | 63.3% | 80.0% | 50.0% | 70.7% | 69.6% | 62.7% | 399 | 1.00 | 4e-63 |
| **M2** | 69.6% | 81.3% | 68.0% | 70.0% | 70.0% | 78.3% | 50.0% | 71.3% | 70.2% | 63.7% | 382 | 1.00 | 5.98e-64 |
| M2_Prec (Aux.) | 69.6% | 81.3% | 68.0% | 70.0% | 70.0% | 75.0% | 53.3% | 71.1% | 69.4% | 62.9% | 413 | 1.00 | 1.3e-64 (raw) |
| **Oracle_UB** | 87.2% | 94.7% | 92.0% | 86.7% | 86.7% | 90.0% | 73.3% | 88.4% | 86.6% | 82.3% | 308 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`strict341`**, n_diag=341) — ver=`v0.5.31` (`f7a0ebf-dirty`) | dtype=`torch.bfloat16` | wall=`27.0s` (calib=0.1s, eval=0.5s, diag=0.1s) | tau=0.650, esc=20.0%
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
| M2_B512 | 86.1% | 92.0% | 90.7% | 82.0% | 86.7% | 81.7% | 83.3% | 65.6% | 80.7% | 76.9% | 445 | 1.04 | 3.05e-64 (raw) |
| **Oracle_UB** | 94.2% | 97.3% | 98.7% | 92.7% | 93.3% | 93.3% | 90.0% | 92.2% | 92.3% | 88.4% | 300 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`diag491`**, n_diag=491) — ver=`v0.5.31` (`f7a0ebf-dirty`) | dtype=`torch.bfloat16` | wall=`26.6s` (calib=0.1s, eval=0.5s, diag=0.1s) | tau=0.650, esc=20.0%
  - **4-Regime**: Helped=0.7%, Hurt=50.0%, Flip=50.7%, TiePass=34.7%, TieFail=14.7%
  - **Section 6 Causal (`|E_hurt|=336`)**: rho_rec=96.1%, rho_int=3.9% | A1_strip=100.0%, B2a_post=96.1%, B1_pre=95.5%, B2b_raw=49.4%, A2_swap=70.2%, B3_cut=86.3%
  - *Eval-450 table identical to `strict341` above (omitted duplicate table).*

- **`Qwen/Qwen3.5-4B` (`strict341`**, n_diag=341) — ver=`v0.5.31` (`bc8c734`) | dtype=`torch.bfloat16` | sample(T=0.6, top_p=0.95, pp=1.5, seed=42) | wall=`25.4s` (calib=0.2s, eval=0.4s, diag=0.1s) | tau=0.625, esc=50.0%
  - **4-Regime**: Helped=0.7%, Hurt=40.4%, Flip=41.1%, TiePass=44.4%, TieFail=14.4%
  - **Section 6 Causal (`|E_hurt|=216`)**: rho_rec=90.7%, rho_int=7.9% | A1_strip=100.0%, B2a_post=90.7%, B1_pre=92.1%, B2b_raw=45.4%, A2_swap=60.6%, B3_cut=86.1%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 84.4% | 94.7% | 90.7% | 81.3% | 81.7% | 81.7% | 76.7% | 84.9% | 84.9% | 78.7% | 308 | 1.00 | 5.17e-49 |
| B1a | 81.7% | 93.3% | 93.3% | 82.0% | 66.7% | 78.3% | 76.7% | 39.1% | 72.2% | 75.3% | 378 | 1.00 | 6.43e-43 |
| **B1b** | 38.5% | 66.7% | 48.0% | 54.7% | 25.0% | 30.0% | 6.7% | 0.0% | 22.0% | 11.6% | 1675 | 1.00 | - |
| B2a | 82.7% | 96.0% | 92.0% | 81.3% | 76.7% | 76.7% | 73.3% | 49.6% | 75.0% | 77.4% | 317 | 1.00 | 1.26e-45 |
| B2b | 64.3% | 78.7% | 65.3% | 68.7% | 58.3% | 55.0% | 60.0% | 44.4% | 55.0% | 46.0% | 999 | 1.50 | 3.03e-28 |
| M1a | 52.9% | 74.7% | 48.0% | 81.3% | 30.0% | 76.7% | 6.7% | 43.1% | 53.1% | 43.2% | 950 | 1.00 | 5.5e-21 |
| M1b | 72.6% | 86.7% | 90.7% | 54.7% | 71.7% | 71.7% | 60.0% | 48.0% | 58.5% | 52.5% | 921 | 1.45 | 2.86e-33 |
| M1c | 57.4% | 86.7% | 88.0% | 54.7% | 30.0% | 78.3% | 6.7% | 36.7% | 48.4% | 39.4% | 1142 | 1.39 | 5.5e-21 |
| M2_B64 | 83.2% | 93.3% | 89.3% | 83.3% | 85.0% | 81.7% | 66.7% | 84.9% | 82.9% | 77.8% | 352 | 1.39 | 4.1e-46 |
| M2_B128 | 80.3% | 93.3% | 89.3% | 76.0% | 76.7% | 83.3% | 63.3% | 81.3% | 78.1% | 73.4% | 397 | 1.39 | 7.92e-42 |
| M2_B256 | 83.4% | 93.3% | 90.7% | 81.3% | 85.0% | 83.3% | 66.7% | 84.7% | 78.8% | 75.6% | 454 | 1.39 | 1.11e-47 |
| M2_Pre | 80.2% | 93.3% | 89.3% | 83.3% | 70.0% | 81.7% | 63.3% | 82.7% | 79.7% | 75.2% | 375 | 1.39 | 2.72e-44 |
| **M2** | 82.9% | 93.3% | 90.7% | 83.3% | 76.7% | 83.3% | 70.0% | 84.4% | 81.3% | 77.0% | 371 | 1.39 | 1.26e-45 |
| M2_Prec (Aux.) | 84.0% | 93.3% | 90.7% | 83.3% | 76.7% | 86.7% | 73.3% | 85.1% | 81.3% | 77.3% | 391 | 1.39 | 2.12e-47 (raw) |
| **Oracle_UB** | 94.5% | 97.3% | 97.3% | 90.7% | 93.3% | 91.7% | 96.7% | 93.8% | 92.0% | 88.1% | 283 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`strict341`**, n_diag=341) — ver=`v0.5.31` (`91bca89`) | dtype=`torch.bfloat16` | sample(T=0.6, top_p=0.95, pp=1.5, seed=43) | wall=`25.1s` (calib=0.1s, eval=0.4s, diag=0.1s) | tau=0.425, esc=83.3%
  - **4-Regime**: Helped=0.9%, Hurt=36.9%, Flip=37.8%, TiePass=46.9%, TieFail=15.3%
  - **Section 6 Causal (`|E_hurt|=195`)**: rho_rec=92.8%, rho_int=7.2% | A1_strip=100.0%, B2a_post=92.8%, B1_pre=91.8%, B2b_raw=47.2%, A2_swap=67.7%, B3_cut=90.3%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 81.8% | 92.0% | 92.0% | 82.0% | 76.7% | 85.0% | 63.3% | 83.8% | 83.8% | 77.5% | 313 | 1.00 | 5.52e-43 |
| B1a | 78.6% | 93.3% | 93.3% | 80.0% | 65.0% | 76.7% | 63.3% | 37.1% | 70.1% | 72.9% | 401 | 1.00 | 9.1e-35 |
| **B1b** | 42.9% | 73.3% | 56.0% | 53.3% | 21.7% | 30.0% | 23.3% | 0.0% | 23.3% | 14.5% | 1662 | 1.00 | - |
| B2a | 81.1% | 96.0% | 94.7% | 80.7% | 73.3% | 68.3% | 73.3% | 46.2% | 73.4% | 76.0% | 322 | 1.00 | 5.22e-36 |
| B2b | 48.4% | 77.3% | 61.3% | 58.7% | 26.7% | 40.0% | 26.7% | 13.1% | 32.2% | 24.3% | 1451 | 1.83 | 3.25e-06 |
| M1a | 58.8% | 78.7% | 58.7% | 82.0% | 26.7% | 80.0% | 26.7% | 43.8% | 54.6% | 47.2% | 951 | 1.00 | 1.78e-23 |
| M1b | 42.9% | 73.3% | 56.0% | 53.3% | 21.7% | 30.0% | 23.3% | 0.0% | 22.6% | 13.9% | 1693 | 2.00 | 1 |
| M1c | 51.3% | 73.3% | 56.0% | 53.3% | 21.7% | 80.0% | 23.3% | 10.0% | 31.3% | 24.7% | 1486 | 1.68 | 5.59e-09 |
| M2_B64 | 79.5% | 84.0% | 85.3% | 76.0% | 76.7% | 85.0% | 70.0% | 79.8% | 76.7% | 71.9% | 396 | 1.68 | 2.42e-31 |
| M2_B128 | 81.0% | 92.0% | 86.7% | 80.7% | 75.0% | 85.0% | 66.7% | 82.4% | 77.0% | 73.8% | 435 | 1.68 | 1.34e-36 |
| M2_B256 | 83.1% | 88.0% | 89.3% | 84.7% | 78.3% | 85.0% | 73.3% | 84.4% | 75.4% | 74.3% | 510 | 1.68 | 1.84e-40 |
| M2_Pre | 79.2% | 88.0% | 89.3% | 76.0% | 73.3% | 85.0% | 63.3% | 80.2% | 74.2% | 71.2% | 451 | 1.68 | 2.96e-32 |
| **M2** | 79.4% | 88.0% | 89.3% | 76.0% | 75.0% | 85.0% | 63.3% | 80.4% | 74.4% | 71.4% | 454 | 1.68 | 1.5e-31 |
| M2_Prec (Aux.) | 80.0% | 88.0% | 89.3% | 76.0% | 75.0% | 88.3% | 63.3% | 80.9% | 74.2% | 71.4% | 475 | 1.68 | 7.45e-33 (raw) |
| **Oracle_UB** | 93.3% | 98.7% | 96.0% | 88.7% | 95.0% | 91.7% | 90.0% | 92.9% | 91.0% | 87.1% | 288 | 1.00 | - |

- **`Qwen/Qwen3.5-4B` (`strict341`**, n_diag=341) — ver=`v0.5.31` (`73e20d2`) | dtype=`torch.bfloat16` | sample(T=0.6, top_p=0.95, pp=1.5, seed=44) | wall=`25.5s` (calib=0.1s, eval=0.6s, diag=0.1s) | tau=0.525, esc=50.0%
  - **4-Regime**: Helped=2.4%, Hurt=37.8%, Flip=40.2%, TiePass=45.1%, TieFail=14.7%
  - **Section 6 Causal (`|E_hurt|=205`)**: rho_rec=92.7%, rho_int=7.3% | A1_strip=100.0%, B2a_post=92.7%, B1_pre=92.2%, B2b_raw=46.8%, A2_swap=62.9%, B3_cut=87.3%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 82.2% | 93.3% | 88.0% | 78.7% | 81.7% | 81.7% | 70.0% | 82.9% | 82.9% | 76.5% | 321 | 1.00 | 5.25e-37 |
| B1a | 79.8% | 96.0% | 96.0% | 80.0% | 65.0% | 78.3% | 63.3% | 39.1% | 72.0% | 74.4% | 381 | 1.00 | 1.73e-38 |
| **B1b** | 41.0% | 69.3% | 60.0% | 53.3% | 31.7% | 28.3% | 3.3% | 0.2% | 23.0% | 14.4% | 1660 | 1.00 | - |
| B2a | 81.8% | 94.7% | 96.0% | 80.0% | 68.3% | 78.3% | 73.3% | 49.8% | 73.6% | 76.6% | 315 | 1.00 | 3.37e-38 |
| B2b | 58.4% | 77.3% | 66.7% | 63.3% | 58.3% | 55.0% | 30.0% | 37.6% | 49.3% | 40.8% | 1071 | 1.54 | 2.58e-16 |
| M1a | 57.1% | 77.3% | 60.0% | 78.7% | 38.3% | 75.0% | 13.3% | 42.4% | 53.2% | 46.0% | 956 | 1.00 | 1.73e-18 |
| M1b | 69.4% | 85.3% | 88.0% | 53.3% | 73.3% | 73.3% | 43.3% | 46.9% | 57.9% | 50.9% | 909 | 1.45 | 1.12e-22 |
| M1c | 58.4% | 85.3% | 86.7% | 53.3% | 38.3% | 76.7% | 10.0% | 36.2% | 48.7% | 39.9% | 1129 | 1.39 | 2.8e-16 |
| M2_B64 | 81.3% | 92.0% | 88.0% | 78.0% | 81.7% | 81.7% | 66.7% | 82.2% | 80.3% | 74.4% | 391 | 1.39 | 3.96e-38 |
| M2_B128 | 84.1% | 92.0% | 88.0% | 81.3% | 91.7% | 81.7% | 70.0% | 84.9% | 81.4% | 76.5% | 417 | 1.39 | 4.76e-42 |
| M2_B256 | 84.9% | 92.0% | 88.0% | 79.3% | 88.3% | 81.7% | 80.0% | 84.4% | 78.6% | 75.3% | 455 | 1.39 | 8.73e-41 |
| M2_Pre | 80.2% | 92.0% | 88.0% | 78.0% | 76.7% | 80.0% | 66.7% | 81.3% | 78.4% | 73.1% | 412 | 1.39 | 1.86e-36 |
| **M2** | 84.7% | 92.0% | 88.0% | 78.0% | 91.7% | 81.7% | 76.7% | 84.2% | 81.0% | 76.0% | 412 | 1.39 | 3.1e-41 |
| M2_Prec (Aux.) | 86.1% | 92.0% | 88.0% | 78.0% | 91.7% | 90.0% | 76.7% | 85.3% | 81.5% | 76.7% | 431 | 1.39 | 2e-45 (raw) |
| **Oracle_UB** | 92.7% | 97.3% | 96.0% | 89.3% | 95.0% | 91.7% | 86.7% | 92.7% | 90.8% | 86.7% | 296 | 1.00 | - |

- **`Qwen/Qwen3.5-9B` (`strict341`**, n_diag=341) — ver=`v0.5.31` (`d4bccf3`) | dtype=`torch.bfloat16` | wall=`25.3s` (calib=0.1s, eval=0.3s, diag=0.1s) | tau=0.475, esc=43.3%
  - **4-Regime**: Helped=0.9%, Hurt=43.8%, Flip=44.7%, TiePass=43.1%, TieFail=12.2%
  - **Section 6 Causal (`|E_hurt|=226`)**: rho_rec=95.1%, rho_int=4.9% | A1_strip=100.0%, B2a_post=95.1%, B1_pre=93.4%, B2b_raw=47.8%, A2_swap=80.1%, B3_cut=92.5%

| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | 87.5% | 92.0% | 93.3% | 81.3% | 88.3% | 86.7% | 83.3% | 86.9% | 86.9% | 80.7% | 310 | 1.00 | 5.03e-52 |
| B1a | 84.3% | 96.0% | 93.3% | 81.3% | 71.7% | 80.0% | 83.3% | 33.6% | 73.0% | 75.5% | 447 | 1.00 | 4.11e-49 |
| **B1b** | 38.3% | 62.7% | 52.0% | 52.0% | 23.3% | 26.7% | 13.3% | 0.0% | 22.2% | 10.6% | 1668 | 1.00 | - |
| B2a | 78.4% | 85.3% | 70.7% | 82.7% | 80.0% | 81.7% | 70.0% | 63.3% | 74.0% | 74.7% | 256 | 1.00 | 3.53e-32 |
| B2b | 70.1% | 81.3% | 76.0% | 66.7% | 68.3% | 65.0% | 63.3% | 51.6% | 60.6% | 52.5% | 895 | 1.42 | 7.17e-33 |
| M1a | 56.8% | 72.0% | 57.3% | 81.3% | 30.0% | 80.0% | 20.0% | 43.8% | 54.2% | 45.6% | 952 | 1.00 | 1.45e-26 |
| M1b | 54.4% | 86.7% | 89.3% | 57.3% | 33.3% | 40.0% | 20.0% | 40.4% | 49.3% | 36.4% | 1158 | 1.56 | 2.94e-19 |
| M1c | 58.8% | 85.3% | 86.7% | 57.3% | 25.0% | 81.7% | 16.7% | 44.2% | 53.2% | 41.9% | 1062 | 1.32 | 1.32e-23 |
| M2_B64 | 86.5% | 93.3% | 93.3% | 80.7% | 85.0% | 86.7% | 80.0% | 86.2% | 84.5% | 79.2% | 353 | 1.32 | 1.97e-49 |
| M2_B128 | 84.6% | 93.3% | 93.3% | 79.3% | 85.0% | 86.7% | 70.0% | 85.1% | 82.2% | 77.6% | 378 | 1.32 | 7.29e-50 |
| M2_B256 | 85.9% | 93.3% | 92.0% | 83.3% | 85.0% | 85.0% | 76.7% | 86.4% | 81.3% | 78.0% | 424 | 1.32 | 1.77e-51 |
| M2_Pre | 86.8% | 93.3% | 92.0% | 80.7% | 91.7% | 86.7% | 76.7% | 86.7% | 83.9% | 79.0% | 382 | 1.32 | 8.51e-51 |
| **M2** | 86.0% | 93.3% | 92.0% | 80.7% | 86.7% | 86.7% | 76.7% | 86.0% | 83.3% | 78.4% | 380 | 1.32 | 5.68e-50 |
| M2_Prec (Aux.) | 86.3% | 93.3% | 92.0% | 80.7% | 86.7% | 88.3% | 76.7% | 86.2% | 82.8% | 78.2% | 401 | 1.32 | 3.24e-51 (raw) |
| **Oracle_UB** | 93.9% | 98.7% | 96.0% | 90.7% | 96.7% | 91.7% | 90.0% | 93.8% | 92.5% | 87.9% | 292 | 1.00 | - |

#### 6. Optional W7 Extensions
- **`HuggingFaceTB__SmolLM3-3B`**: DenseKVEvict(n=80: a=0.00:86.2%) | PrefillMLP(layer_-1, tau=0.850, Macro=66.8%, OAA250=69.6%, U0.05=63.4%, Tok=308, OrigPrecF1=86.0%->81.1%, ParaPrecF1=78.9%->81.7%) | Stage2SolveFirst(replayed=139, Macro=68.5%->67.0%, OAA250=72.0%->69.6%) | IFBenchOOD(N=300: B0=17.7%/B1b=13.0%/M2=18.0%, OAA250=18.0%, Tok=663, p_vs_B1b=0.0275) | StateAdapt(M2_ZeroTok=69.1%, M2_Adapt=77.2%@531t)
- **`Qwen__Qwen3.5-0.8B`**: StateAdapt(M2_ZeroTok=55.0%, M2_Adapt=58.5%@639t)
- **`Qwen__Qwen3.5-2B`**: StateAdapt(M2_ZeroTok=67.7%, M2_Adapt=75.2%@613t)
- **`Qwen__Qwen3.5-4B`**: DeltaNetBlend(n=336: a=0.00:93.2%, a=0.25:95.2%, a=0.50:95.2%, a=0.75:94.0%, a=1.00:91.7%) | DeltaNetConvKV(n=336: a=0.00:85.7%, a=0.25:88.1%, a=0.50:89.9%, a=0.75:91.7%, a=1.00:95.8%) | PrefillMLP(layer_-1, tau=0.525, Macro=84.1%, OAA250=84.9%, U0.05=77.8%, Tok=357, OrigPrecF1=86.0%->81.2%, ParaPrecF1=78.9%->82.6%) | Stage2SolveFirst(replayed=17, Macro=84.6%->84.3%, OAA250=84.9%->84.4%) | IFBenchOOD(N=300: B0=28.7%/B1b=2.0%/M2=31.3%, OAA250=31.3%, Tok=443, p_vs_B1b=1.73e-24) | StateAdapt(M2_ZeroTok=83.7%, M2_Adapt=86.9%@621t)
- **`Qwen__Qwen3.5-9B`**: StateAdapt(M2_ZeroTok=86.8%, M2_Adapt=88.5%@637t)

#### 7. Phase 7 Statistical Audit & Gate-4 Summary
| Model | Version | Clean-446 (max delta) | M2 Macro | B1b Macro | M2 OAA_250 | Delta OAA_250 [95% CI] | Tok Red | p_holm vs B1b | p(M2 vs B0) | p(M2 vs Pre) | Gate 4 | R2 (rec / int) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HuggingFaceTB/SmolLM3-3B` | `v0.5.31+m5` | 448 (0.5 pp) | 68.5% | 69.4% | 72.0% | +68.4 [64.0, 72.7] pp | 62.0% | 1 | 0.1524 | 0.7539 | CHECK | 62.5% / 37.5% |
| `Qwen/Qwen3.5-0.8B` | `v0.5.31+m5` | 448 (0.2 pp) | 53.6% | 2.7% | 56.2% | +56.2 [51.6, 60.9] pp | 78.2% | 7.792e-67 | 1 | 0.2295 | PASS | 76.1% / 23.9% |
| `Qwen/Qwen3.5-2B` | `v0.5.31+m5` | 448 (0.1 pp) | 69.6% | 13.7% | 71.3% | +71.3 [67.1, 75.6] pp | 79.7% | 5.977e-64 | 0.3593 | 0.6072 | PASS | 90.2% / 9.8% |
| `Qwen/Qwen3.5-4B` | `v0.5.31+m7` | 448 (0.4 pp) | 84.6% | 30.2% | 84.9% | +84.7 [81.3, 88.0] pp | 79.2% | 5.104e-62 | 1 | 0.424 | PASS | 96.1% / 3.9% |
| `Qwen/Qwen3.5-9B` | `v0.5.31+m4` | 448 (0.4 pp) | 86.0% | 38.3% | 86.0% | +86.0 [82.7, 89.1] pp | 77.2% | 5.684e-50 | 0.6177 | 0.5488 | PASS | 95.1% / 4.9% |

#### 8. Confirmatory Audit & Appendix B Summary
- **`HuggingFaceTB/SmolLM3-3B` (Confirmatory 4-pair Holm)**: M2_vs_B1b: p_holm4=0.961 (midp=0.817), M2_vs_B0: p_holm4=0.61 (midp=0.476), M2_vs_B2a: p_holm4=0.961 (midp=0.817), M2_vs_M1c: p_holm4=0.961 (midp=0.817)
- **`HuggingFaceTB/SmolLM3-3B` (Conformal & Oracle Audit)**: DevDeg=11.7%, CRC<=13.1%, Hoef<=34.7%, EvalDeg(vs B1b)=10.44% (47/450), EvalDeg(vs B0)=4.22% (19/450), OracleRet(Macro)=77.7% (1.29x tok)
- **`HuggingFaceTB/SmolLM3-3B` (Probe Horizons)**: K=1:68.4%(366t), K=4:68.3%(358t), K=8:67.2%(335t), K=16:67.2%(331t), K=32:68.5%(371t)
- **`HuggingFaceTB/SmolLM3-3B` (Gate-2 L2 Sensitivity)**: C=0.1:68.5%(esc=50%), C=0.25:68.5%(esc=47%), C=0.5:68.7%(esc=47%), C=1:68.5%(esc=48%)
- **`Qwen/Qwen3.5-0.8B` (Confirmatory 4-pair Holm)**: M2_vs_B1b: p_holm4=2.6e-67 (midp=1.31e-67), M2_vs_B0: p_holm4=1 (midp=1), M2_vs_B2a: p_holm4=4.01e-06 (midp=2.89e-06), M2_vs_M1c: p_holm4=6.35e-22 (midp=3.18e-22)
- **`Qwen/Qwen3.5-0.8B` (Conformal & Oracle Audit)**: DevDeg=0.0%, CRC<=1.6%, Hoef<=23.0%, EvalDeg(vs B1b)=0.44% (2/450), EvalDeg(vs B0)=5.11% (23/450), OracleRet(Macro)=73.1% (1.29x tok)
- **`Qwen/Qwen3.5-0.8B` (Probe Horizons)**: K=1:55.4%(429t), K=4:54.1%(439t), K=8:54.9%(435t), K=16:55.4%(428t), K=32:53.6%(436t)
- **`Qwen/Qwen3.5-0.8B` (Gate-2 L2 Sensitivity)**: C=0.1:54.3%(esc=17%), C=0.25:53.6%(esc=32%), C=0.5:53.6%(esc=25%), C=1:53.6%(esc=28%)
- **`Qwen/Qwen3.5-2B` (Confirmatory 4-pair Holm)**: M2_vs_B1b: p_holm4=2.66e-64 (midp=1.36e-64), M2_vs_B0: p_holm4=0.359 (midp=0.263), M2_vs_B2a: p_holm4=0.05 (midp=0.0398), M2_vs_M1c: p_holm4=2.08e-17 (midp=1.04e-17)
- **`Qwen/Qwen3.5-2B` (Conformal & Oracle Audit)**: DevDeg=0.0%, CRC<=1.6%, Hoef<=23.0%, EvalDeg(vs B1b)=1.11% (5/450), EvalDeg(vs B0)=1.56% (7/450), OracleRet(Macro)=79.8% (1.24x tok)
- **`Qwen/Qwen3.5-2B` (Probe Horizons)**: K=1:69.8%(410t), K=4:69.6%(381t), K=8:69.6%(381t), K=16:70.3%(403t), K=32:69.6%(382t)
- **`Qwen/Qwen3.5-2B` (Gate-2 L2 Sensitivity)**: C=0.1:70.5%(esc=28%), C=0.25:69.6%(esc=17%), C=0.5:70.9%(esc=32%), C=1:71.3%(esc=43%)
- **`Qwen/Qwen3.5-4B` (Confirmatory 4-pair Holm)**: M2_vs_B1b: p_holm4=1.86e-62 (midp=9.4e-63), M2_vs_B0: p_holm4=1 (midp=1), M2_vs_B2a: p_holm4=1 (midp=1), M2_vs_M1c: p_holm4=8.13e-20 (midp=4.07e-20)
- **`Qwen/Qwen3.5-4B` (Conformal & Oracle Audit)**: DevDeg=1.7%, CRC<=3.3%, Hoef<=24.7%, EvalDeg(vs B1b)=0.67% (3/450), EvalDeg(vs B0)=1.11% (5/450), OracleRet(Macro)=89.7% (1.20x tok)
- **`Qwen/Qwen3.5-4B` (Probe Horizons)**: K=1:84.7%(356t), K=4:84.4%(369t), K=8:84.7%(356t), K=16:84.5%(358t), K=32:84.6%(360t)
- **`Qwen/Qwen3.5-4B` (Gate-2 L2 Sensitivity)**: C=0.1:84.8%(esc=20%), C=0.25:84.6%(esc=20%), C=0.5:84.6%(esc=20%), C=1:84.6%(esc=20%)
- **`Qwen/Qwen3.5-9B` (Confirmatory 4-pair Holm)**: M2_vs_B1b: p_holm4=2.53e-50 (midp=1.3e-50), M2_vs_B0: p_holm4=0.618 (midp=0.511), M2_vs_B2a: p_holm4=0.00352 (midp=0.00253), M2_vs_M1c: p_holm4=2e-27 (midp=1.03e-27)
- **`Qwen/Qwen3.5-9B` (Conformal & Oracle Audit)**: DevDeg=1.7%, CRC<=3.3%, Hoef<=24.7%, EvalDeg(vs B1b)=1.11% (5/450), EvalDeg(vs B0)=4.44% (20/450), OracleRet(Macro)=91.5% (1.30x tok)
- **`Qwen/Qwen3.5-9B` (Probe Horizons)**: K=1:85.2%(382t), K=4:85.8%(385t), K=8:85.3%(379t), K=16:85.7%(382t), K=32:86.0%(380t)
- **`Qwen/Qwen3.5-9B` (Gate-2 L2 Sensitivity)**: C=0.1:85.9%(esc=50%), C=0.25:86.0%(esc=43%), C=0.5:85.9%(esc=42%), C=1:86.0%(esc=33%)
- **Pooled E_hurt (diag_341_strict)**: A1=100.0% (912/912), B2a=88.6% (808/912), p_midp(B2a vs A1)=4.93e-32
- **Pooled E_hurt (diag_491)**: A1=100.0% (416/416), B2a=89.7% (373/416)
- **R6 Multi-Seed (`Qwen/Qwen3.5-4B`, k=3 [42, 43, 44], T=0.6)**: M2 Macro=82.3+/-2.7%, Tok=412+/-42, Delta_B1b=+41.5 pp
<!-- CBARR_RUN_END:20261120T200335Z_pr6 -->
