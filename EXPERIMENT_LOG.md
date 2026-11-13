# C-BARR Experiment & Runtime Log

Tracks every test, mock dry-run, and live GPU experiment execution (wall-clock duration per stage, hardware configuration, command flags, raw log pointers, and headline metrics).

- Live runs via `bash scripts/run_gpu_phases_4_to_6.sh` automatically stream output to `outputs/logs/cbarr_<RUN_ID>.log`, save per-stage JSONL telemetry (`outputs/logs/timings_<RUN_ID>.jsonl` & `outputs/experiment_history.jsonl`), and append an anonymized summary entry below (even on failure, via `EXIT` trap in `scripts/log_experiment.py`).
- Keep entries free of local usernames, email addresses, or private paths outside the repository.

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
- **Live CUDA Lab Runs**: *Pending execution on lab GPU.*
