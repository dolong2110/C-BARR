"""End-to-end CPU tests for Phases 4-7 automation, metrics, analysis, W7 extensions, and anonymity."""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_matrix import resolve_device_and_dtype, verify_manifest_integrity
from src.metrics import KNOWN_UPSTREAM_DEFECT_KEYS, SLICE_ORDER, filter_clean_446, wilson_ci


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(payload, encoding="utf-8")


def _build_mini_frozen_dir(tmp_data: pathlib.Path) -> None:
    """Slices minimal representative items from frozen data/ splits and seals them with SHA-256."""
    src_data = ROOT / "data"
    eval_full = _read_jsonl(src_data / "eval_450.jsonl")
    seed_full = _read_jsonl(src_data / "seed_60.jsonl")
    dev_full = _read_jsonl(src_data / "dev_60.jsonl")
    diag341 = _read_jsonl(src_data / "ifeval_diag_341_strict.jsonl")
    diag491 = _read_jsonl(src_data / "ifeval_diag_491.jsonl")

    # 2 items per slice for seed/dev (12 each), 3 items per slice + defect item for eval (18-19 items)
    def take_per_slice(rows: list[dict], k: int) -> list[dict]:
        out: list[dict] = []
        for s in SLICE_ORDER:
            out.extend([r for r in rows if r["slice"] == s][:k])
        return out

    eval_mini = take_per_slice(eval_full, 3)
    # Inject any real defect row from eval_450 if not already sampled so Clean-446 delta > 0 is exercised
    eval_uids = {r["uid"] for r in eval_mini}
    defect_rows = [r for r in eval_full if r.get("key") in KNOWN_UPSTREAM_DEFECT_KEYS]
    if defect_rows and defect_rows[0]["uid"] not in eval_uids:
        eval_mini.append(defect_rows[0])

    # Preserve subset invariant: strict341 is a strict prefix of diag491
    d341_mini = diag341[:4]
    d341_uids = {r["uid"] for r in d341_mini}
    extra_491 = [r for r in diag491 if r["uid"] not in d341_uids][:2]
    d491_mini = d341_mini + extra_491

    splits = {
        "seed_60.jsonl": take_per_slice(seed_full, 2),
        "dev_60.jsonl": take_per_slice(dev_full, 2),
        "eval_450.jsonl": eval_mini,
        "ifeval_diag_341_strict.jsonl": d341_mini,
        "ifeval_diag_491.jsonl": d491_mini,
    }
    manifest = {}
    for fname, rows in splits.items():
        p = tmp_data / fname
        _write_jsonl(p, rows)
        manifest[fname] = hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_data / "manifest_sha256.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


class TestPhases4To7(unittest.TestCase):
    def test_real_manifest_integrity_and_tamper_detection(self) -> None:
        manifest = verify_manifest_integrity(ROOT / "data")
        self.assertIn("eval_450.jsonl", manifest)
        with tempfile.TemporaryDirectory() as td:
            td_path = pathlib.Path(td)
            _build_mini_frozen_dir(td_path)
            verify_manifest_integrity(td_path)
            (td_path / "seed_60.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_manifest_integrity(td_path)

    def test_wilson_ci_and_clean_446_filter(self) -> None:
        rate, lo, hi = wilson_ci(30, 40)
        self.assertAlmostEqual(rate, 0.75)
        self.assertGreater(lo, 0.55)
        self.assertLess(hi, 0.90)
        self.assertEqual(wilson_ci(0, 0), (0.0, 0.0, 0.0))

        fake_items = [
            {"uid": "a", "dataset": "gsm8k"},
            {"uid": "b", "dataset": "ifeval", "key": 30, "has_upstream_defect": True},
            {"uid": "c", "dataset": "ifeval", "key": 3311},
            {"uid": "d", "dataset": "ifeval", "key": 9999, "has_upstream_defect": False},
        ]
        fake_runs = [{"z": 1}, {"z": 0}, {"z": 0}, {"z": 1}]
        c_items, c_runs = filter_clean_446(fake_items, fake_runs)
        self.assertEqual([x["uid"] for x in c_items], ["a", "d"])
        self.assertEqual(c_runs, [{"z": 1}, {"z": 1}])

    def test_end_to_end_mock_matrix_analysis_w7_and_anonymity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = pathlib.Path(td)
            tmp_data = td_path / "data"
            tmp_out = td_path / "outputs"
            tmp_figs = td_path / "figures"
            tmp_data.mkdir()

            _build_mini_frozen_dir(tmp_data)

            # 1. Calibration-only must not evaluate eval_450
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_matrix.py"),
                    "--model",
                    "Qwen/Qwen3.5-4B",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(tmp_out),
                    "--calibrate-only",
                    "--mock",
                ],
                check=True,
                cwd=str(ROOT),
            )
            q4_dir = tmp_out / "Qwen__Qwen3.5-4B"
            calib_path = q4_dir / "calibration_summary.json"
            self.assertTrue(calib_path.exists())
            self.assertFalse((q4_dir / "eval_450_cache.jsonl").exists())
            calib_rep = json.loads(calib_path.read_text(encoding="utf-8"))
            clf_m = calib_rep["classifier_metrics"]
            for split_k in ("seed_train_at_05", "seed_train_at_tau", "dev_all_at_tau", "dev_gate2_at_tau"):
                self.assertIn(split_k, clf_m)
                for metric_k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                    val = float(clf_m[split_k][metric_k])
                    self.assertGreaterEqual(val, 0.0)
                    self.assertLessEqual(val, 1.0)
            self.assertGreaterEqual(len(clf_m.get("tau_sweep", [])), 10)

            # 2. strict341 run
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_matrix.py"),
                    "--model",
                    "Qwen/Qwen3.5-4B",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(tmp_out),
                    "--strict-diag",
                    "--mock",
                ],
                check=True,
                cwd=str(ROOT),
            )
            calib_data = json.loads(calib_path.read_text())
            self.assertIn("training_params", calib_data)
            self.assertEqual(calib_data["training_params"]["lam_utility"], 0.05)

            strict_path = q4_dir / "matrix_summary_strict341.json"
            self.assertTrue(strict_path.exists())
            strict_rep = json.loads(strict_path.read_text())
            self.assertIn("training_params", strict_rep)
            strict_mtime_before = strict_path.stat().st_mtime_ns

            # 3. full diag491 + --include-b512 must NOT clobber matrix_summary_strict341.json
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_matrix.py"),
                    "--model",
                    "Qwen/Qwen3.5-4B",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(tmp_out),
                    "--include-b512",
                    "--mock",
                ],
                check=True,
                cwd=str(ROOT),
            )
            self.assertEqual(strict_path.stat().st_mtime_ns, strict_mtime_before)
            diag491_rep = json.loads((q4_dir / "matrix_summary_diag491.json").read_text())
            self.assertEqual(diag491_rep["diag_pool"], "diag491")
            self.assertIn("M2_B512", diag491_rep["auxiliary_systems"])
            self.assertIn("overall_wilson_ci95", diag491_rep["section6_diagnostics"]["A1_strip"])
            self.assertGreaterEqual(diag491_rep["timing_sec"]["total_wall"], 0.0)
            self.assertEqual(diag491_rep["runtime"]["device"], "mock")

            # 4. Dense control SmolLM3-3B mock run for multi-model table/figure parity
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_matrix.py"),
                    "--model",
                    "HuggingFaceTB/SmolLM3-3B",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(tmp_out),
                    "--strict-diag",
                    "--mock",
                ],
                check=True,
                cwd=str(ROOT),
            )

            # 5. Phase 7 analyze_and_plot.py
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/analyze_and_plot.py"),
                    "--out-dir",
                    str(tmp_out),
                    "--fig-dir",
                    str(tmp_figs),
                    "--n-boot",
                    "400",
                ],
                check=True,
                cwd=str(ROOT),
            )
            stats = json.loads((tmp_out / "stats_summary.json").read_text())
            self.assertIn("Qwen/Qwen3.5-4B", stats["models"])
            self.assertIn("HuggingFaceTB/SmolLM3-3B", stats["models"])
            q4_stats = stats["models"]["Qwen/Qwen3.5-4B"]
            self.assertIn("M2_minus_B1b", q4_stats["paired_bootstrap_95ci"])
            self.assertIn("M2_vs_B0", q4_stats["targeted_mcnemar"])
            self.assertIn("conformal_risk_audit", q4_stats)
            self.assertIn("dev_calibration", q4_stats["conformal_risk_audit"])
            self.assertIn("within_hoeffding_ucb", q4_stats["conformal_risk_audit"])
            self.assertIn("oracle_efficiency_audit", q4_stats)
            self.assertIn("variants", q4_stats["oracle_efficiency_audit"])
            self.assertIn("probe_horizon_ablation", q4_stats)
            for k_key in ("K_1", "K_4", "K_8", "K_16", "K_32"):
                self.assertIn(k_key, q4_stats["probe_horizon_ablation"])
            self.assertIn("R4_strict_underpowered_use_diag491", q4_stats["gate_audit"]["contingencies"])

            tex = (tmp_out / "paper_tables.tex").read_text()
            self.assertIn(r"\label{tab:main_results}", tex)
            self.assertIn(r"\label{tab:scale_and_pathology}", tex)
            self.assertIn(r"\label{tab:diagnostic_control}", tex)

            for pdf_name in (
                "fig1_pareto_frontier.pdf",
                "fig2_bsr_sla_curve.pdf",
                "fig3_ehurt_causal_recovery.pdf",
            ):
                p = tmp_figs / pdf_name
                self.assertTrue(p.exists(), f"Missing {p}")
                self.assertTrue(p.read_bytes().startswith(b"%PDF-"))

            # 6. W7 optional extensions (mock mode)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/ext_deltanet_blend.py"),
                    "--model",
                    "Qwen/Qwen3.5-4B",
                    "--out-dir",
                    str(tmp_out),
                    "--mock",
                ],
                check=True,
                cwd=str(ROOT),
            )
            blend_res = json.loads((q4_dir / "ext_deltanet_blend.json").read_text())
            self.assertAlmostEqual(blend_res["sweep"]["alpha_0.50"]["blended_mean_val"], 1.0, places=4)
            self.assertEqual(blend_res["sweep"]["alpha_0.50"]["n_blended_layers"], 1)

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/ext_prefill_mlp.py"),
                    "--model",
                    "Qwen/Qwen3.5-4B",
                    "--out-dir",
                    str(tmp_out),
                    "--mock",
                ],
                check=True,
                cwd=str(ROOT),
            )
            mlp_res = json.loads((q4_dir / "prefill_mlp_summary.json").read_text())
            self.assertIn("macro_acc", mlp_res["eval_table"])
            self.assertIn("gate1_semantic_probe", mlp_res)
            self.assertIn("ifevalpp_paraphrase_stress", mlp_res)
            self.assertTrue((q4_dir / "prefill_hidden_mock.npz").exists())

            # 6b. Experiment logger records stage timings, training params, per-loop tables, and upserts cleanly
            timings_file = tmp_out / "timings_test.jsonl"
            timings_file.write_text(
                json.dumps({"stage": "Gate 0", "status": "PASS", "duration_sec": 1.25}) + "\n",
                encoding="utf-8",
            )
            params_payload = json.dumps({"seed": 42, "lam_utility": 0.05, "logreg_C": 1.0})
            md_log = tmp_out / "test_exp_log.md"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/log_experiment.py"),
                    "--run-id",
                    "test_run_001",
                    "--status",
                    "RUNNING (after Qwen3.5-4B)",
                    "--loop-name",
                    "Phase 4: Qwen/Qwen3.5-4B (strict341)",
                    "--command",
                    "mock test",
                    "--out-dir",
                    str(tmp_out),
                    "--timings-file",
                    str(timings_file),
                    "--params-json",
                    params_payload,
                    "--md-log",
                    str(md_log),
                    "--total-sec",
                    "1.25",
                ],
                check=True,
                cwd=str(ROOT),
            )
            # Second call with same run-id should upsert in-place instead of duplicating headings
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/log_experiment.py"),
                    "--run-id",
                    "test_run_001",
                    "--status",
                    "PASSED",
                    "--loop-name",
                    "Phase 7: Analysis",
                    "--command",
                    "mock test",
                    "--out-dir",
                    str(tmp_out),
                    "--timings-file",
                    str(timings_file),
                    "--params-json",
                    params_payload,
                    "--md-log",
                    str(md_log),
                    "--total-sec",
                    "2.50",
                ],
                check=True,
                cwd=str(ROOT),
            )
            hist_rows = _read_jsonl(tmp_out / "experiment_history.jsonl")
            self.assertEqual(len(hist_rows), 2)
            self.assertEqual(hist_rows[-1]["status"], "PASSED")
            md_txt = md_log.read_text(encoding="utf-8")
            self.assertEqual(md_txt.count("### Run `test_run_001`"), 1)
            self.assertIn("PASSED", md_txt)
            self.assertNotIn("Gate-3 Live 12-Item Smoke Check", md_txt)
            self.assertIn("Training & Experiment Parameters", md_txt)
            self.assertIn("Calibration & Gate-2 Classifier Results", md_txt)
            self.assertIn("Evaluation Matrix & Section 6 Diagnostics", md_txt)
            self.assertIn("Oracle_UB", md_txt)
            self.assertIn("DeltaNetBlend", md_txt)
            self.assertIn("PrefillMLP", md_txt)
            self.assertIn("Phase 7 Statistical Audit", md_txt)
            self.assertIn("Qwen/Qwen3.5-4B", md_txt)

        # 7. Anonymity linter passes on real repo paper/
        subprocess.run(["bash", str(ROOT / "scripts/check_anonymity.sh")], check=True, cwd=str(ROOT))

    def test_resolve_device_and_dtype_hardware_safety(self) -> None:
        """Auto-dtype selects safe CPU fallback and supports explicit dtype overrides."""
        dev, dt = resolve_device_and_dtype("cpu", "auto")
        self.assertEqual(dev, "cpu")
        self.assertEqual(dt, torch.float32)

        self.assertEqual(resolve_device_and_dtype("cpu", "float16")[1], torch.float16)
        self.assertEqual(resolve_device_and_dtype("cpu", "bfloat16")[1], torch.bfloat16)
        self.assertEqual(resolve_device_and_dtype("cpu", "float32")[1], torch.float32)

        with self.assertRaises(ValueError):
            resolve_device_and_dtype("cpu", "fp8")

    def test_colab_kaggle_notebook_valid_json_and_python_syntax(self) -> None:
        """All generated Colab/Kaggle notebooks are valid nbformat 4 JSON with AST-valid Python cells."""
        notebook_expectations = {
            "C_BARR_All_In_One.ipynb": (
                "PIPELINE_MODE = 'all_in_one'",
                "RUN_CALIBRATE_ONLY = True",
                "RUN_PRIMARY_MATRIX = True",
                "RUN_SCALE_SWEEP = True",
                "RUN_PHASE7_ANALYSIS = True",
                "UNPACK_SYNCED_RESULTS_ON_START = True",
                "SYNC_INCLUDE_LEDGER = True",
            ),
            "C_BARR_1_Worker_Qwen4B.ipynb": (
                "PIPELINE_MODE = 'worker1_4b'",
                "RUN_CALIBRATE_ONLY = True",
                "RUN_PRIMARY_MATRIX = True",
                "RUN_SCALE_SWEEP = False",
                "RUN_PHASE7_ANALYSIS = False",
                "UNPACK_SYNCED_RESULTS_ON_START = False",
                "SYNC_INCLUDE_LEDGER = False",
                "Qwen/Qwen3.5-4B",
            ),
            "C_BARR_2_Worker_SmolLM3.ipynb": (
                "PIPELINE_MODE = 'worker2_smollm3'",
                "RUN_CALIBRATE_ONLY = False",
                "RUN_PRIMARY_MATRIX = True",
                "RUN_SCALE_SWEEP = False",
                "RUN_PHASE7_ANALYSIS = False",
                "UNPACK_SYNCED_RESULTS_ON_START = False",
                "SYNC_INCLUDE_LEDGER = False",
                "HuggingFaceTB/SmolLM3-3B",
            ),
            "C_BARR_3_Worker_Qwen2B.ipynb": (
                "PIPELINE_MODE = 'worker3_2b'",
                "RUN_CALIBRATE_ONLY = False",
                "RUN_PRIMARY_MATRIX = False",
                "RUN_SCALE_SWEEP = True",
                "RUN_PHASE7_ANALYSIS = False",
                "UNPACK_SYNCED_RESULTS_ON_START = False",
                "SYNC_INCLUDE_LEDGER = False",
                "Qwen/Qwen3.5-2B",
            ),
            "C_BARR_4_Worker_Qwen08B.ipynb": (
                "PIPELINE_MODE = 'worker4_08b'",
                "RUN_CALIBRATE_ONLY = False",
                "RUN_PRIMARY_MATRIX = False",
                "RUN_SCALE_SWEEP = True",
                "RUN_PHASE7_ANALYSIS = False",
                "UNPACK_SYNCED_RESULTS_ON_START = False",
                "SYNC_INCLUDE_LEDGER = False",
                "Qwen/Qwen3.5-0.8B",
            ),
            "C_BARR_5_Combine_Analyze.ipynb": (
                "PIPELINE_MODE = 'combine_analyze'",
                "RUN_PREFLIGHT_TESTS = False",
                "RUN_CALIBRATE_ONLY = False",
                "RUN_PRIMARY_MATRIX = False",
                "RUN_SCALE_SWEEP = False",
                "RUN_PHASE7_ANALYSIS = True",
                "UNPACK_SYNCED_RESULTS_ON_START = True",
                "SYNC_INCLUDE_LEDGER = True",
                "No finished worker matrix summaries found",
            ),
        }
        actual_nbs = sorted(p.name for p in (ROOT / "notebooks").glob("*.ipynb"))
        self.assertEqual(actual_nbs, sorted(notebook_expectations.keys()))
        shared_tokens = (
            "dolong2110/C-BARR",
            "verify_manifest_integrity",
            "jinja2>=3.1.6",
            "Brotli>=1.2.0",
            "scripts/run_matrix.py",
            "scripts/analyze_and_plot.py",
            "scripts/log_experiment.py",
            "scripts/sync_results.py",
            "record_loop_checkpoint",
            "run_and_tee",
            "RUN_LOG_FILE",
            "matrix_step_flags",
            "--step-checkpoint-every",
            "MATRIX_PARAM_ARGS",
            "AUTO_PUSH_RESULTS_TO_GITHUB",
            "AUTO_PARALLEL_MULTI_GPU",
            "MAX_SHARDS_PER_MODEL = 4",
            "N_ACTIVE_SHARDS",
            "--shard-idx",
            "--num-shards",
            "--merge-shards",
            "--no-ledger",
            "NOTEBOOK_CODE_VERSION",
            "REPO_CODE_VERSION",
            "STRICT_NOTEBOOK_VERSION_MATCH",
            "PREFLIGHT_FAST_ONLY",
            "RAN_LIVE_SMOKE_THIS_RUN",
            "was retired or renamed",
            "ignore::DeprecationWarning",
            "--report-json",
            "scripts.versioning",
            "run_manifest.json",
            "results_manifest.json",
        )
        for nb_name, extra_tokens in notebook_expectations.items():
            nb_path = ROOT / "notebooks" / nb_name
            self.assertTrue(nb_path.exists(), f"Missing notebook: {nb_path}")
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            self.assertEqual(nb.get("nbformat"), 4)
            cells = nb.get("cells", [])
            self.assertGreaterEqual(len(cells), 8)

            combined_code = []
            for idx, cell in enumerate(cells):
                src = (
                    "".join(cell.get("source", []))
                    if isinstance(cell.get("source"), list)
                    else str(cell.get("source", ""))
                )
                self.assertTrue(src.strip(), f"Empty cell at index {idx} in {nb_name}")
                if cell.get("cell_type") == "code":
                    for line in src.splitlines():
                        stripped = line.lstrip()
                        self.assertFalse(
                            stripped.startswith("!") or stripped.startswith("%"),
                            f"{nb_name} cell {idx} uses shell magic instead of pure Python: {line}",
                        )
                    ast.parse(src, filename=f"{nb_name}:cell_{idx}.py")
                    combined_code.append(src)

            all_src = "\n".join(combined_code)
            for token in (*shared_tokens, *extra_tokens):
                self.assertIn(token, all_src, f"Missing token {token!r} in {nb_name}")

    def test_stage1_slicing_and_stage2_memo_parity(self) -> None:
        """Stage-1 B1b token slicing and Stage-2 memoization match uncached greedy outputs bitwise."""
        from scripts.run_matrix import cache_split_primitives
        from src.engine import CBARRDualModeEngine, P_TERM, run_gate1_lexical

        class _Tok:
            eos_token_id = 248044
            pad_token_id = 0

            def __init__(self) -> None:
                self._rev = {248044: "<|im_end|>"}

            def convert_tokens_to_ids(self, tok_str: str) -> int:
                return 248044 if tok_str == "<|im_end|>" else 999999

            def apply_chat_template(
                self,
                msgs: list[dict[str, str]],
                tokenize: bool = False,
                add_generation_prompt: bool = True,
                enable_thinking: bool = False,
            ) -> str:
                return f"SYS:{msgs[0]['content']}|THINK={enable_thinking}|<think>\n" if enable_thinking else f"SYS:{msgs[0]['content']}|FAST\n"

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                words = text.split()
                return [100 + (abs(hash(w)) % 1000) for w in words]

            def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
                return " ".join("</think>" if t == 5000 else ("<|im_end|>" if t == 248044 else f"w{t}") for t in token_ids)

        class _DeterministicGenIdEngine(CBARRDualModeEngine):
            def __init__(self, early_close: bool) -> None:
                self.model_id = "Qwen/Qwen3.5-4B"
                self.is_smollm3 = False
                self.freeze_smollm3_date = "01 November 2026"
                self.device = "cpu"
                self.tok = _Tok()
                self.eos_ids = {248044}
                self.early_close = early_close
                self.decode_calls = 0

            def _decode_raw(
                self, prompt_str: str, max_new_tokens: int, log_entropies: bool = False
            ) -> dict:
                self.decode_calls += 1
                if P_TERM in prompt_str or "</think>" in prompt_str and "|FAST" not in prompt_str:
                    ids = [301, 302, 303, 248044][:max_new_tokens]
                elif "|THINK=True|" in prompt_str:
                    if self.early_close:
                        # 30 think tokens, </think> at idx 30, 60 post tokens, EOS at end (len 92)
                        full = list(range(10, 40)) + [5000] + list(range(200, 260)) + [248044]
                    else:
                        # Long unclosed think trace (300 tokens, no EOS)
                        full = list(range(10, 310))
                    ids = full[:max_new_tokens]
                else:
                    ids = [401, 402, 248044][:max_new_tokens]
                hit_eos = bool(ids and ids[-1] in self.eos_ids)
                content_len = len(ids) - (1 if hit_eos else 0)
                return {
                    "raw_text": self.tok.decode(ids, skip_special_tokens=False),
                    "gen_ids": ids,
                    "total_gen_tokens": len(ids),
                    "content_tokens": content_len,
                    "hit_eos": hit_eos,
                    "step_entropies": [0.4] * min(content_len, 10) if log_entropies else [],
                    "latency_ms": float(len(ids)),
                }

        ifeval_items = [r for r in _read_jsonl(ROOT / "data/eval_450.jsonl") if r["slice"] == "ifeval_precision"][:1]
        item = ifeval_items[0]
        g1 = run_gate1_lexical(item["prompt"])
        self.assertTrue(bool(g1.anchor_clause))

        for early_close in (False, True):
            eng_opt = _DeterministicGenIdEngine(early_close=early_close)
            eng_ref = _DeterministicGenIdEngine(early_close=early_close)
            with tempfile.TemporaryDirectory() as td:
                cache_p = pathlib.Path(td) / "test_cache.jsonl"
                rows = cache_split_primitives(
                    eng_opt, [item], cache_p, full_eval_modes=True, run_calib_tiers=True
                )
                row = rows[0]
                raw_jsonl_line = cache_p.read_text(encoding="utf-8").strip()
                self.assertNotIn("gen_ids", raw_jsonl_line)

            probe_t = int(row["probe"]["T_probe"])
            b256_probe = max(64, min(250, 250 - probe_t))
            pairs = [
                ("m2_b64", 64, "post_stripped"),
                ("m2_b128", 128, "post_stripped"),
                ("m2_b256", b256_probe, "post_stripped"),
                ("m2_b256_direct", 250, "post_stripped"),
                ("m2_pre_b64", 64, "pre_close"),
                ("m2_pre_b128", 128, "pre_close"),
                ("m2_pre_b256", b256_probe, "pre_close"),
                ("m2_pre_b256_direct", 250, "pre_close"),
            ]
            for key, budget, locus in pairs:
                ref = eng_ref.run_capped_think_and_anchor(
                    item["prompt"],
                    budget_b=budget,
                    anchor_clause=g1.anchor_clause,
                    max_ans_tokens=1024,
                    allow_budget_force=True,
                    anchor_locus=locus,
                )
                for cmp_k in (
                    "t_think",
                    "a_final",
                    "T_probe",
                    "T_think",
                    "T_wasted",
                    "T_inj",
                    "T_final",
                    "T_total",
                    "calls",
                    "truncated",
                    "think_exhausted",
                ):
                    self.assertEqual(
                        row[key][cmp_k],
                        ref[cmp_k],
                        f"Mismatch on {key}.{cmp_k} (early_close={early_close})",
                    )

            # Baseline uncached calls: b0(1) + probe(1) + b1b(1) + b1a(1) + b2a(1) + 8*(s1+s2) = 21 calls.
            # With Stage-1 slicing: zero extra Stage-1 calls across all 8 M2/M2_Pre variants!
            # When early_close=True, Stage-2 memoization also eliminates duplicate Stage-2 decodes (<= 8 total).
            max_expected_calls = 7 if early_close else 13
            self.assertLessEqual(eng_opt.decode_calls, max_expected_calls)

    def test_setup_fast_kernels_policy_and_dry_run(self) -> None:
        """Fast DeltaNet kernel installer gates by compute capability and writes safe pip pins."""
        from scripts.setup_fast_kernels import (
            _write_pip_constraints,
            setup_and_verify_kernels,
            should_attempt_kernels,
        )

        self.assertEqual(should_attempt_kernels(None, "auto")[0], False)
        self.assertEqual(should_attempt_kernels((6, 0), "auto")[0], False)  # Pascal P100
        self.assertEqual(should_attempt_kernels((7, 5), "auto")[0], True)   # Turing T4
        self.assertEqual(should_attempt_kernels((8, 0), "auto")[0], True)   # Ampere A100
        self.assertEqual(should_attempt_kernels((7, 5), "off")[0], False)
        self.assertEqual(should_attempt_kernels((6, 0), "always")[0], True)

        dry = setup_and_verify_kernels(mode="auto", dry_run=True)
        self.assertIn("fla_active", dry)
        self.assertIn("causal_conv1d_active", dry)

        with tempfile.TemporaryDirectory() as td:
            cpath = pathlib.Path(td) / "pins.txt"
            _write_pip_constraints(cpath)
            txt = cpath.read_text(encoding="utf-8")
            self.assertIn("torch==", txt)
            self.assertIn("transformers==", txt)

            rpath = pathlib.Path(td) / "nested" / "kernel_rep.json"
            from scripts import setup_fast_kernels

            prev_argv = sys.argv
            try:
                sys.argv = ["setup_fast_kernels.py", "--mode", "off", "--json", "--report-json", str(rpath)]
                setup_fast_kernels.main()
            finally:
                sys.argv = prev_argv
            rep = json.loads(rpath.read_text(encoding="utf-8"))
            self.assertEqual(rep["mode"], "off")
            self.assertIn("actions", rep)
            self.assertIn("decision_reason", rep)

    def test_log_experiment_smoke_report_filtering_and_na(self) -> None:
        """Smoke tables only appear when explicitly provided and use N/A for non-SmolLM3 frozen date."""
        from scripts.log_experiment import _extract_smoke_report, _render_run_markdown

        # Static repo fixture must NOT leak into runs that omit --smoke-report
        self.assertEqual(_extract_smoke_report(ROOT, ROOT / "outputs", ""), [])
        smoke_fixture = ROOT / "outputs" / "smoke_test_gate3_report.json"
        rows = _extract_smoke_report(ROOT, ROOT / "outputs", str(smoke_fixture))
        self.assertEqual(len(rows), 2)

        hw = {"gpu_count": 0, "gpu_name": "CPU", "cuda_available": False, "torch": "2.x", "transformers": "5.x", "fla_available": False, "causal_conv1d_available": False}
        md = _render_run_markdown(
            run_id="smoke_na_test",
            status="PASSED",
            loop_name="Step 0a",
            ts_utc="2026-11-16 00:00:00 UTC",
            git_rev="abc1234",
            total_sec=1.0,
            hw=hw,
            command_clean="test",
            rel_log="",
            notes_clean="",
            training_params={},
            stage_timings=[],
            smoke_rows=rows,
            calibrations=[],
            matrix_loops=[],
            w7_items=[],
            model_rows=[],
            code_version="0.5.3",
        )
        self.assertIn("| `Qwen/Qwen3.5-4B` | 12 | True | True | N/A |", md)
        self.assertIn("| `HuggingFaceTB/SmolLM3-3B` | 12 | True | True | True |", md)

    def test_logfmt_highlighting_toggle(self) -> None:
        """Log formatter emits ANSI highlights when CBARR_COLOR=1 and clean text under NO_COLOR/0."""
        import os
        from src.logfmt import banner, colors_enabled, hl

        prev_c = os.environ.get("CBARR_COLOR")
        prev_n = os.environ.get("NO_COLOR")
        try:
            os.environ.pop("NO_COLOR", None)
            os.environ["CBARR_COLOR"] = "1"
            self.assertTrue(colors_enabled())
            self.assertIn("\033[", hl("PASS", "green"))
            self.assertIn("TITLE", banner("TITLE", "cyan"))

            os.environ["CBARR_COLOR"] = "0"
            self.assertFalse(colors_enabled())
            self.assertEqual(hl("PASS", "green"), "PASS")

            os.environ["CBARR_COLOR"] = "1"
            os.environ["NO_COLOR"] = "1"
            self.assertFalse(colors_enabled())
            self.assertEqual(hl("PASS", "green"), "PASS")
        finally:
            if prev_c is None:
                os.environ.pop("CBARR_COLOR", None)
            else:
                os.environ["CBARR_COLOR"] = prev_c
            if prev_n is None:
                os.environ.pop("NO_COLOR", None)
            else:
                os.environ["NO_COLOR"] = prev_n

    def test_log_experiment_git_push_to_bare_remote(self) -> None:
        """Experiment logger commits and pushes EXPERIMENT_LOG.md cleanly without leaving dirty git state."""
        import os
        from scripts.log_experiment import _git_commit_and_push_ledger, _has_git_push_auth, _upsert_run_block

        # Pre-setup hosts run git auth checks before numpy is installed; import must succeed without numpy.
        bare_import = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.modules['numpy'] = None; from scripts.log_experiment import _ensure_git_identity, _has_git_push_auth",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(bare_import.returncode, 0, bare_import.stderr)

        with tempfile.TemporaryDirectory() as td:
            root_tmp = pathlib.Path(td)
            bare_repo = root_tmp / "origin.git"
            work_repo = root_tmp / "work"
            subprocess.run(["git", "init", "--bare", str(bare_repo)], check=True, capture_output=True)
            subprocess.run(["git", "clone", str(bare_repo), str(work_repo)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(work_repo), "checkout", "-B", "master"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(work_repo), "config", "user.name", "tester"], check=True)
            subprocess.run(["git", "-C", str(work_repo), "config", "user.email", "tester@example.org"], check=True)

            exp_md = work_repo / "EXPERIMENT_LOG.md"
            exp_md.write_text("# C-BARR Log\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(work_repo), "add", "EXPERIMENT_LOG.md"], check=True)
            subprocess.run(["git", "-C", str(work_repo), "commit", "-m", "init"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(work_repo), "push", "-u", "origin", "master"], check=True, capture_output=True)

            block = "<!-- CBARR_RUN_START:r1 -->\n### Run `r1` — **PASSED**\n<!-- CBARR_RUN_END:r1 -->"
            _upsert_run_block(exp_md, "r1", block, upsert=True)
            ok, msg = _git_commit_and_push_ledger(
                repo_root=work_repo,
                md_path=exp_md,
                run_id="r1",
                loop_name="Phase 4: 4B",
                remote="origin",
                branch="master",
                token_env="GITHUB_TOKEN",
            )
            self.assertTrue(ok, msg)
            remote_show = subprocess.check_output(
                ["git", "--git-dir", str(bare_repo), "show", "master:EXPERIMENT_LOG.md"],
                text=True,
            )
            self.assertIn("CBARR_RUN_START:r1", remote_show)

            # Verify _has_git_push_auth cleanly detects SSH, PAT, and git credential.helper
            fake_home = root_tmp / "empty_home"
            fake_home.mkdir()
            prev_env = {k: os.environ.get(k) for k in ("HOME", "GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_GLOBAL")}
            try:
                os.environ["HOME"] = str(fake_home)
                os.environ["GIT_CONFIG_NOSYSTEM"] = "1"
                os.environ["GIT_CONFIG_GLOBAL"] = str(fake_home / ".gitconfig")
                self.assertTrue(_has_git_push_auth(work_repo, "git@github.com:org/repo.git", ""))
                self.assertTrue(_has_git_push_auth(work_repo, "https://github.com/org/repo.git", "tok"))
                self.assertFalse(_has_git_push_auth(work_repo, "https://github.com/org/repo.git", ""))
                subprocess.run(["git", "-C", str(work_repo), "config", "credential.helper", "store"], check=True)
                self.assertTrue(_has_git_push_auth(work_repo, "https://github.com/org/repo.git", ""))
            finally:
                for k, v in prev_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    def test_run_matrix_per_step_checkpoints(self) -> None:
        """run_matrix.py updates progress_checkpoint, timings JSONL, and EXPERIMENT_LOG.md at every internal step."""
        with tempfile.TemporaryDirectory() as td:
            tmp_root = pathlib.Path(td)
            tmp_data = tmp_root / "data"
            tmp_data.mkdir(parents=True, exist_ok=True)
            tmp_out = tmp_root / "outputs"
            tmp_md = tmp_root / "EXPERIMENT_LOG.md"
            tmp_timings = tmp_root / "timings.jsonl"
            tmp_params = tmp_root / "params.json"
            tmp_log = tmp_root / "run.log"
            _build_mini_frozen_dir(tmp_data)
            tmp_md.write_text("# Test Ledger\n", encoding="utf-8")
            tmp_params.write_text(json.dumps({"seed": 42, "step_checkpoint_every": 5}), encoding="utf-8")
            tmp_log.write_text("Mock run log start\n", encoding="utf-8")

            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_matrix.py"),
                "--mock",
                "--model",
                "Qwen/Qwen3.5-4B",
                "--data-dir",
                str(tmp_data),
                "--out-dir",
                str(tmp_out),
                "--strict-diag",
                "--run-id",
                "step_test",
                "--exp-log",
                str(tmp_md),
                "--timings-file",
                str(tmp_timings),
                "--log-file",
                str(tmp_log),
                "--params-json",
                f"@{tmp_params}",
                "--step-checkpoint-every",
                "5",
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            model_dir = tmp_out / "Qwen__Qwen3.5-4B"
            self.assertTrue((model_dir / "calibration_summary.json").exists())
            # In-flight progress_checkpoint.json is cleaned up once eval_450 completes
            self.assertFalse((model_dir / "progress_checkpoint.json").exists())
            summary = json.loads((model_dir / "matrix_summary_strict341.json").read_text(encoding="utf-8"))
            self.assertEqual(summary.get("status"), "completed")

            md_text = tmp_md.read_text(encoding="utf-8")
            self.assertIn("CBARR_RUN_START:step_test", md_text)
            self.assertIn("step_checkpoint_every", md_text)
            self.assertIn("Qwen/Qwen3.5-4B", md_text)

            history_path = tmp_out / "experiment_history.jsonl"
            self.assertTrue(history_path.exists())
            history_rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]
            loop_names = [r.get("loop_name", "") for r in history_rows]
            self.assertTrue(any("Calibration:" in s for s in loop_names), loop_names)
            self.assertTrue(any("Eval-450 progress (" in s for s in loop_names), loop_names)
            self.assertTrue(any("Eval-450:" in s for s in loop_names), loop_names)
            self.assertTrue(any("Section 6 & Matrix Complete:" in s for s in loop_names), loop_names)

            # Verify mid-split live progress rendering when progress_checkpoint.json is active
            from scripts.log_experiment import _extract_live_progress

            (model_dir / "progress_checkpoint.json").write_text(
                json.dumps(
                    {
                        "model": "Qwen/Qwen3.5-4B",
                        "diag_pool": "strict341",
                        "split": "eval_450",
                        "done": 5,
                        "total": 18,
                        "metrics": "Acc B0=0.0% Fast=0.0%",
                        "slices": "gsm8k_easy(3): 0/0/0%",
                    }
                ),
                encoding="utf-8",
            )
            live_rows = _extract_live_progress(tmp_out)
            self.assertEqual(len(live_rows), 1)
            self.assertEqual(live_rows[0]["done"], 5)

    def test_deltanet_blend_qwen35_dynamic_cache_and_r6_sampling(self) -> None:
        """Exercises real tiny Qwen3_5ForCausalLM DynamicCache blending + R6 sampling & bundle isolation."""
        from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        from scripts.analyze_and_plot import discover_bundles
        from scripts.ext_deltanet_blend import generate_with_blended_deltanet
        from src.engine import CBARRDualModeEngine, PresencePenaltyLogitsProcessor

        class _TinyTok:
            eos_token_id = 2
            pad_token_id = 0
            unk_token_id = 1

            def apply_chat_template(
                self,
                msgs: list[dict[str, str]],
                tokenize: bool = False,
                add_generation_prompt: bool = True,
                enable_thinking: bool = False,
            ) -> str:
                suffix = "<think>\n" if enable_thinking else ""
                return " ".join(m["content"] for m in msgs) + " " + suffix

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                toks = [10 + (sum(ord(ch) for ch in w) % 50) for w in text.split()]
                return toks or [10, 11]

            def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
                return " ".join(f"w{t}" for t in token_ids if not skip_special_tokens or t != 2)

            def __call__(self, text: str, return_tensors: str = "pt") -> Any:
                from transformers import BatchEncoding

                ids = torch.tensor([self.encode(text)], dtype=torch.long)
                return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

        cfg = Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
        )
        tiny_qwen = Qwen3_5ForCausalLM(cfg).eval()
        tiny_tok = _TinyTok()

        out_blend = generate_with_blended_deltanet(
            tiny_qwen,
            tiny_tok,
            eos_ids={2},
            prompt_str="Write in lowercase.",
            t_think="Checking constraint step by step.",
            alpha=0.5,
            max_ans_tokens=4,
        )
        self.assertEqual(out_blend["n_blended_layers"], 3)
        self.assertGreaterEqual(out_blend["gen_tokens"], 1)
        self.assertLessEqual(out_blend["gen_tokens"], 4)

        # Verify PresencePenaltyLogitsProcessor subtracts penalty strictly from seen post-prompt tokens
        proc = PresencePenaltyLogitsProcessor(prompt_len=2, penalty=1.5)
        scores = torch.zeros((1, 10), dtype=torch.float32)
        in_ids = torch.tensor([[4, 5, 7, 7]], dtype=torch.long)
        adj = proc(in_ids, scores.clone())
        self.assertAlmostEqual(float(adj[0, 7].item()), -1.5, places=5)
        self.assertAlmostEqual(float(adj[0, 4].item()), 0.0, places=5)

        # Verify stochastic sampling engine path reproduces identically under same seed + isolates output dir
        eng = CBARRDualModeEngine.__new__(CBARRDualModeEngine)
        eng.model_id = "Qwen/Qwen3.5-Tiny"
        eng.is_smollm3 = False
        eng.freeze_smollm3_date = "01 November 2026"
        eng.device = "cpu"
        eng.temperature = 0.6
        eng.top_p = 0.95
        eng.presence_penalty = 1.5
        eng.sampling_seed = 43
        eng.tok = tiny_tok
        eng.model = tiny_qwen
        eng.eos_ids = {2}
        r1 = eng._decode_raw("Test prompt", max_new_tokens=5, log_entropies=True)
        r2 = eng._decode_raw("Test prompt", max_new_tokens=5, log_entropies=True)
        self.assertEqual(r1["gen_ids"], r2["gen_ids"])
        self.assertEqual(len(r1["step_entropies"]), r1["content_tokens"])

        with tempfile.TemporaryDirectory() as td:
            tmp_root = pathlib.Path(td)
            tmp_data = tmp_root / "data"
            tmp_data.mkdir(parents=True, exist_ok=True)
            tmp_out = tmp_root / "outputs"
            _build_mini_frozen_dir(tmp_data)

            # Run greedy baseline first, then sampled R6 run in the same --out-dir
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_matrix.py"),
                    "--mock",
                    "--model",
                    "Qwen/Qwen3.5-4B",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(tmp_out),
                    "--strict-diag",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_matrix.py"),
                    "--mock",
                    "--model",
                    "Qwen/Qwen3.5-4B",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(tmp_out),
                    "--strict-diag",
                    "--temperature",
                    "0.6",
                    "--top-p",
                    "0.95",
                    "--presence-penalty",
                    "1.5",
                    "--seed",
                    "43",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            greedy_dir = tmp_out / "Qwen__Qwen3.5-4B"
            sampled_dir = tmp_out / "Qwen__Qwen3.5-4B__T0.6_tp0.95_pp1.5_s43"
            self.assertTrue((greedy_dir / "matrix_summary_strict341.json").exists())
            self.assertTrue((sampled_dir / "matrix_summary_strict341.json").exists())

            # discover_bundles keeps greedy canonical by default unless include_sampled=True
            canon = discover_bundles(tmp_out)
            self.assertAlmostEqual(canon["Qwen/Qwen3.5-4B"]["report"]["training_params"]["temperature"], 0.0)
            all_b = discover_bundles(tmp_out, include_sampled=True)
            self.assertAlmostEqual(all_b["Qwen/Qwen3.5-4B"]["report"]["training_params"]["temperature"], 0.6)

    def test_stateful_kv_continuation_and_horizon_probes(self) -> None:
        """Verifies Stage-1->Stage-2 KV/DeltaNet continuation parity and K in {1,4,8,16,32} metrics."""
        from transformers import BatchEncoding
        from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        from src.engine import CBARRDualModeEngine

        class _ByteTok:
            eos_token_id = 2
            pad_token_id = 0
            unk_token_id = 1

            def apply_chat_template(
                self,
                msgs: list[dict[str, str]],
                tokenize: bool = False,
                add_generation_prompt: bool = True,
                enable_thinking: bool = False,
            ) -> str:
                suffix = "<think>\n" if enable_thinking else ""
                return "|".join(f"{m['role']}:{m['content']}" for m in msgs) + "|" + suffix

            def convert_tokens_to_ids(self, tok: str) -> int:
                return 2 if tok == "<|im_end|>" else 1

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return [10 + (b % 40) for b in text.encode("utf-8")] or [10]

            def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
                return " ".join(f"t{t}" for t in token_ids if not skip_special_tokens or t != 2)

            def __call__(self, text: str, return_tensors: str = "pt") -> Any:
                ids = torch.tensor([self.encode(text)], dtype=torch.long)
                return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

        torch.manual_seed(7)
        cfg = Qwen3_5TextConfig(
            vocab_size=80,
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=12,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=12,
            linear_value_head_dim=12,
        )
        eng = CBARRDualModeEngine.__new__(CBARRDualModeEngine)
        eng.model_id = "Qwen/Qwen3.5-Tiny"
        eng.is_smollm3 = False
        eng.freeze_smollm3_date = "01 November 2026"
        eng.device = "cpu"
        eng.temperature = 0.0
        eng.top_p = 1.0
        eng.presence_penalty = 0.0
        eng.sampling_seed = 42
        eng.tok = _ByteTok()
        eng.model = Qwen3_5ForCausalLM(cfg).eval()
        eng.eos_ids = {2}

        prompt = "Explain why the sky appears blue."
        probe = eng.run_gate2_fast_probe(prompt, probe_tokens=4)
        self.assertIsNotNone(probe.get("_kv_state"))
        for k_field in ("H_1", "H_4_mean", "H_4_max", "H_8_mean", "H_16_mean", "I_inertia_4", "I_inertia_8", "I_inertia_16"):
            self.assertIn(k_field, probe)

        fast_done = eng.complete_fast_probe(probe, max_ans_tokens=10)
        self.assertNotIn("_kv_state", probe)
        one_shot = eng.run_direct_non_thinking(prompt, max_new_tokens=10)
        self.assertEqual(fast_done["a_final"], one_shot["a_final"])
        self.assertEqual(fast_done["T_total"], one_shot["T_total"])

        # Live capped think+anchor works cleanly with Stage-1 KV continuation
        capped = eng.run_capped_think_and_anchor(
            "Write a short summary without using the letter e.", budget_b=4, anchor_clause="Do not use the letter e.", max_ans_tokens=6
        )
        self.assertTrue(capped["truncated"])
        self.assertGreaterEqual(capped["T_final"], 1)

    def test_t12_conv_blend_and_kv_eviction_on_qwen35_and_llama(self) -> None:
        """Verifies T1.2 conv_states blending & full-attention <think> KV eviction on hybrid & dense GQA."""
        from transformers import BatchEncoding
        from transformers.models.llama import LlamaConfig, LlamaForCausalLM
        from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        from scripts.ext_deltanet_blend import generate_with_blended_deltanet, run_mock_blend_check

        class _SimpleTok:
            eos_token_id = 2

            def apply_chat_template(
                self,
                msgs: list[dict[str, str]],
                tokenize: bool = False,
                add_generation_prompt: bool = True,
                enable_thinking: bool = False,
            ) -> str:
                return " ".join(m["content"] for m in msgs) + (" <think>\n" if enable_thinking else "")

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return [10 + (sum(ord(c) for c in w) % 45) for w in text.split()] or [10]

            def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
                return " ".join(f"w{t}" for t in token_ids if not skip_special_tokens or t != 2)

            def __call__(self, text: str, return_tensors: str = "pt") -> Any:
                ids = torch.tensor([self.encode(text)], dtype=torch.long)
                return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

        mock_sweep = run_mock_blend_check([0.0, 0.5, 1.0], blend_conv=True, evict_think_kv=True)
        self.assertEqual(mock_sweep["sweep"]["alpha_0.50"]["n_blended_layers"], 1)
        self.assertEqual(mock_sweep["sweep"]["alpha_0.50"]["n_blended_conv"], 1)
        self.assertEqual(mock_sweep["sweep"]["alpha_0.50"]["n_evicted_kv"], 1)
        self.assertEqual(mock_sweep["sweep"]["alpha_0.50"]["evicted_kv_seq_len"], 4)
        self.assertAlmostEqual(mock_sweep["sweep"]["alpha_0.50"]["conv_mean_val"], 1.0, places=4)

        tok = _SimpleTok()
        q_cfg = Qwen3_5TextConfig(
            vocab_size=96,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
        )
        q_model = Qwen3_5ForCausalLM(q_cfg).eval()
        out_q = generate_with_blended_deltanet(
            q_model,
            tok,
            eos_ids={2},
            prompt_str="Answer briefly.",
            t_think="Let me think about constraints.",
            alpha=0.5,
            max_ans_tokens=3,
            blend_conv=True,
            evict_think_kv=True,
        )
        self.assertEqual((out_q["n_blended_layers"], out_q["n_blended_conv"], out_q["n_evicted_kv"]), (3, 3, 1))

        # Qwen3.5 style </think>\n\n suffix alignment: alpha=1.0 + conv + KV eviction matches no-think pass
        class _QwenBoundaryTok(_SimpleTok):
            def apply_chat_template(
                self,
                msgs: list[dict[str, str]],
                tokenize: bool = False,
                add_generation_prompt: bool = True,
                enable_thinking: bool = False,
            ) -> str:
                base = " ".join(m["content"] for m in msgs)
                return f"{base} <think>\n" if enable_thinking else f"{base} </think>\n\n"

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                parts = text.split(" ")
                return [10 + (sum(ord(c) for c in w) % 45) for w in parts if w != ""]

        q_tok = _QwenBoundaryTok()
        out_q1 = generate_with_blended_deltanet(
            q_model,
            q_tok,
            eos_ids={2},
            prompt_str="Answer briefly.",
            t_think="Long reasoning tokens.",
            alpha=1.0,
            max_ans_tokens=3,
            blend_conv=True,
            evict_think_kv=True,
        )
        ids_ref = q_tok.encode(q_tok.apply_chat_template([{"role": "user", "content": "Answer briefly."}], False, True, False))
        with torch.no_grad():
            ref_gen = q_model.generate(
                torch.tensor([ids_ref], dtype=torch.long),
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=0,
            )[0, len(ids_ref):].tolist()
        self.assertEqual(out_q1["a_final"], q_tok.decode(ref_gen, skip_special_tokens=True).strip())

        l_cfg = LlamaConfig(
            vocab_size=96,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        out_l = generate_with_blended_deltanet(
            LlamaForCausalLM(l_cfg).eval(),
            tok,
            eos_ids={2},
            prompt_str="Answer briefly.",
            t_think="Let me think about constraints.",
            alpha=1.0,
            max_ans_tokens=3,
            blend_conv=False,
            evict_think_kv=True,
        )
        self.assertEqual((out_l["n_blended_layers"], out_l["n_evicted_kv"]), (0, 2))

    def test_t13_prefill_multihead_npz_cache_and_paraphrase_stress(self) -> None:
        """Verifies T1.3 .npz feature disk cache reuse and IFEval++ paraphrase stress metrics."""
        import numpy as np
        from scripts.ext_prefill_mlp import (
            _load_or_extract_cached_features,
            _mock_prefill_features,
            paraphrase_ifeval_prompt,
            train_and_evaluate_prefill_mlp,
        )

        orig = "Your response must include at least 200 words in all capital letters with no commas and without using JSON."
        para = paraphrase_ifeval_prompt(orig)
        self.assertNotEqual(para, orig)
        self.assertIn("200 words", para)

        def _fake_rows(n: int, prefix: str) -> list[dict]:
            out = []
            for i in range(n):
                is_plan = (i % 2 == 0)
                is_prec = (i % 3 == 0)
                p = "Write at least 300 words " if is_plan else "Solve 2 + 2 "
                if is_prec:
                    p += "with all capital letters and no commas."
                inst_ids = []
                if is_plan:
                    inst_ids.append("length_constraints:number_words")
                if is_prec:
                    inst_ids.append("change_case:english_capital")
                out.append({
                    "item": {"uid": f"{prefix}_{i}", "prompt": p, "slice": "ifeval_conflict" if (is_plan and is_prec) else ("ifeval_planning" if is_plan else ("ifeval_precision" if is_prec else "csqa")), "instruction_id_list": inst_ids},
                    "b0": {"z": 1, "T_total": 40, "calls": 1, "truncated": False, "think_exhausted": False},
                    "fast_done": {"z": 0 if is_plan else 1, "T_total": 45, "calls": 1, "truncated": False, "think_exhausted": False},
                    "probe": {"I_inertia": int(is_plan), "H_mean": 0.6, "T_probe": 12},
                    "m2_b64": {"z": 1, "T_total": 90, "calls": 1, "truncated": False, "think_exhausted": False},
                    "m2_b128": {"z": 1, "T_total": 120, "calls": 1, "truncated": False, "think_exhausted": False},
                    "m2_b250": {"z": 1, "T_total": 140, "calls": 1, "truncated": False, "think_exhausted": False},
                })
            return out

        seed_rows = _fake_rows(12, "s")
        dev_rows = _fake_rows(12, "d")
        eval_rows = _fake_rows(12, "e")
        para_rows = [{"item": {**r["item"], "prompt": paraphrase_ifeval_prompt(r["item"]["prompt"])}} for r in eval_rows]

        with tempfile.TemporaryDirectory() as td:
            cache_p = pathlib.Path(td) / "feats.npz"
            calls = [0]

            def _extract(splits: dict) -> dict[str, np.ndarray]:
                calls[0] += 1
                return {
                    "seed": _mock_prefill_features(splits["seed"], dim=16),
                    "dev": _mock_prefill_features(splits["dev"], dim=16),
                    "eval": _mock_prefill_features(splits["eval"], dim=16),
                    "eval_para": _mock_prefill_features(splits["eval_para"], dim=16, semantic_rows=splits["eval"]),
                }

            splits = {"seed": seed_rows, "dev": dev_rows, "eval": eval_rows, "eval_para": para_rows}
            f1 = _load_or_extract_cached_features(cache_p, splits, _extract, use_cache=True)
            f2 = _load_or_extract_cached_features(cache_p, splits, _extract, use_cache=True)
            self.assertEqual(calls[0], 1)
            self.assertTrue(np.allclose(f1["eval"], f2["eval"]))

            res = train_and_evaluate_prefill_mlp(
                seed_rows,
                dev_rows,
                eval_rows,
                f2["seed"],
                f2["dev"],
                f2["eval"],
                tau_h_high=0.85,
                epochs=40,
                x_eval_para=f2["eval_para"],
                train_multihead=True,
            )
            self.assertIn("gate1_semantic_probe", res)
            self.assertIn("ifevalpp_paraphrase_stress", res)
            self.assertIn("mlp_para", res["ifevalpp_paraphrase_stress"])

    def test_mps_bf16_auto_selection_and_fallback(self) -> None:
        """MPS auto precision prefers BF16 when supported cleanly and falls back to FP16 otherwise."""
        from unittest import mock
        import scripts.run_matrix as rm

        with mock.patch.object(rm, "_mps_supports_bf16", return_value=True):
            self.assertEqual(rm.resolve_device_and_dtype("mps", "auto"), ("mps", torch.bfloat16))
        with mock.patch.object(rm, "_mps_supports_bf16", return_value=False):
            self.assertEqual(rm.resolve_device_and_dtype("mps", "auto"), ("mps", torch.float16))
        self.assertEqual(rm.resolve_device_and_dtype("mps", "float16"), ("mps", torch.float16))

    def test_seed_calib_tier_skip_and_native_probe_b0_fusion(self) -> None:
        """Seed-only calibration skips unused M2 tiers; native greedy engine fuses probe -> b0."""
        from unittest import mock
        from scripts.run_matrix import cache_split_primitives, run_section6_diagnostics
        from src.engine import CBARRDualModeEngine, run_gate1_lexical

        class _NativeDummyTok:
            eos_token_id = 99
            pad_token_id = 0

            def convert_tokens_to_ids(self, tok_str: str) -> int:
                return 99 if tok_str == "<|im_end|>" else -1

            def apply_chat_template(
                self,
                msgs: list[dict[str, str]],
                tokenize: bool = False,
                add_generation_prompt: bool = True,
                enable_thinking: bool = False,
            ) -> str:
                if len(msgs) == 3:
                    return f"U1:{msgs[0]['content']}|A1:{msgs[1]['content']}|U2:{msgs[2]['content']}|GEN:"
                return f"SYS:{msgs[0]['content']}|THINK={enable_thinking}|"

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return [10 + (i % 50) for i, _ in enumerate(text.split())]

            def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
                return " ".join(
                    "<|im_end|>" if t == 99 else f"w{t}" for t in token_ids
                )

        # Construct a native-class instance (does NOT override _decode_raw) with a mocked generate core
        eng = CBARRDualModeEngine.__new__(CBARRDualModeEngine)
        eng.model_id = "Qwen/Qwen3.5-0.8B"
        eng.is_smollm3 = False
        eng.freeze_smollm3_date = "01 November 2026"
        eng.device = "cpu"
        eng.temperature = 0.0
        eng.top_p = 1.0
        eng.presence_penalty = 0.0
        eng.sampling_seed = 42
        eng.tok = _NativeDummyTok()
        eng.eos_ids = {99}

        core_calls: list[dict] = []

        def _fake_core(
            inp_ids,
            attention_mask,
            prompt_len_for_penalty,
            max_new_tokens,
            log_entropies,
            seed_key,
            past_key_values=None,
            return_past=False,
        ):
            core_calls.append({"max_new": max_new_tokens, "has_pkv": past_key_values is not None})
            if past_key_values is not None:
                ids = [201, 202, 99][:max_new_tokens]
                prev_gen = [101, 102]
                seq = torch.cat([inp_ids, torch.tensor([ids])], dim=1)
                return {
                    "raw_text": eng.tok.decode(ids),
                    "gen_ids": ids,
                    "total_gen_tokens": len(ids),
                    "content_tokens": len(ids) - (1 if ids[-1] == 99 else 0),
                    "hit_eos": bool(ids and ids[-1] == 99),
                    "step_entropies": [],
                    "latency_ms": 1.0,
                }
            if max_new_tokens == 32:
                ids = [101, 102]
                seq = torch.cat([inp_ids, torch.tensor([ids])], dim=1)
                out_d = {
                    "raw_text": eng.tok.decode(ids),
                    "gen_ids": ids,
                    "total_gen_tokens": len(ids),
                    "content_tokens": 2,
                    "hit_eos": False,
                    "step_entropies": [0.2, 0.3],
                    "latency_ms": 1.0,
                }
                if return_past:
                    out_d["_kv_state"] = {
                        "sequences": seq,
                        "past_key_values": "MOCK_PKV",
                        "prompt_len_for_penalty": prompt_len_for_penalty,
                        "gen_ids": ids,
                    }
                return out_d
            ids = list(range(300, 300 + min(40, max_new_tokens))) + [99]
            ids = ids[:max_new_tokens]
            return {
                "raw_text": eng.tok.decode(ids),
                "gen_ids": ids,
                "total_gen_tokens": len(ids),
                "content_tokens": len(ids) - (1 if ids[-1] == 99 else 0),
                "hit_eos": bool(ids and ids[-1] == 99),
                "step_entropies": [],
                "latency_ms": 2.0,
            }

        item = _read_jsonl(ROOT / "data/seed_60.jsonl")[0]
        with mock.patch.object(eng, "_decode_generate_core", side_effect=_fake_core):
            def _fake_raw_stateful(prompt_str, max_new_tokens, log_entropies=False):
                return _fake_core(
                    torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long),
                    3, max_new_tokens, log_entropies, prompt_str, return_past=True
                )

            with mock.patch.object(eng, "_decode_raw_stateful", side_effect=_fake_raw_stateful), mock.patch.object(
                eng, "_decode_raw", side_effect=lambda p, max_new_tokens, log_entropies=False: _fake_core(
                    torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long),
                    3, max_new_tokens, log_entropies, p, return_past=False
                )
            ):
                with tempfile.TemporaryDirectory() as td:
                    rows = cache_split_primitives(
                        eng, [item], pathlib.Path(td) / "seed.jsonl",
                        full_eval_modes=False, run_calib_tiers=True, calib_seed_only=True,
                    )
                    self.assertIn("m2_b128", rows[0])
                    self.assertNotIn("m2_b64", rows[0])
                    self.assertNotIn("m2_b256", rows[0])
                    # Probe(32) + complete_fast_probe(has_pkv=True) fused b0 without extra separate b0 decode!
                    self.assertTrue(any(c["has_pkv"] for c in core_calls))
                    self.assertFalse(rows[0]["b0"]["truncated"])

                # Also verify run_section6_diagnostics reuses B2a_post_anchor for B2b_post_raw
                ehurt_item = [r for r in _read_jsonl(ROOT / "data/eval_450.jsonl") if r["slice"] == "ifeval_precision"][0]
                diag_calls: list[str] = []

                def _fake_diag_raw(prompt_str, max_new_tokens, log_entropies=False):
                    diag_calls.append(prompt_str)
                    return {
                        "raw_text": "clean response",
                        "gen_ids": [1, 2, 99],
                        "total_gen_tokens": 3,
                        "content_tokens": 2,
                        "hit_eos": True,
                        "step_entropies": [],
                        "latency_ms": 1.0,
                    }

                with tempfile.TemporaryDirectory() as td2:
                    tp = pathlib.Path(td2)
                    # Pre-populate diag_ehurt_cache with one E_hurt row
                    (tp / "diag_ehurt_cache.jsonl").write_text(
                        json.dumps({
                            "item": ehurt_item,
                            "b0": {"z": 1, "T_total": 10},
                            "b1b": {"z": 0, "T_total": 50, "t_think": "long thinking"},
                            "probe": {"T_probe": 10},
                            "fast_done": {"z": 1, "T_total": 10},
                        }) + "\n",
                        encoding="utf-8",
                    )
                    with mock.patch.object(eng, "_decode_raw", side_effect=_fake_diag_raw):
                        summary = run_section6_diagnostics(eng, [ehurt_item], [{"rationale_gold": "step 1"}], tp)
                    self.assertEqual(len(diag_calls), 5)  # 5 instead of 6 (B2b reused B2a!)
                    g1_e = run_gate1_lexical(ehurt_item["prompt"])
                    self.assertIn("B2b_post_raw", summary)
                    self.assertEqual(summary["B2a_post_anchor"]["n_ehurt"], 1)
                    controls_recs = {
                        (r["uid"], r["mode"]): r["a_final"]
                        for r in (json.loads(l) for l in (tp / "diag_controls_cache.jsonl").read_text().splitlines() if l)
                    }
                    self.assertEqual(controls_recs[(ehurt_item["uid"], "B2a_post_anchor")], "clean response")
                    self.assertEqual(
                        controls_recs[(ehurt_item["uid"], "B2b_post_raw")],
                        f"[Reminder: strictly follow: {g1_e.anchor_clause}]\nclean response",
                    )

    def test_detect_hardware_mps_telemetry(self) -> None:
        """Hardware probe records Apple Silicon MPS VRAM and BF16 capability cleanly."""
        from unittest import mock
        from types import SimpleNamespace
        from scripts.log_experiment import _detect_hardware

        fake_torch = SimpleNamespace(
            __version__="2.14.1",
            version=SimpleNamespace(cuda=None),
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
            bfloat16=torch.bfloat16,
            ones=lambda *a, **kw: torch.ones(*a),
            mps=SimpleNamespace(recommended_max_memory=lambda: int(24 * (1024**3))),
        )
        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            hw = _detect_hardware()
        self.assertTrue(hw["mps_available"])
        self.assertIn("Apple MPS", hw["gpu_name"])
        self.assertEqual(hw["vram_gb"], 24.0)
        self.assertTrue(hw["bf16_native"])

    def test_qwen35_device_safe_kernel_routing_with_mocked_triton(self) -> None:
        """Simulated Triton/FLA CPU crash is bypassed on CPU while CUDA tensors keep fast path."""
        import functools
        import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_mod
        from transformers import BatchEncoding
        from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        from scripts.ext_deltanet_blend import generate_with_blended_deltanet
        from src.engine import CBARRDualModeEngine, ensure_qwen35_device_safe_kernels

        orig_chunk = qwen_mod.torch_chunk_gated_delta_rule
        orig_rec = qwen_mod.torch_recurrent_gated_delta_rule
        orig_conv_fn = qwen_mod.causal_conv1d_fn
        orig_conv_upd = qwen_mod.causal_conv1d_update

        def _unwrap(fn):
            raw = fn
            while hasattr(raw, "__wrapped__"):
                raw = raw.__wrapped__
            return raw

        raw_chunk = _unwrap(orig_chunk)
        raw_rec = _unwrap(orig_rec)
        raw_conv_fn = _unwrap(orig_conv_fn)
        raw_conv_upd = _unwrap(orig_conv_upd)

        cuda_calls: list[str] = []

        def _make_triton_stub(tag: str, raw_fn):
            @functools.wraps(raw_fn)
            def _triton_stub(*args, **kwargs):
                cuda_calls.append(tag)
                raise ValueError("Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)")

            return _triton_stub

        class _SimpleTok:
            eos_token_id = 2
            pad_token_id = 0

            def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
                return " ".join(m["content"] for m in msgs) + (" <think>\n" if enable_thinking else "")

            def encode(self, text, add_special_tokens=False):
                return [10 + (sum(ord(c) for c in w) % 40) for w in text.split()] or [10, 11]

            def decode(self, token_ids, skip_special_tokens=False):
                return " ".join(f"w{t}" for t in token_ids if not skip_special_tokens or t != 2)

            def __call__(self, text, return_tensors="pt"):
                ids = torch.tensor([self.encode(text)], dtype=torch.long)
                return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

        try:
            qwen_mod.torch_chunk_gated_delta_rule = _make_triton_stub("chunk", raw_chunk)
            qwen_mod.torch_recurrent_gated_delta_rule = _make_triton_stub("rec", raw_rec)
            qwen_mod.causal_conv1d_fn = _make_triton_stub("conv_fn", raw_conv_fn)
            qwen_mod.causal_conv1d_update = _make_triton_stub("conv_upd", raw_conv_upd)

            self.assertTrue(ensure_qwen35_device_safe_kernels())

            cfg = Qwen3_5TextConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                linear_num_value_heads=4,
                linear_num_key_heads=2,
                linear_key_head_dim=8,
                linear_value_head_dim=8,
            )
            tiny_qwen = Qwen3_5ForCausalLM(cfg).eval()
            tok = _SimpleTok()

            # Both ext_deltanet_blend and CBARRDualModeEngine must execute on CPU without hitting Triton stubs
            out_blend = generate_with_blended_deltanet(
                tiny_qwen,
                tok,
                eos_ids={2},
                prompt_str="Answer concisely.",
                t_think="Checking steps.",
                alpha=0.5,
                max_ans_tokens=3,
                blend_conv=True,
                evict_think_kv=True,
            )
            self.assertGreaterEqual(out_blend["gen_tokens"], 1)

            eng = CBARRDualModeEngine.__new__(CBARRDualModeEngine)
            eng.model_id = "Qwen/Qwen3.5-Tiny"
            eng.is_smollm3 = False
            eng.freeze_smollm3_date = "01 November 2026"
            eng.device = "cpu"
            eng.temperature = 0.0
            eng.top_p = 1.0
            eng.presence_penalty = 0.0
            eng.sampling_seed = 42
            eng.tok = tok
            eng.model = tiny_qwen
            eng.eos_ids = {2}
            probe = eng.run_gate2_fast_probe("Explain gravity briefly.", probe_tokens=3)
            self.assertGreaterEqual(probe["T_probe"], 1)
            self.assertEqual(cuda_calls, [])

            # CUDA tensors still preserve the fast kernel path
            class _FakeCudaTensor(torch.Tensor):
                @property
                def is_cuda(self) -> bool:  # type: ignore[override]
                    return True

            fake_cuda_q = torch.randn(1, 2, 2, 4).as_subclass(_FakeCudaTensor)
            with self.assertRaisesRegex(ValueError, "Triton"):
                qwen_mod.torch_chunk_gated_delta_rule(
                    fake_cuda_q, fake_cuda_q, fake_cuda_q, fake_cuda_q[..., 0], fake_cuda_q[..., 0]
                )
            self.assertEqual(cuda_calls, ["chunk"])
        finally:
            qwen_mod.torch_chunk_gated_delta_rule = orig_chunk
            qwen_mod.torch_recurrent_gated_delta_rule = orig_rec
            qwen_mod.causal_conv1d_fn = orig_conv_fn
            qwen_mod.causal_conv1d_update = orig_conv_upd
            ensure_qwen35_device_safe_kernels()

    def test_sync_results_split_part1_part2_git_combine(self) -> None:
        """End-to-end simulation of Part 1 push -> Part 2 pull/unpack -> combined Phase 7 push."""
        import shutil
        from scripts.log_experiment import _upsert_run_block
        from scripts.sync_results import sync_and_push_results, unpack_repo_caches

        with tempfile.TemporaryDirectory() as td:
            root_tmp = pathlib.Path(td)
            tmp_data = root_tmp / "data"
            tmp_data.mkdir()
            _build_mini_frozen_dir(tmp_data)

            bare_repo = root_tmp / "origin.git"
            seed_repo = root_tmp / "seed_repo"
            subprocess.run(["git", "init", "--bare", str(bare_repo)], check=True, capture_output=True)
            subprocess.run(["git", "clone", str(bare_repo), str(seed_repo)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(seed_repo), "checkout", "-B", "master"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(seed_repo), "config", "user.name", "tester"], check=True)
            subprocess.run(["git", "-C", str(seed_repo), "config", "user.email", "tester@example.org"], check=True)

            shutil.copy2(ROOT / ".gitignore", seed_repo / ".gitignore")
            shutil.copy2(ROOT / "VERSION", seed_repo / "VERSION")
            (seed_repo / "scripts").mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "scripts" / "check_anonymity.sh", seed_repo / "scripts" / "check_anonymity.sh")
            (seed_repo / "paper" / "figures").mkdir(parents=True, exist_ok=True)
            (seed_repo / "paper" / "main.tex").write_text("\\author{Anonymous Authors}\n", encoding="utf-8")
            (seed_repo / "EXPERIMENT_LOG.md").write_text("# C-BARR Log\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(seed_repo), "add", "-A"], check=True)
            subprocess.run(["git", "-C", str(seed_repo), "commit", "-m", "seed"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(seed_repo), "push", "-u", "origin", "master"], check=True, capture_output=True)

            # Part 1 session (separate OUT_DIR simulating Colab Drive or working dir)
            p1_repo = root_tmp / "p1_repo"
            p1_out = root_tmp / "p1_drive_out"
            subprocess.run(["git", "clone", "-b", "master", str(bare_repo), str(p1_repo)], check=True, capture_output=True)
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/run_matrix.py"), "--model", "Qwen/Qwen3.5-4B", "--data-dir", str(tmp_data), "--out-dir", str(p1_out), "--strict-diag", "--mock"],
                check=True,
                cwd=str(ROOT),
            )
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/run_matrix.py"), "--model", "Qwen/Qwen3.5-4B", "--data-dir", str(tmp_data), "--out-dir", str(p1_out), "--mock"],
                check=True,
                cwd=str(ROOT),
            )
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/run_matrix.py"), "--model", "HuggingFaceTB/SmolLM3-3B", "--data-dir", str(tmp_data), "--out-dir", str(p1_out), "--strict-diag", "--mock"],
                check=True,
                cwd=str(ROOT),
            )
            _upsert_run_block(
                p1_repo / "EXPERIMENT_LOG.md",
                "r_p1",
                "<!-- CBARR_RUN_START:r_p1 -->\n### Run `r_p1` — **PASSED (PART1)**\n<!-- CBARR_RUN_END:r_p1 -->",
                upsert=True,
            )
            ok1, msg1 = sync_and_push_results(
                repo_root=p1_repo,
                out_dir=p1_out,
                label="part1-primary",
                models=["Qwen/Qwen3.5-4B", "HuggingFaceTB/SmolLM3-3B"],
                include_phase7=False,
                remote="origin",
                branch="master",
                git_push=True,
            )
            self.assertTrue(ok1, msg1)

            # Part 2 session (fresh Kaggle-style clone where OUT_DIR == p2_repo/outputs)
            p2_repo = root_tmp / "p2_repo"
            p2_out = p2_repo / "outputs"
            subprocess.run(["git", "clone", "-b", "master", str(bare_repo), str(p2_repo)], check=True, capture_output=True)
            unpacked = unpack_repo_caches(repo_root=p2_repo, out_dir=p2_out)
            self.assertGreaterEqual(unpacked["caches_unpacked"], 6)
            self.assertEqual(
                (p2_out / "Qwen__Qwen3.5-4B" / "eval_450_cache.jsonl").read_bytes(),
                (p1_out / "Qwen__Qwen3.5-4B" / "eval_450_cache.jsonl").read_bytes(),
            )
            self.assertTrue((p2_out / "Qwen__Qwen3.5-4B" / "diag_ehurt_cache.jsonl").exists())
            # Unpacked raw *.jsonl must remain ignored by git
            st = subprocess.check_output(["git", "-C", str(p2_repo), "status", "--short"], text=True).strip()
            self.assertEqual(st, "")

            for m_scale in ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"):
                subprocess.run(
                    [sys.executable, str(ROOT / "scripts/run_matrix.py"), "--model", m_scale, "--data-dir", str(tmp_data), "--out-dir", str(p2_out), "--strict-diag", "--mock"],
                    check=True,
                    cwd=str(ROOT),
                )

            p2_figs = p2_repo / "paper" / "figures"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/analyze_and_plot.py"), "--out-dir", str(p2_out), "--fig-dir", str(p2_figs)],
                check=True,
                cwd=str(ROOT),
            )
            stats = json.loads((p2_out / "stats_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                set(stats["models"].keys()),
                {
                    "Qwen/Qwen3.5-4B",
                    "HuggingFaceTB/SmolLM3-3B",
                    "Qwen/Qwen3.5-0.8B",
                    "Qwen/Qwen3.5-2B",
                },
            )
            _upsert_run_block(
                p2_repo / "EXPERIMENT_LOG.md",
                "r_p2",
                "<!-- CBARR_RUN_START:r_p2 -->\n### Run `r_p2` — **PASSED**\n<!-- CBARR_RUN_END:r_p2 -->",
                upsert=True,
            )
            ok2, msg2 = sync_and_push_results(
                repo_root=p2_repo,
                out_dir=p2_out,
                label="part2-combined",
                include_phase7=True,
                remote="origin",
                branch="master",
                git_push=True,
            )
            self.assertTrue(ok2, msg2)

            remote_files = set(
                subprocess.check_output(
                    ["git", "--git-dir", str(bare_repo), "ls-tree", "-r", "--name-only", "master"],
                    text=True,
                ).splitlines()
            )
            self.assertIn("outputs/stats_summary.json", remote_files)
            self.assertIn("outputs/paper_tables.tex", remote_files)
            for slug in (
                "Qwen__Qwen3.5-4B",
                "HuggingFaceTB__SmolLM3-3B",
                "Qwen__Qwen3.5-0.8B",
                "Qwen__Qwen3.5-2B",
            ):
                self.assertIn(f"outputs/{slug}/eval_450_cache.jsonl.gz", remote_files)
                self.assertIn(f"outputs/{slug}/matrix_summary_strict341.json", remote_files)
            self.assertFalse(any(p.endswith("_cache.jsonl") for p in remote_files))
            self.assertGreaterEqual(len([p for p in remote_files if p.startswith("paper/figures/") and p.endswith(".pdf")]), 3)

            remote_md = subprocess.check_output(
                ["git", "--git-dir", str(bare_repo), "show", "master:EXPERIMENT_LOG.md"],
                text=True,
            )
            self.assertIn("CBARR_RUN_START:r_p1", remote_md)
            self.assertIn("CBARR_RUN_START:r_p2", remote_md)

    def test_run_matrix_n_gpu_sharding_parity(self) -> None:
        """N=3 intra-model sharding (--shard-idx/--num-shards/--merge-shards) matches 1-GPU baseline bitwise."""
        with tempfile.TemporaryDirectory() as td:
            root_tmp = pathlib.Path(td)
            tmp_data = root_tmp / "data"
            tmp_data.mkdir()
            _build_mini_frozen_dir(tmp_data)

            out_single = root_tmp / "out_single"
            out_sharded = root_tmp / "out_sharded"

            # 1-GPU baseline: strict341 -> incremental diag491
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_matrix.py"),
                    "--model",
                    "Qwen/Qwen3.5-4B",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(out_single),
                    "--strict-diag",
                    "--mock",
                ],
                check=True,
                cwd=str(ROOT),
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_matrix.py"),
                    "--model",
                    "Qwen/Qwen3.5-4B",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(out_single),
                    "--mock",
                ],
                check=True,
                cwd=str(ROOT),
            )

            # N=3 sharded run: calibrate-only -> strict341 -> incremental diag491
            n_shards = 3
            for s_idx in range(n_shards):
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/run_matrix.py"),
                        "--model",
                        "Qwen/Qwen3.5-4B",
                        "--data-dir",
                        str(tmp_data),
                        "--out-dir",
                        str(out_sharded),
                        "--calibrate-only",
                        "--shard-idx",
                        str(s_idx),
                        "--num-shards",
                        str(n_shards),
                        "--mock",
                    ],
                    check=True,
                    cwd=str(ROOT),
                )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_matrix.py"),
                    "--model",
                    "Qwen/Qwen3.5-4B",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(out_sharded),
                    "--calibrate-only",
                    "--merge-shards",
                    "--mock",
                ],
                check=True,
                cwd=str(ROOT),
            )
            slug_dir_sharded = out_sharded / "Qwen__Qwen3.5-4B"
            self.assertFalse((slug_dir_sharded / "shards").exists())

            for diag_flags in (["--strict-diag"], []):
                for s_idx in range(n_shards):
                    subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "scripts/run_matrix.py"),
                            "--model",
                            "Qwen/Qwen3.5-4B",
                            "--data-dir",
                            str(tmp_data),
                            "--out-dir",
                            str(out_sharded),
                            "--shard-idx",
                            str(s_idx),
                            "--num-shards",
                            str(n_shards),
                            *diag_flags,
                            "--mock",
                        ],
                        check=True,
                        cwd=str(ROOT),
                    )
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/run_matrix.py"),
                        "--model",
                        "Qwen/Qwen3.5-4B",
                        "--data-dir",
                        str(tmp_data),
                        "--out-dir",
                        str(out_sharded),
                        "--merge-shards",
                        *diag_flags,
                        "--mock",
                    ],
                    check=True,
                    cwd=str(ROOT),
                )
                self.assertFalse((slug_dir_sharded / "shards").exists())

            slug_dir_single = out_single / "Qwen__Qwen3.5-4B"
            for tag in ("strict341", "diag491"):
                rep_single = json.loads((slug_dir_single / f"matrix_summary_{tag}.json").read_text(encoding="utf-8"))
                rep_sharded = json.loads((slug_dir_sharded / f"matrix_summary_{tag}.json").read_text(encoding="utf-8"))
                self.assertEqual(rep_single["calibrated_params"], rep_sharded["calibrated_params"])
                self.assertEqual(rep_single["systems"], rep_sharded["systems"])
                self.assertEqual(rep_single["oracle_and_4regime"], rep_sharded["oracle_and_4regime"])
                self.assertEqual(rep_single["section6_diagnostics"], rep_sharded["section6_diagnostics"])
                self.assertEqual(
                    (slug_dir_single / f"tables_rows_{tag}.tex").read_text(encoding="utf-8"),
                    (slug_dir_sharded / f"tables_rows_{tag}.tex").read_text(encoding="utf-8"),
                )

    def test_sync_results_concurrent_workers_no_ledger_combine(self) -> None:
        """Three concurrent workers (cloned from same commit, include_ledger=False) rebase-merge conflict-free -> Combiner."""
        import shutil
        from scripts.log_experiment import _upsert_run_block
        from scripts.sync_results import sync_and_push_results, unpack_repo_caches

        with tempfile.TemporaryDirectory() as td:
            root_tmp = pathlib.Path(td)
            tmp_data = root_tmp / "data"
            tmp_data.mkdir()
            _build_mini_frozen_dir(tmp_data)

            bare_repo = root_tmp / "origin.git"
            seed_repo = root_tmp / "seed_repo"
            subprocess.run(["git", "init", "--bare", str(bare_repo)], check=True, capture_output=True)
            subprocess.run(["git", "clone", str(bare_repo), str(seed_repo)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(seed_repo), "checkout", "-B", "master"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(seed_repo), "config", "user.name", "tester"], check=True)
            subprocess.run(["git", "-C", str(seed_repo), "config", "user.email", "tester@example.org"], check=True)

            shutil.copy2(ROOT / ".gitignore", seed_repo / ".gitignore")
            shutil.copy2(ROOT / "VERSION", seed_repo / "VERSION")
            (seed_repo / "scripts").mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "scripts" / "check_anonymity.sh", seed_repo / "scripts" / "check_anonymity.sh")
            (seed_repo / "paper" / "figures").mkdir(parents=True, exist_ok=True)
            (seed_repo / "paper" / "main.tex").write_text("\\author{Anonymous Authors}\n", encoding="utf-8")
            (seed_repo / "EXPERIMENT_LOG.md").write_text("# C-BARR Log\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(seed_repo), "add", "-A"], check=True)
            subprocess.run(["git", "-C", str(seed_repo), "commit", "-m", "seed"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(seed_repo), "push", "-u", "origin", "master"], check=True, capture_output=True)

            # Clone all 3 concurrent workers from the SAME base commit before any worker pushes
            worker_specs = [
                ("w_4b", "worker-4b", ["Qwen/Qwen3.5-4B"], []),
                ("w_smol", "worker-smollm3", ["HuggingFaceTB/SmolLM3-3B"], ["--strict-diag"]),
                ("w_scale", "worker-scale", ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"], ["--strict-diag"]),
            ]
            worker_repos: list[tuple[pathlib.Path, str, list[str]]] = []
            for dir_name, lbl, models, extra_flags in worker_specs:
                w_repo = root_tmp / dir_name
                w_out = w_repo / "outputs"
                subprocess.run(["git", "clone", "-b", "master", str(bare_repo), str(w_repo)], check=True, capture_output=True)
                for m_id in models:
                    subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "scripts/run_matrix.py"),
                            "--model",
                            m_id,
                            "--data-dir",
                            str(tmp_data),
                            "--out-dir",
                            str(w_out),
                            *extra_flags,
                            "--mock",
                        ],
                        check=True,
                        cwd=str(ROOT),
                    )
                # Dirty local EXPERIMENT_LOG.md must NOT get committed when include_ledger=False
                _upsert_run_block(
                    w_repo / "EXPERIMENT_LOG.md",
                    lbl,
                    f"<!-- CBARR_RUN_START:{lbl} -->\n### `{lbl}`\n<!-- CBARR_RUN_END:{lbl} -->",
                    upsert=True,
                )
                worker_repos.append((w_repo, lbl, models))

            # Push sequentially from stale clones (forces real rebase across disjoint outputs/<model>/* paths)
            for w_repo, lbl, models in worker_repos:
                ok, msg = sync_and_push_results(
                    repo_root=w_repo,
                    out_dir=w_repo / "outputs",
                    label=lbl,
                    models=models,
                    include_phase7=False,
                    include_ledger=False,
                    remote="origin",
                    branch="master",
                    git_push=True,
                )
                self.assertTrue(ok, f"{lbl} failed: {msg}")

            # Verify remote EXPERIMENT_LOG.md & outputs/RESULTS_VERSION remained untouched during concurrent worker pushes
            mid_md = subprocess.check_output(
                ["git", "--git-dir", str(bare_repo), "show", "master:EXPERIMENT_LOG.md"],
                text=True,
            )
            self.assertEqual(mid_md, "# C-BARR Log\n")
            mid_files = set(
                subprocess.check_output(
                    ["git", "--git-dir", str(bare_repo), "ls-tree", "-r", "--name-only", "master"],
                    text=True,
                ).splitlines()
            )
            self.assertNotIn("outputs/RESULTS_VERSION", mid_files)
            self.assertIn("outputs/Qwen__Qwen3.5-4B/run_manifest.json", mid_files)
            q4_mf = json.loads(
                subprocess.check_output(
                    ["git", "--git-dir", str(bare_repo), "show", "master:outputs/Qwen__Qwen3.5-4B/run_manifest.json"],
                    text=True,
                )
            )
            self.assertEqual(q4_mf["sync_seq"], 1)
            self.assertEqual(q4_mf["code_version"], (ROOT / "VERSION").read_text(encoding="utf-8").strip())

            # Combiner (Part 3) clones fresh, unpacks all 4 worker models, runs Phase 7 + ledger push
            comb_repo = root_tmp / "comb_repo"
            comb_out = comb_repo / "outputs"
            subprocess.run(["git", "clone", "-b", "master", str(bare_repo), str(comb_repo)], check=True, capture_output=True)
            unpacked = unpack_repo_caches(repo_root=comb_repo, out_dir=comb_out)
            self.assertEqual(unpacked["models"], 4)
            self.assertGreaterEqual(unpacked["caches_unpacked"], 12)

            comb_figs = comb_repo / "paper" / "figures"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/analyze_and_plot.py"), "--out-dir", str(comb_out), "--fig-dir", str(comb_figs)],
                check=True,
                cwd=str(ROOT),
            )
            _upsert_run_block(
                comb_repo / "EXPERIMENT_LOG.md",
                "r_comb",
                "<!-- CBARR_RUN_START:r_comb -->\n### Run `r_comb` — **PASSED (COMBINED 4 MODELS)**\n<!-- CBARR_RUN_END:r_comb -->",
                upsert=True,
            )
            ok_comb, msg_comb = sync_and_push_results(
                repo_root=comb_repo,
                out_dir=comb_out,
                label="part3-combined",
                include_phase7=True,
                include_ledger=True,
                remote="origin",
                branch="master",
                git_push=True,
            )
            self.assertTrue(ok_comb, msg_comb)

            stats = json.loads((comb_out / "stats_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                set(stats["models"].keys()),
                {
                    "Qwen/Qwen3.5-4B",
                    "HuggingFaceTB/SmolLM3-3B",
                    "Qwen/Qwen3.5-0.8B",
                    "Qwen/Qwen3.5-2B",
                },
            )
            self.assertTrue(stats["version_audit"]["consistent_across_models"])
            self.assertTrue(stats["version_audit"]["matches_analyzer_version"])
            final_md = subprocess.check_output(
                ["git", "--git-dir", str(bare_repo), "show", "master:EXPERIMENT_LOG.md"],
                text=True,
            )
            self.assertIn("CBARR_RUN_START:r_comb", final_md)
            res_ver_remote = subprocess.check_output(
                ["git", "--git-dir", str(bare_repo), "show", "master:outputs/RESULTS_VERSION"],
                text=True,
            ).strip()
            self.assertEqual(res_ver_remote, "1")
            res_mf = json.loads(
                subprocess.check_output(
                    ["git", "--git-dir", str(bare_repo), "show", "master:outputs/results_manifest.json"],
                    text=True,
                )
            )
            self.assertEqual(res_mf["results_seq"], 1)
            self.assertTrue(res_mf["consistent_code_versions"])
            self.assertEqual(len(res_mf["models"]), 4)

    def test_versioning_bump_hook_and_mismatch_audit(self) -> None:
        """Verifies code VERSION checks/pre-push hook, per-model sync_seq increments, and mismatch detection."""
        import shutil
        from scripts.sync_results import pack_outputs_for_repo, summarize_unpacked_versions
        from scripts.versioning import (
            bump_repo_version,
            bump_results_seq,
            bump_semver,
            check_version_increment,
            install_git_hooks,
            read_repo_version,
            read_results_seq,
        )

        self.assertEqual(bump_semver("0.3.0", "patch"), "0.3.1")
        self.assertEqual(bump_semver("0.3.9", "minor"), "0.4.0")
        self.assertEqual(bump_semver("1.2.3", "major"), "2.0.0")

        with tempfile.TemporaryDirectory() as td:
            root_tmp = pathlib.Path(td)
            bare_repo = root_tmp / "origin.git"
            work_repo = root_tmp / "work_repo"
            subprocess.run(["git", "init", "--bare", str(bare_repo)], check=True, capture_output=True)
            subprocess.run(["git", "clone", str(bare_repo), str(work_repo)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(work_repo), "checkout", "-B", "master"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(work_repo), "config", "user.name", "tester"], check=True)
            subprocess.run(["git", "-C", str(work_repo), "config", "user.email", "tester@example.org"], check=True)

            # Populate scripts/versioning.py and .githooks/pre-push in temp clone so real pre-push hook runs end-to-end
            (work_repo / "scripts").mkdir(parents=True, exist_ok=True)
            (work_repo / ".githooks").mkdir(parents=True, exist_ok=True)
            (work_repo / "src").mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "scripts" / "versioning.py", work_repo / "scripts" / "versioning.py")
            shutil.copy2(ROOT / ".githooks" / "pre-push", work_repo / ".githooks" / "pre-push")
            (work_repo / "VERSION").write_text("0.3.0\n", encoding="utf-8")
            (work_repo / "src" / "mod.py").write_text("X = 1\n", encoding="utf-8")
            (work_repo / "EXPERIMENT_LOG.md").write_text("# Log\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(work_repo), "add", "-A"], check=True)
            subprocess.run(["git", "-C", str(work_repo), "commit", "-m", "init"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(work_repo), "push", "-u", "origin", "master"], check=True, capture_output=True)

            install_git_hooks(work_repo)

            # 1. Code edit WITHOUT VERSION bump must fail check & pre-push hook
            (work_repo / "src" / "mod.py").write_text("X = 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(work_repo), "add", "src/mod.py"], check=True)
            subprocess.run(["git", "-C", str(work_repo), "commit", "-m", "code change"], check=True, capture_output=True)
            ok_chk, msg_chk = check_version_increment(work_repo, base_ref="origin/master", target_ref="HEAD")
            self.assertFalse(ok_chk, msg_chk)
            push_blocked = subprocess.run(
                ["git", "-C", str(work_repo), "push", "origin", "master"],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(push_blocked.returncode, 0)
            self.assertIn("[version-check] FAIL", push_blocked.stdout + push_blocked.stderr)

            # 2. Bump VERSION -> check passes and pre-push hook allows push
            new_v = bump_repo_version(work_repo, part="patch")
            self.assertEqual(new_v, "0.3.1")
            self.assertEqual(read_repo_version(work_repo), "0.3.1")
            subprocess.run(["git", "-C", str(work_repo), "add", "VERSION"], check=True)
            subprocess.run(["git", "-C", str(work_repo), "commit", "-m", "bump version"], check=True, capture_output=True)
            ok_chk2, msg_chk2 = check_version_increment(work_repo, base_ref="origin/master", target_ref="HEAD")
            self.assertTrue(ok_chk2, msg_chk2)
            push_ok = subprocess.run(
                ["git", "-C", str(work_repo), "push", "origin", "master"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(push_ok.returncode, 0, push_ok.stderr)

            # 3. Result-only push (RESULTS_VERSION + EXPERIMENT_LOG.md) must push cleanly WITHOUT bumping VERSION
            self.assertEqual(read_results_seq(work_repo), 0)
            self.assertEqual(bump_results_seq(work_repo), 1)
            (work_repo / "EXPERIMENT_LOG.md").write_text("# Log\nUpdated\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(work_repo), "add", "outputs/RESULTS_VERSION", "EXPERIMENT_LOG.md"], check=True)
            subprocess.run(["git", "-C", str(work_repo), "commit", "-m", "results sync"], check=True, capture_output=True)
            push_res = subprocess.run(
                ["git", "-C", str(work_repo), "push", "origin", "master"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(push_res.returncode, 0, push_res.stderr)

            # 4. Separate-training differentiation & mismatch audit across two model versions
            tmp_data = root_tmp / "data"
            tmp_data.mkdir()
            _build_mini_frozen_dir(tmp_data)
            split_out = root_tmp / "split_out"
            pack_repo = root_tmp / "pack_repo"
            (pack_repo / "outputs").mkdir(parents=True, exist_ok=True)
            (pack_repo / "VERSION").write_text((ROOT / "VERSION").read_text(encoding="utf-8"), encoding="utf-8")

            for m_id in ("Qwen/Qwen3.5-4B", "HuggingFaceTB/SmolLM3-3B"):
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/run_matrix.py"),
                        "--model",
                        m_id,
                        "--data-dir",
                        str(tmp_data),
                        "--out-dir",
                        str(split_out),
                        "--strict-diag",
                        "--mock",
                    ],
                    check=True,
                    cwd=str(ROOT),
                )

            # First pack of 4B -> sync_seq=1
            pack_outputs_for_repo(
                out_dir=split_out,
                repo_root=pack_repo,
                models=["Qwen/Qwen3.5-4B"],
                include_phase7=False,
                include_ledger=False,
                label="run1",
            )
            mf1 = json.loads((split_out / "Qwen__Qwen3.5-4B" / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(mf1["sync_seq"], 1)

            # Modified 4B artifact -> re-pack must increment sync_seq=2
            q4_sum_path = split_out / "Qwen__Qwen3.5-4B" / "matrix_summary_strict341.json"
            q4_sum = json.loads(q4_sum_path.read_text(encoding="utf-8"))
            q4_sum["run_id"] = "retrain_r2"
            q4_sum_path.write_text(json.dumps(q4_sum, indent=2) + "\n", encoding="utf-8")
            pack_outputs_for_repo(
                out_dir=split_out,
                repo_root=pack_repo,
                models=["Qwen/Qwen3.5-4B"],
                include_phase7=False,
                include_ledger=False,
                label="run2",
            )
            mf2 = json.loads((split_out / "Qwen__Qwen3.5-4B" / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(mf2["sync_seq"], 2)
            self.assertEqual(mf2["result_version"], f"v{read_repo_version(ROOT)}+m2")

            # Simulate SmolLM3-3B trained against an updated code version (e.g., 9.9.9)
            sm_sum_path = split_out / "HuggingFaceTB__SmolLM3-3B" / "matrix_summary_strict341.json"
            sm_sum = json.loads(sm_sum_path.read_text(encoding="utf-8"))
            sm_sum["code_version"] = "9.9.9"
            sm_sum_path.write_text(json.dumps(sm_sum, indent=2) + "\n", encoding="utf-8")
            pack_outputs_for_repo(
                out_dir=split_out,
                repo_root=pack_repo,
                models=["HuggingFaceTB/SmolLM3-3B"],
                include_phase7=False,
                include_ledger=False,
                label="smol_new_ver",
            )

            v_summary = summarize_unpacked_versions(split_out, expected_code_version=read_repo_version(ROOT))
            self.assertFalse(v_summary["consistent"])
            self.assertEqual(sorted(v_summary["versions_seen"]), sorted([read_repo_version(ROOT), "9.9.9"]))

            # Non-strict Phase 7 runs and records mismatch; --strict-version-match exits non-zero
            figs_dir = root_tmp / "figs"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/analyze_and_plot.py"), "--out-dir", str(split_out), "--fig-dir", str(figs_dir)],
                check=True,
                cwd=str(ROOT),
            )
            stats_mm = json.loads((split_out / "stats_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(stats_mm["version_audit"]["consistent_across_models"])
            self.assertIn("HuggingFaceTB/SmolLM3-3B", stats_mm["version_audit"]["mismatched_models"])

            strict_proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/analyze_and_plot.py"),
                    "--out-dir",
                    str(split_out),
                    "--fig-dir",
                    str(figs_dir),
                    "--strict-version-match",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(strict_proc.returncode, 0)

    def test_four_part_shell_scripts_workflow(self) -> None:
        """Runs Parts 1/2/3/4 1-model worker bash scripts concurrently in --mock mode, then Part 5 combiner."""
        import os

        from scripts.versioning import read_repo_version

        scripts_expected = (
            "run_part1_worker_qwen4b.sh",
            "run_part2_worker_smollm3.sh",
            "run_part3_worker_qwen2b.sh",
            "run_part4_worker_qwen08b.sh",
            "run_part5_combine_analyze.sh",
        )
        for s_name in scripts_expected:
            sp = ROOT / "scripts" / s_name
            self.assertTrue(sp.is_file(), f"Missing script: {sp}")
            self.assertTrue(os.access(sp, os.X_OK), f"Script not executable: {sp}")

        with tempfile.TemporaryDirectory() as td:
            root_tmp = pathlib.Path(td)
            tmp_data = root_tmp / "data"
            tmp_out = root_tmp / "out"
            tmp_figs = root_tmp / "figs"
            tmp_data.mkdir()
            _build_mini_frozen_dir(tmp_data)

            env = {**os.environ, "PYTHON_BIN": sys.executable, "CBARR_COLOR": "0"}
            worker_cmds = [
                [
                    "bash",
                    str(ROOT / "scripts" / s_name),
                    "--mock",
                    "--skip-preflight",
                    "--strict-only",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(tmp_out),
                ]
                for s_name in scripts_expected[:4]
            ]
            procs = [
                subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for cmd in worker_cmds
            ]
            for s_name, proc in zip(scripts_expected[:4], procs):
                out_s, err_s = proc.communicate(timeout=180)
                self.assertEqual(proc.returncode, 0, f"{s_name} failed:\nSTDOUT:\n{out_s}\nSTDERR:\n{err_s}")

            # Verify each worker populated only its own model directory + run logs in tmp_out
            for slug in (
                "Qwen__Qwen3.5-4B",
                "HuggingFaceTB__SmolLM3-3B",
                "Qwen__Qwen3.5-2B",
                "Qwen__Qwen3.5-0.8B",
            ):
                self.assertTrue(
                    (tmp_out / slug / "matrix_summary_strict341.json").is_file(),
                    f"Missing strict341 summary for {slug}",
                )

            part5 = subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts" / "run_part5_combine_analyze.sh"),
                    "--mock",
                    "--skip-preflight",
                    "--no-unpack",
                    "--strict-version-match",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(tmp_out),
                    "--fig-dir",
                    str(tmp_figs),
                    "--exp-log",
                    str(tmp_out / "combined_exp_log.md"),
                ],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(part5.returncode, 0, f"Part 5 failed:\n{part5.stdout}\n{part5.stderr}")
            stats = json.loads((tmp_out / "stats_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                sorted(stats["models"].keys()),
                [
                    "HuggingFaceTB/SmolLM3-3B",
                    "Qwen/Qwen3.5-0.8B",
                    "Qwen/Qwen3.5-2B",
                    "Qwen/Qwen3.5-4B",
                ],
            )
            self.assertTrue(stats["version_audit"]["consistent_across_models"])
            self.assertEqual(stats["version_audit"]["analyzer_code_version"], read_repo_version(ROOT))
            self.assertTrue((tmp_out / "paper_tables.tex").is_file())
            self.assertTrue((tmp_figs / "fig1_pareto_frontier.pdf").is_file())
            self.assertTrue((tmp_figs / "fig2_bsr_sla_curve.pdf").is_file())
            self.assertTrue((tmp_figs / "fig3_ehurt_causal_recovery.pdf").is_file())

    def test_run_all_a_to_z_script_mock_and_git_auth(self) -> None:
        """A-Z runner script runs full pipeline end-to-end in mock mode and scrubs token secrets."""
        import os

        a2z_path = ROOT / "scripts" / "run_all_a_to_z.sh"
        self.assertTrue(a2z_path.is_file())
        self.assertTrue(os.access(a2z_path, os.X_OK))
        for s_name in ("run_all_a_to_z.sh", "run_gpu_phases_4_to_6.sh", "setup_env.sh"):
            syn = subprocess.run(["bash", "-n", str(ROOT / "scripts" / s_name)], capture_output=True, text=True)
            self.assertEqual(syn.returncode, 0, syn.stderr)

        with tempfile.TemporaryDirectory() as td:
            root_tmp = pathlib.Path(td)
            tmp_data = root_tmp / "data"
            tmp_out = root_tmp / "out"
            tmp_figs = root_tmp / "figs"
            tmp_data.mkdir()
            _build_mini_frozen_dir(tmp_data)

            secret_tok = "ghp_test_super_secret_token_xyz"
            proc = subprocess.run(
                [
                    "bash",
                    str(a2z_path),
                    "--mock",
                    "--no-push",
                    "--no-setup",
                    "--skip-preflight",
                    "--strict-only",
                    "--models",
                    "Qwen/Qwen3.5-0.8B",
                    "--github-token",
                    secret_tok,
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(tmp_out),
                    "--fig-dir",
                    str(tmp_figs),
                    "--run-id",
                    "a2z_mock_01",
                ],
                cwd=str(ROOT),
                env={**os.environ, "PYTHON_BIN": sys.executable, "CBARR_COLOR": "0"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, f"A-Z failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertTrue((tmp_out / "Qwen__Qwen3.5-0.8B" / "matrix_summary_strict341.json").is_file())
            self.assertTrue((tmp_out / "Qwen__Qwen3.5-0.8B" / "ext_deltanet_blend.json").is_file())
            self.assertTrue((tmp_out / "Qwen__Qwen3.5-0.8B" / "prefill_mlp_summary.json").is_file())
            self.assertTrue((tmp_out / "stats_summary.json").is_file())
            self.assertTrue((tmp_out / "paper_tables.tex").is_file())
            self.assertTrue((tmp_figs / "fig1_pareto_frontier.pdf").is_file())

            log_txt = (tmp_out / "logs" / "cbarr_a2z_mock_01.log").read_text(encoding="utf-8")
            md_txt = (tmp_out / "experiment_log.md").read_text(encoding="utf-8")
            self.assertNotIn(secret_tok, log_txt)
            self.assertNotIn(secret_tok, md_txt)
            self.assertIn("***", md_txt)
            self.assertIn("PASSED", md_txt)

    def test_sharded_live_progress_monitor_and_error_tail(self) -> None:
        """Multi-GPU sharded runner streams GPU0/milestones, prints progress snapshots, and surfaces shard errors."""
        import os

        with tempfile.TemporaryDirectory() as td:
            root_tmp = pathlib.Path(td)
            tmp_data = root_tmp / "data"
            tmp_out = root_tmp / "out"
            tmp_logs = tmp_out / "logs"
            tmp_data.mkdir()
            _build_mini_frozen_dir(tmp_data)

            proc = subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts" / "run_gpu_phases_4_to_6.sh"),
                    "--mock",
                    "--part",
                    "3",
                    "--shards",
                    "3",
                    "--shard-progress-sec",
                    "0.2",
                    "--shard-log",
                    "auto",
                    "--skip-preflight",
                    "--skip-smoke",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(tmp_out),
                    "--run-id",
                    "shard_mon_ok",
                ],
                cwd=str(ROOT),
                env={**os.environ, "PYTHON_BIN": sys.executable, "CBARR_COLOR": "0"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, f"Sharded run failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertTrue((tmp_out / "Qwen__Qwen3.5-2B" / "matrix_summary_strict341.json").is_file())
            self.assertFalse((tmp_out / "Qwen__Qwen3.5-2B" / "shards").exists())

            unified_log = (tmp_logs / "cbarr_shard_mon_ok.log").read_text(encoding="utf-8")
            self.assertIn("[sharded] GPU map (3):", unified_log)
            self.assertIn("[sharded] Live follow : tail -f", unified_log)
            self.assertIn("[GPU0]", unified_log)
            self.assertIn("GPU0 (dev=0) DONE in", unified_log)
            self.assertIn("GPU2 (dev=2) DONE in", unified_log)
            for s_idx in range(3):
                s_log = tmp_logs / f"cbarr_shard_mon_ok_gpu{s_idx}_Qwen__Qwen3.5-2B.log"
                self.assertTrue(s_log.is_file(), f"Missing per-GPU log {s_log}")
                self.assertIn(f"COMPLETED SHARD {s_idx}/3", s_log.read_text(encoding="utf-8"))

            # Multi-worker per GPU round-robin test (--shards 2 --workers-per-gpu 2 => 4 shards on devs 0,1,0,1)
            out_rr = root_tmp / "out_rr"
            logs_rr = out_rr / "logs"
            proc_rr = subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts" / "run_gpu_phases_4_to_6.sh"),
                    "--mock",
                    "--part",
                    "3",
                    "--shards",
                    "2",
                    "--workers-per-gpu",
                    "2",
                    "--shard-progress-sec",
                    "0",
                    "--shard-log",
                    "milestones",
                    "--skip-preflight",
                    "--skip-smoke",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(out_rr),
                    "--run-id",
                    "shard_rr_ok",
                ],
                cwd=str(ROOT),
                env={**os.environ, "PYTHON_BIN": sys.executable, "CBARR_COLOR": "0"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc_rr.returncode, 0, f"Round-robin sharded run failed:\n{proc_rr.stdout}\n{proc_rr.stderr}")
            log_rr = (logs_rr / "cbarr_shard_rr_ok.log").read_text(encoding="utf-8")
            self.assertIn("4 shards across 2 GPU(s), 2 workers/GPU", log_rr)
            self.assertIn("GPU map (4): shard0->dev0, shard1->dev1, shard2->dev0, shard3->dev1", log_rr)
            self.assertTrue((out_rr / "Qwen__Qwen3.5-2B" / "matrix_summary_strict341.json").is_file())

            # Direct unit check of snapshot rendering + immediate error tailing on failing shard & thread caps
            dummy_worker = root_tmp / "dummy_shard.py"
            dummy_worker.write_text(
                "\n".join(
                    [
                        "import json, os, pathlib, sys, time",
                        "args = sys.argv[1:]",
                        "s_idx = int(args[args.index('--shard-idx') + 1])",
                        "n_sh = int(args[args.index('--num-shards') + 1])",
                        "out_dir = pathlib.Path(args[args.index('--out-dir') + 1])",
                        "s_dir = out_dir / 'Qwen__Qwen3.5-2B' / 'shards' / f'shard_{s_idx}_of_{n_sh}'",
                        "s_dir.mkdir(parents=True, exist_ok=True)",
                        "if s_idx == 0:",
                        "    print(f\"OMP={os.environ.get('OMP_NUM_THREADS')} [eval_450 1/5] running ok\", flush=True)",
                        "    (s_dir / 'eval_450_cache.jsonl').write_text(json.dumps({'item': {'uid': 'u0'}}) + '\\n', encoding='utf-8')",
                        "    time.sleep(0.35)",
                        "    print('COMPLETED SHARD 0/2', flush=True)",
                        "else:",
                        "    sys.stdout.write('Loading shards: 50%\\rLoading shards: 100%\\r\\n')",
                        "    sys.stdout.flush()",
                        "    time.sleep(0.15)",
                        "    print('Simulated OOM failure on GPU1', flush=True)",
                        "    sys.exit(2)",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fail_out = root_tmp / "fail_out"
            fail_logs = root_tmp / "fail_logs"
            fail_proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_sharded.py"),
                    "--stage-name",
                    "Test Sharded Monitor",
                    "--model",
                    "Qwen/Qwen3.5-2B",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(fail_out),
                    "--log-dir",
                    str(fail_logs),
                    "--run-id",
                    "fail_case",
                    "--shards",
                    "2",
                    "--progress-sec",
                    "0.12",
                    "--shard-log",
                    "auto",
                    "--",
                    sys.executable,
                    str(dummy_worker),
                    "--out-dir",
                    str(fail_out),
                    "--strict-diag",
                ],
                cwd=str(ROOT),
                env={k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"} | {"CBARR_COLOR": "0"},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(fail_proc.returncode, 0)
            self.assertIn("[Shards Progress | Test Sharded Monitor]", fail_proc.stdout)
            self.assertIn("Total (canon+shards):", fail_proc.stdout)
            self.assertIn("OMP=4", fail_proc.stdout)
            self.assertIn("GPU1 (dev=1) FAILED exit=2", fail_proc.stdout)
            self.assertIn("Simulated OOM failure on GPU1", fail_proc.stdout)
            gpu1_log = (fail_logs / "cbarr_fail_case_gpu1_Qwen__Qwen3.5-2B.log").read_text(encoding="utf-8")
            self.assertIn("Loading shards: 100%", gpu1_log)

    def test_tokenizer_only_merge_finalizes_cached_section6_without_gpu(self) -> None:
        """Tokenizer-only --merge-shards finalization succeeds when 100% cached without _decode_raw."""
        from unittest import mock

        from scripts import run_matrix as rm
        from scripts.run_smoke_test import MockDualModeEngine

        with tempfile.TemporaryDirectory() as td:
            root_tmp = pathlib.Path(td)
            tmp_data = root_tmp / "data"
            tmp_out = root_tmp / "out"
            tmp_data.mkdir()
            _build_mini_frozen_dir(tmp_data)

            mock_eng = MockDualModeEngine("Qwen/Qwen3.5-2B")
            tok_only = object.__new__(rm._TokenizerOnlyEngine)
            tok_only.model_id = "Qwen/Qwen3.5-2B"
            tok_only.device = "cpu"
            tok_only.tok = mock_eng.tok
            self.assertFalse(rm._is_native_engine(tok_only))

            # 1) Populate shards via mock mode, then patch a cached E_hurt + controls entry
            for s_idx in range(2):
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "run_matrix.py"),
                        "--model",
                        "Qwen/Qwen3.5-2B",
                        "--data-dir",
                        str(tmp_data),
                        "--out-dir",
                        str(tmp_out),
                        "--shard-idx",
                        str(s_idx),
                        "--num-shards",
                        "2",
                        "--strict-diag",
                        "--mock",
                    ],
                    check=True,
                    cwd=str(ROOT),
                )

            model_dir = tmp_out / "Qwen__Qwen3.5-2B"
            diag_cache_0 = model_dir / "shards" / "shard_0_of_2" / "diag_ehurt_cache.jsonl"
            diag_rows_0 = _read_jsonl(diag_cache_0)
            self.assertGreater(len(diag_rows_0), 0)
            diag_rows_0[0]["b0"]["z"] = 1
            diag_rows_0[0]["b1b"]["z"] = 0
            diag_rows_0[0]["b1b"]["t_think"] = "Let me check constraint carefully."
            _write_jsonl(diag_cache_0, diag_rows_0)
            ehurt_uid = str(diag_rows_0[0]["item"]["uid"])
            ctrl_rows = [
                {
                    "uid": ehurt_uid,
                    "mode": m,
                    "a_final": "ok",
                    "grade": {"z": 1, "inhibition_strict_all": 1, "structural_strict_all": 1},
                }
                for m in rm._DIAG_MODES
            ]
            _write_jsonl(model_dir / "shards" / "shard_0_of_2" / "diag_controls_cache.jsonl", ctrl_rows)

            def _fake_tok_init(inst: object, model_id: str, local_files_only: bool = False) -> None:
                inst.model_id = model_id
                inst.device = "cpu"
                inst.tok = mock_eng.tok

            with mock.patch.object(rm._TokenizerOnlyEngine, "__init__", _fake_tok_init):
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_matrix.py",
                        "--model",
                        "Qwen/Qwen3.5-2B",
                        "--data-dir",
                        str(tmp_data),
                        "--out-dir",
                        str(tmp_out),
                        "--merge-shards",
                        "--strict-diag",
                    ],
                ):
                    rm.main()

            self.assertFalse((model_dir / "shards").exists())
            summary = json.loads((model_dir / "matrix_summary_strict341.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["section6_diagnostics"]["A1_strip"]["n_ehurt"], 1)
            self.assertEqual(summary["section6_diagnostics"]["B2b_post_raw"]["overall_recovery_pct"], 100.0)

    def test_gate2_clean_labels_cost_aware_tau_and_m2_prec(self) -> None:
        import numpy as np

        from scripts import run_matrix as rm
        from src.routers import (
            _gate2_target_label,
            fit_gate2_logistic_weights,
            resolve_m2_prec_variant,
            select_cost_aware_tau,
        )

        # Clean label ignores false positives where fast=0, slow=0, I_inertia=1
        self.assertEqual(
            _gate2_target_label({"fast_done": {"z": 0}, "m2_b128": {"z": 0}, "probe": {"I_inertia": 1}}),
            0,
        )
        self.assertEqual(
            _gate2_target_label({"fast_done": {"z": 0}, "m2_b128": {"z": 1}, "probe": {"I_inertia": 0}}),
            1,
        )

        # Near-tie threshold selection prefers lower escalation rate within tie_tol
        sweep = [
            {"tau": 0.45, "utility_u005": 0.7700, "esc_rate": 0.80},
            {"tau": 0.70, "utility_u005": 0.7695, "esc_rate": 0.20},
            {"tau": 0.80, "utility_u005": 0.7650, "esc_rate": 0.10},
        ]
        chosen = select_cost_aware_tau(sweep, tie_tol=0.002)
        self.assertEqual(chosen["tau"], 0.70)

        # Standardized LogisticRegression projection preserves predicted probability sign
        x = np.array(
            [
                [0.1, 0.1, 0.2, 0.0, 0.0],
                [0.2, 0.15, 0.25, 0.0, 0.0],
                [1.4, 1.2, 1.9, 1.0, 1.0],
                [1.5, 1.3, 2.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        y = np.array([0, 0, 1, 1], dtype=int)
        coef, intercept = fit_gate2_logistic_weights(x, y, seed=42, reg_c=1.0)
        p_lo = 1.0 / (1.0 + np.exp(-float(np.dot(coef, x[0]) + intercept)))
        p_hi = 1.0 / (1.0 + np.exp(-float(np.dot(coef, x[3]) + intercept)))
        self.assertGreater(p_hi, p_lo)

        # M2_Prec routes pure phi_prec items to m2_b128 while resolve_all_variants stays 13-key table
        prec_row = {
            "item": {
                "uid": "prec_1",
                "dataset": "ifeval",
                "slice": "ifeval_precision",
                "prompt": "Do not include any commas in your explanation of gravity.",
                "gold": {},
            },
            "b0": {"z": 0, "T_total": 80, "calls": 1, "T_probe": 0, "T_think": 0, "latency_ms": 1.0},
            "b1a": {"z": 0, "T_total": 80, "calls": 1, "T_probe": 0, "T_think": 0, "latency_ms": 1.0},
            "b1b": {"z": 0, "T_total": 300, "calls": 1, "T_probe": 0, "T_think": 200, "latency_ms": 2.0},
            "b2a": {"z": 0, "T_total": 80, "calls": 1, "T_probe": 0, "T_think": 0, "latency_ms": 1.0},
            "fast_done": {"z": 0, "T_total": 80, "calls": 1, "T_probe": 32, "T_think": 0, "latency_ms": 1.0},
            "probe": {"H_first": 0.2, "H_mean": 0.2, "H_max": 0.4, "I_inertia": 0, "T_probe": 32, "latency_ms": 0.5},
            "m2_b64": {"z": 1, "T_total": 120, "calls": 1, "T_probe": 0, "T_think": 64, "latency_ms": 1.5},
            "m2_b128": {"z": 1, "T_total": 140, "calls": 1, "T_probe": 0, "T_think": 90, "latency_ms": 1.5},
            "m2_b256": {"z": 1, "T_total": 200, "calls": 1, "T_probe": 0, "T_think": 150, "latency_ms": 2.0},
            "m2_b256_direct": {"z": 1, "T_total": 200, "calls": 1, "T_probe": 0, "T_think": 150, "latency_ms": 2.0},
            "m2_pre_b64": {"z": 0, "T_total": 120, "calls": 1, "T_probe": 0, "T_think": 64, "latency_ms": 1.5},
            "m2_pre_b128": {"z": 0, "T_total": 140, "calls": 1, "T_probe": 0, "T_think": 90, "latency_ms": 1.5},
            "m2_pre_b256": {"z": 0, "T_total": 200, "calls": 1, "T_probe": 0, "T_think": 150, "latency_ms": 2.0},
            "m2_pre_b256_direct": {"z": 0, "T_total": 200, "calls": 1, "T_probe": 0, "T_think": 150, "latency_ms": 2.0},
        }
        dummy_params = rm.CalibratedRouterParams(
            coef=[0.1, 0.1, 0.1, 0.1, 0.1],
            intercept=-1.0,
            tau_prob=0.5,
            tau_h_high=0.8,
            m2_escalation_rate_dev=0.2,
        )
        base_vars = rm.resolve_all_variants_for_item(
            prec_row, dummy_params, np.random.default_rng(42), target_esc_prob=0.2
        )
        self.assertEqual(len(base_vars), 13)
        self.assertNotIn("M2_Prec", base_vars)
        self.assertEqual(base_vars["M2"]["z"], 0)
        self.assertEqual(resolve_m2_prec_variant(prec_row, dummy_params)["z"], 1)

    def test_stage2_solve_first_bridge_and_compare_m2_v2_mock(self) -> None:
        from unittest import mock

        from scripts import run_matrix as rm
        from scripts.compare_m2_v2 import replay_stage2_with_bridge, run_comparison
        from scripts.run_smoke_test import MockDualModeEngine
        from src.engine import P_TERM, run_gate1_lexical

        eng_classic = MockDualModeEngine("Qwen/Qwen3.5-4B", stage2_bridge="classic")
        eng_v2 = MockDualModeEngine("Qwen/Qwen3.5-4B", stage2_bridge="solve_first")
        c_classic, v_classic = eng_classic._build_reanchor_for_engine(True, "", "post_stripped")
        c_v2, v_v2 = eng_v2._build_reanchor_for_engine(True, "", "post_stripped")
        self.assertEqual(c_classic, f"{P_TERM}\n</think>\n")
        self.assertEqual(v_classic, "")
        self.assertIn("work through any remaining steps", c_v2)
        self.assertIn("<|im_start|>assistant", c_v2)
        self.assertEqual(v_v2, "")

        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            tmp_data = tmp / "data"
            tmp_out = tmp / "outputs"
            tmp_data.mkdir()
            _build_mini_frozen_dir(tmp_data)
            with mock.patch.object(
                sys,
                "argv",
                [
                    "run_matrix.py",
                    "--model",
                    "Qwen/Qwen3.5-4B",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(tmp_out),
                    "--strict-diag",
                    "--mock",
                ],
            ):
                rm.main()

            model_dir = tmp_out / "Qwen__Qwen3.5-4B"
            eval_cache_p = model_dir / "eval_450_cache.jsonl"
            eval_rows = _read_jsonl(eval_cache_p)
            # Inject a stale GSM8K false-negative row matching the real bug + truncated m2_b256
            gsm_idx = next(i for i, r in enumerate(eval_rows) if r["item"]["dataset"] == "gsm8k")
            gold_val = float(eval_rows[gsm_idx]["item"]["gold"])
            gold_str = str(int(gold_val)) if gold_val.is_integer() else str(gold_val)
            for sk in ("b0", "b1b", "m2_b128", "m2_b256"):
                eval_rows[gsm_idx][sk]["a_final"] = f"She will be **{gold_str}** years old in 3 years."
                eval_rows[gsm_idx][sk]["z"] = 0
                eval_rows[gsm_idx][sk]["pred"] = "3"
            eval_rows[gsm_idx]["m2_b256"]["truncated"] = True
            _write_jsonl(eval_cache_p, eval_rows)
            orig_eval_text = eval_cache_p.read_text(encoding="utf-8")

            report = run_comparison(
                "Qwen/Qwen3.5-4B",
                model_dir,
                use_mock=True,
            )
            self.assertGreaterEqual(report["gsm8k_cache_rows_regraded"], 1)
            replayed_row = replay_stage2_with_bridge(
                eng_v2,
                eval_rows[gsm_idx]["item"],
                eval_rows[gsm_idx]["m2_b256"],
                run_gate1_lexical(eval_rows[gsm_idx]["item"]["prompt"]),
            )
            self.assertTrue(replayed_row["truncated"])
            self.assertGreater(replayed_row["T_inj"], 0)
            self.assertIn("M2_classic", report["systems"])
            self.assertIn("M2_solve_first", report["systems"])
            self.assertIn("M2_Prec_solve_first", report["systems"])
            self.assertTrue((model_dir / "eval_450_m2v2_comparison.json").is_file())
            self.assertEqual(eval_cache_p.read_text(encoding="utf-8"), orig_eval_text)

            # Verify run_matrix.py itself logs M2_Prec under auxiliary_systems and re-grades cached GSM8K
            summary = json.loads((model_dir / "matrix_summary_strict341.json").read_text(encoding="utf-8"))
            self.assertIn("M2_Prec", summary["auxiliary_systems"])

    def test_populated_slice_macro_acc_and_check_fully_cached(self) -> None:
        from src.metrics import evaluate_system_table

        # 5-slice split (omitting ifeval_conflict, like dev_60) averages only over populated slices
        items_5 = [
            {"slice": s}
            for s in ("gsm8k_easy", "gsm8k_hard", "csqa", "ifeval_planning", "ifeval_precision")
        ]
        runs_5 = [
            {"z": 1, "T_total": 50, "T_probe": 0, "T_think": 0, "calls": 1}
            for _ in items_5
        ]
        tab_5 = evaluate_system_table(items_5, runs_5)
        self.assertAlmostEqual(tab_5["macro_acc"], 1.0, places=6)

        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            tmp_data = tmp / "data"
            tmp_out = tmp / "outputs"
            tmp_data.mkdir()
            _build_mini_frozen_dir(tmp_data)

            pre_chk = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_matrix.py"),
                    "--model",
                    "Qwen/Qwen3.5-2B",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(tmp_out),
                    "--strict-diag",
                    "--check-fully-cached",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(pre_chk.returncode, 1)

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_matrix.py"),
                    "--model",
                    "Qwen/Qwen3.5-2B",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(tmp_out),
                    "--strict-diag",
                    "--mock",
                ],
                check=True,
                cwd=str(ROOT),
            )
            post_chk = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_matrix.py"),
                    "--model",
                    "Qwen/Qwen3.5-2B",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(tmp_out),
                    "--strict-diag",
                    "--check-fully-cached",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(post_chk.returncode, 0)

    def test_w7_deltanet_cache_fastpath_and_prefill_mlp_clean_labels(self) -> None:
        from scripts import ext_deltanet_blend as edb
        from scripts import ext_prefill_mlp as epm

        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            tmp_data = tmp / "data"
            tmp_out = tmp / "outputs"
            tmp_data.mkdir()
            _build_mini_frozen_dir(tmp_data)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_matrix.py"),
                    "--model",
                    "Qwen/Qwen3.5-4B",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(tmp_out),
                    "--strict-diag",
                    "--mock",
                ],
                check=True,
                cwd=str(ROOT),
            )
            model_dir = tmp_out / "Qwen__Qwen3.5-4B"
            diag_p = model_dir / "diag_ehurt_cache.jsonl"
            d_rows = _read_jsonl(diag_p)
            d_rows[0]["b0"]["z"] = 1
            d_rows[0]["b1b"]["z"] = 0
            d_rows[0]["b1b"]["t_think"] = "think trace"
            _write_jsonl(diag_p, d_rows)
            uid0 = str(d_rows[0]["item"]["uid"])
            _write_jsonl(
                model_dir / "ext_deltanet_blend_cache.jsonl",
                [
                    {
                        "uid": uid0,
                        "alpha": 0.5,
                        "blend_conv_state": False,
                        "evict_think_kv": False,
                        "n_blended_layers": 18,
                        "n_blended_conv": 0,
                        "n_evicted_kv": 0,
                        "a_final": "ok",
                        "grade": {"z": 1, "inhibition_strict_all": 1, "structural_strict_all": 1},
                    }
                ],
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "ext_deltanet_blend.py"),
                    "--model",
                    "Qwen/Qwen3.5-4B",
                    "--out-dir",
                    str(tmp_out),
                    "--alphas",
                    "0.5",
                ],
                check=True,
                cwd=str(ROOT),
            )
            blend_rep = json.loads((model_dir / "ext_deltanet_blend.json").read_text(encoding="utf-8"))
            self.assertEqual(blend_rep["alphas"]["0.50"]["overall_recovery_pct"], 100.0)
            self.assertEqual(edb._blend_key(uid0, 0.5, False, False)[1], "0.5000")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "ext_prefill_mlp.py"),
                    "--model",
                    "Qwen/Qwen3.5-4B",
                    "--out-dir",
                    str(tmp_out),
                    "--mock",
                ],
                check=True,
                cwd=str(ROOT),
            )
            mlp_rep = json.loads((model_dir / "prefill_mlp_summary.json").read_text(encoding="utf-8"))
            self.assertIn("best_tau", mlp_rep)
            self.assertTrue(callable(epm.train_and_evaluate_prefill_mlp))

    def test_non_mock_sharded_fast_skip_bash_and_kernel_error_extractor(self) -> None:
        import os

        from scripts import setup_fast_kernels as sfk

        sample_pip_err = (
            "  × Building wheel for causal-conv1d (pyproject.toml) did not run successfully.\n"
            "  ╰─> [3 lines of output]\n"
            "      RuntimeError: The detected CUDA version (12.4) mismatches PyTorch (13.2)\n"
            "      [end of output]\n"
            "  note: This error originates from a subprocess.\n"
            "╰─> causal-conv1d\n"
        )
        extracted = sfk._extract_pip_build_error(sample_pip_err)
        self.assertIn("RuntimeError: The detected CUDA version", extracted)
        self.assertNotIn("╰─> causal-conv1d", extracted)

        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            tmp_data = tmp / "data"
            tmp_out = tmp / "outputs"
            tmp_figs = tmp / "figs"
            tmp_exp = tmp / "exp.md"
            tmp_patch = tmp / "py_patch"
            tmp_data.mkdir()
            tmp_patch.mkdir()
            _build_mini_frozen_dir(tmp_data)

            # Stub AutoTokenizer inside subprocess so IS_MOCK=0 --merge-shards runs 100% offline on CPU
            (tmp_patch / "sitecustomize.py").write_text(
                "import transformers\n"
                "from scripts.run_smoke_test import MockTokenizer\n"
                "transformers.AutoTokenizer.from_pretrained = staticmethod(lambda m, **kw: MockTokenizer('smollm3' in str(m).lower()))\n",
                encoding="utf-8",
            )

            # Populate canonical cache via mock, then run non-mock bash with --shards 2 to exercise
            # run_matrix_sharded's live IS_MOCK=0 --check-fully-cached fast-skip under `set -euo pipefail`.
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_matrix.py"),
                    "--model",
                    "Qwen/Qwen3.5-2B",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(tmp_out),
                    "--strict-diag",
                    "--mock",
                ],
                check=True,
                cwd=str(ROOT),
            )
            prev_pp = os.environ.get("PYTHONPATH", "")
            sub_pp = f"{tmp_patch}:{ROOT}:{prev_pp}" if prev_pp else f"{tmp_patch}:{ROOT}"
            proc = subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts" / "run_gpu_phases_4_to_6.sh"),
                    "--part",
                    "all",
                    "--shards",
                    "2",
                    "--no-setup",
                    "--skip-preflight",
                    "--skip-smoke",
                    "--no-w7",
                    "--no-9b",
                    "--no-push",
                    "--strict-only",
                    "--models",
                    "Qwen/Qwen3.5-2B",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(tmp_out),
                    "--fig-dir",
                    str(tmp_figs),
                    "--exp-log",
                    str(tmp_exp),
                ],
                cwd=str(ROOT),
                env={**os.environ, "PYTHON_BIN": sys.executable, "PYTHONPATH": sub_pp},
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
            self.assertIn("All items already cached — skipping", proc.stdout)
            self.assertIn("M2_Prec (Aux.)", tmp_exp.read_text(encoding="utf-8"))

    def test_gpu_pool_live_free_vram_slots_and_partial_blend_recovery(self) -> None:
        from scripts.ext_deltanet_blend import _harvest_partial_blend_shards
        from scripts.gpu_pool import compute_slots_from_free_mem

        # Simulate shared 8x L40 server where 7 GPUs are occupied (~8.5G free) and only GPU 4 is free (44.4G)
        shared_lab_gpus = [
            {"id": str(i), "free_gb": 44.4 if i == 4 else 8.5, "total_gb": 44.4}
            for i in range(8)
        ]
        self.assertEqual(
            compute_slots_from_free_mem(shared_lab_gpus, "Qwen/Qwen3.5-4B", workers_per_gpu="auto"),
            ["4", "4"],
        )
        self.assertEqual(
            compute_slots_from_free_mem(shared_lab_gpus, "Qwen/Qwen3.5-9B", workers_per_gpu="auto"),
            ["4"],
        )
        # Mixed free VRAM: GPU 4 has 44.4G (2 slots), GPU 7 has 18.0G (1 slot), others <12.5G (0 slots)
        mixed_gpus = [
            {"id": "0", "free_gb": 3.5, "total_gb": 44.4},
            {"id": "4", "free_gb": 44.4, "total_gb": 44.4},
            {"id": "7", "free_gb": 18.0, "total_gb": 44.4},
        ]
        self.assertEqual(
            compute_slots_from_free_mem(mixed_gpus, "Qwen/Qwen3.5-4B", workers_per_gpu="auto"),
            ["4", "7", "4"],
        )

        # Verify interrupted deltanet_blend shard files are harvested cleanly into canonical cache
        with tempfile.TemporaryDirectory() as td:
            model_dir = pathlib.Path(td)
            shard_dir = model_dir / "shards"
            shard_dir.mkdir()
            _write_jsonl(
                shard_dir / "deltanet_blend_shard_1_of_8.jsonl",
                [
                    {
                        "uid": "ifeval_100",
                        "alpha": 0.25,
                        "blend_conv_state": False,
                        "evict_think_kv": False,
                        "n_blended_layers": 24,
                        "a_final": "ok",
                        "grade": {"z": 1},
                    }
                ],
            )
            harvested = _harvest_partial_blend_shards(
                model_dir / "ext_deltanet_blend_cache.jsonl", shard_dir
            )
            self.assertEqual(len(harvested), 1)
            self.assertTrue((model_dir / "ext_deltanet_blend_cache.jsonl").is_file())
            self.assertFalse(shard_dir.exists())

    def test_v0513_phase7_regrade_k32_parity_tables_and_resilience(self) -> None:
        from unittest import mock

        from scripts import analyze_and_plot as ap
        from scripts import ext_deltanet_blend as edb
        from scripts import ext_prefill_mlp as epm
        from scripts import run_sharded as rs
        from scripts import sync_results as sr
        from scripts.versioning import is_result_only_path

        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            tmp_data = tmp / "data"
            tmp_out = tmp / "outputs"
            tmp_figs = tmp / "figures"
            tmp_tabs = tmp / "tables"
            tmp_data.mkdir()
            _build_mini_frozen_dir(tmp_data)

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_matrix.py"),
                    "--model",
                    "Qwen/Qwen3.5-4B",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(tmp_out),
                    "--strict-diag",
                    "--mock",
                ],
                check=True,
                cwd=str(ROOT),
            )
            q4_dir = tmp_out / "Qwen__Qwen3.5-4B"

            # 1. Verify K=32 matches M2 identically and Phase 7 re-grades legacy GSM8K rows in memory
            eval_p = q4_dir / "eval_450_cache.jsonl"
            eval_rows = _read_jsonl(eval_p)
            orig_summary = json.loads((q4_dir / "matrix_summary_strict341.json").read_text(encoding="utf-8"))
            clean_bundle = ap.load_model_bundle(q4_dir)
            self.assertIsNotNone(clean_bundle)
            clean_analysis = ap.analyze_model_bundle(clean_bundle, n_boot=50, seed=42)
            k32 = clean_analysis["probe_horizon_ablation"]["K_32"]
            m2_sys = clean_bundle["report"]["systems"]["M2"]
            self.assertAlmostEqual(k32["macro_acc"], orig_summary["systems"]["M2"]["macro_acc"], places=9)
            self.assertAlmostEqual(k32["oaa_250"], m2_sys["oaa_250"], places=9)
            self.assertAlmostEqual(k32["u_005"], m2_sys["u_005"], places=9)
            self.assertAlmostEqual(k32["mean_t_tot"], m2_sys["mean_t_tot"], places=9)

            gsm_idx = next(i for i, r in enumerate(eval_rows) if r["item"].get("dataset") == "gsm8k")
            gsm_ans = str(int(float(eval_rows[gsm_idx]["item"]["gold"])))
            orig_b0_z = int(eval_rows[gsm_idx]["b0"]["z"])
            eval_rows[gsm_idx]["b0"]["a_final"] = f"Conclusion: the total is **{gsm_ans}**."
            eval_rows[gsm_idx]["b0"]["z"] = 0
            eval_rows[gsm_idx]["b0"]["pred"] = -999999.0
            _write_jsonl(eval_p, eval_rows)

            regraded_bundle = ap.load_model_bundle(q4_dir)
            self.assertIsNotNone(regraded_bundle)
            self.assertEqual(int(regraded_bundle["per_system_runs"]["B0"][gsm_idx]["z"]), 1)
            if orig_b0_z == 0:
                self.assertGreater(
                    regraded_bundle["report"]["systems"]["B0"]["macro_acc"],
                    orig_summary["systems"]["B0"]["macro_acc"],
                )
            # Restore clean mock cache before checking W7 / tables
            eval_rows[gsm_idx]["b0"]["z"] = 1
            _write_jsonl(eval_p, eval_rows)

            # 2. Verify ext_prefill_mlp --check-cached & modular Table 1-4 generation with W7 data
            chk_fail = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "ext_prefill_mlp.py"),
                    "--model",
                    "Qwen/Qwen3.5-4B",
                    "--out-dir",
                    str(tmp_out),
                    "--mock",
                    "--check-cached",
                ],
                cwd=str(ROOT),
            )
            self.assertEqual(chk_fail.returncode, 1)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "ext_prefill_mlp.py"),
                    "--model",
                    "Qwen/Qwen3.5-4B",
                    "--out-dir",
                    str(tmp_out),
                    "--mock",
                ],
                check=True,
                cwd=str(ROOT),
            )
            chk_ok = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "ext_prefill_mlp.py"),
                    "--model",
                    "Qwen/Qwen3.5-4B",
                    "--out-dir",
                    str(tmp_out),
                    "--mock",
                    "--check-cached",
                ],
                cwd=str(ROOT),
            )
            self.assertEqual(chk_ok.returncode, 0)
            self.assertTrue(epm._cache_covers_keys(q4_dir / "prefill_hidden_mock.npz", ["eval::" + eval_rows[0]["item"]["uid"]]))

            # Seed realistic DeltaNetBlend report + cache so analyze_and_plot and sync_results cover Table 4 & W7 cache
            diag_p = q4_dir / "diag_ehurt_cache.jsonl"
            d_rows = _read_jsonl(diag_p)
            for idx_d, dr in enumerate(d_rows[:4]):
                dr["b0"]["z"] = 1
                dr["b1b"]["z"] = 0
                dr["b1b"]["t_think"] = f"trace {idx_d}"
            _write_jsonl(diag_p, d_rows)
            _write_jsonl(
                q4_dir / "ext_deltanet_blend_cache.jsonl",
                [
                    {
                        "uid": str(dr["item"]["uid"]),
                        "alpha": 0.5,
                        "blend_conv_state": False,
                        "evict_think_kv": False,
                        "n_blended_layers": 24,
                        "n_blended_conv": 0,
                        "n_evicted_kv": 0,
                        "a_final": "ok",
                        "grade": {"z": 1, "inhibition_strict_all": 1, "structural_strict_all": 1},
                    }
                    for dr in d_rows[:4]
                ],
            )
            (q4_dir / "ext_deltanet_blend.json").write_text(
                json.dumps(
                    {
                        "model": "Qwen/Qwen3.5-4B",
                        "n_ehurt": 4,
                        "blend_conv_state": False,
                        "evict_think_kv": False,
                        "alphas": {
                            "0.50": {
                                "alpha": 0.5,
                                "n_ehurt": 4,
                                "overall_recovery_pct": 100.0,
                                "inhibition_recovery_pct": 100.0,
                                "structural_recovery_pct": 100.0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "analyze_and_plot.py"),
                    "--out-dir",
                    str(tmp_out),
                    "--fig-dir",
                    str(tmp_figs),
                    "--table-dir",
                    str(tmp_tabs),
                    "--n-boot",
                    "50",
                ],
                check=True,
                cwd=str(ROOT),
            )
            for tname, label_tok in (
                ("tab1_main_results.tex", r"\label{tab:main_results}"),
                ("tab2_scale_pathology.tex", r"\label{tab:scale_and_pathology}"),
                ("tab3_diagnostic_control.tex", r"\label{tab:diagnostic_control}"),
                ("tab4_extensions_w7.tex", r"\label{tab:extensions_w7}"),
            ):
                tp = tmp_tabs / tname
                self.assertTrue(tp.is_file(), f"Missing generated table {tname}")
                txt = tp.read_text(encoding="utf-8")
                self.assertIn(label_tok, txt)
            self.assertIn("DeltaNet Blend", (tmp_tabs / "tab4_extensions_w7.tex").read_text(encoding="utf-8"))
            self.assertIn("IFEval++", (tmp_tabs / "tab4_extensions_w7.tex").read_text(encoding="utf-8"))

            # 3. Verify run_sharded cross-shard-count harvest sanitizes truncated JSONL tails before merging
            s8_dir = q4_dir / "shards" / "shard_0_of_8"
            s8_dir.mkdir(parents=True, exist_ok=True)
            seed_first = _read_jsonl(q4_dir / "seed_60_cache.jsonl")[0]
            (s8_dir / "seed_60_cache.jsonl").write_text(
                json.dumps(seed_first) + "\n" + '{"truncated_incomplete": ',
                encoding="utf-8",
            )
            harvested = rs._merge_and_prune_matrix_shards(tmp_data, q4_dir, strict_diag=True)
            self.assertIn("seed_60_cache.jsonl", harvested)
            self.assertFalse(s8_dir.exists())

            # 4. Verify ext_deltanet_blend non-recursion (--worker on 1 non-zero GPU), balanced uncached slicing & retry
            (q4_dir / "ext_deltanet_blend.json").unlink()
            # Leave 2 items cached (indices 1, 3) and 2 uncached (indices 0, 2) via pre-existing shard_1_of_2
            _write_jsonl(
                q4_dir / "ext_deltanet_blend_cache.jsonl",
                [],
            )
            (q4_dir / "shards").mkdir(parents=True, exist_ok=True)
            _write_jsonl(
                q4_dir / "shards" / "deltanet_blend_shard_1_of_2.jsonl",
                [
                    {
                        "uid": str(dr["item"]["uid"]),
                        "alpha": 0.5,
                        "blend_conv_state": False,
                        "evict_think_kv": False,
                        "n_blended_layers": 24,
                        "a_final": "ok",
                        "grade": {"z": 1, "inhibition_strict_all": 1, "structural_strict_all": 1},
                    }
                    for dr in (d_rows[1], d_rows[3])
                ],
            )
            spawn_cmds: list[list[str]] = []

            class _FakePopen:
                def __init__(self, cmd: list[str], env: dict[str, str] | None = None, **_kwargs) -> None:
                    spawn_cmds.append(cmd)
                    self.stdout = None
                    self._rc = 1 if len(spawn_cmds) == 1 else 0
                    s_idx = int(cmd[cmd.index("--shard-idx") + 1])
                    n_sh = int(cmd[cmd.index("--num-shards") + 1])
                    cur_cache = edb._load_blend_cache(q4_dir / "ext_deltanet_blend_cache.jsonl")
                    rem = edb._filter_uncached_ehurt(d_rows[:4], cur_cache, [0.5], False, False)
                    assigned = rem[s_idx::n_sh]
                    sp = q4_dir / "shards" / f"deltanet_blend_shard_{s_idx}_of_{n_sh}.jsonl"
                    sp.parent.mkdir(parents=True, exist_ok=True)
                    # Attempt 1 writes 1 row then fails; Attempt 2 finishes all remaining uncached rows
                    to_write = assigned[:1] if self._rc != 0 else assigned
                    _write_jsonl(
                        sp,
                        [
                            {
                                "uid": str(dr["item"]["uid"]),
                                "alpha": 0.5,
                                "blend_conv_state": False,
                                "evict_think_kv": False,
                                "n_blended_layers": 24,
                                "a_final": "ok",
                                "grade": {"z": 1, "inhibition_strict_all": 1, "structural_strict_all": 1},
                            }
                            for dr in to_write
                        ],
                    )

                def poll(self) -> int:
                    return self._rc

                def terminate(self) -> None:
                    pass

                def wait(self, timeout: float | None = None) -> int:
                    return self._rc

                def kill(self) -> None:
                    pass

            with (
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch.object(edb, "_detect_auto_slots", return_value=["4"]),
                mock.patch("subprocess.Popen", side_effect=_FakePopen),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "ext_deltanet_blend.py",
                        "--model",
                        "Qwen/Qwen3.5-4B",
                        "--out-dir",
                        str(tmp_out),
                        "--alphas",
                        "0.5",
                        "--max-retries",
                        "1",
                        "--retry-wait-sec",
                        "0",
                    ],
                ),
            ):
                edb.main()
            # Exactly 2 spawns: attempt 1 (failed after 1 row, passing --worker with eff_shards=1 on GPU 4), attempt 2 (succeeded)
            self.assertEqual(len(spawn_cmds), 2)
            self.assertIn("--worker", spawn_cmds[0])
            self.assertIn("--worker", spawn_cmds[1])
            bl_out = json.loads((q4_dir / "ext_deltanet_blend.json").read_text(encoding="utf-8"))
            self.assertEqual(bl_out["n_ehurt"], 4)
            self.assertEqual(bl_out["alphas"]["0.50"]["overall_recovery_pct"], 100.0)

            # 5. Verify sync_results packs/unpacks diag_controls_cache & ext_deltanet_blend_cache + paper/tables versioning exemption
            _write_jsonl(
                q4_dir / "diag_controls_cache.jsonl",
                [{"uid": str(d_rows[0]["item"]["uid"]), "mode": "A1_strip", "a_final": "ok", "grade": {"z": 1}}],
            )
            self.assertTrue(is_result_only_path("paper/tables/tab1_main_results.tex"))
            fake_repo = tmp / "fake_repo"
            (fake_repo / "paper" / "tables").mkdir(parents=True)
            (fake_repo / "paper" / "figures").mkdir(parents=True)
            (fake_repo / "VERSION").write_text("0.5.13\n", encoding="utf-8")
            (fake_repo / "EXPERIMENT_LOG.md").write_text("# Log\n", encoding="utf-8")
            (fake_repo / "paper" / "tables" / "tab1_main_results.tex").write_text(
                (tmp_tabs / "tab1_main_results.tex").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            staged = sr.pack_outputs_for_repo(out_dir=tmp_out, repo_root=fake_repo, include_phase7=True)
            staged_strs = {str(p) for p in staged}
            self.assertIn("outputs/Qwen__Qwen3.5-4B/diag_controls_cache.jsonl.gz", staged_strs)
            self.assertIn("outputs/Qwen__Qwen3.5-4B/ext_deltanet_blend_cache.jsonl.gz", staged_strs)
            self.assertIn("paper/tables/tab1_main_results.tex", staged_strs)

            unpacked_out = tmp / "unpacked_out"
            counts = sr.unpack_repo_caches(fake_repo, unpacked_out)
            self.assertGreaterEqual(counts["caches_unpacked"], 5)
            self.assertTrue((unpacked_out / "Qwen__Qwen3.5-4B" / "diag_controls_cache.jsonl").is_file())
            self.assertTrue((unpacked_out / "Qwen__Qwen3.5-4B" / "ext_deltanet_blend_cache.jsonl").is_file())

    def test_v0514_logging_dedup_dirty_guard_and_blend_logs(self) -> None:
        import copy
        import os
        from typing import Any
        from unittest import mock

        from scripts import ext_deltanet_blend as edb
        from scripts.log_experiment import (
            _collect_training_params,
            _detect_git_rev,
            _has_post_smoke_stage,
            _render_run_markdown,
        )

        # 1. Code-only git rev ignores EXPERIMENT_LOG.md & outputs/* modifications
        with tempfile.TemporaryDirectory() as td:
            repo_dir = pathlib.Path(td) / "repo"
            repo_dir.mkdir()
            subprocess.run(["git", "-C", str(repo_dir), "init"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(repo_dir), "config", "user.name", "t"], check=True)
            subprocess.run(["git", "-C", str(repo_dir), "config", "user.email", "t@example.com"], check=True)
            (repo_dir / "scripts").mkdir()
            (repo_dir / "outputs" / "Qwen__Qwen3.5-4B").mkdir(parents=True)
            (repo_dir / "VERSION").write_text("0.5.14\n", encoding="utf-8")
            (repo_dir / "scripts" / "x.py").write_text("x = 1\n", encoding="utf-8")
            (repo_dir / "EXPERIMENT_LOG.md").write_text("# Log\n", encoding="utf-8")
            (repo_dir / "outputs" / "Qwen__Qwen3.5-4B" / "matrix_summary.json").write_text("{}\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo_dir), "commit", "-m", "init"], check=True, capture_output=True)

            # Mutating outputs & EXPERIMENT_LOG.md must NOT mark _detect_git_rev as -dirty
            (repo_dir / "EXPERIMENT_LOG.md").write_text("# Updated\n", encoding="utf-8")
            (repo_dir / "outputs" / "Qwen__Qwen3.5-4B" / "matrix_summary.json").write_text('{"a": 1}\n', encoding="utf-8")
            rev_clean = _detect_git_rev(repo_dir)
            self.assertFalse(rev_clean.endswith("-dirty"), rev_clean)

            # Mutating tracked code under scripts/ MUST mark _detect_git_rev as -dirty
            (repo_dir / "scripts" / "x.py").write_text("x = 2\n", encoding="utf-8")
            rev_dirty = _detect_git_rev(repo_dir)
            self.assertTrue(rev_dirty.endswith("-dirty"), rev_dirty)

        # 2. _collect_training_params ignores stale run provenance & dedup omits identical diag491 Eval-450 table
        collected = _collect_training_params(
            {"seed": 42},
            [{"training_params": {"seed": 42, "git_rev": "91efc2f", "run_id": "", "code_version": "0.5.8", "dtype": "bfloat16"}}],
            [],
        )
        self.assertNotIn("git_rev", collected)
        self.assertNotIn("run_id", collected)
        self.assertNotIn("code_version", collected)
        self.assertEqual(collected["dtype"], "bfloat16")
        self.assertFalse(_has_post_smoke_stage([{"stage": "Gate 0: Preflight Tests", "status": "PASS", "duration_sec": 1.0}], "Step 0a: Live GPU 12-Item Smoke"))
        self.assertTrue(_has_post_smoke_stage([{"stage": "Phase 4 / W4: Qwen/Qwen3.5-4B (strict341)", "status": "PASS", "duration_sec": 10.0}], "W7: DeltaNet Blend"))

        def _fake_sys(macro: float) -> dict[str, object]:
            return {
                "slice_acc": {"gsm8k_easy": macro, "gsm8k_hard": macro, "csqa": macro, "ifeval_planning": macro, "ifeval_precision": macro, "ifeval_conflict": macro},
                "macro_acc": macro,
                "oaa_250": macro,
                "auc_oaa_2048": macro,
                "u_005": macro - 0.05,
                "mean_t_tot": 300.0,
                "mean_calls": 1.0,
                "p_holm_vs_b1b": 1e-10,
            }

        identical_sys = {"B0": _fake_sys(0.84), "B1b": _fake_sys(0.30), "M2": _fake_sys(0.85)}
        rep_s = {
            "model": "Qwen/Qwen3.5-4B",
            "diag_pool": "strict341",
            "diag_n_items": 341,
            "runtime": {"dtype": "torch.bfloat16"},
            "timing_sec": {"total_wall": 10.0, "calibration": 0.1, "eval_matrix": 0.3, "section6_diagnostics": 0.1},
            "calibrated_params": {"tau_prob": 0.65, "m2_escalation_rate_dev": 0.2},
            "oracle_and_4regime": {"helped_rate": 0.01, "hurt_rate": 0.5, "flip_rate": 0.51, "tie_pass_rate": 0.35, "tie_fail_rate": 0.14, "oracle_summary": _fake_sys(0.92)},
            "section6_diagnostics": {"causal_attribution": {"rho_recency": 0.96, "rho_interference": 0.04}, "A1_strip": {"n_ehurt": 237, "overall_recovery_pct": 100.0}},
            "systems": identical_sys,
            "auxiliary_systems": {},
        }
        rep_d = copy.deepcopy(rep_s)
        rep_d["diag_pool"] = "diag491"
        rep_d["diag_n_items"] = 491
        rep_d["section6_diagnostics"]["A1_strip"]["n_ehurt"] = 336
        hw = {"gpu_count": 0, "gpu_name": "CPU", "cuda_available": False, "torch": "2.x", "transformers": "5.x", "fla_available": False, "causal_conv1d_available": False}
        md_dedup = _render_run_markdown(
            run_id="dedup_test",
            status="RUNNING (after diag491)",
            loop_name="Phase 4",
            ts_utc="2026-11-17 00:00:00 UTC",
            git_rev="abc1234",
            total_sec=10.0,
            hw=hw,
            command_clean="test",
            rel_log="",
            notes_clean="",
            training_params={},
            stage_timings=[],
            smoke_rows=[],
            calibrations=[],
            matrix_loops=[rep_s, rep_d],
            w7_items=[],
            model_rows=[],
            code_version="0.5.14",
        )
        self.assertEqual(md_dedup.count("| Variant | Macro |"), 1)
        self.assertIn("Eval-450 table identical to `strict341` above (omitted duplicate table).", md_dedup)

        # 3. ext_deltanet_blend streams worker logs into per-worker log files & surfaces failure tail
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            tmp_out = tmp / "outputs"
            tmp_logs = tmp / "logs"
            m_dir = tmp_out / "Qwen__Qwen3.5-4B"
            m_dir.mkdir(parents=True)
            _write_jsonl(
                m_dir / "diag_ehurt_cache.jsonl",
                [{"item": {"uid": "eh_0", "prompt": "p"}, "b0": {"z": 1}, "b1b": {"z": 0, "t_think": "think"}}],
            )
            dummy_blend_worker = tmp / "dummy_blend_worker.py"
            dummy_blend_worker.write_text(
                "import sys\n"
                "print('[deltanet_blend shard=0/1] Model=Qwen/Qwen3.5-4B | items=1 remaining (1 total)', flush=True)\n"
                "print('Simulated blend OOM on physical GPU 3', file=sys.stderr, flush=True)\n"
                "sys.exit(2)\n",
                encoding="utf-8",
            )
            orig_popen = subprocess.Popen

            def _piped_dummy_popen(cmd: list[str], **kwargs) -> subprocess.Popen[Any]:
                remapped = [sys.executable, str(dummy_blend_worker)]
                return orig_popen(remapped, **kwargs)

            with (
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch.object(edb, "_detect_auto_slots", return_value=["3"]),
                mock.patch("subprocess.Popen", side_effect=_piped_dummy_popen),
                mock.patch.dict(os.environ, {"CBARR_LOG_DIR": str(tmp_logs), "CBARR_RUN_ID": "blend_log_test"}),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "ext_deltanet_blend.py",
                        "--model",
                        "Qwen/Qwen3.5-4B",
                        "--out-dir",
                        str(tmp_out),
                        "--alphas",
                        "0.5",
                        "--max-retries",
                        "0",
                    ],
                ),
            ):
                with self.assertRaises(RuntimeError):
                    edb.main()
            w_log_p = tmp_logs / "deltanet_blend_blend_log_test_a1_gpu0_Qwen__Qwen3.5-4B.log"
            self.assertTrue(w_log_p.is_file())
            w_log_txt = w_log_p.read_text(encoding="utf-8")
            self.assertIn("dev=3", w_log_txt)
            self.assertIn("Simulated blend OOM on physical GPU 3", w_log_txt)

        # 4. Early Step-0a FAILED run omits stale calibration/matrix/progress from prior runs in out_dir
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            tmp_out = tmp / "outputs"
            m_dir = tmp_out / "Qwen__Qwen3.5-4B"
            m_dir.mkdir(parents=True)
            (m_dir / "calibration_summary.json").write_text(
                json.dumps(
                    {
                        "model": "Qwen/Qwen3.5-4B",
                        "calibrated_params": {"tau_prob": 0.65, "tau_h_high": 2.5, "m2_escalation_rate_dev": 0.2, "coef": [0.1] * 5, "intercept": 0.0},
                        "training_params": {"git_rev": "old999", "run_id": "old_run", "seed": 42},
                    }
                ),
                encoding="utf-8",
            )
            (m_dir / "progress_checkpoint.json").write_text(
                json.dumps({"model": "Qwen/Qwen3.5-4B", "diag_pool": "strict341", "split": "eval_450", "done": 10, "total": 450, "metrics": "x"}),
                encoding="utf-8",
            )
            md_early = tmp / "EXP.md"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "log_experiment.py"),
                    "--run-id",
                    "early_fail_run",
                    "--status",
                    "FAILED (exit=1 at Step 0a: Live GPU 12-Item Smoke)",
                    "--loop-name",
                    "Step 0a: Live GPU 12-Item Smoke",
                    "--command",
                    "test",
                    "--out-dir",
                    str(tmp_out),
                    "--md-log",
                    str(md_early),
                    "--notes",
                    "torch.OutOfMemoryError: mock",
                ],
                check=True,
                cwd=str(ROOT),
            )
            early_txt = md_early.read_text(encoding="utf-8")
            self.assertNotIn("Calibration & Gate-2 Classifier Results", early_txt)
            self.assertNotIn("Live Step Checkpoints", early_txt)
            self.assertIn('"run_id": "early_fail_run"', early_txt)
            self.assertNotIn("old999", early_txt)
            self.assertIn("torch.OutOfMemoryError: mock", early_txt)

        # 5. Bash syntax audit passes on updated runner script
        subprocess.run(["bash", "-n", str(ROOT / "scripts" / "run_gpu_phases_4_to_6.sh")], check=True)

    def test_v0515_gpu_pool_adaptive_bandwidth_util_and_cpu_bounds(self) -> None:
        from types import SimpleNamespace
        from unittest import mock

        from scripts.gpu_pool import (
            compute_slots_from_free_mem,
            model_vram_per_worker_gb,
            query_physical_gpus,
        )

        # Stage-aware VRAM budget adds blend dual-cache overhead cleanly
        self.assertEqual(model_vram_per_worker_gb("Qwen/Qwen3.5-4B", stage="matrix"), 12.5)
        self.assertGreater(model_vram_per_worker_gb("Qwen/Qwen3.5-4B", stage="blend"), 12.5)

        l40_pair = [
            {"id": "3", "free_gb": 44.4, "total_gb": 44.4, "util_pct": 0.0},
            {"id": "4", "free_gb": 44.4, "total_gb": 44.4, "util_pct": 0.0},
        ]
        # Model-bandwidth cap scales small models up (0.8B->4/GPU, 2B->3/GPU) while keeping 4B=2, 9B=1
        self.assertEqual(
            compute_slots_from_free_mem(l40_pair, "Qwen/Qwen3.5-0.8B", cpu_count=64),
            ["3", "4"] * 4,
        )
        self.assertEqual(
            compute_slots_from_free_mem(l40_pair, "Qwen/Qwen3.5-2B", cpu_count=64),
            ["3", "4"] * 3,
        )
        self.assertEqual(
            compute_slots_from_free_mem(l40_pair, "Qwen/Qwen3.5-4B", cpu_count=64),
            ["3", "4"] * 2,
        )
        self.assertEqual(
            compute_slots_from_free_mem(l40_pair, "Qwen/Qwen3.5-9B", cpu_count=64),
            ["3", "4"],
        )

        # Blend stage tightens partially loaded 4B cards from 2 slots to 1 when free_gb=28.0G
        half_busy = [{"id": "3", "free_gb": 28.0, "total_gb": 44.4, "util_pct": 0.0}]
        self.assertEqual(compute_slots_from_free_mem(half_busy, "Qwen/Qwen3.5-4B", stage="matrix"), ["3", "3"])
        self.assertEqual(compute_slots_from_free_mem(half_busy, "Qwen/Qwen3.5-4B", stage="blend"), ["3"])

        # Shared GPU already busy at >=60% compute utilization gets capped to 1 slot even for 0.8B
        busy_gpus = [
            {"id": "3", "free_gb": 44.4, "total_gb": 44.4, "util_pct": 75.0},
            {"id": "4", "free_gb": 44.4, "total_gb": 44.4, "util_pct": 0.0},
        ]
        self.assertEqual(
            compute_slots_from_free_mem(busy_gpus, "Qwen/Qwen3.5-0.8B", cpu_count=64),
            ["4", "3", "4", "4", "4"],
        )

        # Low host CPU count prevents thrashing (e.g., 4 CPUs with 2 CPUs/worker => max 2 total workers)
        self.assertEqual(
            compute_slots_from_free_mem(l40_pair, "Qwen/Qwen3.5-0.8B", cpu_count=4, cpus_per_worker=2),
            ["3", "4"],
        )

        # query_physical_gpus parses utilization.gpu cleanly and handles [N/A]
        fake_smi = SimpleNamespace(
            returncode=0,
            stdout="3, 45460, 46068, 12\n4, 28000, 46068, [N/A]\n",
        )
        with mock.patch("subprocess.run", return_value=fake_smi):
            parsed = query_physical_gpus()
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["util_pct"], 12.0)
        self.assertIsNone(parsed[1]["util_pct"])

    def test_v0516_stage_oom_hardening_and_all_left_phases(self) -> None:
        from types import SimpleNamespace
        from unittest import mock
        import os

        from scripts.ext_deltanet_blend import _detect_auto_slots, _supports_logits_to_keep
        from scripts.gpu_pool import (
            compute_slots_from_free_mem,
            host_ram_per_worker_gb,
            model_vram_per_worker_gb,
            query_host_avail_ram_gb,
        )

        # 1. Stage-aware VRAM differentiation: mlp < replay < b512 < calibrate < matrix == diag < smoke < blend
        v_mlp = model_vram_per_worker_gb("Qwen/Qwen3.5-4B", stage="mlp")
        v_rep = model_vram_per_worker_gb("Qwen/Qwen3.5-4B", stage="replay")
        v_b512 = model_vram_per_worker_gb("Qwen/Qwen3.5-4B", stage="b512")
        v_cal = model_vram_per_worker_gb("Qwen/Qwen3.5-4B", stage="calibrate")
        v_mat = model_vram_per_worker_gb("Qwen/Qwen3.5-4B", stage="matrix")
        v_diag = model_vram_per_worker_gb("Qwen/Qwen3.5-4B", stage="diag")
        v_smk = model_vram_per_worker_gb("Qwen/Qwen3.5-4B", stage="smoke")
        v_bld = model_vram_per_worker_gb("Qwen/Qwen3.5-4B", stage="blend")
        self.assertLess(v_mlp, v_rep)
        self.assertLess(v_rep, v_b512)
        self.assertLess(v_b512, v_cal)
        self.assertLess(v_cal, v_mat)
        self.assertEqual(v_mat, v_diag)
        self.assertGreater(v_smk, v_mat)
        self.assertGreater(v_bld, v_smk)

        # 2. Mandatory reserve blocks near-full GPUs (9B requires 24 + 2 = 26G; 24.5G rejected)
        tight_9b = [{"id": "3", "free_gb": 24.5, "total_gb": 44.4, "util_pct": 0.0}]
        self.assertEqual(compute_slots_from_free_mem(tight_9b, "Qwen/Qwen3.5-9B", stage="matrix"), [])
        self.assertEqual(compute_slots_from_free_mem(tight_9b, "Qwen/Qwen3.5-9B", stage="mlp"), ["3"])

        # Per-extra-worker CUDA context cushion prevents packing 2 workers when headroom is marginal
        marginal_4b = [{"id": "3", "free_gb": 27.2, "total_gb": 44.4, "util_pct": 0.0}]
        self.assertEqual(compute_slots_from_free_mem(marginal_4b, "Qwen/Qwen3.5-4B", stage="matrix"), ["3"])

        # Retry safety_mult backoff shrinks workers/GPU on subsequent attempts
        loose_4b = [{"id": "3", "free_gb": 29.0, "total_gb": 44.4, "util_pct": 0.0}]
        self.assertEqual(compute_slots_from_free_mem(loose_4b, "Qwen/Qwen3.5-4B", safety_mult=1.0), ["3", "3"])
        self.assertEqual(compute_slots_from_free_mem(loose_4b, "Qwen/Qwen3.5-4B", safety_mult=1.25), ["3"])

        with mock.patch("torch.cuda.is_available", return_value=True):
            with mock.patch("scripts.ext_deltanet_blend.select_worker_gpu_slots", return_value=["3"]) as sw:
                _detect_auto_slots("Qwen/Qwen3.5-4B", 50, attempt=2)
                self.assertEqual(sw.call_args.kwargs["workers_per_gpu"], "1")
                self.assertAlmostEqual(sw.call_args.kwargs["safety_mult"], 1.25)

        class _WithLogitsKeep:
            def forward(self, input_ids, logits_to_keep=0):
                return input_ids

        class _NoLogitsKeep:
            def forward(self, input_ids):
                return input_ids

        self.assertTrue(_supports_logits_to_keep(_WithLogitsKeep()))
        self.assertFalse(_supports_logits_to_keep(_NoLogitsKeep()))

        # 3. Host CPU RAM clamp trims concurrent weight-loading workers before SIGKILL (137)
        l40_quad = [
            {"id": str(i), "free_gb": 44.4, "total_gb": 44.4, "util_pct": 0.0}
            for i in (3, 4, 6, 7)
        ]
        host_req_4b = host_ram_per_worker_gb("Qwen/Qwen3.5-4B", stage="matrix")
        host_req_9b = host_ram_per_worker_gb("Qwen/Qwen3.5-9B", stage="matrix")
        self.assertGreaterEqual(host_req_4b, 10.0)
        self.assertGreaterEqual(host_req_9b, 20.0)
        tight_host = compute_slots_from_free_mem(
            l40_quad,
            "Qwen/Qwen3.5-4B",
            stage="matrix",
            cpu_count=64,
            host_avail_gb=20.0,
            host_reserve_gb=4.0,
        )
        self.assertEqual(len(tight_host), 1)
        tight_host_9b = compute_slots_from_free_mem(
            l40_quad,
            "Qwen/Qwen3.5-9B",
            stage="matrix",
            cpu_count=64,
            host_avail_gb=50.0,
            host_reserve_gb=4.0,
        )
        self.assertEqual(len(tight_host_9b), 2)

        fake_vm = SimpleNamespace(
            returncode=0,
            stdout=(
                "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                "Pages free:                             1048576.\n"
                "Pages inactive:                          524288.\n"
                "Pages speculative:                            0.\n"
            ),
        )
        with mock.patch("pathlib.Path.is_file", return_value=False):
            with mock.patch("subprocess.run", return_value=fake_vm):
                avail = query_host_avail_ram_gb()
        self.assertIsNotNone(avail)
        self.assertAlmostEqual(float(avail), 24.0, places=1)

        # 4. End-to-end --part all --with-r6 --mock runs all left phases: W7 conv+kv, PrefillMLP,
        # compare_m2_v2 replay, R6 sampled validation, Phase 7 stats, Tab 4, and Fig 1-4.
        with tempfile.TemporaryDirectory() as td:
            root_tmp = pathlib.Path(td)
            tmp_data = root_tmp / "data"
            tmp_out = root_tmp / "out"
            tmp_figs = root_tmp / "figs"
            tmp_tables = root_tmp / "tables"
            tmp_data.mkdir()
            _build_mini_frozen_dir(tmp_data)

            proc = subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts" / "run_gpu_phases_4_to_6.sh"),
                    "--mock",
                    "--part",
                    "all",
                    "--no-setup",
                    "--skip-preflight",
                    "--skip-smoke",
                    "--with-w7",
                    "--with-r6",
                    "--r6-models",
                    "Qwen/Qwen3.5-0.8B",
                    "--models",
                    "Qwen/Qwen3.5-0.8B",
                    "--data-dir",
                    str(tmp_data),
                    "--out-dir",
                    str(tmp_out),
                    "--fig-dir",
                    str(tmp_figs),
                    "--table-dir",
                    str(tmp_tables),
                    "--exp-log",
                    str(tmp_out / "exp_log.md"),
                    "--run-id",
                    "all_left_mock",
                ],
                cwd=str(ROOT),
                env={**os.environ, "PYTHON_BIN": sys.executable, "CBARR_COLOR": "0"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, f"Full all-phases mock failed:\n{proc.stdout}\n{proc.stderr}")

            q08_dir = tmp_out / "Qwen__Qwen3.5-0.8B"
            self.assertTrue((q08_dir / "ext_deltanet_blend.json").is_file())
            self.assertTrue((q08_dir / "ext_deltanet_blend_conv_kv.json").is_file())
            self.assertTrue((q08_dir / "prefill_mlp_summary.json").is_file())
            self.assertTrue((q08_dir / "eval_450_m2v2_comparison.json").is_file())
            self.assertTrue(
                (tmp_out / "Qwen__Qwen3.5-0.8B__T0.6_tp0.95_pp1.5_s42" / "matrix_summary_strict341.json").is_file()
            )
            self.assertTrue((tmp_figs / "fig4_horizons_and_w7.pdf").read_bytes().startswith(b"%PDF-"))

            stats = json.loads((tmp_out / "stats_summary.json").read_text(encoding="utf-8"))
            self.assertIn("r6_sampled_validation", stats)
            self.assertIn("w7_extensions", stats)
            self.assertIn("m2v2_comparison", stats["w7_extensions"]["Qwen/Qwen3.5-0.8B"])
            tab4 = (tmp_tables / "tab4_extensions_w7.tex").read_text(encoding="utf-8")
            self.assertIn("DeltaNet Conv+KV", tab4)
            self.assertIn("Stage-2 Bridge", tab4)
            self.assertIn("R6 Sampled M2", tab4)

            # 5. Also verify live alpha-format conv+kv / dense evict_kv + R3 M2_B512 rendering in analyze_and_plot
            live_conv_kv = {
                "model": "Qwen/Qwen3.5-0.8B",
                "n_ehurt": 12,
                "blend_conv_state": True,
                "evict_think_kv": True,
                "alphas": {
                    "0.50": {
                        "alpha": 0.5,
                        "n_ehurt": 12,
                        "overall_recovery_pct": 91.7,
                        "inhibition_recovery_pct": 90.0,
                        "structural_recovery_pct": 93.0,
                    }
                },
            }
            (q08_dir / "ext_deltanet_blend_conv_kv.json").write_text(
                json.dumps(live_conv_kv, indent=2) + "\n", encoding="utf-8"
            )
            sm_dir = tmp_out / "HuggingFaceTB__SmolLM3-3B"
            sm_dir.mkdir(parents=True, exist_ok=True)
            sm_rep = json.loads((q08_dir / "matrix_summary_strict341.json").read_text(encoding="utf-8"))
            sm_rep["model"] = "HuggingFaceTB/SmolLM3-3B"
            sm_rep["auxiliary_systems"] = {
                **sm_rep.get("auxiliary_systems", {}),
                "M2_B512": {
                    **sm_rep["systems"]["M2"],
                    "macro_acc": min(1.0, float(sm_rep["systems"]["M2"]["macro_acc"]) + 0.05),
                    "mean_t_tot": float(sm_rep["systems"]["M2"]["mean_t_tot"]) + 60.0,
                },
            }
            (sm_dir / "matrix_summary_strict341.json").write_text(
                json.dumps(sm_rep, indent=2) + "\n", encoding="utf-8"
            )
            (sm_dir / "ext_deltanet_blend_evict_kv.json").write_text(
                json.dumps(
                    {
                        "model": "HuggingFaceTB/SmolLM3-3B",
                        "n_ehurt": 8,
                        "blend_conv_state": False,
                        "evict_think_kv": True,
                        "alphas": {
                            "0.00": {
                                "alpha": 0.0,
                                "n_ehurt": 8,
                                "overall_recovery_pct": 75.0,
                                "inhibition_recovery_pct": 75.0,
                                "structural_recovery_pct": 75.0,
                            }
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "analyze_and_plot.py"),
                    "--out-dir",
                    str(tmp_out),
                    "--fig-dir",
                    str(tmp_figs),
                    "--table-dir",
                    str(tmp_tables),
                    "--primary-model",
                    "HuggingFaceTB/SmolLM3-3B",
                    "--dense-model",
                    "HuggingFaceTB/SmolLM3-3B",
                    "--n-boot",
                    "100",
                ],
                check=True,
                cwd=str(ROOT),
            )
            tab4_live = (tmp_tables / "tab4_extensions_w7.tex").read_text(encoding="utf-8")
            self.assertIn("DeltaNet Conv+KV", tab4_live)
            self.assertIn("Dense KV Evict", tab4_live)
            self.assertIn("M2-512: C-BARR Aux.", (tmp_tables / "tab1_main_results.tex").read_text(encoding="utf-8"))
            self.assertIn(
                "R3_m2_trails_b1b_gsm8k_hard_gt_3pp",
                stats["models"]["Qwen/Qwen3.5-0.8B"]["gate_audit"]["contingencies"],
            )
            chk_v2 = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "compare_m2_v2.py"),
                    "--model",
                    "Qwen/Qwen3.5-0.8B",
                    "--out-dir",
                    str(tmp_out),
                    "--check-cached",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(chk_v2.returncode, 0, chk_v2.stderr)


if __name__ == "__main__":
    unittest.main()
