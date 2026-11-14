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
        """Generated Colab/Kaggle notebook is valid nbformat 4 JSON with pure AST-valid Python cells."""
        nb_path = ROOT / "notebooks" / "C_BARR_Colab_Kaggle.ipynb"
        self.assertTrue(nb_path.exists(), f"Missing notebook: {nb_path}")
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb.get("nbformat"), 4)
        cells = nb.get("cells", [])
        self.assertGreaterEqual(len(cells), 8)

        combined_code = []
        for idx, cell in enumerate(cells):
            src = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
            self.assertTrue(src.strip(), f"Empty cell at index {idx}")
            if cell.get("cell_type") == "code":
                for line in src.splitlines():
                    stripped = line.lstrip()
                    self.assertFalse(
                        stripped.startswith("!") or stripped.startswith("%"),
                        f"Cell {idx} uses shell magic instead of pure Python: {line}",
                    )
                ast.parse(src, filename=f"cell_{idx}.py")
                combined_code.append(src)

        all_src = "\n".join(combined_code)
        for token in (
            "dolong2110/C-BARR",
            "verify_manifest_integrity",
            "scripts/run_matrix.py",
            "scripts/analyze_and_plot.py",
            "scripts/log_experiment.py",
            "record_loop_checkpoint",
            "run_and_tee",
            "RUN_LOG_FILE",
            "matrix_step_flags",
            "--step-checkpoint-every",
            "MATRIX_PARAM_ARGS",
            "AUTO_PUSH_RESULTS_TO_GITHUB",
        ):
            self.assertIn(token, all_src)

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
        from scripts.log_experiment import _git_commit_and_push_ledger, _upsert_run_block

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


if __name__ == "__main__":
    unittest.main()
