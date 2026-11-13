"""End-to-end CPU tests for Phases 4-7 automation, metrics, analysis, W7 extensions, and anonymity."""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_matrix import verify_manifest_integrity
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
            self.assertTrue((q4_dir / "calibration_summary.json").exists())
            self.assertFalse((q4_dir / "eval_450_cache.jsonl").exists())

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
            strict_path = q4_dir / "matrix_summary_strict341.json"
            self.assertTrue(strict_path.exists())
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

        # 7. Anonymity linter passes on real repo paper/
        subprocess.run(["bash", str(ROOT / "scripts/check_anonymity.sh")], check=True, cwd=str(ROOT))


if __name__ == "__main__":
    unittest.main()
