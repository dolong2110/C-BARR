"""Pre-flight verification suite for C-BARR Phases 1-3 (W1-W3) and Patches D1-D16."""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
import numpy as np
from langdetect import DetectorFactory

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.freeze_datasets import (
    INHIBITION_IDS,
    KNOWN_UPSTREAM_DEFECT_KEYS,
    PLANNING_IDS,
    PRECISION_IDS,
    STRUCTURAL_IDS,
)
from src.engine import (
    CBARRDualModeEngine,
    DualModeEngine,
    P_TERM,
    _build_reanchor_block,
    run_gate1_lexical,
)
from src.graders import (
    grade_csqa,
    grade_gsm8k,
    grade_ifeval_item,
    has_4gram_repetition_loop,
    has_cot_spillover,
    split_think_and_answer,
)
from src.metrics import (
    SLICE_ORDER,
    compute_auc_oaa,
    compute_bsr_curve,
    compute_causal_decomposition,
    compute_conformal_risk_bounds,
    compute_oaa_t,
    compute_oracle_efficiency_audit,
    exact_and_midp_mcnemar,
    holm_bonferroni,
    paired_bootstrap_diff_ci,
)
from src.routers import CBARRCalibrator, select_tiered_budget
from scripts.run_smoke_test import MockDualModeEngine, build_smoke_12, verify_gate_3_on_engine


# Canonical sample prompts for all 25 Zhou et al. (2023) IFEval instruction types
CANONICAL_25_TEMPLATES: dict[str, str] = {
    # 10 Planning templates
    "length_constraints:number_words": "Write an essay of at least 300 words about renewable energy.",
    "length_constraints:number_sentences": "Explain photosynthesis in less than 5 sentences.",
    "length_constraints:number_paragraphs": "Write a story that contains 4 paragraphs.",
    "length_constraints:nth_paragraph_first_word": "Write 3 paragraphs. The 2nd paragraph must start with the word However.",
    "detectable_format:number_bullet_lists": "Provide exactly 5 bullet points summarizing the article.",
    "detectable_format:multiple_sections": "Organize your guide into multiple sections starting with Section X.",
    "detectable_format:number_highlighted_sections": "Include at least 3 highlighted sections using *highlighted* markdown.",
    "keywords:frequency": "In your response, the word 'quantum' should appear at least 3 times.",
    "keywords:letter_frequency": "Write a short paragraph where the letter 'e' should appear less than 4 times.",
    "combination:two_responses": "Give two different responses separated by ******.",
    # 5 Inhibition Precision templates
    "punctuation:no_comma": "Explain how a steam engine works without using any commas.",
    "keywords:forbidden_words": "Describe the ocean but do not include the forbidden words blue or water.",
    "change_case:english_lowercase": "Write your entire response in all lowercase letters with no capital letters.",
    "change_case:english_capital": "Write your answer in all capital letters only.",
    "language:response_language": "Your entire response should be in Spanish, no other language is allowed.",
    # 10 Structural Precision templates
    "detectable_format:json_format": "Format your entire output as a valid JSON object.",
    "detectable_format:constrained_response": "Answer with one of the following options: My answer is yes.",
    "detectable_format:title": "Include a title wrapped in double angular brackets, like <<My Title>>.",
    "detectable_content:postscript": "At the end of your letter, include a postscript starting with P.S.",
    "detectable_content:number_placeholders": "Include at least 3 placeholders represented by square brackets like [name].",
    "startend:end_checker": "Finish your entire response with this exact phrase: Any other questions?",
    "startend:quotation": "Wrap your entire response with double quotation marks.",
    "combination:repeat_prompt": "First, repeat the request word for word without change, then give your answer.",
    "keywords:existence": "Make sure to include the keywords 'climate' and 'policy' in your text.",
    "change_case:capital_word_frequency": "Include at least 5 words with all capital letters in your response.",
}


class TestCBARRPreflight(unittest.TestCase):
    """Verifies W1-W3 offline gates: IFEval template recall, graders, D12-D16 fixes, splits, and smoke harness."""

    def test_w1_detector_factory_seed_and_symbol_renaming(self) -> None:
        self.assertEqual(DetectorFactory.seed, 0)
        self.assertIs(DualModeEngine, CBARRDualModeEngine)

    def test_w1_25_of_25_canonical_ifeval_regex_templates(self) -> None:
        from instruction_following_eval import instructions_registry

        self.assertEqual(len(CANONICAL_25_TEMPLATES), 25)
        self.assertEqual(set(CANONICAL_25_TEMPLATES.keys()), PLANNING_IDS | PRECISION_IDS)
        self.assertEqual(set(instructions_registry.INSTRUCTION_DICT.keys()), PLANNING_IDS | PRECISION_IDS)

        # 1. Verify against official google-research instruction_following_eval build_description() templates
        for inst_id, cls in sorted(instructions_registry.INSTRUCTION_DICT.items()):
            inst = cls(inst_id)
            desc = (
                inst.build_description(prompt_to_repeat="Sample prompt")
                if inst_id == "combination:repeat_prompt"
                else inst.build_description()
            )
            g1_official = run_gate1_lexical(desc)
            if inst_id in PLANNING_IDS:
                self.assertEqual(
                    g1_official.phi_plan,
                    1,
                    f"Gate 1 missed official planning template {inst_id}: {desc!r}",
                )
            if inst_id in PRECISION_IDS:
                self.assertEqual(
                    g1_official.phi_prec,
                    1,
                    f"Gate 1 missed official precision template {inst_id}: {desc!r}",
                )
            self.assertTrue(len(g1_official.anchor_clause) > 0)

        # 2. Verify against C-BARR canonical sample prompts
        for inst_id, sample_prompt in CANONICAL_25_TEMPLATES.items():
            g1 = run_gate1_lexical(sample_prompt)
            if inst_id in PLANNING_IDS:
                self.assertEqual(
                    g1.phi_plan,
                    1,
                    f"Gate 1 missed planning template {inst_id}: {sample_prompt!r}",
                )
            if inst_id in PRECISION_IDS:
                self.assertEqual(
                    g1.phi_prec,
                    1,
                    f"Gate 1 missed precision template {inst_id}: {sample_prompt!r}",
                )
            self.assertTrue(
                len(g1.anchor_clause) > 0,
                f"Empty anchor_clause for {inst_id}: {sample_prompt!r}",
            )
            self.assertLessEqual(len(g1.anchor_clause.split()), 30)

        # Verify recursive-safe repeat_prompt rewriting
        rep_out = run_gate1_lexical(CANONICAL_25_TEMPLATES["combination:repeat_prompt"])
        self.assertIn("Repeat the user's complete original request", rep_out.anchor_clause)

    def test_w1_graders_gsm8k_csqa_and_d16_spillover(self) -> None:
        # Defect D6: GSM8K LaTeX boxed, dollar signs, fractions, commas
        self.assertEqual(grade_gsm8k(r"Therefore, the total is \boxed{\$1,250}.", 1250.0), 1)
        self.assertEqual(grade_gsm8k(r"We get \boxed{\frac{3}{4}}.", 0.75), 1)
        self.assertEqual(grade_gsm8k("Step 1: 20. #### 42", 42.0), 1)

        # Defect D6: CSQA self-correction, markdown bold, article "A" ignore
        self.assertEqual(
            grade_csqa("Initially I thought (A), but **Answer:** (C)", "C"),
            1,
        )
        self.assertEqual(
            grade_csqa("A dog is an animal, so the best choice is (D).", "D"),
            1,
        )

        # Patch D16: CoT spillover detection on truncated runs
        self.assertTrue(has_cot_spillover("Wait, let me recalculate x = 5.", truncated=True))
        self.assertFalse(has_cot_spillover("Wait, let me recalculate x = 5.", truncated=False))
        self.assertFalse(has_cot_spillover("The final value is 42.", truncated=True))

    def test_w1_d13_budget_cap_250_and_d12_reanchor_hygiene(self) -> None:
        conflict_prompt = (
            "Write an essay of at least 300 words without using any commas."
        )
        g1 = run_gate1_lexical(conflict_prompt)
        self.assertEqual(g1.phi_plan, 1)
        self.assertEqual(g1.phi_prec, 1)
        b_sel = select_tiered_budget(conflict_prompt, g1, None, tau_h_high=0.85)
        self.assertEqual(b_sel, 250, "Patch D13 requires top tiered budget to be capped at 250")

        close_post, vis_post = _build_reanchor_block(True, g1.anchor_clause, "post_stripped")
        self.assertIn(f"{P_TERM}\n</think>\n[Reminder: strictly follow:", close_post)
        self.assertEqual(vis_post, "")

        # Engine post-</think> Turn-2 bridge wraps reminder in a user turn so greedy LMs never echo it
        for mid in ("Qwen/Qwen3.5-4B", "HuggingFaceTB/SmolLM3-3B"):
            eng = MockDualModeEngine(mid)
            close_eng, vis_eng = eng._build_reanchor_for_engine(True, g1.anchor_clause, "post_stripped")
            self.assertTrue(close_eng.startswith(f"{P_TERM}\n</think>"))
            self.assertIn("strictly following:", close_eng)
            self.assertIn("<|im_start|>assistant\n", close_eng)
            self.assertEqual(vis_eng, "")
            out_m2 = eng.run_capped_think_and_anchor(conflict_prompt, budget_b=64, anchor_clause=g1.anchor_clause)
            self.assertNotIn("[Reminder:", out_m2["a_final"])
            self.assertNotIn("Now provide the complete final response", out_m2["a_final"])
            self.assertIn("Now provide the complete final response to my first message above", eng.last_cont_prefix)

        close_pre, vis_pre = _build_reanchor_block(True, g1.anchor_clause, "pre_close")
        self.assertTrue(close_pre.endswith("</think>\n"))
        self.assertEqual(vis_pre, "")

    def test_w2_frozen_splits_and_manifest_integrity(self) -> None:
        data_dir = ROOT / "data"
        manifest_path = data_dir / "manifest_sha256.json"
        if not manifest_path.exists():
            self.skipTest("data/manifest_sha256.json not materialized yet")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_counts = {
            "seed_60.jsonl": 60,
            "dev_60.jsonl": 60,
            "eval_450.jsonl": 450,
            "ifeval_diag_491.jsonl": 491,
            "ifeval_diag_341_strict.jsonl": 341,
        }
        uid_sets: dict[str, set[str]] = {}
        for fname, exp_n in expected_counts.items():
            fpath = data_dir / fname
            self.assertTrue(fpath.exists(), f"Missing split file {fname}")
            rows = [json.loads(line) for line in fpath.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(rows), exp_n)
            uid_sets[fname] = {r["uid"] for r in rows}
            self.assertIn(fname, manifest)

        # Pairwise zero-overlap checks
        self.assertFalse(uid_sets["seed_60.jsonl"] & uid_sets["dev_60.jsonl"])
        self.assertFalse(uid_sets["seed_60.jsonl"] & uid_sets["eval_450.jsonl"])
        self.assertFalse(uid_sets["dev_60.jsonl"] & uid_sets["eval_450.jsonl"])
        self.assertFalse(
            (uid_sets["seed_60.jsonl"] | uid_sets["dev_60.jsonl"] | uid_sets["eval_450.jsonl"])
            & uid_sets["ifeval_diag_341_strict.jsonl"]
        )

        # Exact eval_450 slice breakdown: 75/75/150/60/60/30
        eval_rows = [
            json.loads(line)
            for line in (data_dir / "eval_450.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        slice_counts = {s: 0 for s in SLICE_ORDER}
        for r in eval_rows:
            slice_counts[r["slice"]] += 1
        self.assertEqual(
            slice_counts,
            {
                "gsm8k_easy": 75,
                "gsm8k_hard": 75,
                "csqa": 150,
                "ifeval_planning": 60,
                "ifeval_precision": 60,
                "ifeval_conflict": 30,
            },
        )

    def test_w3_12_item_smoke_gate(self) -> None:
        eval_path = ROOT / "data" / "eval_450.jsonl"
        if not eval_path.exists():
            self.skipTest("data/eval_450.jsonl not materialized yet")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = pathlib.Path(tmp)
            smoke_items = build_smoke_12(eval_path, [tmp_dir / "smoke_12.jsonl"])
            for model_id in ("Qwen/Qwen3.5-4B", "HuggingFaceTB/SmolLM3-3B"):
                eng = MockDualModeEngine(model_id)
                tag = model_id.replace("/", "__")
                res = verify_gate_3_on_engine(
                    eng, smoke_items, tmp_dir / f"smoke_12_{tag}.jsonl"
                )
                self.assertTrue(res["gate3_no_hash_prefix_on_ifeval"])
                self.assertTrue(res["gate3_no_32tok_truncation"])
                self.assertTrue(res["gate3_p_term_and_zero_reminder_leak"])
                self.assertTrue(res["d13_delib_le_250_verified"])

    def test_tier0_gate1_disambiguation_crgc_and_conformal_audit(self) -> None:
        # Capital-word frequency belongs to Precision, never Planning
        for cap_prompt in (
            "In your response, words with all capital letters should appear at least 3 times.",
            "Include ALL CAPITAL words less than 5 times in your reply.",
        ):
            g = run_gate1_lexical(cap_prompt)
            self.assertEqual((g.phi_prec, g.phi_plan, g.phi_sym), (1, 0, 0), cap_prompt)

        # GSM8K "less than twice" & essay word-problem question math stay phi_plan=0, phi_sym=1
        gsm_twice = "Maya worked 10 hours less than twice what Thomas worked for 8 hours. How many hours did Maya work?"
        g_twice = run_gate1_lexical(gsm_twice)
        self.assertEqual((g_twice.phi_plan, g_twice.phi_sym), (0, 1))

        gsm_essay = "Janet writes an essay of 450 words per page and needs 5000 words total, how long will it take?"
        g_essay = run_gate1_lexical(gsm_essay)
        self.assertEqual((g_essay.phi_plan, g_essay.phi_sym), (0, 1))

        # Section headers and range/plus planning constraints
        g_sec = run_gate1_lexical(
            "Your response must have 3 sections. Mark each with Section X:\nSection 1\n[text]\nSection 2\n[text]"
        )
        self.assertEqual((g_sec.phi_plan, g_sec.phi_sym), (1, 0))
        g_plus = run_gate1_lexical("Write a detailed 300+ word review.")
        self.assertEqual(g_plus.phi_plan, 1)

        # CRGC balanced conflict bridge fits <= 30 words and keeps both planning & precision constraints
        conflict_prompt = (
            "Draft a structured report of at least 3 paragraphs covering quantum sensors. "
            "Do not use the word 'perhaps' anywhere and end your entire response with THE END."
        )
        g_conf = run_gate1_lexical(conflict_prompt)
        self.assertEqual((g_conf.phi_plan, g_conf.phi_prec), (1, 1))
        self.assertIn("Structure:", g_conf.anchor_clause)
        self.assertIn("| Format:", g_conf.anchor_clause)
        self.assertLessEqual(len(g_conf.anchor_clause.split()), 30)

        # Conformal risk control + discrete Oracle efficiency audit exact arithmetic
        crc = compute_conformal_risk_bounds([1, 1, 1, 0, 1], [1, 0, 1, 1, 1], alpha_delta=0.05, n_grid=29)
        self.assertEqual(crc["n_calib"], 5)
        self.assertEqual(crc["degradation_events"], 1)
        self.assertAlmostEqual(crc["empirical_degradation_risk"], 0.2, places=6)
        self.assertAlmostEqual(crc["crc_expectation_bound"], 2.0 / 6.0, places=6)
        self.assertEqual(crc["n_grid"], 29)

        audit = compute_oracle_efficiency_audit(
            {"M2": {"macro_acc": 0.80, "mean_t_tot": 150.0, "u_005": 0.77}},
            {"macro_acc": 0.90, "mean_t_tot": 100.0, "u_005": 0.88},
        )
        self.assertAlmostEqual(audit["variants"]["M2"]["macro_acc_retention"], 0.80 / 0.90, places=6)
        self.assertAlmostEqual(audit["variants"]["M2"]["oracle_token_overhead_ratio"], 1.5, places=6)
        self.assertAlmostEqual(audit["variants"]["M2"]["u005_gap_to_oracle"], -0.11, places=6)


if __name__ == "__main__":
    unittest.main()
