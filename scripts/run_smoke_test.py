"""Phase 3 (W3): End-to-End 12-Example Pipeline Smoke Test and Verification Gate 3."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine import CBARRDualModeEngine, P_TERM, _build_reanchor_block, run_gate1_lexical
from src.graders import grade_item, has_4gram_repetition_loop, has_cot_spillover
from src.logfmt import banner, hl
from src.metrics import SLICE_ORDER
from scripts.run_matrix import cache_split_primitives, resolve_device_and_dtype


class MockTokenizer:
    """Deterministic tokenizer for offline/CPU smoke testing of CBARRDualModeEngine."""

    def __init__(self, is_smollm3: bool = False):
        self.is_smollm3 = is_smollm3
        self.eos_token_id = [248046, 248044] if not is_smollm3 else 128001
        self.pad_token_id = 0
        self.unk_token_id = 999999

    def convert_tokens_to_ids(self, tok_str: str) -> int:
        mapping = {
            "<|im_end|>": 248044,
            "<|endoftext|>": 248046,
            "<|end_of_text|>": 128001,
            "</s>": 2,
        }
        return mapping.get(tok_str, self.unk_token_id)

    def apply_chat_template(
        self,
        msgs: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        parts = []
        if self.is_smollm3:
            parts.append("<|im_start|>system\nMetadata: Today Date: 10 November 2026\n")
            for m in msgs:
                parts.append(f"<{m['role']}>{m['content']}</{m['role']}>\n")
            parts.append("<|im_start|>assistant\n")
            if enable_thinking:
                parts.append("<think>\n")
        else:
            for m in msgs:
                parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
            parts.append("<|im_start|>assistant\n")
            if enable_thinking:
                parts.append("<think>\n")
        return "".join(parts)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [idx + 10 for idx, _ in enumerate(text.split())]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return " ".join(f"tok{t}" for t in token_ids)


class MockDualModeEngine(CBARRDualModeEngine):
    """Subclasses CBARRDualModeEngine to exercise exact pipeline logic without 16GB GPU weights."""

    def __init__(
        self,
        model_id: str,
        freeze_smollm3_date: str = "01 November 2026",
        temperature: float = 0.0,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        sampling_seed: int = 42,
    ):
        self.model_id = model_id
        self.is_smollm3 = "smollm3" in model_id.lower()
        self.freeze_smollm3_date = freeze_smollm3_date
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.presence_penalty = float(presence_penalty)
        self.sampling_seed = int(sampling_seed)
        self.device = "cpu"
        self.tok = MockTokenizer(is_smollm3=self.is_smollm3)
        self.eos_ids = {2, 128001, 248044, 248046}
        self.last_cont_prefix: str = ""

    def _decode_raw(
        self, prompt_str: str, max_new_tokens: int, log_entropies: bool = False
    ) -> dict[str, Any]:
        """Simulates deterministic dual-mode decoding across thinking caps, probes, and answers."""
        # Record continuation prompts containing P_TERM for Gate 3 verification
        if P_TERM in prompt_str:
            self.last_cont_prefix = prompt_str

        # Case 1: Thinking mode stage 1 without "</think>" yet -> simulate truncation at budget_b <= 250
        if prompt_str.rstrip().endswith("<think>") and max_new_tokens <= 250:
            words = ["Let", "us", "carefully", "derive", "each", "constraint", "step", "by", "step."]
            n_toks = max_new_tokens
            text = " ".join(words * ((n_toks // len(words)) + 1))
            return {
                "raw_text": text,
                "total_gen_tokens": n_toks,
                "content_tokens": n_toks,
                "hit_eos": False,
                "step_entropies": [0.9] * n_toks if log_entropies else [],
                "latency_ms": 1.5,
            }

        # Case 2: B1b unbounded thinking (max_new_tokens == 2048)
        if prompt_str.rstrip().endswith("<think>") and max_new_tokens > 250:
            return {
                "raw_text": "Step 1: analyze problem. Step 2: compute result.\n</think>\nFinal response text without commas.<|im_end|>",
                "total_gen_tokens": 96,
                "content_tokens": 95,
                "hit_eos": True,
                "step_entropies": [],
                "latency_ms": 2.0,
            }

        # Case 3: Gate-2 32-token non-thinking probe on open-ended / IFEval prompt (no EOS at 32 tokens)
        if log_entropies and max_new_tokens == 32 and not prompt_str.endswith(("Answer: ", "#### ")):
            return {
                "raw_text": "Here is the first part of a detailed multi-sentence response that continues past thirty-two tokens",
                "total_gen_tokens": 32,
                "content_tokens": 32,
                "hit_eos": False,
                "step_entropies": [0.45] * 32,
                "latency_ms": 0.8,
            }

        # Case 4: Continuation of non-EOS Gate-2 fast probe (complete_fast_probe up to 1024 tokens)
        if "Here is the first part of a detailed multi-sentence response" in prompt_str:
            return {
                "raw_text": " and completes cleanly with a full paragraph satisfying the user's request.<|im_end|>",
                "total_gen_tokens": 24,
                "content_tokens": 23,
                "hit_eos": True,
                "step_entropies": [],
                "latency_ms": 0.7,
            }

        # Case 5: Math / MC / Stage-2 post-</think> answer generation
        if prompt_str.endswith("#### "):
            return {
                "raw_text": "42<|im_end|>",
                "total_gen_tokens": 3,
                "content_tokens": 2,
                "hit_eos": True,
                "step_entropies": [0.15, 0.10] if log_entropies else [],
                "latency_ms": 0.5,
            }
        if prompt_str.endswith("Answer: "):
            return {
                "raw_text": "(C)<|im_end|>",
                "total_gen_tokens": 3,
                "content_tokens": 2,
                "hit_eos": True,
                "step_entropies": [0.20, 0.12] if log_entropies else [],
                "latency_ms": 0.5,
            }
        return {
            "raw_text": "clean compliant response text generated after thinking.<|im_end|>",
            "total_gen_tokens": 44,
            "content_tokens": 43,
            "hit_eos": True,
            "step_entropies": [0.3] * 10 if log_entropies else [],
            "latency_ms": 1.0,
        }


def build_smoke_12(
    eval_path: pathlib.Path,
    out_paths: list[pathlib.Path],
) -> list[dict[str, Any]]:
    """Slices 2 items per eval_450 category (12 items total) into smoke_12.jsonl."""
    rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line]
    smoke_items: list[dict[str, Any]] = []
    for sl in SLICE_ORDER:
        sl_rows = [r for r in rows if r["slice"] == sl][:2]
        assert len(sl_rows) == 2, f"Expected 2 items for slice {sl}, found {len(sl_rows)}"
        smoke_items.extend(sl_rows)

    assert len(smoke_items) == 12
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in smoke_items)
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(payload, encoding="utf-8")
    return smoke_items


def verify_gate_3_on_engine(
    engine: CBARRDualModeEngine,
    smoke_items: list[dict[str, Any]],
    out_cache_path: pathlib.Path,
) -> dict[str, Any]:
    """Runs 12-item smoke suite and verifies all Phase 3 / W3 Verification Gate 3 conditions."""
    if out_cache_path.exists():
        out_cache_path.unlink()

    cached_rows = cache_split_primitives(
        engine,
        smoke_items,
        out_cache_path,
        full_eval_modes=True,
        run_calib_tiers=True,
    )
    assert len(cached_rows) == 12, f"Expected 12 cached smoke rows, got {len(cached_rows)}"

    ifeval_rows = [r for r in cached_rows if r["item"]["dataset"] == "ifeval"]
    assert len(ifeval_rows) == 6, f"Expected 6 IFEval smoke rows, got {len(ifeval_rows)}"

    # Gate 3 Check 1: B0 and non-escalated Gate-2 IFEval outputs do not start with '#### '
    # and complete_fast_probe continues past 32 tokens without truncating at 32 tokens.
    for r in ifeval_rows:
        assert r["probe"]["ans_prefix"] == "", (
            f"IFEval item {r['item']['uid']} got non-empty probe ans_prefix: {r['probe']['ans_prefix']!r}"
        )
        assert not r["b0"]["a_final"].startswith("####"), (
            f"IFEval B0 output started with '#### ': {r['b0']['a_final'][:40]!r}"
        )
        assert not r["fast_done"]["a_final"].startswith("####"), (
            f"IFEval fast_done output started with '#### ': {r['fast_done']['a_final'][:40]!r}"
        )

    # Also explicitly test complete_fast_probe continuation when 32-token probe did not hit EOS
    sample_ife = ifeval_rows[0]["item"]["prompt"]
    g1_ife = run_gate1_lexical(sample_ife)
    probe_ife = engine.run_gate2_fast_probe(sample_ife, g1=g1_ife, probe_tokens=32)
    completed_ife = engine.complete_fast_probe(probe_ife, max_ans_tokens=1024)
    assert not completed_ife["a_final"].startswith("####")
    if not probe_ife["hit_eos"]:
        assert completed_ife["T_total"] > 32, (
            f"Non-EOS 32-token probe failed to continue past 32 tokens: T_total={completed_ife['T_total']}"
        )

    # Gate 3 Check 2: M2 truncated traces contain '\nOkay, I think I have finished thinking.\n</think>\n'
    # inside the context prompt, while a_final under post_stripped contains zero leaked '[Reminder: ...]' text.
    close_block, vis_prefix = _build_reanchor_block(
        truncated=True,
        anchor_clause=g1_ife.anchor_clause or "Respond in lowercase.",
        anchor_locus="post_stripped",
    )
    assert f"{P_TERM}\n</think>\n" in close_block, (
        f"Expected P_TERM + </think> in close_block, got: {close_block!r}"
    )
    assert vis_prefix == "", f"post_stripped vis_prefix must be empty, got: {vis_prefix!r}"

    for r in cached_rows:
        for k in ("m2_b64", "m2_b128", "m2_b256", "m2_b256_direct"):
            a_fin = r[k]["a_final"]
            assert "[Reminder:" not in a_fin, (
                f"Leaked '[Reminder:' wrapper in {k} a_final for {r['item']['uid']}: {a_fin!r}"
            )
            assert "Constraint check:" not in a_fin, (
                f"Leaked 'Constraint check:' wrapper in {k} a_final for {r['item']['uid']}"
            )
            assert "Now provide the complete final response to my first message above" not in a_fin, (
                f"Leaked Turn-2 user bridge in {k} a_final for {r['item']['uid']}: {a_fin!r}"
            )
        # Verify Patch D13: T_probe + T_think <= 250 on m2_b256 and m2_b256_direct
        assert int(r["probe"]["T_probe"]) + int(r["m2_b256"]["T_think"]) <= 250
        assert int(r["m2_b256_direct"]["T_think"]) <= 250

    # Gate 3 Check 3: SmolLM3-3B rendered prompts show frozen 'Today Date: 01 November 2026'
    if engine.is_smollm3:
        rendered_think = engine.build_prompt_prefix("Solve 2+2.", enable_thinking=True)
        rendered_nothink = engine.build_prompt_prefix("Solve 2+2.", enable_thinking=False)
        assert "Today Date: 01 November 2026" in rendered_think, (
            f"Missing frozen Today Date in SmolLM3 thinking prompt: {rendered_think!r}"
        )
        assert "Today Date: 01 November 2026" in rendered_nothink, (
            f"Missing frozen Today Date in SmolLM3 non-thinking prompt: {rendered_nothink!r}"
        )

    return {
        "model": engine.model_id,
        "n_items": len(cached_rows),
        "gate3_no_hash_prefix_on_ifeval": True,
        "gate3_no_32tok_truncation": True,
        "gate3_p_term_and_zero_reminder_leak": True,
        "gate3_smollm3_frozen_date": engine.is_smollm3,
        "d13_delib_le_250_verified": True,
        "cache_path": str(out_cache_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 12-example pipeline smoke test.")
    parser.add_argument(
        "--eval-path",
        type=str,
        default="data/eval_450.jsonl",
        help="Path to frozen eval_450.jsonl",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen/Qwen3.5-4B", "HuggingFaceTB/SmolLM3-3B"],
        help="Models to test",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Output directory for smoke caches and gate3 report",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device ('auto', 'cuda', 'mps', 'cpu', or 'cuda:N')",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Floating dtype ('auto', 'bfloat16', 'float16', 'float32')",
    )
    parser.add_argument(
        "--live-gpu",
        action="store_true",
        help="Load full HuggingFace model checkpoints instead of deterministic smoke harness",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = pathlib.Path(args.eval_path)
    out_smoke_paths = [
        pathlib.Path("/tmp/smoke_12.jsonl"),
        out_dir / "smoke_12.jsonl",
    ]
    smoke_items = build_smoke_12(eval_path, out_smoke_paths)

    results = []
    for model_id in args.models:
        tag = model_id.replace("/", "__")
        cache_out = out_dir / f"smoke_12_{tag}.jsonl"
        if args.live_gpu:
            device, dtype = resolve_device_and_dtype(args.device, args.dtype)
            print(f"[smoke] Loading {model_id} on device={device} dtype={dtype}", flush=True)
            engine = CBARRDualModeEngine(model_id, device=device, dtype=dtype)
        else:
            engine = MockDualModeEngine(model_id)
        res = verify_gate_3_on_engine(engine, smoke_items, cache_out)
        results.append(res)
        if args.live_gpu:
            import gc
            from scripts.run_matrix import _release_accelerator_cache

            dev_name = str(getattr(engine, "device", "cpu"))
            del engine
            gc.collect()
            _release_accelerator_cache(dev_name)

    summary_path = out_dir / "smoke_test_gate3_report.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(banner(f"Phase 3 (W3) 12-Example Pipeline Smoke Test: {hl('PASSED', 'green')}", "green"))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
