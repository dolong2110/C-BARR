#!/usr/bin/env python3
"""OOD Generalization Extension: Evaluates frozen B0, B1b, and C-BARR M2 on allenai/IFBench_test (300 prompts, 58 unseen constraints)."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import time
from typing import Any

# Set PyTorch CUDA allocator fragmentation guard before importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gpu_pool import _estimate_model_billions, select_worker_gpu_slots
from scripts.run_matrix import (
    _build_engine,
    _is_native_engine,
    _release_accelerator_cache,
    resolve_device_and_dtype,
)
from scripts.sync_results import model_slug
from scripts.versioning import get_version_provenance
from src.engine import run_gate1_lexical
from src.graders import has_4gram_repetition_loop, has_cot_spillover
from src.metrics import (
    compute_auc_oaa,
    compute_bsr_curve,
    compute_oaa_t,
    exact_and_midp_mcnemar,
    paired_bootstrap_diff_ci,
    wilson_ci,
)
from src.routers import (
    CalibratedRouterParams,
    score_gate2_probability,
    select_tiered_budget,
)

# Deterministic 12-item synthetic OOD benchmark used under --mock for offline CI validation
_MOCK_IFBENCH_ROWS: tuple[dict[str, Any], ...] = (
    {
        "key": 1001,
        "prompt": "Write a concise two-sentence description of mountain rivers. Do not repeat any word twice.",
        "instruction_id_list": ["repeat:repeat_change"],
        "kwargs": [{}],
    },
    {
        "key": 1002,
        "prompt": "Compose a 4-sentence paragraph with strictly increasing alliteration per sentence on winter mornings.",
        "instruction_id_list": ["sentence:alliteration_increment"],
        "kwargs": [{}],
    },
    {
        "key": 1003,
        "prompt": "Provide a short factual answer on why tides occur. Use at most 3 conjunctions in total.",
        "instruction_id_list": ["count:conjunctions"],
        "kwargs": [{"max_conjunctions": 3}],
    },
    {
        "key": 1004,
        "prompt": "Summarize photosynthesis cleanly. Format every heading in Title Case without commas.",
        "instruction_id_list": ["format:title_case", "punctuation:no_comma"],
        "kwargs": [{}, {}],
    },
    {
        "key": 1005,
        "prompt": "First outline a three-step study schedule, then write a concise takeaway in plain ASCII.",
        "instruction_id_list": ["structure:step_outline", "format:plain_ascii"],
        "kwargs": [{}, {}],
    },
    {
        "key": 1006,
        "prompt": "Explain binary search vs linear search clearly. Respond with a valid JSON object containing keys complexity and intuition.",
        "instruction_id_list": ["detectable_format:json_format", "count:word_range"],
        "kwargs": [{}, {"min_words": 15, "max_words": 80}],
    },
    {
        "key": 1007,
        "prompt": "Draft a brief policy note with all lowercase text and zero exclamation marks regarding open datasets.",
        "instruction_id_list": ["change_case:english_lowercase", "punctuation:no_exclamation"],
        "kwargs": [{}, {}],
    },
    {
        "key": 1008,
        "prompt": "Compare iterative deepening and breadth-first search carefully, ending with a single standalone takeaway sentence.",
        "instruction_id_list": ["sentence:trailing_takeaway"],
        "kwargs": [{}],
    },
    {
        "key": 1009,
        "prompt": "Write a crisp 40 to 90 word note describing why checksums protect file downloads against corruption.",
        "instruction_id_list": ["count:word_range"],
        "kwargs": [{"min_words": 40, "max_words": 90}],
    },
    {
        "key": 1010,
        "prompt": "List three distinct ecological roles of wetlands without using passive voice verbs.",
        "instruction_id_list": ["grammar:no_passive"],
        "kwargs": [{}],
    },
    {
        "key": 1011,
        "prompt": "Plan a four-stage code review checklist with numbered sections and no parenthetical remarks.",
        "instruction_id_list": ["structure:numbered_sections", "punctuation:no_parens"],
        "kwargs": [{}, {}],
    },
    {
        "key": 1012,
        "prompt": "Give a compact intuition for Bayes theorem using two sentences and zero Markdown headings.",
        "instruction_id_list": ["count:exact_sentences", "format:no_markdown_headers"],
        "kwargs": [{"num_sentences": 2}, {}],
    },
)


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_ifbench_items(
    *,
    dataset_path: pathlib.Path | None = None,
    mock: bool = False,
    limit: int = 0,
) -> tuple[list[dict[str, Any]], str]:
    """Loads allenai/IFBench_test rows (from JSONL, HuggingFace datasets, ifbench wheel, or mock)."""
    rows: list[dict[str, Any]] = []
    source_tag = "mock_ifbench_12"

    if mock:
        rows = [dict(r) for r in _MOCK_IFBENCH_ROWS]
    elif dataset_path is not None:
        if not dataset_path.is_file():
            raise FileNotFoundError(f"IFBench dataset file not found: {dataset_path}")
        rows = _load_jsonl(dataset_path)
        source_tag = str(dataset_path)
    else:
        try:
            from datasets import load_dataset

            ds = None
            try:
                ds = load_dataset("allenai/IFBench_test", split="test")
            except Exception:
                ds_dict = load_dataset("allenai/IFBench_test")
                first_split = next(iter(ds_dict.keys()))
                ds = ds_dict[first_split]
            rows = [dict(r) for r in ds]
            source_tag = "hf:allenai/IFBench_test"
        except Exception as hf_exc:
            try:
                import ifbench

                if hasattr(ifbench, "data_path"):
                    bundled_p = pathlib.Path(str(ifbench.data_path()))
                    if bundled_p.is_file():
                        rows = _load_jsonl(bundled_p)
                        source_tag = f"ifbench_pkg:{bundled_p.name}"
            except Exception:
                rows = []
            if not rows:
                raise RuntimeError(
                    "Unable to load allenai/IFBench_test via datasets or bundled ifbench.data_path(); "
                    f"original error: {hf_exc}"
                ) from hf_exc

    normalized: list[dict[str, Any]] = []
    for idx, raw in enumerate(rows):
        prompt = str(raw["prompt"])
        key = raw.get("key", idx)
        inst_ids = [str(x) for x in raw.get("instruction_id_list", [])]
        raw_kw = raw.get("kwargs", [])
        clean_kwargs: list[dict[str, Any]] = []
        for kw in raw_kw:
            if isinstance(kw, dict):
                clean_kwargs.append({k: v for k, v in kw.items() if v is not None})
            else:
                clean_kwargs.append({})
        while len(clean_kwargs) < len(inst_ids):
            clean_kwargs.append({})
        uid_hash = hashlib.sha256(f"{key}::{prompt}".encode("utf-8")).hexdigest()[:10]
        normalized.append(
            {
                "uid": f"ifbench_{key}_{uid_hash}",
                "key": key,
                "prompt": prompt,
                "instruction_id_list": inst_ids,
                "kwargs": clean_kwargs[: len(inst_ids)],
            }
        )

    if limit and limit > 0:
        normalized = normalized[:limit]
    return normalized, source_tag


def _load_ifbench_registry(mock: bool) -> dict[str, Any] | None:
    """Imports allenai ifbench instructions_registry in live mode; returns None in --mock."""
    if mock:
        return None
    try:
        from ifbench import instructions_registry
    except ImportError as exc:
        raise RuntimeError(
            "The 'ifbench' Python package is required for live allenai/IFBench_test verification. "
            "Install it with `pip install ifbench` (or re-run bash scripts/setup_env.sh)."
        ) from exc
    return dict(instructions_registry.INSTRUCTION_DICT)


def _strip_markdown_wrapping(text: str) -> str:
    """Produces IFEval-style loose response candidate with outer emphasis/fences removed."""
    cleaned = text.strip()
    lines = cleaned.splitlines()
    if len(lines) >= 2 and (lines[0].startswith("```") or lines[0].startswith("***")):
        lines = lines[1:]
    if lines and (lines[-1].startswith("```") or lines[-1].startswith("***")):
        lines = lines[:-1]
    out = "\n".join(lines).replace("**", "").replace("*", "")
    return out.strip()


def _grade_ifbench_mock(item: dict[str, Any], response: str) -> list[bool]:
    """Deterministic lightweight constraint checks used exclusively under --mock."""
    txt = str(response or "").strip()
    if not txt:
        return [False for _ in item["instruction_id_list"]]
    spillover = has_cot_spillover(txt, truncated=False)
    results: list[bool] = []
    for inst_id, kw in zip(item["instruction_id_list"], item["kwargs"]):
        if spillover or "[Reminder:" in txt or "</think>" in txt:
            results.append(False)
            continue
        if inst_id == "punctuation:no_comma":
            results.append("," not in txt)
        elif inst_id == "change_case:english_lowercase":
            results.append(txt == txt.lower())
        elif inst_id == "detectable_format:json_format":
            results.append("{" in txt and "}" in txt)
        elif inst_id == "count:word_range":
            wc = len(txt.split())
            lo = int(kw.get("min_words", 5))
            hi = int(kw.get("max_words", 250))
            results.append(lo <= wc <= hi)
        else:
            # Deterministic pass for clean non-spillover mock outputs of sufficient substance
            results.append(len(txt.split()) >= 3 and "mock" in txt.lower() or len(txt) >= 12)
    return results


def _eval_single_ifbench_inst(
    registry: dict[str, Any],
    inst_id: str,
    kw: dict[str, Any],
    prompt: str,
    candidate: str,
) -> bool:
    cls = registry[inst_id]
    checker = cls(inst_id)
    build_kw = dict(kw)
    try:
        sig = inspect.signature(checker.build_description)
        accepted = set(sig.parameters.keys())
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not has_var_kw:
            build_kw = {k: v for k, v in build_kw.items() if k in accepted}
        checker.build_description(**build_kw)
    except TypeError:
        checker.build_description()

    try:
        args = checker.get_instruction_args()
        if isinstance(args, dict) and "prompt" in args and args["prompt"] is None:
            checker.build_description(prompt=prompt, **build_kw)
    except Exception:
        pass
    return bool(checker.check_following(candidate))


def grade_ifbench_response(
    item: dict[str, Any],
    run_rec: dict[str, Any],
    registry: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluates prompt-level strict/loose & instruction-level compliance on IFBench_test."""
    final_answer = str(run_rec.get("a_final", ""))
    inst_ids = list(item["instruction_id_list"])
    kwargs_list = list(item["kwargs"])

    if registry is None:
        strict_flags = _grade_ifbench_mock(item, final_answer)
        loose_flags = strict_flags
    else:
        random.seed(0)
        try:
            from langdetect import DetectorFactory

            DetectorFactory.seed = 0
        except Exception:
            pass
        if not final_answer.strip():
            strict_flags = [False for _ in inst_ids]
            loose_flags = [False for _ in inst_ids]
        else:
            loose_cand = _strip_markdown_wrapping(final_answer)
            variants = [
                final_answer,
                loose_cand,
                "\n".join(final_answer.splitlines()[1:]).strip() if "\n" in final_answer else final_answer,
            ]
            strict_flags = [
                _eval_single_ifbench_inst(registry, iid, kw, item["prompt"], final_answer)
                for iid, kw in zip(inst_ids, kwargs_list)
            ]
            loose_flags = [
                any(
                    _eval_single_ifbench_inst(registry, iid, kw, item["prompt"], cand)
                    for cand in variants
                    if cand
                )
                for iid, kw in zip(inst_ids, kwargs_list)
            ]

    out = dict(run_rec)
    out["follow_strict_list"] = strict_flags
    out["follow_loose_list"] = loose_flags
    out["z"] = int(bool(strict_flags) and all(strict_flags))
    out["z_loose"] = int(bool(loose_flags) and all(loose_flags))
    out["inst_pass"] = int(sum(1 for f in strict_flags if f))
    out["inst_total"] = int(len(strict_flags))
    combined_txt = f"{out.get('t_think', '')} {final_answer}".strip()
    out["has_4gram_loop"] = has_4gram_repetition_loop(combined_txt)
    out["has_cot_spillover"] = has_cot_spillover(
        final_answer, truncated=bool(out.get("truncated", False))
    )
    return out


def resolve_calibration_params(
    model_id: str,
    out_root: pathlib.Path,
    calib_json: pathlib.Path | None,
    *,
    mock: bool,
    allow_default_params: bool = False,
) -> tuple[CalibratedRouterParams, str]:
    """Loads frozen C-BARR Gate-2 weights without fitting on OOD data."""
    m_dir = out_root / model_slug(model_id)
    candidates: list[pathlib.Path] = []
    if calib_json is not None:
        candidates.append(calib_json)
    candidates.extend(
        [
            m_dir / "calibration_summary.json",
            m_dir / "matrix_summary.json",
            m_dir / "matrix_summary_strict341.json",
            m_dir / "matrix_summary_diag491.json",
        ]
    )
    for cand in candidates:
        if cand.is_file():
            payload = json.loads(cand.read_text(encoding="utf-8"))
            cp = payload.get("calibrated_params")
            if isinstance(cp, dict) and "coef" in cp and "tau_prob" in cp:
                return (
                    CalibratedRouterParams(
                        coef=[float(x) for x in cp["coef"]],
                        intercept=float(cp["intercept"]),
                        tau_prob=float(cp["tau_prob"]),
                        tau_h_high=float(cp["tau_h_high"]),
                        m2_escalation_rate_dev=float(cp.get("m2_escalation_rate_dev", 0.5)),
                    ),
                    str(cand),
                )

    if mock or allow_default_params:
        return (
            CalibratedRouterParams(
                coef=[0.6, 0.8, 0.4, 1.0, 0.5],
                intercept=-1.0,
                tau_prob=0.5,
                tau_h_high=0.85,
                m2_escalation_rate_dev=0.45,
            ),
            "default_fallback",
        )
    raise FileNotFoundError(
        f"No frozen calibration_summary.json or matrix_summary*.json found under {m_dir}; "
        "run matrix calibration first or pass --calib-json / --allow-default-params."
    )


def _merge_probe_and_slow(probe: dict[str, Any], slow_run: dict[str, Any]) -> dict[str, Any]:
    merged = dict(slow_run)
    merged["T_probe"] = int(probe["T_probe"])
    merged["T_total"] = int(probe["T_probe"]) + int(slow_run["T_total"])
    merged["calls"] = 2
    merged["latency_ms"] = float(probe["latency_ms"]) + float(slow_run["latency_ms"])
    return merged


def _run_ifbench_item(
    engine: Any,
    item: dict[str, Any],
    params: CalibratedRouterParams,
    registry: dict[str, Any] | None,
) -> dict[str, Any]:
    prompt = item["prompt"]
    g1 = run_gate1_lexical(prompt)
    native_greedy = _is_native_engine(engine) and float(getattr(engine, "temperature", 0.0)) == 0.0

    if native_greedy:
        probe = engine.run_gate2_fast_probe(prompt, g1=g1, probe_tokens=32)
        if probe["ans_prefix"] == "":
            fast_done = grade_ifbench_response(
                item, engine.complete_fast_probe(probe, max_ans_tokens=1024), registry
            )
            b0 = dict(fast_done)
            b0["truncated"] = False
        else:
            probe.pop("_kv_state", None)
            b0 = grade_ifbench_response(
                item, engine.run_direct_non_thinking(prompt, max_new_tokens=1024), registry
            )
            fast_done = grade_ifbench_response(
                item, engine.complete_fast_probe(probe, max_ans_tokens=1024), registry
            )
    else:
        b0 = grade_ifbench_response(
            item, engine.run_direct_non_thinking(prompt, max_new_tokens=1024), registry
        )
        probe = engine.run_gate2_fast_probe(prompt, g1=g1, probe_tokens=32)
        if probe["ans_prefix"] == "":
            probe.pop("_kv_state", None)
            fast_done = dict(b0)
        else:
            fast_done = grade_ifbench_response(
                item, engine.complete_fast_probe(probe, max_ans_tokens=1024), registry
            )

    b1b_s1: dict[str, Any] = {}
    s2_memo: dict[tuple[str, int], dict[str, Any]] = {}
    b1b = grade_ifbench_response(
        item,
        engine.run_capped_think_and_anchor(
            prompt,
            budget_b=2048,
            anchor_clause="",
            max_ans_tokens=1024,
            allow_budget_force=False,
            out_s1_cache=b1b_s1,
            s2_memo=s2_memo,
        ),
        registry,
    )

    def _run_m2_budget(b_val: int) -> dict[str, Any]:
        if (
            not g1.anchor_clause
            and not b1b["think_exhausted"]
            and not b1b["truncated"]
            and int(b1b["T_think"]) <= b_val
        ):
            return dict(b1b)
        return grade_ifbench_response(
            item,
            engine.run_capped_think_and_anchor(
                prompt,
                budget_b=b_val,
                anchor_clause=g1.anchor_clause,
                max_ans_tokens=1024,
                allow_budget_force=True,
                anchor_locus="post_stripped",
                precomputed_b1b_s1=b1b_s1,
                s2_memo=s2_memo,
            ),
            registry,
        )

    p_esc = score_gate2_probability(g1, probe, params)
    if g1.phi_prec == 1 and g1.phi_plan == 0 and g1.phi_sym == 0:
        m2 = dict(b0)
        branch = "gate1_prec_fast"
        gate2_esc = False
    elif g1.phi_plan == 1:
        b_dir = select_tiered_budget(prompt, g1, None, params.tau_h_high)
        m2 = _run_m2_budget(b_dir)
        branch = f"gate1_plan_b{b_dir}"
        gate2_esc = False
    elif p_esc >= params.tau_prob:
        b_esc = select_tiered_budget(prompt, g1, probe, params.tau_h_high)
        b_eff = max(64, min(250, 250 - int(probe["T_probe"]))) if b_esc == 250 else b_esc
        slow = _run_m2_budget(b_eff)
        m2 = _merge_probe_and_slow(probe, slow)
        branch = f"gate2_esc_b{b_eff}"
        gate2_esc = True
    else:
        m2 = dict(fast_done)
        branch = "gate2_fast_exit"
        gate2_esc = False

    probe_clean = {k: v for k, v in probe.items() if k not in ("step_entropies", "_kv_state")}
    return {
        "item": item,
        "g1": {
            "phi_prec": int(g1.phi_prec),
            "phi_plan": int(g1.phi_plan),
            "phi_sym": int(g1.phi_sym),
            "has_anchor": bool(g1.anchor_clause),
        },
        "probe": probe_clean,
        "gate2_p_esc": float(p_esc),
        "gate2_escalated": bool(gate2_esc),
        "m2_branch": branch,
        "b0": b0,
        "b1b": b1b,
        "m2": m2,
    }


def summarize_ifbench_rows(
    rows: list[dict[str, Any]],
    *,
    model_id: str,
    source_tag: str,
    calib_source: str,
    params: CalibratedRouterParams,
) -> dict[str, Any]:
    """Aggregates prompt-level, instruction-level, and constraint-subset metrics cleanly."""
    def _sys_stats(key: str, sub_rows: list[dict[str, Any]]) -> dict[str, Any]:
        runs = [r[key] for r in sub_rows]
        n = len(runs)
        if n == 0:
            return {"n": 0}
        z_vec = [int(r["z"]) for r in runs]
        z_loose = [int(r.get("z_loose", r["z"])) for r in runs]
        t_delib = [int(r.get("T_probe", 0)) + int(r.get("T_think", 0)) for r in runs]
        t_tot = [int(r["T_total"]) for r in runs]
        calls = [int(r.get("calls", 1)) for r in runs]
        k_pass = int(sum(z_vec))
        k_loose = int(sum(z_loose))
        p_strict, lo_s, hi_s = wilson_ci(k_pass, n)
        p_loose, lo_l, hi_l = wilson_ci(k_loose, n)
        inst_pass = int(sum(int(r.get("inst_pass", r["z"])) for r in runs))
        inst_tot = int(sum(int(r.get("inst_total", 1)) for r in runs))
        p_inst, _, _ = wilson_ci(inst_pass, inst_tot)
        u_vec = np.asarray(z_vec, dtype=np.float64) - 0.05 * (
            np.asarray(t_tot, dtype=np.float64) / 250.0
        )
        return {
            "n": n,
            "prompt_pass": k_pass,
            "prompt_strict_pct": round(100.0 * p_strict, 2),
            "wilson_lo_pct": round(100.0 * lo_s, 2),
            "wilson_hi_pct": round(100.0 * hi_s, 2),
            "prompt_loose_pct": round(100.0 * p_loose, 2),
            "loose_wilson_lo_pct": round(100.0 * lo_l, 2),
            "loose_wilson_hi_pct": round(100.0 * hi_l, 2),
            "inst_pass": inst_pass,
            "inst_total": inst_tot,
            "inst_strict_pct": round(100.0 * p_inst, 2),
            "oaa_250_pct": round(100.0 * compute_oaa_t(z_vec, t_delib, 250), 2),
            "auc_oaa_2048_pct": round(100.0 * compute_auc_oaa(z_vec, t_delib, 2048), 2),
            "u_005_pct": round(100.0 * float(np.mean(u_vec)), 2),
            "mean_t_tot": round(float(np.mean(t_tot)), 2),
            "mean_t_delib": round(float(np.mean(t_delib)), 2),
            "mean_calls": round(float(np.mean(calls)), 3),
            "loop_rate_pct": round(100.0 * float(np.mean([int(bool(r.get("has_4gram_loop"))) for r in runs])), 2),
            "exhaust_rate_pct": round(100.0 * float(np.mean([int(bool(r.get("think_exhausted"))) for r in runs])), 2),
            "spillover_rate_pct": round(100.0 * float(np.mean([int(bool(r.get("has_cot_spillover"))) for r in runs])), 2),
            "bsr": {str(b): round(100.0 * v, 2) for b, v in compute_bsr_curve(z_vec, t_tot).items()},
        }

    systems = {name: _sys_stats(name.lower(), rows) for name in ("B0", "B1b", "M2")}
    b1b_tok = float(systems["B1b"].get("mean_t_tot", 0.0) or 0.0)
    m2_tok = float(systems["M2"].get("mean_t_tot", 0.0) or 0.0)
    if b1b_tok > 1e-6:
        systems["M2"]["tok_savings_vs_b1b_pct"] = round(100.0 * (b1b_tok - m2_tok) / b1b_tok, 2)

    zb0 = [int(r["b0"]["z"]) for r in rows]
    zb1b = [int(r["b1b"]["z"]) for r in rows]
    zm2 = [int(r["m2"]["z"]) for r in rows]
    tb0 = [float(r["b0"]["T_total"]) for r in rows]
    tb1b = [float(r["b1b"]["T_total"]) for r in rows]
    tm2 = [float(r["m2"]["T_total"]) for r in rows]

    def _u_vec(zv: list[int], tv: list[float]) -> np.ndarray:
        return np.asarray(zv, dtype=np.float64) - 0.05 * (np.asarray(tv, dtype=np.float64) / 250.0)

    mcnemar = {
        "M2_vs_B0": exact_and_midp_mcnemar(zb0, zm2),
        "M2_vs_B1b": exact_and_midp_mcnemar(zb1b, zm2),
        "B1b_vs_B0": exact_and_midp_mcnemar(zb0, zb1b),
    }
    diff_m2_b1b_acc, acc_lo, acc_hi = paired_bootstrap_diff_ci(
        np.asarray(zb1b, dtype=np.float64), np.asarray(zm2, dtype=np.float64), seed=42
    )
    diff_m2_b1b_u, u_lo, u_hi = paired_bootstrap_diff_ci(
        _u_vec(zb1b, tb1b), _u_vec(zm2, tm2), seed=42
    )
    mcnemar["M2_vs_B1b"]["acc_diff_pp"] = round(100.0 * diff_m2_b1b_acc, 2)
    mcnemar["M2_vs_B1b"]["acc_ci95_pp"] = [round(100.0 * acc_lo, 2), round(100.0 * acc_hi, 2)]
    mcnemar["M2_vs_B1b"]["u005_diff_pp"] = round(100.0 * diff_m2_b1b_u, 2)
    mcnemar["M2_vs_B1b"]["u005_ci95_pp"] = [round(100.0 * u_lo, 2), round(100.0 * u_hi, 2)]

    g1_rows = [r for r in rows if int(r["g1"]["phi_prec"]) == 1 or int(r["g1"]["phi_plan"]) == 1]
    g2_only_rows = [r for r in rows if int(r["g1"]["phi_prec"]) == 0 and int(r["g1"]["phi_plan"]) == 0]
    branch_counts: dict[str, int] = {}
    for r in rows:
        br = str(r.get("m2_branch", "unknown"))
        branch_counts[br] = branch_counts.get(br, 0) + 1

    domain_rows: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        doms = {str(iid).split(":", 1)[0] for iid in r["item"]["instruction_id_list"]}
        for d in sorted(doms):
            domain_rows.setdefault(d, []).append(r)

    by_domain = {
        dom: {
            "n": len(sub),
            "B0_strict_pct": round(100.0 * float(np.mean([x["b0"]["z"] for x in sub])), 2),
            "B1b_strict_pct": round(100.0 * float(np.mean([x["b1b"]["z"] for x in sub])), 2),
            "M2_strict_pct": round(100.0 * float(np.mean([x["m2"]["z"] for x in sub])), 2),
        }
        for dom, sub in sorted(domain_rows.items())
    }

    return {
        "extension": "E5_ifbench_ood",
        "model": model_id,
        "dataset": source_tag,
        "calibration_source": calib_source,
        "calibrated_params": params.__dict__,
        "n_prompts": len(rows),
        "n_unique_constraints": len(
            {str(iid) for r in rows for iid in r["item"]["instruction_id_list"]}
        ),
        "systems": systems,
        "mcnemar": mcnemar,
        "gate_stats": {
            "phi_prec_rate_pct": round(100.0 * float(np.mean([r["g1"]["phi_prec"] for r in rows])), 2) if rows else 0.0,
            "phi_plan_rate_pct": round(100.0 * float(np.mean([r["g1"]["phi_plan"] for r in rows])), 2) if rows else 0.0,
            "phi_sym_rate_pct": round(100.0 * float(np.mean([r["g1"]["phi_sym"] for r in rows])), 2) if rows else 0.0,
            "gate1_any_rate_pct": round(100.0 * (len(g1_rows) / max(1, len(rows))), 2),
            "gate2_routed_n": len(g2_only_rows),
            "gate2_escalated_n": int(sum(1 for r in rows if r.get("gate2_escalated"))),
            "branch_counts": dict(sorted(branch_counts.items())),
        },
        "subsets": {
            "gate1_triggered": {name: _sys_stats(name.lower(), g1_rows) for name in ("B0", "B1b", "M2")},
            "gate2_fallback_unmatched_regex": {
                name: _sys_stats(name.lower(), g2_only_rows) for name in ("B0", "B1b", "M2")
            },
        },
        "by_constraint_domain": by_domain,
    }


def _contiguous_shard_slice(items: list[dict[str, Any]], shard_id: int, num_shards: int) -> list[dict[str, Any]]:
    if num_shards <= 1:
        return list(items)
    n = len(items)
    base, rem = divmod(n, num_shards)
    start = shard_id * base + min(shard_id, rem)
    size = base + (1 if shard_id < rem else 0)
    return items[start : start + size]


def _run_multi_shard(
    args: argparse.Namespace,
    items: list[dict[str, Any]],
    out_dir: pathlib.Path,
    summary_path: pathlib.Path,
) -> None:
    cands = [g.strip() for g in str(args.gpu_ids).split(",") if g.strip()] or None
    cur_safety = float(args.safety_mult)
    cur_wpg = str(args.workers_per_gpu)
    worker_map: list[str] = []
    if not args.mock:
        try:
            worker_map = select_worker_gpu_slots(
                args.model,
                candidate_ids=cands,
                workers_per_gpu=cur_wpg,
                max_gpus=8,
                wait_timeout_sec=int(args.wait_gpu_min) * 60,
                stage="matrix",
                safety_mult=cur_safety,
            )
        except Exception as exc:
            print(f"[ext_ifbench] gpu_pool auto-sizing failed ({exc}); falling back to 1 worker.", flush=True)
            worker_map = [cands[0]] if cands else ["0"]

    if args.mock:
        num_shards = 1 if str(args.shards).lower() == "auto" else max(1, int(args.shards))
        worker_map = ["cpu"] * num_shards
    elif str(args.shards).lower() == "auto":
        num_shards = max(1, len(worker_map))
    else:
        num_shards = max(1, int(args.shards))
        if not worker_map:
            worker_map = cands or ["0"]

    num_shards = min(num_shards, max(1, len(items)))
    if num_shards <= 1:
        # Delegate directly to single-process worker inside same process when num_shards == 1
        args.shards = "1"
        args.num_shards = 1
        args.shard_id = 0
        _run_worker(args, items, out_dir, summary_path)
        return

    print(
        f"[ext_ifbench] Multi-shard launch: {num_shards} shards across slots {worker_map[:num_shards]}",
        flush=True,
    )
    shard_dir = out_dir / "shards" / "ifbench"
    shard_dir.mkdir(parents=True, exist_ok=True)
    max_retries = max(0, int(args.max_retries))
    pending_shards = list(range(num_shards))
    attempt = 0

    if args.mock or num_shards <= 1:
        stagger_sec = 0.0
    elif float(args.stagger_sec) >= 0.0:
        stagger_sec = min(15.0, float(args.stagger_sec))
    else:
        pb = _estimate_model_billions(args.model)
        stagger_sec = 8.0 if pb >= 7.0 else (3.0 if pb >= 2.8 else 1.5)

    while pending_shards:
        if not args.mock and attempt > 0:
            try:
                worker_map = select_worker_gpu_slots(
                    args.model,
                    candidate_ids=cands,
                    workers_per_gpu=cur_wpg,
                    max_gpus=8,
                    wait_timeout_sec=int(args.wait_gpu_min) * 60,
                    stage="matrix",
                    safety_mult=cur_safety,
                ) or worker_map
            except Exception:
                pass
        procs: list[tuple[int, subprocess.Popen[str], pathlib.Path]] = []
        for idx, sid in enumerate(pending_shards):
            gpu_id = worker_map[sid % len(worker_map)]
            s_cache = shard_dir / f"shard_{sid:02d}_cache.jsonl"
            s_log = shard_dir / f"shard_{sid:02d}_attempt{attempt}.log"
            cmd = [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                "--model", args.model,
                "--out-dir", str(args.out_dir),
                "--device", "cuda:0" if not args.mock else "cpu",
                "--dtype", args.dtype,
                "--shard-id", str(sid),
                "--num-shards", str(num_shards),
                "--limit", str(args.limit),
                "--safety-mult", f"{cur_safety:.3f}",
            ]
            if args.dataset_path:
                cmd.extend(["--dataset-path", str(args.dataset_path)])
            if args.calib_json:
                cmd.extend(["--calib-json", str(args.calib_json)])
            if args.allow_default_params:
                cmd.append("--allow-default-params")
            if args.mock:
                cmd.append("--mock")
            if args.local_files_only:
                cmd.append("--local-files-only")
            if args.resume:
                cmd.append("--resume")
            cmd.extend(["--shard-cache-override", str(s_cache)])

            env = dict(os.environ)
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            if not args.mock and gpu_id != "cpu":
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            if idx > 0 and stagger_sec > 0.0:
                time.sleep(stagger_sec)
            log_f = s_log.open("w", encoding="utf-8")
            try:
                proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, text=True)
            finally:
                log_f.close()
            procs.append((sid, proc, s_log))

        failed_shards: list[int] = []
        for sid, proc, s_log in procs:
            rc = proc.wait()
            if rc != 0:
                print(f"[ext_ifbench] Shard {sid} exited {rc} (see {s_log})", flush=True)
                failed_shards.append(sid)

        if not failed_shards:
            break
        attempt += 1
        if attempt > max_retries:
            raise RuntimeError(
                f"ext_ifbench_ood failed on shards {failed_shards} after {max_retries} retries"
            )
        cur_safety = round(cur_safety * 1.25, 3)
        if attempt >= 2 and cur_wpg.lower() != "1":
            cur_wpg = "1"
        pending_shards = failed_shards
        print(
            f"[ext_ifbench] Retrying {len(pending_shards)} failed shard(s) with safety_mult={cur_safety:.2f}, wpg={cur_wpg}",
            flush=True,
        )

    # Merge shard caches + existing canonical cache cleanly
    cache_path = out_dir / "ext_ifbench_ood_cache.jsonl"
    merged_by_uid: dict[str, dict[str, Any]] = {}
    if cache_path.is_file():
        for row in _load_jsonl(cache_path):
            merged_by_uid[str(row["item"]["uid"])] = row
    for sid in range(num_shards):
        sc = shard_dir / f"shard_{sid:02d}_cache.jsonl"
        if sc.is_file():
            for row in _load_jsonl(sc):
                merged_by_uid[str(row["item"]["uid"])] = row

    ordered_rows = [merged_by_uid[it["uid"]] for it in items if it["uid"] in merged_by_uid]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered_rows),
        encoding="utf-8",
    )
    if len(ordered_rows) != len(items):
        raise RuntimeError(
            f"Merged IFBench cache incomplete: {len(ordered_rows)}/{len(items)} items"
        )
    params, calib_src = resolve_calibration_params(
        args.model,
        pathlib.Path(args.out_dir),
        pathlib.Path(args.calib_json) if args.calib_json else None,
        mock=args.mock,
        allow_default_params=args.allow_default_params,
    )
    source_tag = str(args.dataset_path) if args.dataset_path else ("mock_ifbench_12" if args.mock else "hf:allenai/IFBench_test")
    summary = summarize_ifbench_rows(
        ordered_rows,
        model_id=args.model,
        source_tag=source_tag,
        calib_source=calib_src,
        params=params,
    )
    prov = get_version_provenance(ROOT)
    summary["code_version"] = prov["code_version"]
    summary["git_rev"] = prov["git_rev"]
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not args.keep_shards and shard_dir.is_dir():
        shutil.rmtree(shard_dir, ignore_errors=True)
    print(f"[ext_ifbench] Merged {len(ordered_rows)} IFBench rows -> {summary_path}", flush=True)


def _run_worker(
    args: argparse.Namespace,
    items: list[dict[str, Any]],
    out_dir: pathlib.Path,
    summary_path: pathlib.Path,
) -> None:
    params, calib_src = resolve_calibration_params(
        args.model,
        pathlib.Path(args.out_dir),
        pathlib.Path(args.calib_json) if args.calib_json else None,
        mock=args.mock,
        allow_default_params=args.allow_default_params,
    )
    is_subshard = bool(args.shard_cache_override) or int(args.num_shards) > 1
    cache_path = (
        pathlib.Path(args.shard_cache_override)
        if args.shard_cache_override
        else (out_dir / "ext_ifbench_ood_cache.jsonl")
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing_by_uid: dict[str, dict[str, Any]] = {}
    main_cache = out_dir / "ext_ifbench_ood_cache.jsonl"
    if args.resume and main_cache.is_file():
        for r in _load_jsonl(main_cache):
            existing_by_uid[str(r["item"]["uid"])] = r
    if args.resume and cache_path.is_file() and cache_path != main_cache:
        for r in _load_jsonl(cache_path):
            existing_by_uid[str(r["item"]["uid"])] = r

    my_items = _contiguous_shard_slice(items, int(args.shard_id), int(args.num_shards))
    todo = [it for it in my_items if it["uid"] not in existing_by_uid]
    registry = _load_ifbench_registry(mock=bool(args.mock))

    t0 = time.perf_counter()
    if todo:
        engine = _build_engine(
            args.model,
            args.device,
            use_mock=bool(args.mock),
            dtype_arg=args.dtype,
            local_files_only=bool(args.local_files_only),
        )
        try:
            with cache_path.open("a", encoding="utf-8") as f_out:
                for i, item in enumerate(todo, start=1):
                    row = _run_ifbench_item(engine, item, params, registry)
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f_out.flush()
                    existing_by_uid[item["uid"]] = row
                    if i % 16 == 0:
                        _release_accelerator_cache(str(getattr(engine, "device", "cpu")))
                    if i == len(todo) or i % 25 == 0:
                        print(
                            f"[ext_ifbench] Shard {args.shard_id}: {i}/{len(todo)} new items ({time.perf_counter()-t0:.1f}s)",
                            flush=True,
                        )
        finally:
            try:
                dev_s = str(getattr(engine, "device", "cpu"))
                del engine
                _release_accelerator_cache(dev_s)
            except Exception:
                pass

    if is_subshard:
        return

    ordered_rows = [existing_by_uid[it["uid"]] for it in items if it["uid"] in existing_by_uid]
    source_tag = str(args.dataset_path) if args.dataset_path else ("mock_ifbench_12" if args.mock else "hf:allenai/IFBench_test")
    summary = summarize_ifbench_rows(
        ordered_rows,
        model_id=args.model,
        source_tag=source_tag,
        calib_source=calib_src,
        params=params,
    )
    summary["wall_time_sec"] = round(time.perf_counter() - t0, 2)
    prov = get_version_provenance(ROOT)
    summary["code_version"] = prov["code_version"]
    summary["git_rev"] = prov["git_rev"]
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    b0s = summary["systems"]["B0"]
    b1bs = summary["systems"]["B1b"]
    m2s = summary["systems"]["M2"]
    print(
        f"[ext_ifbench] {args.model} (N={len(ordered_rows)}): "
        f"B0={b0s['prompt_strict_pct']:.1f}% ({b0s['mean_t_tot']:.0f} tok) | "
        f"B1b={b1bs['prompt_strict_pct']:.1f}% ({b1bs['mean_t_tot']:.0f} tok) | "
        f"M2={m2s['prompt_strict_pct']:.1f}% ({m2s['mean_t_tot']:.0f} tok, "
        f"OAA250={m2s['oaa_250_pct']:.1f}%) -> {summary_path}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate frozen B0/B1b/M2 on allenai/IFBench_test OOD constraints.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--dataset-path", default="")
    ap.add_argument("--calib-json", default="")
    ap.add_argument("--allow-default-params", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--check-cached", action="store_true")
    ap.add_argument("--shards", default="1")
    ap.add_argument("--workers-per-gpu", default="auto")
    ap.add_argument("--gpu-ids", default="")
    ap.add_argument("--wait-gpu-min", type=int, default=60)
    ap.add_argument("--stagger-sec", type=float, default=-1.0)
    ap.add_argument("--safety-mult", type=float, default=1.0)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--keep-shards", action="store_true")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-cache-override", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir) / model_slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "ext_ifbench_ood.json"
    cache_path = out_dir / "ext_ifbench_ood_cache.jsonl"

    items, _ = load_ifbench_items(
        dataset_path=pathlib.Path(args.dataset_path) if args.dataset_path else None,
        mock=bool(args.mock),
        limit=int(args.limit),
    )

    if args.check_cached:
        cached_uids = (
            {str(r["item"]["uid"]) for r in _load_jsonl(cache_path)}
            if cache_path.is_file()
            else set()
        )
        ok = summary_path.is_file() and all(it["uid"] in cached_uids for it in items)
        raise SystemExit(0 if ok else 1)

    is_sub = bool(args.shard_cache_override) or int(args.num_shards) > 1
    if not is_sub and (str(args.shards).lower() == "auto" or int(args.shards) > 1):
        _run_multi_shard(args, items, out_dir, summary_path)
    else:
        _run_worker(args, items, out_dir, summary_path)


if __name__ == "__main__":
    main()
