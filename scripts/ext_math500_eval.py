#!/usr/bin/env python3
"""Spotlight Extension U3: Competition-Level Hard Math Evaluation (`HuggingFaceH4/MATH-500`, Stratified Levels 1-5).

Evaluates frozen `B0`, `B1b` (B=2048), `M2_250` (interactive SLA), and `M2_Adapt` (adaptive B in {0, 250, 1024}
gated by Gate-2 uncertainty and Thinking Inertia) with ZERO recalibration on `MATH-500` (N=200, 40/level, seed=42).
Writes strictly disjoint `outputs/<model>/ext_math500_cache.jsonl` and `outputs/<model>/ext_math500_eval.json`.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ext_ifbench_ood import resolve_calibration_params
from scripts.gpu_pool import (
    _estimate_model_billions,
    prewarm_hf_model_weights,
    select_worker_gpu_slots,
)
from scripts.run_matrix import _build_engine, _load_jsonl, _release_accelerator_cache
from scripts.sync_results import model_slug
from scripts.versioning import get_version_provenance
from src.engine import run_gate1_lexical
from src.graders import has_4gram_repetition_loop, has_cot_spillover
from src.logfmt import colorize_log_line, fmt_progress_bar, gpu_color, hl, status_badge
from src.metrics import (
    compute_oaa_t,
    exact_and_midp_mcnemar,
    paired_bootstrap_diff_ci,
)
from src.routers import CalibratedRouterParams, score_gate2_probability

_MOCK_MATH500_ROWS: tuple[dict[str, Any], ...] = tuple(
    {
        "uid": f"math500_mock_{lvl}_{idx}",
        "problem": f"Compute {lvl * 10 + idx} + {lvl * 3} and place the final integer inside \\boxed{{}}.",
        "answer": str(lvl * 13 + idx),
        "subject": "Algebra",
        "level": lvl,
    }
    for lvl in (1, 2, 3, 4, 5)
    for idx in (1, 2, 3)
)


def extract_last_boxed(text: str) -> str | None:
    """Extracts the content of the last `\\boxed{...}` or `\\fbox{...}` with balanced brace tracking."""
    if not text:
        return None
    idx = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if idx < 0:
        m = re.findall(r"\\boxed\s+([^\s.$]+)", text)
        return m[-1].strip() if m else None
    brace_open = text.find("{", idx)
    if brace_open < 0:
        return None
    depth = 0
    for pos in range(brace_open, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_open + 1 : pos].strip()
    return text[brace_open + 1 :].strip()


def extract_math_candidate(text: str) -> str:
    """Extracts boxed answer or falls back to `#### <ans>` / final answer phrase / trailing expression."""
    s = str(text or "").strip()
    if not s:
        return ""
    boxed = extract_last_boxed(s)
    if boxed is not None and boxed != "":
        return boxed
    m_hash = re.findall(r"####\s*([^\n]+)", s)
    if m_hash:
        return m_hash[-1].strip()
    m_ans = re.findall(
        r"(?:final answer is|answer is|therefore,?\s+the answer is)\s*:?\s*\$?([^\n$]+)\$?",
        s,
        flags=re.IGNORECASE,
    )
    if m_ans:
        return m_ans[-1].strip().rstrip(".")
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", s)
    return nums[-1] if nums else s.splitlines()[-1].strip()


def _normalize_latex_expr(expr: str) -> str:
    s = str(expr or "").strip()
    s = s.replace("$", "").strip()
    s = re.sub(r"\\(?:left|right|!|,|;|quad|qquad|displaystyle)\b", "", s)
    s = re.sub(r"\\(?:text|mathrm|textbf|mbox|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\frac\s*([0-9a-zA-Z])\s*([0-9a-zA-Z])", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s*([0-9a-zA-Z])", r"sqrt(\1)", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = s.replace("^\\circ", "").replace("\\circ", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = s.rstrip(".")
    s = re.sub(r"\s+", "", s)
    while len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        depth = 0
        valid = True
        for idx, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and idx < len(s) - 1:
                    valid = False
                    break
        if valid and depth == 0:
            s = s[1:-1]
        else:
            break
    return s


def _try_numeric_eval(norm_s: str) -> float | None:
    expr = norm_s.replace("\\pi", "pi").replace("^", "**")
    expr = re.sub(r"(\d)(pi|sqrt\()", r"\1*\2", expr)
    if not re.fullmatch(r"[0-9+\-*/().pisqrt]+", expr):
        return None
    try:
        val = eval(expr, {"__builtins__": {}}, {"pi": math.pi, "sqrt": math.sqrt})  # noqa: S307
        f_val = float(val)
        return f_val if math.isfinite(f_val) else None
    except Exception:
        return None


def grade_math500_answer(pred_text: str, gold_answer: str) -> bool:
    """Grades competition math response via exact normalized LaTeX, numeric float, or optional SymPy."""
    cand = extract_math_candidate(pred_text)
    if not cand:
        return False
    n_pred = _normalize_latex_expr(cand)
    n_gold = _normalize_latex_expr(gold_answer)
    if not n_pred or not n_gold:
        return False
    if n_pred == n_gold or n_pred.lower() == n_gold.lower():
        return True
    v_pred = _try_numeric_eval(n_pred)
    v_gold = _try_numeric_eval(n_gold)
    if v_pred is not None and v_gold is not None:
        if abs(v_pred - v_gold) <= 1e-6 * max(1.0, abs(v_gold)):
            return True
    try:
        import sympy as sp

        sp_p = sp.sympify(n_pred.replace("\\pi", "pi").replace("^", "**"))
        sp_g = sp.sympify(n_gold.replace("\\pi", "pi").replace("^", "**"))
        if sp.simplify(sp_p - sp_g) == 0:
            return True
    except Exception:
        pass
    return False


def load_stratified_math500(
    *,
    per_level: int = 40,
    seed: int = 42,
    mock: bool = False,
    dataset_path: pathlib.Path | None = None,
) -> list[dict[str, Any]]:
    """Loads deterministic stratified MATH-500 subset (`per_level` items across Levels 1..5)."""
    if mock:
        return [
            {
                "uid": r["uid"],
                "slice": f"math_level_{r['level']}",
                "level": int(r["level"]),
                "subject": r["subject"],
                "prompt": f"{r['problem']}\nPut your final answer within \\boxed{{}}.",
                "gold": r["answer"],
            }
            for r in _MOCK_MATH500_ROWS
        ]
    raw_rows: list[dict[str, Any]] = []
    if dataset_path and dataset_path.is_file():
        raw_rows = _load_jsonl(dataset_path)
    else:
        from datasets import load_dataset

        last_err: Exception | None = None
        for ds_id, split_name in (
            ("HuggingFaceH4/MATH-500", "test"),
            ("Qwen/MATH-500", "test"),
        ):
            try:
                ds = load_dataset(ds_id, split=split_name)
                raw_rows = [dict(x) for x in ds]
                if raw_rows:
                    break
            except Exception as exc:
                last_err = exc
        if not raw_rows:
            raise RuntimeError(f"Unable to load MATH-500 dataset: {last_err}")

    by_level: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    for idx, row in enumerate(raw_rows):
        lvl_raw = str(row.get("level", "3"))
        m_lvl = re.search(r"([1-5])", lvl_raw)
        lvl = int(m_lvl.group(1)) if m_lvl else 3
        uid = str(row.get("unique_id") or row.get("id") or f"math500_{idx:04d}").replace("/", "_")
        prob = str(row.get("problem") or row.get("question") or "").strip()
        ans = str(row.get("answer") or extract_last_boxed(str(row.get("solution", ""))) or "").strip()
        if not prob or not ans:
            continue
        by_level[lvl].append(
            {
                "uid": f"math500_{uid}",
                "slice": f"math_level_{lvl}",
                "level": lvl,
                "subject": str(row.get("subject", "Math")),
                "prompt": f"{prob}\nPlease reason concise and put your final answer within \\boxed{{}}.",
                "gold": ans,
            }
        )

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for lvl in (1, 2, 3, 4, 5):
        pool = sorted(by_level[lvl], key=lambda x: x["uid"])
        rng.shuffle(pool)
        selected.extend(sorted(pool[:per_level], key=lambda x: x["uid"]))
    return selected


def _pack_math_run(item: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    a_fin = str(raw.get("a_final", ""))
    t_th = str(raw.get("t_think", ""))
    z = 1 if grade_math500_answer(a_fin, str(item["gold"])) else 0
    return {
        "a_final": a_fin,
        "t_think": t_th,
        "z": z,
        "correct": bool(z == 1),
        "T_probe": int(raw.get("T_probe", 0)),
        "T_think": int(raw.get("T_think", 0)),
        "T_wasted": int(raw.get("T_wasted", 0)),
        "T_inj": int(raw.get("T_inj", 0)),
        "T_final": int(raw.get("T_final", 0)),
        "T_total": int(raw.get("T_total", 0)),
        "calls": int(raw.get("calls", 1)),
        "truncated": bool(raw.get("truncated", False)),
        "think_exhausted": bool(raw.get("think_exhausted", False)),
        "has_loop": bool(has_4gram_repetition_loop(f"{t_th}\n{a_fin}")),
        "has_cot_spillover": bool(has_cot_spillover(a_fin)),
        "latency_ms": float(raw.get("latency_ms", 0.0)),
    }


def _run_math_item(
    engine: Any,
    item: dict[str, Any],
    params: CalibratedRouterParams,
    *,
    mock: bool = False,
) -> dict[str, Any]:
    prompt_str = item["prompt"]
    if mock:
        lvl = int(item["level"])
        gold = str(item["gold"])
        b0 = _pack_math_run(
            item,
            {"a_final": f"\\boxed{{{gold}}}" if lvl <= 2 else "\\boxed{0}", "T_final": 40, "T_total": 40},
        )
        b1b = _pack_math_run(
            item,
            {
                "t_think": "step " * 50,
                "a_final": f"\\boxed{{{gold}}}" if lvl <= 4 else "",
                "T_think": 900,
                "T_final": 40,
                "T_total": 940,
                "think_exhausted": lvl == 5,
            },
        )
        probe = {"T_probe": 16, "H_first": 0.4, "H_mean": 0.35 * lvl, "H_max": 0.4 * lvl, "I_inertia": int(lvl >= 3), "latency_ms": 1.0}
        fast_done = dict(b0)
        m2_250 = _pack_math_run(
            item,
            {"t_think": "plan", "a_final": f"\\boxed{{{gold}}}" if lvl <= 3 else "\\boxed{0}", "T_think": 200, "T_final": 40, "T_total": 240, "truncated": lvl >= 3},
        )
        m2_1024 = _pack_math_run(
            item,
            {"t_think": "deep plan", "a_final": f"\\boxed{{{gold}}}", "T_think": 520, "T_final": 45, "T_total": 565, "truncated": False},
        )
        return {"item": item, "probe": probe, "b0": b0, "fast_done": fast_done, "b1b": b1b, "m2_b250": m2_250, "m2_b1024": m2_1024}

    g1 = run_gate1_lexical(prompt_str)
    probe_raw = engine.run_gate2_fast_probe(prompt_str, g1=g1, probe_tokens=32)
    fast_raw = engine.complete_fast_probe(dict(probe_raw), max_ans_tokens=1024)
    b0_rec = _pack_math_run(item, fast_raw)

    b1b_s1_cache: dict[str, Any] = {}
    b1b_raw = engine.run_capped_think_and_anchor(
        prompt_str,
        budget_b=2048,
        anchor_clause="",
        max_ans_tokens=1024,
        allow_budget_force=False,
        out_s1_cache=b1b_s1_cache,
    )
    b1b_rec = _pack_math_run(item, b1b_raw)

    b_250 = max(32, 250 - int(probe_raw["T_probe"]))
    b_1024 = max(128, 1024 - int(probe_raw["T_probe"]))
    s1_ref = b1b_s1_cache if float(getattr(engine, "temperature", 0.0)) == 0.0 else None
    m2_250 = _pack_math_run(
        item,
        engine.run_capped_think_and_anchor(
            prompt_str,
            budget_b=b_250,
            anchor_clause="",
            max_ans_tokens=1024,
            anchor_locus="post_stripped",
            precomputed_b1b_s1=s1_ref,
        ),
    )
    m2_1024 = _pack_math_run(
        item,
        engine.run_capped_think_and_anchor(
            prompt_str,
            budget_b=b_1024,
            anchor_clause="",
            max_ans_tokens=1024,
            anchor_locus="post_stripped",
            precomputed_b1b_s1=s1_ref,
        ),
    )
    probe_clean = {k: v for k, v in probe_raw.items() if k not in ("_kv_state", "base_pref", "probe_raw_text")}
    return {
        "item": item,
        "probe": probe_clean,
        "b0": b0_rec,
        "fast_done": dict(b0_rec),
        "b1b": b1b_rec,
        "m2_b250": m2_250,
        "m2_b1024": m2_1024,
    }


def _merge_probe_slow(probe: dict[str, Any], slow: dict[str, Any]) -> dict[str, Any]:
    out = dict(slow)
    tp = int(probe.get("T_probe", 0))
    out["T_probe"] = tp
    out["T_total"] = tp + int(slow["T_total"])
    out["calls"] = 2
    out["latency_ms"] = float(probe.get("latency_ms", 0.0)) + float(slow.get("latency_ms", 0.0))
    return out


def summarize_math500_rows(
    rows: list[dict[str, Any]],
    model_id: str,
    params: CalibratedRouterParams,
    calib_source: str,
) -> dict[str, Any]:
    items = [r["item"] for r in rows]
    sys_runs: dict[str, list[dict[str, Any]]] = {"B0": [], "B1b": [], "M2_250": [], "M2_Adapt": []}
    for r in rows:
        g1 = run_gate1_lexical(r["item"]["prompt"])
        probe = r["probe"]
        p_esc = score_gate2_probability(g1, probe, params)
        i_in = int(probe.get("I_inertia") or 0)
        h_max = float(probe.get("H_max") or 0.0) if np.isfinite(float(probe.get("H_max") or 0.0)) else 0.0

        esc_250 = (p_esc >= params.tau_prob) or (i_in == 1)
        m2_250 = _merge_probe_slow(probe, r["m2_b250"]) if esc_250 else r["fast_done"]

        # M2_Adapt: fast-exits on low-entropy confident items; escalates to B=250 on moderate uncertainty
        # and B=1024 when Thinking Inertia or high entropy signals multi-step competition derivation.
        if i_in == 1 or h_max >= max(0.75, float(params.tau_h_high)):
            m2_ad = _merge_probe_slow(probe, r["m2_b1024"])
        elif p_esc >= params.tau_prob or h_max >= 0.45:
            m2_ad = _merge_probe_slow(probe, r["m2_b250"])
        else:
            m2_ad = r["fast_done"]

        sys_runs["B0"].append(r["b0"])
        sys_runs["B1b"].append(r["b1b"])
        sys_runs["M2_250"].append(m2_250)
        sys_runs["M2_Adapt"].append(m2_ad)

    systems_out: dict[str, Any] = {}
    for sk, rlist in sys_runs.items():
        by_lvl: dict[str, list[int]] = {}
        for it, rk in zip(items, rlist):
            by_lvl.setdefault(f"level_{it['level']}", []).append(int(rk["z"]))
        lvl_acc = {lk: 100.0 * float(np.mean(lv)) for lk, lv in sorted(by_lvl.items())}
        z_list = [int(x["z"]) for x in rlist]
        td_list = [int(x.get("T_probe", 0)) + int(x.get("T_think", 0)) for x in rlist]
        acc_pct = 100.0 * float(np.mean(z_list))
        mean_tok = float(np.mean([float(x["T_total"]) for x in rlist]))
        systems_out[sk] = {
            "n": len(rlist),
            "acc_pct": acc_pct,
            "level_acc_pct": lvl_acc,
            "easy_l1_l2_pct": float(np.mean([lvl_acc.get("level_1", 0.0), lvl_acc.get("level_2", 0.0)])),
            "hard_l4_l5_pct": float(np.mean([lvl_acc.get("level_4", 0.0), lvl_acc.get("level_5", 0.0)])),
            "oaa_250_pct": 100.0 * compute_oaa_t(z_list, td_list, threshold_t=250),
            "oaa_1024_pct": 100.0 * compute_oaa_t(z_list, td_list, threshold_t=1024),
            "u_005_pct": 100.0 * float(np.mean([float(x["z"]) - 0.05 * (float(x["T_total"]) / 250.0) for x in rlist])),
            "mean_t_tot": mean_tok,
            "loop_rate_pct": 100.0 * float(np.mean([int(bool(x.get("has_loop"))) for x in rlist])),
            "exhaust_rate_pct": 100.0 * float(np.mean([int(bool(x.get("think_exhausted"))) for x in rlist])),
        }

    z_map = {k: [int(x["z"]) for x in v] for k, v in sys_runs.items()}
    mcnemar_out: dict[str, Any] = {}
    for cand in ("M2_250", "M2_Adapt"):
        for base in ("B0", "B1b"):
            mc = exact_and_midp_mcnemar(z_map[base], z_map[cand])
            d_mean, ci_lo, ci_hi = paired_bootstrap_diff_ci(
                np.asarray(z_map[base], dtype=np.float64),
                np.asarray(z_map[cand], dtype=np.float64),
                n_boot=2000,
                seed=42,
            )
            mcnemar_out[f"{cand}_vs_{base}"] = {
                **mc,
                "delta_acc_pp": 100.0 * d_mean,
                "ci95_acc_pp": [100.0 * ci_lo, 100.0 * ci_hi],
                "tok_savings_pct": 100.0
                * (1.0 - float(systems_out[cand]["mean_t_tot"]) / max(1.0, float(systems_out[base]["mean_t_tot"]))),
            }

    prov = get_version_provenance(ROOT)
    return {
        "model": model_id,
        "code_version": prov["code_version"],
        "git_rev": prov["git_rev"],
        "calibration_source": calib_source,
        "n_problems": len(rows),
        "systems": systems_out,
        "mcnemar": mcnemar_out,
    }


def _contiguous_slice(items: list[dict[str, Any]], shard_id: int, num_shards: int) -> list[dict[str, Any]]:
    n = len(items)
    return items[(shard_id * n) // num_shards : ((shard_id + 1) * n) // num_shards]


def _run_worker(
    args: argparse.Namespace,
    items: list[dict[str, Any]],
    model_dir: pathlib.Path,
) -> None:
    params, calib_src = resolve_calibration_params(
        args.model, pathlib.Path(args.out_dir), None, mock=bool(args.mock), allow_default_params=True
    )
    main_cache = model_dir / "ext_math500_cache.jsonl"
    cache_path = pathlib.Path(args.shard_cache_override) if args.shard_cache_override else main_cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict[str, Any]] = {}
    if main_cache.is_file():
        for r in _load_jsonl(main_cache):
            existing[str(r["item"]["uid"])] = r
    if cache_path.is_file() and cache_path != main_cache:
        for r in _load_jsonl(cache_path):
            existing[str(r["item"]["uid"])] = r

    my_items = _contiguous_slice(items, int(args.shard_id), int(args.num_shards))
    todo = [it for it in my_items if it["uid"] not in existing]
    if todo:
        t0 = time.perf_counter()
        engine = (
            None
            if args.mock
            else _build_engine(
                args.model,
                args.device,
                use_mock=False,
                dtype_arg=args.dtype,
                local_files_only=bool(args.offline),
            )
        )
        try:
            with cache_path.open("a", encoding="utf-8") as f_out:
                for i, it in enumerate(todo, start=1):
                    row = _run_math_item(engine, it, params, mock=bool(args.mock))
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f_out.flush()
                    existing[it["uid"]] = row
                    if engine is not None and i % 12 == 0:
                        _release_accelerator_cache(str(getattr(engine, "device", "cpu")))
                    if i == len(todo) or i % 20 == 0:
                        print(
                            f"[ext_math500] Shard {args.shard_id}: {i}/{len(todo)} problems ({time.perf_counter()-t0:.1f}s)",
                            flush=True,
                        )
        finally:
            if engine is not None:
                dev_s = str(getattr(engine, "device", "cpu"))
                del engine
                _release_accelerator_cache(dev_s)

    if int(args.num_shards) == 1 and not args.shard_cache_override:
        ordered = [existing[it["uid"]] for it in items]
        main_cache.write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ordered),
            encoding="utf-8",
        )
        summary = summarize_math500_rows(ordered, args.model, params, calib_src)
        out_json = model_dir / "ext_math500_eval.json"
        out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        s_ad = summary["systems"]["M2_Adapt"]
        s_b0 = summary["systems"]["B0"]
        s_b1 = summary["systems"]["B1b"]
        ad_lbl = hl(f"{s_ad['acc_pct']:.1f}%", "green")
        print(
            f"{hl('[ext_math500]', 'cyan')} {status_badge('done', 'DONE')} "
            f"{args.model} (N={len(ordered)}): B0={s_b0['acc_pct']:.1f}% | B1b={s_b1['acc_pct']:.1f}% | "
            f"M2_Adapt={ad_lbl} ({s_ad['mean_t_tot']:.0f} tok) -> {out_json}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified MATH-500 (Levels 1-5) hard reasoning evaluation.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--per-level", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--shards", type=str, default="auto")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-cache-override", type=str, default="")
    parser.add_argument("--gpu-ids", type=str, default="")
    parser.add_argument("--workers-per-gpu", type=str, default=os.environ.get("CBARR_WORKERS_PER_GPU", "auto"))
    parser.add_argument("--wait-gpu-min", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--keep-shards", action="store_true")
    args = parser.parse_args()

    model_dir = pathlib.Path(args.out_dir) / model_slug(args.model)
    model_dir.mkdir(parents=True, exist_ok=True)
    ds_p = pathlib.Path(args.dataset_path) if args.dataset_path else None
    items = load_stratified_math500(
        per_level=int(args.per_level), seed=int(args.seed), mock=bool(args.mock), dataset_path=ds_p
    )

    # Save frozen local problem list so shard workers never hit HF Hub concurrently
    frozen_ds = model_dir / "shards" / "math500_items_frozen.jsonl"
    if not args.shard_cache_override and int(args.num_shards) == 1:
        frozen_ds.parent.mkdir(parents=True, exist_ok=True)
        frozen_ds.write_text(
            "".join(
                json.dumps(
                    {
                        "unique_id": it["uid"].removeprefix("math500_"),
                        "level": it["level"],
                        "subject": it["subject"],
                        "problem": it["prompt"].rsplit("\n", 1)[0],
                        "answer": it["gold"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for it in items
            ),
            encoding="utf-8",
        )

    if args.shard_cache_override or int(args.num_shards) > 1:
        _run_worker(args, items, model_dir)
        return

    main_cache = model_dir / "ext_math500_cache.jsonl"
    existing = {str(r["item"]["uid"]): r for r in _load_jsonl(main_cache)} if main_cache.is_file() else {}
    remaining = [it for it in items if it["uid"] not in existing]
    if not remaining or args.mock or str(args.shards).strip() == "1":
        args.num_shards = 1
        args.shard_id = 0
        _run_worker(args, items, model_dir)
        return

    cands = [x.strip() for x in args.gpu_ids.split(",") if x.strip()] if args.gpu_ids else None
    worker_map = select_worker_gpu_slots(
        args.model,
        candidate_ids=cands,
        workers_per_gpu=args.workers_per_gpu,
        max_gpus=8,
        wait_timeout_sec=int(args.wait_gpu_min) * 60,
        stage="matrix",
    )
    n_slots = max(1, len(worker_map)) if str(args.shards).strip().lower() == "auto" else min(max(1, int(args.shards)), max(1, len(worker_map or ["0"])))
    if n_slots <= 1:
        args.num_shards = 1
        args.shard_id = 0
        _run_worker(args, items, model_dir)
        return

    num_shards = min(len(items), n_slots * 2)
    prewarmed = prewarm_hf_model_weights(args.model, offline=bool(args.offline))
    shard_dir = model_dir / "shards" / "math500"
    shard_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"{hl('[ext_math500]', 'cyan')} Multi-shard launch: {hl(f'{n_slots} GPU slots', 'green')} "
        f"({num_shards} micro-shards, {len(remaining)} remaining problems) across {worker_map[:n_slots]}",
        flush=True,
    )

    pending = list(range(num_shards))
    attempt = 0
    pb = _estimate_model_billions(args.model)
    stagger_sec = 4.0 if pb >= 7.0 else 1.5
    while pending:
        eff_slots = max(1, min(len(worker_map), n_slots, len(pending)))
        todo_q: deque[int] = deque(pending)
        active: dict[int, tuple[int, subprocess.Popen[str], pathlib.Path]] = {}

        def _launch(sid: int, slot_i: int) -> None:
            gpu_id = worker_map[slot_i % len(worker_map)]
            s_cache = shard_dir / f"shard_{sid:02d}_cache.jsonl"
            s_log = shard_dir / f"shard_{sid:02d}_attempt{attempt}.log"
            cmd = [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                "--model", args.model,
                "--out-dir", str(args.out_dir),
                "--dataset-path", str(frozen_ds),
                "--per-level", str(args.per_level),
                "--seed", str(args.seed),
                "--device", "cuda:0",
                "--dtype", args.dtype,
                "--shard-id", str(sid),
                "--num-shards", str(num_shards),
                "--shard-cache-override", str(s_cache),
            ]
            if args.offline or prewarmed:
                cmd.append("--offline")
            env = dict(os.environ)
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            if prewarmed:
                env["HF_HUB_OFFLINE"] = "1"
            if gpu_id != "cpu":
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            with s_log.open("w", encoding="utf-8") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, text=True)
            active[slot_i] = (sid, proc, s_log)

        for slot_i in range(eff_slots):
            if slot_i > 0:
                time.sleep(stagger_sec)
            _launch(todo_q.popleft(), slot_i)

        failed: list[int] = []
        done_cnt = 0
        total_wave = len(pending)
        last_prog = time.perf_counter()
        t_start = last_prog
        while active:
            for slot_i in list(active.keys()):
                sid, proc, s_log = active[slot_i]
                rc = proc.poll()
                if rc is None:
                    continue
                dev_lbl = worker_map[slot_i % len(worker_map)]
                if rc == 0:
                    done_cnt += 1
                    if todo_q and not failed:
                        nxt = todo_q.popleft()
                        print(
                            f"{hl('[ext_math500]', 'cyan')} {status_badge('done', 'DONE')} "
                            f"{hl(f'GPU{slot_i}(dev={dev_lbl})', gpu_color(slot_i))} shard {sid}/{num_shards} -> {nxt}/{num_shards}",
                            flush=True,
                        )
                        _launch(nxt, slot_i)
                    else:
                        del active[slot_i]
                else:
                    del active[slot_i]
                    failed.append(sid)
                    failed.extend(list(todo_q))
                    todo_q.clear()
                    print(f"[ext_math500] Shard {sid} exited {rc} ({s_log})", flush=True)
                    if s_log.is_file():
                        for tln in s_log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]:
                            print(f"  {colorize_log_line(tln)}", flush=True)
            now_t = time.perf_counter()
            if active and (now_t - last_prog) >= 12.0:
                done_rows = sum(
                    sum(1 for ln in (shard_dir / f"shard_{s:02d}_cache.jsonl").open("rb") if ln.strip())
                    for s in range(num_shards)
                    if (shard_dir / f"shard_{s:02d}_cache.jsonl").is_file()
                )
                pbar = fmt_progress_bar(min(len(items), len(existing) + done_rows), len(items), width=18)
                print(
                    f"[MATH-500 | {args.model}] elapsed={int(now_t - t_start)}s | "
                    f"shards={done_cnt}/{total_wave} | {pbar} items={min(len(items), len(existing)+done_rows)}/{len(items)}",
                    flush=True,
                )
                last_prog = now_t
            time.sleep(0.5)

        if not failed:
            break
        attempt += 1
        if attempt > int(args.max_retries):
            raise RuntimeError(f"ext_math500_eval failed on shards {failed}")
        pending = failed

    for sid in range(num_shards):
        sc = shard_dir / f"shard_{sid:02d}_cache.jsonl"
        if sc.is_file():
            for r in _load_jsonl(sc):
                existing[str(r["item"]["uid"])] = r
    ordered = [existing[it["uid"]] for it in items if it["uid"] in existing]
    if len(ordered) != len(items):
        raise RuntimeError(f"Incomplete MATH-500 merge: {len(ordered)}/{len(items)}")
    main_cache.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ordered),
        encoding="utf-8",
    )
    if not args.keep_shards and shard_dir.is_dir():
        shutil.rmtree(shard_dir, ignore_errors=True)
    if frozen_ds.is_file():
        frozen_ds.unlink()

    params, calib_src = resolve_calibration_params(
        args.model, pathlib.Path(args.out_dir), None, mock=False, allow_default_params=True
    )
    summary = summarize_math500_rows(ordered, args.model, params, calib_src)
    out_json = model_dir / "ext_math500_eval.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    s_ad = summary["systems"]["M2_Adapt"]
    s_b0 = summary["systems"]["B0"]
    s_b1 = summary["systems"]["B1b"]
    ad_lbl = hl(f"{s_ad['acc_pct']:.1f}%", "green")
    print(
        f"{hl('[ext_math500]', 'cyan')} {status_badge('done', 'DONE')} "
        f"{args.model} (N={len(ordered)}): B0={s_b0['acc_pct']:.1f}% | B1b={s_b1['acc_pct']:.1f}% | "
        f"M2_Adapt={ad_lbl} ({s_ad['mean_t_tot']:.0f} tok) -> {out_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
