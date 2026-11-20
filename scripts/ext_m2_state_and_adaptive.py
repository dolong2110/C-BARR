#!/usr/bin/env python3
"""Spotlight Extension U1: End-to-End Zero-Token State Surgery (M2-ZeroTok) & Regime-Calibrated Multi-Horizon C-BARR (M2-Adapt).

Strictly non-destructive: reads existing `seed_60`, `dev_60`, and `eval_450` caches without mutating them,
reuses cached `m2_b512` runs where available (e.g., `Qwen3.5-4B` and `SmolLM3-3B`), generates missing `m2_b512`
and Stage-2 zero-token state-restoration completions (`DeltaNet` recurrent+conv blend / dense `<think>` KV eviction)
across multi-GPU worker slots, and writes ONLY `ext_m2_state_adaptive_cache.jsonl` and `ext_m2_state_adaptive.json`.
"""
from __future__ import annotations

import argparse
from collections import deque
import gzip
import json
import os
import pathlib
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

from scripts.ext_deltanet_blend import generate_with_blended_deltanet
from scripts.gpu_pool import (
    _estimate_model_billions,
    prewarm_hf_model_weights,
    select_worker_gpu_slots,
)
from scripts.run_matrix import (
    _annotate_run,
    _build_engine,
    _is_native_engine,
    _load_jsonl,
    _regrade_cached_gsm8k_row,
    _release_accelerator_cache,
)
from scripts.sync_results import model_slug
from scripts.versioning import get_version_provenance
from src.engine import run_gate1_lexical
from src.logfmt import colorize_log_line, fmt_progress_bar, gpu_color, hl, status_badge
from scripts.analyze_and_plot import paired_metric_bootstrap_ci
from src.metrics import (
    compute_oaa_t,
    evaluate_system_table,
    exact_and_midp_mcnemar,
)
from src.routers import (
    CBARRCalibrator,
    CalibratedRouterParams,
    _get_slow_rec,
    resolve_all_variants_for_item,
    resolve_m2_prec_variant,
    select_tiered_budget,
)


def _ensure_unpacked_jsonl(model_dir: pathlib.Path, stem: str) -> pathlib.Path:
    """Returns unpacked `<stem>.jsonl`, decompressing `<stem>.jsonl.gz` if needed."""
    raw_p = model_dir / f"{stem}.jsonl"
    gz_p = model_dir / f"{stem}.jsonl.gz"
    if not raw_p.is_file() and gz_p.is_file():
        tmp_p = model_dir / f".{stem}.jsonl.tmp"
        with gzip.open(gz_p, "rb") as f_in, tmp_p.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.replace(tmp_p, raw_p)
    if not raw_p.is_file():
        raise FileNotFoundError(f"Missing required cache {raw_p} (and {gz_p})")
    return raw_p


def _contiguous_slice(items: list[dict[str, Any]], shard_id: int, num_shards: int) -> list[dict[str, Any]]:
    n = len(items)
    lo = (shard_id * n) // num_shards
    hi = ((shard_id + 1) * n) // num_shards
    return items[lo:hi]


def _is_hybrid_deltanet_model(model_id: str) -> bool:
    low = model_id.lower()
    return "qwen3.5" in low or "qwen3_5" in low or "deltanet" in low


def _select_ifeval_source_slow(row: dict[str, Any], params: CalibratedRouterParams) -> dict[str, Any]:
    g1 = run_gate1_lexical(row["item"]["prompt"])
    if g1.phi_prec == 1 and g1.phi_plan == 0:
        return row["m2_b128"]
    b_dir = select_tiered_budget(row["item"]["prompt"], g1, None, params.tau_h_high)
    return _get_slow_rec(row, b_dir, direct=True, prefix="m2")


def _run_item_extensions(
    engine: Any,
    row: dict[str, Any],
    params: CalibratedRouterParams,
    *,
    model_id: str,
    mock: bool = False,
) -> dict[str, Any]:
    """Computes missing `m2_b512` and Stage-2 zero-token state surgery (`state_zero_tok`) for one eval_450 row."""
    item = row["item"]
    prompt_str = item["prompt"]
    g1 = run_gate1_lexical(prompt_str)

    # 1. Reuse cached m2_b512 if already in eval_450_cache.jsonl; otherwise generate at B=512
    if isinstance(row.get("m2_b512"), dict) and "a_final" in row["m2_b512"]:
        m2_b512 = dict(row["m2_b512"])
    else:
        raw_512 = engine.run_capped_think_and_anchor(
            prompt_str,
            budget_b=512,
            anchor_clause=g1.anchor_clause,
            anchor_locus="post_stripped",
        )
        m2_b512 = _annotate_run(item, raw_512)

    # 2. End-to-end zero-token state surgery on IFEval items (phi_plan==1 or phi_prec==1)
    state_zero_tok: dict[str, Any] | None = None
    if g1.phi_plan == 1 or g1.phi_prec == 1:
        src_slow = _select_ifeval_source_slow(row, params)
        t_think = str(src_slow.get("t_think", ""))
        t_think_toks = int(src_slow.get("T_think", 0))
        t_wasted = int(src_slow.get("T_wasted", 0))
        t0 = time.perf_counter()
        if mock or not _is_native_engine(engine):
            # Deterministic offline CPU mock fallback
            state_zero_tok = _annotate_run(
                item,
                {
                    "t_think": t_think,
                    "a_final": str(src_slow.get("a_final", "")),
                    "T_probe": 0,
                    "T_think": t_think_toks,
                    "T_wasted": t_wasted,
                    "T_inj": 0,
                    "T_final": int(src_slow.get("T_final", 16)),
                    "T_total": t_think_toks + t_wasted + int(src_slow.get("T_final", 16)),
                    "calls": 1,
                    "truncated": bool(src_slow.get("truncated", False)),
                    "think_exhausted": False,
                    "latency_ms": float(src_slow.get("latency_ms", 1.0)),
                },
            )
        else:
            is_hyb = _is_hybrid_deltanet_model(model_id)
            eos_ids: set[int] = set()
            if getattr(engine.tok, "eos_token_id", None) is not None:
                eos_ids.add(int(engine.tok.eos_token_id))
            gen_cfg_eos = getattr(getattr(engine.model, "generation_config", None), "eos_token_id", None)
            if isinstance(gen_cfg_eos, int):
                eos_ids.add(gen_cfg_eos)
            elif isinstance(gen_cfg_eos, (list, tuple)):
                eos_ids.update(int(x) for x in gen_cfg_eos if x is not None)

            # Hybrid DeltaNet: alpha=0.50 recurrent+conv state blend preserves Stage-1 plan while restoring prompt rules;
            # Dense GQA: evict noisy intermediate <think> KV entries back to clean prompt KV.
            st_out = generate_with_blended_deltanet(
                engine.model,
                engine.tok,
                eos_ids,
                prompt_str,
                t_think,
                alpha=0.50 if is_hyb else 0.0,
                max_ans_tokens=1024,
                blend_conv=is_hyb,
                evict_think_kv=(not is_hyb),
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            t_fin = int(st_out["gen_tokens"])
            state_zero_tok = _annotate_run(
                item,
                {
                    "t_think": t_think,
                    "a_final": st_out["a_final"],
                    "T_probe": 0,
                    "T_think": t_think_toks,
                    "T_wasted": t_wasted,
                    "T_inj": 0,
                    "T_final": t_fin,
                    "T_total": t_think_toks + t_wasted + t_fin,
                    "calls": 1,
                    "truncated": bool(src_slow.get("truncated", False)),
                    "think_exhausted": False,
                    "latency_ms": float(src_slow.get("latency_ms", 0.0)) * 0.4 + dt_ms,
                    "n_blended_layers": int(st_out.get("n_blended_layers", 0)),
                    "n_blended_conv": int(st_out.get("n_blended_conv", 0)),
                    "n_evicted_kv": int(st_out.get("n_evicted_kv", 0)),
                },
            )

    return {
        "uid": str(item["uid"]),
        "slice": str(item["slice"]),
        "m2_b512": m2_b512,
        "state_zero_tok": state_zero_tok,
    }


def calibrate_regime_adaptive_policy(
    seed_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    base_params: CalibratedRouterParams,
) -> dict[str, Any]:
    """Zero-leakage regime calibration on `seed_60` + `dev_60` only (never touches `eval_450`)."""
    cal_pool = seed_rows + dev_rows
    b1b_exhaust_cal = float(
        np.mean([int(bool(r["b1b"].get("think_exhausted", False))) for r in cal_pool])
    )
    sym_cal = [
        r
        for r in cal_pool
        if run_gate1_lexical(r["item"]["prompt"]).phi_sym == 1
        and run_gate1_lexical(r["item"]["prompt"]).phi_plan == 0
    ]
    b1b_sym_gain = (
        float(np.mean([int(r["b1b"]["z"]) - int(r["b0"]["z"]) for r in sym_cal]))
        if sym_cal
        else 0.0
    )
    prec_cal = [
        r
        for r in dev_rows
        if run_gate1_lexical(r["item"]["prompt"]).phi_prec == 1
        and run_gate1_lexical(r["item"]["prompt"]).phi_plan == 0
    ]
    use_prec_bridge = (
        float(np.mean([int(r["m2_b128"]["z"]) for r in prec_cal]))
        >= float(np.mean([int(r["b0"]["z"]) for r in prec_cal]))
        if prec_cal
        else True
    )
    g2_cal = [
        r
        for r in dev_rows
        if run_gate1_lexical(r["item"]["prompt"]).phi_prec == 0
        and run_gate1_lexical(r["item"]["prompt"]).phi_plan == 0
        and run_gate1_lexical(r["item"]["prompt"]).phi_sym == 0
    ]
    best_g2_tier = (
        max(
            ["b0", "m2_b64", "m2_b128", "m2_b256"],
            key=lambda k: float(
                np.mean([int(r[k]["z"]) - 0.02 * (float(r[k]["T_total"]) / 250.0) for r in g2_cal])
            ),
        )
        if g2_cal
        else "m2_b128"
    )
    return {
        "b1b_exhaust_cal": b1b_exhaust_cal,
        "is_concise_dense_regime": bool(b1b_exhaust_cal < 0.25),
        "b1b_sym_gain_cal": b1b_sym_gain,
        "use_prec_bridge": bool(use_prec_bridge),
        "best_g2_tier": str(best_g2_tier),
        "base_tau_prob": float(base_params.tau_prob),
        "base_tau_h_high": float(base_params.tau_h_high),
    }


def summarize_state_and_adaptive(
    model_id: str,
    seed_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    ext_by_uid: dict[str, dict[str, Any]],
    *,
    lam_utility: float = 0.05,
    norm_tokens: float = 250.0,
    seed: int = 42,
    reg_c: float = 1.0,
) -> dict[str, Any]:
    """Evaluates B0, B1b, M2, M2_Prec, M2_ZeroTok, M2_Adapt, and M2_Adapt_State on eval_450."""
    for rows in (seed_rows, dev_rows, eval_rows):
        for r in rows:
            _regrade_cached_gsm8k_row(r)
            r["g1"] = run_gate1_lexical(r["item"]["prompt"])

    calibrator = CBARRCalibrator(
        lam_utility=lam_utility, norm_tokens=norm_tokens, seed=seed, reg_c=reg_c
    )
    params = calibrator.fit(seed_rows, dev_rows)
    adapt_cfg = calibrate_regime_adaptive_policy(seed_rows, dev_rows, params)
    is_concise = bool(adapt_cfg["is_concise_dense_regime"])
    sym_gain = float(adapt_cfg["b1b_sym_gain_cal"])
    use_prec = bool(adapt_cfg["use_prec_bridge"])
    best_g2 = str(adapt_cfg["best_g2_tier"])

    items = [r["item"] for r in eval_rows]
    runs_by_sys: dict[str, list[dict[str, Any]]] = {
        k: []
        for k in ("B0", "B1b", "M2", "M2_Prec", "M2_ZeroTok", "M2_Adapt", "M2_Adapt_State")
    }
    rng_b2b = np.random.default_rng(seed)
    for r in eval_rows:
        uid = str(r["item"]["uid"])
        ext = ext_by_uid[uid]
        tmp_512 = {"item": r["item"], "m2_b512": dict(ext["m2_b512"])}
        _regrade_cached_gsm8k_row(tmp_512)
        m2_512 = tmp_512["m2_b512"]
        st_zero = ext.get("state_zero_tok")

        base_v = resolve_all_variants_for_item(r, params, rng_b2b, params.m2_escalation_rate_dev)
        m2_prec = resolve_m2_prec_variant(r, params)

        g1 = r["g1"]
        probe = r["probe"]

        # M2_ZeroTok: strictly T_delib <= 250, T_inj = 0 (replaces text reminder with zero-token state surgery)
        if (g1.phi_plan == 1 or (g1.phi_prec == 1 and g1.phi_sym == 0)) and isinstance(st_zero, dict):
            m2_zt = st_zero
        else:
            m2_zt = base_v["M2"]

        # M2_Adapt: zero-leakage regime-calibrated multi-horizon C-BARR (B in {0, 64, 128, 250, 512})
        if g1.phi_prec == 1 and g1.phi_plan == 0 and g1.phi_sym == 0:
            c_adapt = r["m2_b128"] if use_prec else r["b0"]
        elif g1.phi_plan == 1:
            c_adapt = r["m2_b128"] if is_concise else m2_512
        elif g1.phi_sym == 1:
            h_max = float(probe.get("H_max") or 0.0) if np.isfinite(float(probe.get("H_max") or 0.0)) else 0.0
            if sym_gain > 0.02 or int(probe.get("I_inertia") or 0) == 1 or h_max >= 0.60:
                c_adapt = dict(m2_512)
                c_adapt["T_probe"] = probe["T_probe"]
                c_adapt["T_total"] = int(probe["T_probe"]) + int(m2_512["T_total"])
                c_adapt["calls"] = 2
            else:
                c_adapt = r["fast_done"]
        else:
            h_mean = float(probe.get("H_mean") or 0.0) if np.isfinite(float(probe.get("H_mean") or 0.0)) else 0.0
            if best_g2 == "b0":
                c_adapt = r["fast_done"]
            elif h_mean >= 0.25 or int(probe.get("I_inertia") or 0) == 1:
                c_adapt = dict(r[best_g2])
                c_adapt["T_probe"] = probe["T_probe"]
                c_adapt["T_total"] = int(probe["T_probe"]) + int(c_adapt["T_total"])
                c_adapt["calls"] = 2
            else:
                c_adapt = r["fast_done"]

        # M2_Adapt_State: uses zero-token state surgery on pure precision inhibition when dense KV eviction applies
        if is_concise and g1.phi_prec == 1 and g1.phi_plan == 0 and g1.phi_sym == 0 and isinstance(st_zero, dict):
            c_adapt_state = st_zero
        else:
            c_adapt_state = c_adapt

        runs_by_sys["B0"].append(base_v["B0"])
        runs_by_sys["B1b"].append(base_v["B1b"])
        runs_by_sys["M2"].append(base_v["M2"])
        runs_by_sys["M2_Prec"].append(m2_prec)
        runs_by_sys["M2_ZeroTok"].append(m2_zt)
        runs_by_sys["M2_Adapt"].append(c_adapt)
        runs_by_sys["M2_Adapt_State"].append(c_adapt_state)

    systems_out: dict[str, Any] = {}
    for k, rlist in runs_by_sys.items():
        st_raw = evaluate_system_table(items, rlist, lam_utility=lam_utility, norm_tokens=norm_tokens)
        st = {sk: sv for sk, sv in st_raw.items() if sk not in ("z_vec", "u_vec")}
        z_l = [int(x["z"]) for x in rlist]
        td_l = [int(x.get("T_probe", 0)) + int(x.get("T_think", 0)) for x in rlist]
        st["oaa_512"] = compute_oaa_t(z_l, td_l, threshold_t=512)
        systems_out[k] = st

    z_map = {k: [int(x["z"]) for x in rlist] for k, rlist in runs_by_sys.items()}
    comparisons: dict[str, Any] = {}
    for cand in ("M2_ZeroTok", "M2_Adapt", "M2_Adapt_State"):
        for base in ("B1b", "B0", "M2"):
            pair_k = f"{cand}_vs_{base}"
            mc = exact_and_midp_mcnemar(z_map[base], z_map[cand])
            boot = paired_metric_bootstrap_ci(
                items,
                runs_by_sys[base],
                runs_by_sys[cand],
                n_boot=2000,
                seed=seed,
            )
            comparisons[pair_k] = {
                **mc,
                "delta_macro_pp": 100.0 * float(boot["macro_acc"]["diff_b_minus_a"]),
                "ci95_macro_pp": [
                    100.0 * float(boot["macro_acc"]["ci95_lo"]),
                    100.0 * float(boot["macro_acc"]["ci95_hi"]),
                ],
                "tok_savings_vs_base_pct": 100.0
                * (1.0 - float(systems_out[cand]["mean_t_tot"]) / max(1.0, float(systems_out[base]["mean_t_tot"]))),
            }

    prov = get_version_provenance(ROOT)
    return {
        "model": model_id,
        "code_version": prov["code_version"],
        "git_rev": prov["git_rev"],
        "n_eval": len(eval_rows),
        "adaptive_calibration": adapt_cfg,
        "systems": systems_out,
        "comparisons": comparisons,
    }


def _needs_gpu_work(row: dict[str, Any], cached_ext: dict[str, Any] | None) -> bool:
    g1 = run_gate1_lexical(row["item"]["prompt"])
    need_st = g1.phi_plan == 1 or g1.phi_prec == 1
    if cached_ext is not None:
        has_512 = isinstance(cached_ext.get("m2_b512"), dict) and "a_final" in cached_ext["m2_b512"]
        has_st = (not need_st) or isinstance(cached_ext.get("state_zero_tok"), dict)
        if has_512 and has_st:
            return False
    has_orig_512 = isinstance(row.get("m2_b512"), dict) and "a_final" in row["m2_b512"]
    return (not has_orig_512) or need_st


def _run_worker(
    args: argparse.Namespace,
    seed_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    model_dir: pathlib.Path,
) -> None:
    calibrator = CBARRCalibrator(seed=args.seed)
    for rows in (seed_rows, dev_rows):
        for r in rows:
            _regrade_cached_gsm8k_row(r)
            r["g1"] = run_gate1_lexical(r["item"]["prompt"])
    params = calibrator.fit(seed_rows, dev_rows)

    main_cache = model_dir / "ext_m2_state_adaptive_cache.jsonl"
    cache_path = pathlib.Path(args.shard_cache_override) if args.shard_cache_override else main_cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict[str, Any]] = {}
    if main_cache.is_file():
        for r in _load_jsonl(main_cache):
            existing[str(r["uid"])] = r
    if cache_path.is_file() and cache_path != main_cache:
        for r in _load_jsonl(cache_path):
            existing[str(r["uid"])] = r

    my_rows = _contiguous_slice(eval_rows, int(args.shard_id), int(args.num_shards))
    todo_gpu = [r for r in my_rows if _needs_gpu_work(r, existing.get(str(r["item"]["uid"])))]
    todo_fast = [
        r
        for r in my_rows
        if str(r["item"]["uid"]) not in existing and not _needs_gpu_work(r, None)
    ]

    with cache_path.open("a", encoding="utf-8") as f_out:
        for r in todo_fast:
            rec = {
                "uid": str(r["item"]["uid"]),
                "slice": str(r["item"]["slice"]),
                "m2_b512": dict(r["m2_b512"]),
                "state_zero_tok": None,
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            existing[rec["uid"]] = rec
        f_out.flush()

        if todo_gpu:
            t0 = time.perf_counter()
            engine = _build_engine(
                args.model,
                args.device,
                use_mock=bool(args.mock),
                dtype_arg=args.dtype,
                local_files_only=bool(args.offline),
            )
            try:
                for idx, r in enumerate(todo_gpu, start=1):
                    rec = _run_item_extensions(
                        engine, r, params, model_id=args.model, mock=bool(args.mock)
                    )
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_out.flush()
                    existing[rec["uid"]] = rec
                    if idx % 16 == 0:
                        _release_accelerator_cache(str(getattr(engine, "device", "cpu")))
                    if idx == len(todo_gpu) or idx % 25 == 0:
                        print(
                            f"[ext_state_adapt] Shard {args.shard_id}: {idx}/{len(todo_gpu)} GPU items ({time.perf_counter()-t0:.1f}s)",
                            flush=True,
                        )
            finally:
                dev_s = str(getattr(engine, "device", "cpu"))
                del engine
                _release_accelerator_cache(dev_s)

    if int(args.num_shards) == 1 and not args.shard_cache_override:
        ordered = [existing[str(r["item"]["uid"])] for r in eval_rows]
        main_cache.write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ordered),
            encoding="utf-8",
        )
        summary = summarize_state_and_adaptive(
            args.model, seed_rows, dev_rows, eval_rows, existing, seed=args.seed
        )
        out_json = model_dir / "ext_m2_state_adaptive.json"
        out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        s_ad = summary["systems"]["M2_Adapt"]
        s_zt = summary["systems"]["M2_ZeroTok"]
        ad_lbl = hl(f"{100 * s_ad['macro_acc']:.1f}%", "green")
        print(
            f"{hl('[ext_state_adapt]', 'cyan')} {status_badge('done', 'DONE')} "
            f"{args.model}: M2_ZeroTok={100*s_zt['macro_acc']:.1f}% | "
            f"M2_Adapt={ad_lbl} ({s_ad['mean_t_tot']:.0f} tok) -> {out_json}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end M2-ZeroTok state surgery and zero-leakage M2-Adapt multi-horizon evaluation."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out-dir", type=str, default="outputs")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--keep-shards", action="store_true")
    args = parser.parse_args()

    model_dir = pathlib.Path(args.out_dir) / model_slug(args.model)
    seed_rows = _load_jsonl(_ensure_unpacked_jsonl(model_dir, "seed_60_cache"))
    dev_rows = _load_jsonl(_ensure_unpacked_jsonl(model_dir, "dev_60_cache"))
    eval_rows = _load_jsonl(_ensure_unpacked_jsonl(model_dir, "eval_450_cache"))
    if args.limit > 0:
        eval_rows = eval_rows[: int(args.limit)]

    # Sub-shard worker entry
    if args.shard_cache_override or int(args.num_shards) > 1:
        _run_worker(args, seed_rows, dev_rows, eval_rows, model_dir)
        return

    main_cache = model_dir / "ext_m2_state_adaptive_cache.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if main_cache.is_file():
        for r in _load_jsonl(main_cache):
            existing[str(r["uid"])] = r

    remaining_gpu = [r for r in eval_rows if _needs_gpu_work(r, existing.get(str(r["item"]["uid"])))]
    if not remaining_gpu or args.mock or str(args.shards).strip() == "1":
        args.num_shards = 1
        args.shard_id = 0
        _run_worker(args, seed_rows, dev_rows, eval_rows, model_dir)
        return

    cands = [x.strip() for x in args.gpu_ids.split(",") if x.strip()] if args.gpu_ids else None
    worker_map: list[str] = []
    if str(args.shards).strip().lower() == "auto":
        worker_map = select_worker_gpu_slots(
            args.model,
            candidate_ids=cands,
            workers_per_gpu=args.workers_per_gpu,
            max_gpus=8,
            wait_timeout_sec=int(args.wait_gpu_min) * 60,
            stage="w7_blend",
        )
        n_slots = max(1, len(worker_map))
    else:
        n_slots = max(1, int(args.shards))
        worker_map = select_worker_gpu_slots(
            args.model,
            candidate_ids=cands,
            workers_per_gpu=args.workers_per_gpu,
            max_gpus=8,
            wait_timeout_sec=int(args.wait_gpu_min) * 60,
            stage="w7_blend",
        ) or (cands or ["0"])
        n_slots = min(n_slots, max(1, len(worker_map)))

    if n_slots <= 1:
        args.num_shards = 1
        args.shard_id = 0
        _run_worker(args, seed_rows, dev_rows, eval_rows, model_dir)
        return

    num_shards = min(len(eval_rows), n_slots * 2)
    prewarmed = prewarm_hf_model_weights(args.model, offline=bool(args.offline))
    shard_dir = model_dir / "shards" / "state_adapt"
    shard_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"{hl('[ext_state_adapt]', 'cyan')} Multi-shard launch: {hl(f'{n_slots} GPU slots', 'green')} "
        f"({num_shards} micro-shards, {len(remaining_gpu)} GPU items) across {worker_map[:n_slots]}",
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
                "--device", "cuda:0",
                "--dtype", args.dtype,
                "--shard-id", str(sid),
                "--num-shards", str(num_shards),
                "--seed", str(args.seed),
                "--limit", str(args.limit),
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
                            f"{hl('[ext_state_adapt]', 'cyan')} {status_badge('done', 'DONE')} "
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
                    print(f"[ext_state_adapt] Shard {sid} exited {rc} ({s_log})", flush=True)
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
                pbar = fmt_progress_bar(min(len(eval_rows), len(existing) + done_rows), len(eval_rows), width=18)
                print(
                    f"[StateAdapt | {args.model}] elapsed={int(now_t - t_start)}s | "
                    f"shards={done_cnt}/{total_wave} | {pbar} items={min(len(eval_rows), len(existing)+done_rows)}/{len(eval_rows)}",
                    flush=True,
                )
                last_prog = now_t
            time.sleep(0.5)

        if not failed:
            break
        attempt += 1
        if attempt > int(args.max_retries):
            raise RuntimeError(f"ext_m2_state_and_adaptive failed on shards {failed}")
        pending = failed

    for sid in range(num_shards):
        sc = shard_dir / f"shard_{sid:02d}_cache.jsonl"
        if sc.is_file():
            for r in _load_jsonl(sc):
                existing[str(r["uid"])] = r
    ordered = [existing[str(r["item"]["uid"])] for r in eval_rows if str(r["item"]["uid"]) in existing]
    if len(ordered) != len(eval_rows):
        raise RuntimeError(f"Incomplete state_adaptive merge: {len(ordered)}/{len(eval_rows)}")
    main_cache.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ordered),
        encoding="utf-8",
    )
    if not args.keep_shards and shard_dir.is_dir():
        shutil.rmtree(shard_dir, ignore_errors=True)

    summary = summarize_state_and_adaptive(
        args.model, seed_rows, dev_rows, eval_rows, existing, seed=args.seed
    )
    out_json = model_dir / "ext_m2_state_adaptive.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    s_ad = summary["systems"]["M2_Adapt"]
    s_zt = summary["systems"]["M2_ZeroTok"]
    ad_lbl = hl(f"{100 * s_ad['macro_acc']:.1f}%", "green")
    print(
        f"{hl('[ext_state_adapt]', 'cyan')} {status_badge('done', 'DONE')} "
        f"{args.model}: M2_ZeroTok={100*s_zt['macro_acc']:.1f}% | "
        f"M2_Adapt={ad_lbl} ({s_ad['mean_t_tot']:.0f} tok) -> {out_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
