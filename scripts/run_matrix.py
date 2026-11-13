"""End-to-end cached execution matrix for Cascaded Budgeted Adaptive Reasoning Routing (C-BARR)."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from typing import Any
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine import CBARRDualModeEngine, run_gate1_lexical
from src.graders import grade_item, has_4gram_repetition_loop, has_cot_spillover
from src.metrics import (
    SLICE_ORDER,
    compute_4regime_and_oracle,
    compute_causal_decomposition,
    evaluate_system_table,
    exact_and_midp_mcnemar,
    holm_bonferroni,
    wilson_ci,
)
from src.routers import CBARRCalibrator, CalibratedRouterParams, resolve_all_variants_for_item

REQUIRED_DATA_SPLITS: tuple[str, ...] = (
    "seed_60.jsonl",
    "dev_60.jsonl",
    "eval_450.jsonl",
    "ifeval_diag_491.jsonl",
    "ifeval_diag_341_strict.jsonl",
)


def verify_manifest_integrity(data_dir: pathlib.Path) -> dict[str, str]:
    """Verifies frozen dataset split files against data/manifest_sha256.json."""
    manifest_path = data_dir / "manifest_sha256.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing frozen split manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for fname in REQUIRED_DATA_SPLITS:
        if fname not in manifest:
            raise RuntimeError(f"Manifest {manifest_path} missing required entry: {fname}")
    for fname, expected_sha in manifest.items():
        fpath = data_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing frozen split file listed in manifest: {fpath}")
        actual_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"SHA-256 mismatch for {fpath}: expected {expected_sha}, got {actual_sha}. "
                "Do NOT re-run freeze_datasets.py; restore the frozen artifact."
            )
    return manifest


def resolve_device_and_dtype(
    device_arg: str = "auto", dtype_arg: str = "auto"
) -> tuple[str, torch.dtype]:
    """Resolves target compute device and hardware-appropriate floating dtype (BF16 on sm>=80, FP16 on T4/P100/MPS)."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    else:
        dev = device_arg

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_arg != "auto":
        if dtype_arg not in dtype_map:
            raise ValueError(f"Unsupported --dtype {dtype_arg!r}; choose from auto, bfloat16, float16, float32")
        return dev, dtype_map[dtype_arg]

    if dev.startswith("cuda"):
        dev_idx = int(dev.split(":", 1)[1]) if ":" in dev else 0
        has_native_bf16 = (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            and torch.cuda.get_device_capability(dev_idx)[0] >= 8
        )
        return dev, (torch.bfloat16 if has_native_bf16 else torch.float16)
    if dev == "mps":
        return dev, torch.float16
    return dev, torch.float32


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _annotate_run(item: dict[str, Any], run_dict: dict[str, Any]) -> dict[str, Any]:
    out = dict(run_dict)
    verdict = grade_item(item, out["a_final"])
    out.update(verdict)
    combined_txt = f"{out.get('t_think', '')} {out.get('a_final', '')}".strip()
    out["has_4gram_loop"] = has_4gram_repetition_loop(combined_txt)
    out["has_cot_spillover"] = has_cot_spillover(
        out["a_final"], truncated=bool(out.get("truncated", False))
    )
    return out


def cache_split_primitives(
    engine: CBARRDualModeEngine,
    items: list[dict[str, Any]],
    cache_path: pathlib.Path,
    full_eval_modes: bool = True,
    run_calib_tiers: bool = True,
    include_b512: bool = False,
) -> list[dict[str, Any]]:
    """Executes and incrementally caches atomic primitives per item to prevent GPU re-runs (with D13-D16)."""
    existing: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        for row in _load_jsonl(cache_path):
            existing[row["item"]["uid"]] = row

    def _run_or_reuse_m2_row(
        item: dict[str, Any], g1: Any, b1b: dict[str, Any], budget_val: int, locus: str
    ) -> dict[str, Any]:
        if (
            not g1.anchor_clause
            and not b1b["think_exhausted"]
            and not b1b["truncated"]
            and int(b1b["T_think"]) <= budget_val
        ):
            return dict(b1b)
        return _annotate_run(
            item,
            engine.run_capped_think_and_anchor(
                item["prompt"],
                budget_b=budget_val,
                anchor_clause=g1.anchor_clause,
                max_ans_tokens=1024,
                allow_budget_force=True,
                anchor_locus=locus,
            ),
        )

    # Caching atomic per-item primitives (rather than routed decisions) lets us evaluate all
    # B0-B2/M1/M2 variants offline and backfill --include-b512 without re-running completed items.
    rows_out: list[dict[str, Any]] = []
    upgraded_existing = False
    total_items = len(items)
    n_new = 0
    n_cached = 0
    t_start = time.perf_counter()
    t_last_log = t_start
    split_tag = cache_path.stem.replace("_cache", "")

    with cache_path.open("a", encoding="utf-8") as f_append:
        for idx, item in enumerate(items, start=1):
            uid = item["uid"]
            try:
                if uid in existing:
                    cached_row = existing[uid]
                    if include_b512 and full_eval_modes and "m2_b512" not in cached_row:
                        g1 = run_gate1_lexical(item["prompt"])
                        cached_row["m2_b512"] = _run_or_reuse_m2_row(
                            item, g1, cached_row["b1b"], 512, "post_stripped"
                        )
                        upgraded_existing = True
                    n_cached += 1
                    rows_out.append(cached_row)
                else:
                    prompt = item["prompt"]
                    g1 = run_gate1_lexical(prompt)
                    # Patch D14: max_new_tokens=1024
                    b0 = _annotate_run(
                        item, engine.run_direct_non_thinking(prompt, max_new_tokens=1024)
                    )
                    probe = engine.run_gate2_fast_probe(prompt, g1=g1, probe_tokens=32)
                    # Patch D15: Reuse b0 when probe has empty ans_prefix under deterministic greedy decoding
                    fast_done = (
                        dict(b0)
                        if probe["ans_prefix"] == ""
                        else _annotate_run(
                            item, engine.complete_fast_probe(probe, max_ans_tokens=1024)
                        )
                    )
                    b1b = _annotate_run(
                        item,
                        engine.run_capped_think_and_anchor(
                            prompt,
                            budget_b=2048,
                            anchor_clause="",
                            max_ans_tokens=1024,
                            allow_budget_force=False,
                        ),
                    )

                    rec: dict[str, Any] = {
                        "item": item,
                        "b0": b0,
                        "probe": {k: v for k, v in probe.items() if k != "step_entropies"},
                        "fast_done": fast_done,
                        "b1b": b1b,
                    }

                    if run_calib_tiers or full_eval_modes:
                        # Patch D13: Deduct T_probe at B=250 when escalated from Gate 2 so T_probe + T_think <= 250 (OAA_250)
                        b256_after_probe = max(64, min(250, 250 - int(probe["T_probe"])))
                        rec["m2_b64"] = _run_or_reuse_m2_row(item, g1, b1b, 64, "post_stripped")
                        rec["m2_b128"] = _run_or_reuse_m2_row(item, g1, b1b, 128, "post_stripped")
                        rec["m2_b256"] = _run_or_reuse_m2_row(
                            item, g1, b1b, b256_after_probe, "post_stripped"
                        )
                        # Patch D13: Cap m2_b256_direct at budget_b=250
                        rec["m2_b256_direct"] = _run_or_reuse_m2_row(
                            item, g1, b1b, 250, "post_stripped"
                        )

                    if full_eval_modes:
                        b256_after_probe = max(64, min(250, 250 - int(probe["T_probe"])))
                        rec["b1a"] = _annotate_run(item, engine.run_prompted_mode(prompt, "B1a_cot"))
                        rec["b2a"] = _annotate_run(
                            item, engine.run_prompted_mode(prompt, "B2a_self_select")
                        )
                        # Patch D15: Alias m2_pre_* = m2_* when anchor_clause == ""
                        if not g1.anchor_clause:
                            for k_suf in ("b64", "b128", "b256", "b256_direct"):
                                rec[f"m2_pre_{k_suf}"] = dict(rec[f"m2_{k_suf}"])
                        else:
                            for b_val, k_suf in (
                                (64, "b64"),
                                (128, "b128"),
                                (b256_after_probe, "b256"),
                                (250, "b256_direct"),
                            ):
                                rec[f"m2_pre_{k_suf}"] = _run_or_reuse_m2_row(
                                    item, g1, b1b, b_val, "pre_close"
                                )
                        if include_b512:
                            rec["m2_b512"] = _run_or_reuse_m2_row(
                                item, g1, b1b, 512, "post_stripped"
                            )

                    f_append.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_append.flush()
                    existing[uid] = rec
                    n_new += 1
                    rows_out.append(rec)

                now = time.perf_counter()
                if (
                    total_items >= 40
                    and (idx % 25 == 0 or (now - t_last_log) >= 30.0)
                    and idx < total_items
                ):
                    elapsed = now - t_start
                    eta_str = (
                        f" | eta={(elapsed / n_new) * (total_items - idx):.0f}s"
                        if n_new > 0 and n_cached == 0
                        else ""
                    )
                    print(
                        f"[{split_tag} {idx}/{total_items}] new={n_new} cached={n_cached} "
                        f"elapsed={elapsed:.1f}s{eta_str}",
                        flush=True,
                    )
                    t_last_log = now
            except Exception as exc:
                print(
                    f"[ERROR] Failed in {split_tag} at item {idx}/{total_items} uid={uid}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                raise

    # Persist backfilled m2_b512 entries atomically so interrupted rewrites cannot corrupt cache
    if upgraded_existing:
        tmp_path = cache_path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8") as f_tmp:
            for row in existing.values():
                f_tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp_path.replace(cache_path)

    total_elapsed = time.perf_counter() - t_start
    print(
        f"[{split_tag}] completed {total_items} items (new={n_new}, cached={n_cached}) "
        f"in {total_elapsed:.1f}s",
        flush=True,
    )
    return rows_out


def run_section6_diagnostics(
    engine: CBARRDualModeEngine,
    diag_items: list[dict[str, Any]],
    gsm_seed_items: list[dict[str, Any]],
    out_dir: pathlib.Path,
) -> dict[str, Any]:
    """Runs Causal Diagnostic Controls (Trace Stripping vs. Post-Think Re-Anchoring) on E_hurt."""
    filler_text = " ".join(x.get("rationale_gold", "") for x in gsm_seed_items)
    filler_ids = engine.tok.encode(filler_text, add_special_tokens=False)

    # Identify E_hurt: items where thinking directly caused failure (Pass(B0)=1 and Fail(B1b)=0).
    # strict341 is a subset of diag491, so sharing diag_ehurt_cache + diag_controls_cache avoids recompute.
    e_hurt: list[dict[str, Any]] = []
    cache_path = out_dir / "diag_ehurt_cache.jsonl"
    cached_rows = cache_split_primitives(
        engine, diag_items, cache_path, full_eval_modes=False, run_calib_tiers=False
    )
    for row in cached_rows:
        if int(row["b0"]["z"]) == 1 and int(row["b1b"]["z"]) == 0 and row["b1b"]["t_think"]:
            e_hurt.append(row)

    controls_cache_path = out_dir / "diag_controls_cache.jsonl"
    cached_controls: dict[tuple[str, str], dict[str, Any]] = {}
    if controls_cache_path.exists():
        for rec in _load_jsonl(controls_cache_path):
            cached_controls[(str(rec["uid"]), str(rec["mode"]))] = rec

    modes = (
        "A1_strip",
        "B2a_post_anchor",
        "B1_pre_anchor",
        "B2b_post_raw",
        "A2_math_swap",
        "B3_early_cut_no_anchor",
    )
    per_mode_z: dict[str, list[int]] = {}
    summary: dict[str, Any] = {}

    with controls_cache_path.open("a", encoding="utf-8") as f_ctrl:
        print(
            f"[diag_controls] E_hurt={len(e_hurt)} items across {len(modes)} causal controls",
            flush=True,
        )
        for m in modes:
            t_mode = time.perf_counter()
            all_pass, inh_pass, str_pass = [], [], []
            for row in e_hurt:
                uid = str(row["item"]["uid"])
                cached = cached_controls.get((uid, m))
                if cached is None:
                    try:
                        ans = engine.run_diagnostic_control(
                            row["item"]["prompt"], row["b1b"]["t_think"], filler_ids, mode=m
                        )
                    except Exception as exc:
                        print(
                            f"[ERROR] diag control {m} failed at uid={uid}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        raise
                    g = grade_item(row["item"], ans)
                    cached = {"uid": uid, "mode": m, "a_final": ans, "grade": g}
                    cached_controls[(uid, m)] = cached
                    f_ctrl.write(json.dumps(cached, ensure_ascii=False) + "\n")
                    f_ctrl.flush()
                else:
                    g = cached["grade"]
                all_pass.append(int(g["z"]))
                if g.get("inhibition_strict_all") is not None:
                    inh_pass.append(int(g["inhibition_strict_all"]))
                if g.get("structural_strict_all") is not None:
                    str_pass.append(int(g["structural_strict_all"]))

            _, w_lo, w_hi = wilson_ci(sum(all_pass), len(all_pass))
            per_mode_z[m] = all_pass
            summary[m] = {
                "n_ehurt": len(e_hurt),
                "overall_pass_count": int(sum(all_pass)),
                "overall_recovery_pct": 100.0 * float(np.mean(all_pass)) if all_pass else 0.0,
                "overall_wilson_ci95": [100.0 * w_lo, 100.0 * w_hi],
                "inhibition_recovery_pct": 100.0 * float(np.mean(inh_pass)) if inh_pass else 0.0,
                "structural_recovery_pct": 100.0 * float(np.mean(str_pass)) if str_pass else 0.0,
            }
            print(
                f"[diag_controls] {m}: {summary[m]['overall_recovery_pct']:.1f}% "
                f"({summary[m]['overall_pass_count']}/{len(e_hurt)}) in {time.perf_counter() - t_mode:.1f}s",
                flush=True,
            )

    summary["causal_attribution"] = compute_causal_decomposition(
        z_strip=per_mode_z["A1_strip"],
        z_post_anchor=per_mode_z["B2a_post_anchor"],
        z_pre_anchor=per_mode_z["B1_pre_anchor"],
        z_post_raw=per_mode_z["B2b_post_raw"],
    )
    return summary


def _format_latex_row(label: str, t: dict[str, Any], sig: str = "") -> str:
    s_cols = " & ".join(f"{100.0 * t['slice_acc'][s]:.1f}" for s in SLICE_ORDER)
    return (
        f"{label} & {s_cols} & {100.0 * t['macro_acc']:.1f}{sig} & "
        f"{100.0 * t['oaa_250']:.1f} & {100.0 * t['auc_oaa_2048']:.1f} & "
        f"{100.0 * t['u_005']:.1f} & {t['mean_t_tot']:.0f} / {t['mean_calls']:.2f} \\\\"
    )


def _build_engine(
    model_id: str, device_arg: str, use_mock: bool, dtype_arg: str = "auto"
) -> Any:
    if use_mock:
        from scripts.run_smoke_test import MockDualModeEngine

        return MockDualModeEngine(model_id)
    dev, dtype = resolve_device_and_dtype(device_arg, dtype_arg)
    return CBARRDualModeEngine(model_id, device=dev, dtype=dtype)


def _resolve_m2_b512(
    row: dict[str, Any], params: CalibratedRouterParams
) -> dict[str, Any]:
    """Resolves Contingency R3 M2_B512 variant reusing Gate 1/2 routing rules with B=512 slow cap."""
    from src.routers import score_gate2_probability

    g1 = run_gate1_lexical(row["item"]["prompt"])
    probe = row["probe"]
    if g1.phi_prec == 1 and g1.phi_plan == 0 and g1.phi_sym == 0:
        return row["b0"]
    slow = row["m2_b512"]
    if g1.phi_plan == 1:
        return slow
    if score_gate2_probability(g1, probe, params) >= params.tau_prob:
        merged = dict(slow)
        merged["T_probe"] = probe["T_probe"]
        merged["T_total"] = probe["T_probe"] + slow["T_total"]
        merged["calls"] = 2
        merged["latency_ms"] = probe["latency_ms"] + slow["latency_ms"]
        return merged
    return row["fast_done"]


def _build_calibration_summary(
    model_id: str,
    seed_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    params: CalibratedRouterParams,
) -> dict[str, Any]:
    """Summarizes Stage-0/1 calibration, Gate 2 escalation rate, and Contingency R1 status."""
    dev_items = [r["item"] for r in dev_rows]
    # Note: dev_rows lack full_eval_modes ('b1a'/'b2a'/'m2_pre_*'), so evaluate dev M2/B0/B1b directly
    dev_m2_runs: list[dict[str, Any]] = []
    dev_b1b_runs: list[dict[str, Any]] = []
    dev_b0_runs: list[dict[str, Any]] = []
    from src.routers import _get_slow_rec, score_gate2_probability, select_tiered_budget

    gate1_skip_dev = 0
    gate2_esc_dev = 0
    for row in dev_rows:
        prompt_str = row["item"]["prompt"]
        g1 = row["g1"]
        probe = row["probe"]
        dev_b0_runs.append(row["b0"])
        dev_b1b_runs.append(row["b1b"])
        if g1.phi_prec == 1 and g1.phi_plan == 0 and g1.phi_sym == 0:
            gate1_skip_dev += 1
            dev_m2_runs.append(row["b0"])
        elif g1.phi_plan == 1:
            gate2_esc_dev += 1
            b_dir = select_tiered_budget(prompt_str, g1, None, params.tau_h_high)
            dev_m2_runs.append(_get_slow_rec(row, b_dir, direct=True, prefix="m2"))
        elif score_gate2_probability(g1, probe, params) >= params.tau_prob:
            gate2_esc_dev += 1
            b_esc = select_tiered_budget(prompt_str, g1, probe, params.tau_h_high)
            slow = _get_slow_rec(row, b_esc, direct=False, prefix="m2")
            merged = dict(slow)
            merged["T_probe"] = probe["T_probe"]
            merged["T_total"] = probe["T_probe"] + slow["T_total"]
            merged["calls"] = 2
            merged["latency_ms"] = probe["latency_ms"] + slow["latency_ms"]
            dev_m2_runs.append(merged)
        else:
            dev_m2_runs.append(row["fast_done"])

    dev_m2_table = evaluate_system_table(dev_items, dev_m2_runs)
    dev_b1b_table = evaluate_system_table(dev_items, dev_b1b_runs)
    dev_b0_table = evaluate_system_table(dev_items, dev_b0_runs)

    r1_in_range = 0.20 <= params.m2_escalation_rate_dev <= 0.85
    tau_at_boundary = params.tau_prob <= 0.16 or params.tau_prob >= 0.84
    return {
        "model": model_id,
        "n_seed": len(seed_rows),
        "n_dev": len(dev_rows),
        "calibrated_params": params.__dict__,
        "dev_gate1_skip_rate": gate1_skip_dev / max(1, len(dev_rows)),
        "dev_escalation_rate": gate2_esc_dev / max(1, len(dev_rows)),
        "contingency_r1_ok": r1_in_range,
        "tau_prob_near_boundary": tau_at_boundary,
        "dev_summary": {
            "B0": {k: v for k, v in dev_b0_table.items() if k not in ("z_vec", "u_vec")},
            "B1b": {k: v for k, v in dev_b1b_table.items() if k not in ("z_vec", "u_vec")},
            "M2": {k: v for k, v in dev_m2_table.items() if k not in ("z_vec", "u_vec")},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full C-BARR evaluation matrix.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--strict-diag", action="store_true", help="Use 341 eval-disjoint pool")
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Run seed_60 + dev_60 calibration, write calibration_summary.json, and exit.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device ('auto', 'cuda', 'mps', 'cpu', or 'cuda:N').",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model precision ('auto', 'bfloat16', 'float16', or 'float32').",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use offline MockDualModeEngine for CPU end-to-end verification.",
    )
    parser.add_argument(
        "--include-b512",
        action="store_true",
        help="Also cache/evaluate auxiliary M2_B512 sweep point (Contingency R3).",
    )
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    verify_manifest_integrity(data_dir)

    tag = args.model.replace("/", "__")
    out_dir = pathlib.Path(args.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_items = _load_jsonl(data_dir / "seed_60.jsonl")
    dev_items = _load_jsonl(data_dir / "dev_60.jsonl")
    eval_items = _load_jsonl(data_dir / "eval_450.jsonl")
    diag_tag = "strict341" if args.strict_diag else "diag491"
    diag_file = "ifeval_diag_341_strict.jsonl" if args.strict_diag else "ifeval_diag_491.jsonl"
    diag_items = _load_jsonl(data_dir / diag_file)

    t_run_start = time.perf_counter()
    resolved_runtime = (
        ("mock", "mock")
        if args.mock
        else resolve_device_and_dtype(device_arg=args.device, dtype_arg=args.dtype)
    )
    print(
        f"[run_matrix] model={args.model} | diag_pool={diag_tag} ({len(diag_items)}) | "
        f"device={resolved_runtime[0]} | dtype={resolved_runtime[1]}",
        flush=True,
    )

    engine = _build_engine(
        args.model, device_arg=args.device, use_mock=args.mock, dtype_arg=args.dtype
    )

    t_calib_start = time.perf_counter()
    seed_rows = cache_split_primitives(
        engine, seed_items, out_dir / "seed_60_cache.jsonl", False, True
    )
    dev_rows = cache_split_primitives(
        engine, dev_items, out_dir / "dev_60_cache.jsonl", False, True
    )
    for r in seed_rows + dev_rows:
        r["g1"] = run_gate1_lexical(r["item"]["prompt"])

    calibrator = CBARRCalibrator()
    params = calibrator.fit(seed_rows, dev_rows)
    t_calib_sec = round(time.perf_counter() - t_calib_start, 2)
    calib_summary = _build_calibration_summary(args.model, seed_rows, dev_rows, params)
    calib_summary["wall_time_sec"] = t_calib_sec
    calib_summary["runtime"] = {"device": str(resolved_runtime[0]), "dtype": str(resolved_runtime[1])}
    calib_path = out_dir / "calibration_summary.json"
    calib_path.write_text(json.dumps(calib_summary, indent=2) + "\n", encoding="utf-8")

    if not calib_summary["contingency_r1_ok"]:
        print(
            f"[WARN] Contingency R1 check: dev escalation rate={params.m2_escalation_rate_dev:.3f} "
            f"(outside [0.20, 0.85]), tau_prob={params.tau_prob:.3f}."
        )
    if args.calibrate_only:
        print(f"Saved calibration summary for {args.model} to {calib_path} ({t_calib_sec:.1f}s)")
        print(json.dumps(calib_summary["calibrated_params"], indent=2))
        return

    t_eval_start = time.perf_counter()
    eval_rows = cache_split_primitives(
        engine,
        eval_items,
        out_dir / "eval_450_cache.jsonl",
        True,
        True,
        include_b512=args.include_b512,
    )
    t_eval_sec = round(time.perf_counter() - t_eval_start, 2)
    rng_b2b = np.random.default_rng(42)

    variant_names = (
        "B0",
        "B1a",
        "B1b",
        "B2a",
        "B2b",
        "M1a",
        "M1b",
        "M1c",
        "M2_B64",
        "M2_B128",
        "M2_B256",
        "M2_Pre",
        "M2",
    )
    per_system_runs: dict[str, list[dict[str, Any]]] = {k: [] for k in variant_names}

    for row in eval_rows:
        resolved = resolve_all_variants_for_item(
            row, params, rng_b2b, target_esc_prob=params.m2_escalation_rate_dev
        )
        for k in variant_names:
            per_system_runs[k].append(resolved[k])

    tables: dict[str, Any] = {
        k: evaluate_system_table(eval_items, per_system_runs[k]) for k in variant_names
    }

    comp_keys = [k for k in variant_names if k != "B1b"]
    raw_p = [
        exact_and_midp_mcnemar(tables["B1b"]["z_vec"], tables[k]["z_vec"])["p_exact"]
        for k in comp_keys
    ]
    adj_p = holm_bonferroni(raw_p)
    for k, p_adj in zip(comp_keys, adj_p):
        tables[k]["p_holm_vs_b1b"] = p_adj

    # Evaluate M2_B512 separately under auxiliary_systems so Contingency R3 never alters
    # the 12-hypothesis Holm-Bonferroni FWER correction of the pre-registered table.
    aux_systems: dict[str, Any] = {}
    if args.include_b512:
        m2_b512_runs = [_resolve_m2_b512(r, params) for r in eval_rows]
        m2_b512_table = evaluate_system_table(eval_items, m2_b512_runs)
        m2_b512_table["mcnemar_vs_b1b"] = exact_and_midp_mcnemar(
            tables["B1b"]["z_vec"], m2_b512_table["z_vec"]
        )
        aux_systems["M2_B512"] = {
            mk: mv for mk, mv in m2_b512_table.items() if mk not in ("z_vec", "u_vec")
        }

    oracle_candidates = [
        per_system_runs["B0"],
        per_system_runs["B1b"],
        [r["m2_b64"] for r in eval_rows],
        [r["m2_b128"] for r in eval_rows],
        [r["m2_b256_direct"] for r in eval_rows],
    ]
    if args.include_b512:
        oracle_candidates.append([r["m2_b512"] for r in eval_rows])

    oracle_info = compute_4regime_and_oracle(
        eval_items,
        per_system_runs["B0"],
        per_system_runs["B1b"],
        oracle_candidates,
    )

    t_diag_start = time.perf_counter()
    gsm_seed = [x for x in seed_items if x["dataset"] == "gsm8k"]
    diag_summary = run_section6_diagnostics(engine, diag_items, gsm_seed, out_dir)
    t_diag_sec = round(time.perf_counter() - t_diag_start, 2)
    t_total_sec = round(time.perf_counter() - t_run_start, 2)

    report = {
        "model": args.model,
        "diag_pool": diag_tag,
        "diag_n_items": len(diag_items),
        "runtime": {"device": str(resolved_runtime[0]), "dtype": str(resolved_runtime[1])},
        "timing_sec": {
            "calibration": t_calib_sec,
            "eval_matrix": t_eval_sec,
            "section6_diagnostics": t_diag_sec,
            "total_wall": t_total_sec,
        },
        "calibrated_params": params.__dict__,
        "systems": {
            k: {mk: mv for mk, mv in v.items() if mk not in ("z_vec", "u_vec")}
            for k, v in tables.items()
        },
        "auxiliary_systems": aux_systems,
        "oracle_and_4regime": {
            k: (
                {mk: mv for mk, mv in v.items() if mk not in ("z_vec", "u_vec")}
                if k == "oracle_summary"
                else v
            )
            for k, v in oracle_info.items()
        },
        "section6_diagnostics": diag_summary,
    }
    # Write both pool-specific (non-clobbering) outputs and canonical latest aliases
    report_json = json.dumps(report, indent=2) + "\n"
    (out_dir / f"matrix_summary_{diag_tag}.json").write_text(report_json, encoding="utf-8")
    (out_dir / "matrix_summary.json").write_text(report_json, encoding="utf-8")

    tex_lines: list[str] = []
    for k in variant_names:
        sig = r"$^\dagger$" if tables[k].get("p_holm_vs_b1b", 1.0) < 0.05 else ""
        tex_lines.append(_format_latex_row(k, tables[k], sig=sig))
    if "M2_B512" in aux_systems:
        tex_lines.append(_format_latex_row(r"M2\_B512 (R3)", aux_systems["M2_B512"]))
    tex_lines.append(_format_latex_row(r"\textit{Oracle\_UB}", oracle_info["oracle_summary"]))
    tex_payload = "\n".join(tex_lines) + "\n"
    (out_dir / f"tables_rows_{diag_tag}.tex").write_text(tex_payload, encoding="utf-8")
    (out_dir / "tables_rows.tex").write_text(tex_payload, encoding="utf-8")
    print(
        f"Completed matrix run for {args.model} ({diag_tag}) in {t_total_sec:.1f}s. "
        f"Saved to {out_dir / f'matrix_summary_{diag_tag}.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
