"""End-to-end cached execution matrix for Cascaded Budgeted Adaptive Reasoning Routing (C-BARR)."""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any
import numpy as np

from src.engine import CBARRDualModeEngine, run_gate1_lexical
from src.graders import grade_item, has_4gram_repetition_loop, has_cot_spillover
from src.metrics import (
    SLICE_ORDER,
    compute_4regime_and_oracle,
    compute_causal_decomposition,
    evaluate_system_table,
    exact_and_midp_mcnemar,
    holm_bonferroni,
)
from src.routers import CBARRCalibrator, resolve_all_variants_for_item


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
) -> list[dict[str, Any]]:
    """Executes and incrementally caches atomic primitives per item to prevent GPU re-runs (with D13-D16)."""
    existing: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        for row in _load_jsonl(cache_path):
            existing[row["item"]["uid"]] = row

    rows_out: list[dict[str, Any]] = []
    with cache_path.open("a", encoding="utf-8") as f_append:
        for item in items:
            uid = item["uid"]
            if uid in existing:
                rows_out.append(existing[uid])
                continue

            prompt = item["prompt"]
            g1 = run_gate1_lexical(prompt)
            # Patch D14: max_new_tokens=1024
            b0 = _annotate_run(item, engine.run_direct_non_thinking(prompt, max_new_tokens=1024))
            probe = engine.run_gate2_fast_probe(prompt, g1=g1, probe_tokens=32)
            # Patch D15: Reuse b0 when probe has empty ans_prefix under deterministic greedy decoding
            fast_done = (
                dict(b0)
                if probe["ans_prefix"] == ""
                else _annotate_run(item, engine.complete_fast_probe(probe, max_ans_tokens=1024))
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

            # Patch D15: Reuse natural b1b if anchor_clause == "" and b1b finished within budget_val
            def _run_or_reuse_m2(budget_val: int, locus: str) -> dict[str, Any]:
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
                        prompt,
                        budget_b=budget_val,
                        anchor_clause=g1.anchor_clause,
                        max_ans_tokens=1024,
                        allow_budget_force=True,
                        anchor_locus=locus,
                    ),
                )

            if run_calib_tiers or full_eval_modes:
                # Patch D13: Deduct T_probe at B=250 when escalated from Gate 2 so T_probe + T_think <= 250 (OAA_250)
                b256_after_probe = max(64, min(250, 250 - int(probe["T_probe"])))
                rec["m2_b64"] = _run_or_reuse_m2(64, "post_stripped")
                rec["m2_b128"] = _run_or_reuse_m2(128, "post_stripped")
                rec["m2_b256"] = _run_or_reuse_m2(b256_after_probe, "post_stripped")
                # Patch D13: Cap m2_b256_direct at budget_b=250
                rec["m2_b256_direct"] = _run_or_reuse_m2(250, "post_stripped")

            if full_eval_modes:
                b256_after_probe = max(64, min(250, 250 - int(probe["T_probe"])))
                rec["b1a"] = _annotate_run(item, engine.run_prompted_mode(prompt, "B1a_cot"))
                rec["b2a"] = _annotate_run(item, engine.run_prompted_mode(prompt, "B2a_self_select"))
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
                        rec[f"m2_pre_{k_suf}"] = _run_or_reuse_m2(b_val, "pre_close")

            f_append.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_append.flush()
            rows_out.append(rec)
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

    e_hurt: list[dict[str, Any]] = []
    cache_path = out_dir / "diag_ehurt_cache.jsonl"
    cached_rows = cache_split_primitives(
        engine, diag_items, cache_path, full_eval_modes=False, run_calib_tiers=False
    )
    for row in cached_rows:
        if int(row["b0"]["z"]) == 1 and int(row["b1b"]["z"]) == 0 and row["b1b"]["t_think"]:
            e_hurt.append(row)

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

    for m in modes:
        all_pass, inh_pass, str_pass = [], [], []
        for row in e_hurt:
            ans = engine.run_diagnostic_control(
                row["item"]["prompt"], row["b1b"]["t_think"], filler_ids, mode=m
            )
            g = grade_item(row["item"], ans)
            all_pass.append(int(g["z"]))
            if g.get("inhibition_strict_all") is not None:
                inh_pass.append(int(g["inhibition_strict_all"]))
            if g.get("structural_strict_all") is not None:
                str_pass.append(int(g["structural_strict_all"]))
        per_mode_z[m] = all_pass
        summary[m] = {
            "n_ehurt": len(e_hurt),
            "overall_recovery_pct": 100.0 * float(np.mean(all_pass)) if all_pass else 0.0,
            "inhibition_recovery_pct": 100.0 * float(np.mean(inh_pass)) if inh_pass else 0.0,
            "structural_recovery_pct": 100.0 * float(np.mean(str_pass)) if str_pass else 0.0,
        }

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full C-BARR evaluation matrix.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--strict-diag", action="store_true", help="Use 341 eval-disjoint pool")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    tag = args.model.replace("/", "__")
    out_dir = pathlib.Path(args.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_items = _load_jsonl(data_dir / "seed_60.jsonl")
    dev_items = _load_jsonl(data_dir / "dev_60.jsonl")
    eval_items = _load_jsonl(data_dir / "eval_450.jsonl")
    diag_file = "ifeval_diag_341_strict.jsonl" if args.strict_diag else "ifeval_diag_491.jsonl"
    diag_items = _load_jsonl(data_dir / diag_file)

    engine = CBARRDualModeEngine(args.model)

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

    eval_rows = cache_split_primitives(
        engine, eval_items, out_dir / "eval_450_cache.jsonl", True, True
    )
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

    oracle_info = compute_4regime_and_oracle(
        eval_items,
        per_system_runs["B0"],
        per_system_runs["B1b"],
        [
            per_system_runs["B0"],
            per_system_runs["B1b"],
            [r["m2_b64"] for r in eval_rows],
            [r["m2_b128"] for r in eval_rows],
            [r["m2_b256_direct"] for r in eval_rows],
        ],
    )

    gsm_seed = [x for x in seed_items if x["dataset"] == "gsm8k"]
    diag_summary = run_section6_diagnostics(engine, diag_items, gsm_seed, out_dir)

    report = {
        "model": args.model,
        "calibrated_params": params.__dict__,
        "systems": {
            k: {mk: mv for mk, mv in v.items() if mk not in ("z_vec", "u_vec")}
            for k, v in tables.items()
        },
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
    (out_dir / "matrix_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    tex_lines: list[str] = []
    for k in variant_names:
        sig = r"$^\dagger$" if tables[k].get("p_holm_vs_b1b", 1.0) < 0.05 else ""
        tex_lines.append(_format_latex_row(k, tables[k], sig=sig))
    tex_lines.append(_format_latex_row(r"\textit{Oracle\_UB}", oracle_info["oracle_summary"]))
    (out_dir / "tables_rows.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(f"Completed matrix run for {args.model}. Saved to {out_dir / 'matrix_summary.json'}")


if __name__ == "__main__":
    main()
