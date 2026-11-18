"""Stage-2 replay & A/B benchmark comparator for upgraded Gate-2 calibration, M2_Prec, and solve_first bridge.

Reuses cached Stage-1 primitives from outputs/<model>/eval_450_cache.jsonl (zero full-matrix
re-inference) and replays Stage-2 ONLY on escalated/direct truncated non-anchor runs under
--stage2-bridge solve_first. Does not mutate original canonical cache JSONL files.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_matrix import _annotate_run, _build_engine, _load_jsonl, _regrade_cached_gsm8k_row
from src.engine import CBARRDualModeEngine, run_gate1_lexical, strip_special_end_tokens
from src.logfmt import banner, hl
from src.metrics import evaluate_system_table, exact_and_midp_mcnemar
from src.routers import (
    CBARRCalibrator,
    CalibratedRouterParams,
    resolve_all_variants_for_item,
    resolve_m2_prec_variant,
    score_gate2_probability,
    select_tiered_budget,
)


def _slow_cache_key(budget: int, direct: bool) -> str:
    if budget == 250 and direct:
        return "m2_b256_direct"
    if budget == 250:
        return "m2_b256"
    return f"m2_b{budget}"


def replay_stage2_with_bridge(
    engine: CBARRDualModeEngine,
    item: dict[str, Any],
    cached_slow: dict[str, Any],
    g1: Any,
    max_ans_tokens: int = 1024,
) -> dict[str, Any]:
    """Replays only Stage-2 completion from cached Stage-1 t_think using engine's Stage-2 bridge."""
    if not cached_slow.get("truncated", False) or g1.anchor_clause:
        return dict(cached_slow)
    pref = engine.build_prompt_prefix(item["prompt"], enable_thinking=True)
    t_think = str(cached_slow.get("t_think", ""))
    close_block, vis_prefix = engine._build_reanchor_for_engine(
        True, g1.anchor_clause, "post_stripped"
    )
    cont_prefix = pref + t_think + close_block
    t_inj_toks = len(engine.tok.encode(close_block, add_special_tokens=False))
    s2 = engine._decode_raw(cont_prefix, max_new_tokens=max_ans_tokens, log_entropies=False)
    s2_clean = strip_special_end_tokens(s2["raw_text"]).lstrip("\n").strip()
    a_final = f"{vis_prefix}{s2_clean}".strip() if vis_prefix else s2_clean
    t_think_toks = int(cached_slow.get("T_think", 0))
    t_wasted = int(cached_slow.get("T_wasted", 0))
    tot = t_think_toks + t_wasted + t_inj_toks + int(s2["total_gen_tokens"])
    replayed = {
        "t_think": t_think,
        "a_final": a_final,
        "T_probe": 0,
        "T_think": t_think_toks,
        "T_wasted": t_wasted,
        "T_inj": t_inj_toks,
        "T_final": int(s2["total_gen_tokens"]),
        "T_total": tot,
        "calls": 1,
        "truncated": True,
        "think_exhausted": False,
        "latency_ms": float(cached_slow.get("latency_ms", 0.0)) + float(s2["latency_ms"]),
    }
    return _annotate_run(item, replayed)


def _target_slow_keys_for_row(
    row: dict[str, Any], params: CalibratedRouterParams
) -> list[str]:
    """Returns cache keys of Stage-2 slow runs that solve_first replay can affect for this item."""
    prompt_str = row["item"]["prompt"]
    g1 = run_gate1_lexical(prompt_str)
    if g1.phi_prec == 1 and g1.phi_plan == 0 and g1.phi_sym == 0:
        return ["m2_b128"]
    if g1.phi_plan == 1:
        b_dir = select_tiered_budget(prompt_str, g1, None, params.tau_h_high)
        return [_slow_cache_key(b_dir, direct=True)]
    if score_gate2_probability(g1, row["probe"], params) >= params.tau_prob:
        b_esc = select_tiered_budget(prompt_str, g1, row["probe"], params.tau_h_high)
        return [_slow_cache_key(b_esc, direct=False)]
    return []


def run_comparison(
    model_id: str,
    model_dir: pathlib.Path,
    *,
    use_mock: bool = False,
    device: str = "auto",
    dtype: str = "auto",
    offline: bool = False,
    max_items: int = 0,
    lam_utility: float = 0.05,
    norm_tokens: float = 250.0,
    seed: int = 42,
    reg_c: float = 1.0,
) -> dict[str, Any]:
    """Runs non-destructive re-calibration + Stage-2 solve_first replay on cached model outputs."""
    seed_path = model_dir / "seed_60_cache.jsonl"
    dev_path = model_dir / "dev_60_cache.jsonl"
    eval_path = model_dir / "eval_450_cache.jsonl"
    for req_p in (seed_path, dev_path, eval_path):
        if not req_p.exists():
            raise FileNotFoundError(f"Missing required cached split: {req_p}")

    seed_rows = [copy.deepcopy(r) for r in _load_jsonl(seed_path)]
    dev_rows = [copy.deepcopy(r) for r in _load_jsonl(dev_path)]
    eval_rows = [copy.deepcopy(r) for r in _load_jsonl(eval_path)]
    if max_items > 0:
        eval_rows = eval_rows[:max_items]

    gsm_regraded = 0
    for split_rows in (seed_rows, dev_rows, eval_rows):
        for r in split_rows:
            if _regrade_cached_gsm8k_row(r):
                gsm_regraded += 1

    for r in seed_rows + dev_rows:
        r["g1"] = run_gate1_lexical(r["item"]["prompt"])

    calibrator = CBARRCalibrator(
        lam_utility=lam_utility,
        norm_tokens=norm_tokens,
        reg_c=reg_c,
        seed=seed,
    )
    params = calibrator.fit(seed_rows, dev_rows)

    solve_engine = _build_engine(
        model_id,
        device_arg=device,
        use_mock=use_mock,
        dtype_arg=dtype,
        local_files_only=offline,
        stage2_bridge="solve_first",
    )

    eval_items = [r["item"] for r in eval_rows]
    rng_b2b = np.random.default_rng(seed)
    runs_by_system: dict[str, list[dict[str, Any]]] = {
        "B0": [],
        "B1b": [],
        "B2a": [],
        "M2_classic": [],
        "M2_Prec_classic": [],
        "M2_solve_first": [],
        "M2_Prec_solve_first": [],
    }

    t0 = time.perf_counter()
    n_replayed = 0
    for idx, row in enumerate(eval_rows, start=1):
        base_resolved = resolve_all_variants_for_item(
            row, params, rng_b2b, target_esc_prob=params.m2_escalation_rate_dev
        )
        runs_by_system["B0"].append(base_resolved["B0"])
        runs_by_system["B1b"].append(base_resolved["B1b"])
        runs_by_system["B2a"].append(base_resolved["B2a"])
        runs_by_system["M2_classic"].append(base_resolved["M2"])
        runs_by_system["M2_Prec_classic"].append(resolve_m2_prec_variant(row, params))

        g1 = run_gate1_lexical(row["item"]["prompt"])
        row_v2 = dict(row)
        for key in _target_slow_keys_for_row(row, params):
            slow_rec = row[key]
            if slow_rec.get("truncated", False) and not g1.anchor_clause:
                row_v2[key] = replay_stage2_with_bridge(solve_engine, row["item"], slow_rec, g1)
                n_replayed += 1

        rng_dummy = np.random.default_rng(seed)
        v2_resolved = resolve_all_variants_for_item(
            row_v2, params, rng_dummy, target_esc_prob=params.m2_escalation_rate_dev
        )
        runs_by_system["M2_solve_first"].append(v2_resolved["M2"])
        runs_by_system["M2_Prec_solve_first"].append(resolve_m2_prec_variant(row_v2, params))
        if idx % 50 == 0 or idx == len(eval_rows):
            print(
                f"[compare_m2_v2 {idx}/{len(eval_rows)}] stage2_replayed={n_replayed} "
                f"elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    elapsed = round(time.perf_counter() - t0, 2)
    tables = {
        k: evaluate_system_table(eval_items, v, lam_utility=lam_utility, norm_tokens=norm_tokens)
        for k, v in runs_by_system.items()
    }
    paired_tests = {
        "M2_solve_first_vs_M2_classic": exact_and_midp_mcnemar(
            tables["M2_classic"]["z_vec"], tables["M2_solve_first"]["z_vec"]
        ),
        "M2_solve_first_vs_B0": exact_and_midp_mcnemar(
            tables["B0"]["z_vec"], tables["M2_solve_first"]["z_vec"]
        ),
        "M2_Prec_solve_first_vs_M2_Prec_classic": exact_and_midp_mcnemar(
            tables["M2_Prec_classic"]["z_vec"], tables["M2_Prec_solve_first"]["z_vec"]
        ),
        "M2_Prec_solve_first_vs_B0": exact_and_midp_mcnemar(
            tables["B0"]["z_vec"], tables["M2_Prec_solve_first"]["z_vec"]
        ),
    }

    report = {
        "model": model_id,
        "n_eval": len(eval_rows),
        "gsm8k_cache_rows_regraded": gsm_regraded,
        "stage2_items_replayed": n_replayed,
        "replay_elapsed_sec": elapsed,
        "calibrated_params": params.__dict__,
        "systems": {
            k: {mk: mv for mk, mv in v.items() if mk not in ("z_vec", "u_vec")}
            for k, v in tables.items()
        },
        "paired_mcnemar": paired_tests,
    }
    out_path = model_dir / "eval_450_m2v2_comparison.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    del solve_engine
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    print(banner(f"M2 v2 HEAD-TO-HEAD COMPARISON: {model_id} (n={len(eval_rows)}, replayed={n_replayed})", "cyan"), flush=True)
    for k in (
        "B0",
        "B1b",
        "B2a",
        "M2_classic",
        "M2_solve_first",
        "M2_Prec_classic",
        "M2_Prec_solve_first",
    ):
        t = tables[k]
        ip = 100.0 * float(t["slice_acc"].get("ifeval_precision", 0.0))
        row_s = (
            f"  {k:20s}: Macro={100*t['macro_acc']:.1f}% | Micro={100*t['micro_acc']:.1f}% | "
            f"Prec={ip:.1f}% | OAA_250={100*t['oaa_250']:.1f}% | "
            f"U_0.05={100*t['u_005']:.1f}% | Tok={t['mean_t_tot']:.1f}"
        )
        print(hl(row_s, "green") if "solve_first" in k else row_s, flush=True)
    for pname, pv in paired_tests.items():
        print(
            f"  McNemar {pname}: base_wins={pv['base_wins']} method_wins={pv['method_wins']} "
            f"p_exact={pv['p_exact']:.4g} p_midp={pv['p_midp']:.4g}",
            flush=True,
        )
    print(f"Saved comparison report -> {out_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay Stage-2 solve_first bridge against cached classic C-BARR results."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--lam-utility", type=float, default=0.05)
    parser.add_argument("--norm-tokens", type=float, default=250.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reg-c", type=float, default=1.0)
    parser.add_argument(
        "--check-cached",
        action="store_true",
        help="Exit 0 immediately if eval_450_m2v2_comparison.json exists and is valid JSON.",
    )
    args = parser.parse_args()

    model_dir = (
        pathlib.Path(args.cache_dir)
        if args.cache_dir
        else pathlib.Path(args.out_dir) / args.model.replace("/", "__")
    )
    if args.check_cached:
        out_p = model_dir / "eval_450_m2v2_comparison.json"
        if not out_p.is_file():
            sys.exit(1)
        try:
            data = json.loads(out_p.read_text(encoding="utf-8"))
            sys.exit(0 if isinstance(data, dict) and "systems" in data else 1)
        except Exception:
            sys.exit(1)
    run_comparison(
        args.model,
        model_dir,
        use_mock=args.mock,
        device=args.device,
        dtype=args.dtype,
        offline=args.offline,
        max_items=args.max_items,
        lam_utility=args.lam_utility,
        norm_tokens=args.norm_tokens,
        seed=args.seed,
        reg_c=args.reg_c,
    )


if __name__ == "__main__":
    main()
