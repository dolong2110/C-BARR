"""Phase 7 (W8): Statistical analysis, Clean-446 sensitivity, LaTeX tables, and PDF figures."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_matrix import _regrade_cached_gsm8k_row
from scripts.versioning import get_version_provenance
from src.engine import run_gate1_lexical
from src.logfmt import banner, hl
from src.metrics import (
    SLICE_ORDER,
    compute_conformal_risk_bounds,
    compute_oracle_efficiency_audit,
    evaluate_system_table,
    exact_and_midp_mcnemar,
    filter_clean_446,
    holm_bonferroni,
    wilson_ci,
)
from src.routers import (
    CalibratedRouterParams,
    _build_gate2_feature_vec,
    _finite_float,
    _gate2_target_label,
    _get_slow_rec,
    fit_gate2_logistic_weights,
    resolve_all_variants_for_item,
    score_gate2_probability,
    select_cost_aware_tau,
    select_tiered_budget,
)

# Pre-registered 4-pair confirmatory family (Nov 17 audit: avoids m=12 dilution to alpha=0.00417)
CONFIRMATORY_HOLM_PAIRS: tuple[tuple[str, str], ...] = (
    ("B1b", "M2"),
    ("B0", "M2"),
    ("B2a", "M2"),
    ("M1c", "M2"),
)

VARIANT_ORDER: tuple[str, ...] = (
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

VARIANT_LATEX_LABELS: dict[str, str] = {
    "B0": r"B0: Always Non-Thinking ($B=0$)",
    "B1a": r"B1a: Prompted CoT (\textit{Step-by-step})",
    "B1b": r"B1b: Always Thinking ($B=2048$)",
    "B2a": r"B2a: Self-Selective Prompting",
    "B2b": r"B2b: Budget-Matched Random Esc.",
    "M1a": r"M1a: Regex Constraint Router (Gate 1)",
    "M1b": r"M1b: Fast Entropy + Inertia Router (Gate 2)",
    "M1c": r"M1c: Dual-Signal Cascade (Gate 1 + Gate 2)",
    "M2_B64": r"M2-64: C-BARR Fixed Cap ($B=64$ + Post-Anc.)",
    "M2_B128": r"M2-128: C-BARR Fixed Cap ($B=128$ + Post-Anc.)",
    "M2_B256": r"M2-250: C-BARR Fixed Cap ($B=250$ + Post-Anc.)",
    "M2_Pre": r"M2-Pre: Tiered $\{64,128,250\}$ + Pre-Anc.",
    "M2": r"\textbf{M2: Full C-BARR ($\{64,128,250\}$ + Post-Anc.)}",
}

DEFAULT_SCALE_ORDER: tuple[tuple[str, str, str], ...] = (
    ("Qwen/Qwen3.5-0.8B", r"\texttt{Qwen3.5-0.8B}", "18 DeltaNet + 6 Attn"),
    ("Qwen/Qwen3.5-2B", r"\texttt{Qwen3.5-2B}", "18 DeltaNet + 6 Attn"),
    ("Qwen/Qwen3.5-4B", r"\texttt{Qwen3.5-4B}", "24 DeltaNet + 8 Attn"),
    ("Qwen/Qwen3.5-9B", r"\texttt{Qwen3.5-9B}", "24 DeltaNet + 8 Attn"),
    ("HuggingFaceTB/SmolLM3-3B", r"\texttt{SmolLM3-3B}", "36 Full GQA (3:1 NoPE)"),
)

DIAG_MODES: tuple[tuple[str, str, str], ...] = (
    ("A1_strip", r"\textbf{Control A1: Trace Stripping} ($0$ toks)", r"Deletes \texttt{<think>}; removes decay + interference"),
    ("B2a_post_anchor", r"\textbf{Control B2a: Post-Think Constraint Re-Anchor (Stripped)}", r"Keeps \texttt{<think>}; restores post-\texttt{</think>} recency"),
    ("B1_pre_anchor", r"Control B1: Pre-\texttt{</think>} Clause Re-Anchor", r"Tests pre- vs.\ post-\texttt{</think>} tag locus"),
    ("B2b_post_raw", r"Control B2b: Post-\texttt{</think>} Unstripped Visible Prefill", r"Quantifies wrapper verifier corruption"),
    ("A2_math_swap", r"Control A2: Length-Matched Math Filler ($|t_{\text{think}}|$)", r"Keeps distance; removes task priming"),
    ("B3_early_cut_no_anchor", r"Control B3: Intra-Trace Cut (No Re-Anchor)", r"Halves distance; keeps early priming"),
)

# Colorblind-safe Okabe-Ito palette
OKABE_ITO: dict[str, str] = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#E69F00",
    "grey": "#6E6E6E",
    "black": "#1A1A1A",
}


def _configure_matplotlib() -> None:
    """Configures headless, publication-safe typography (Type 42 TrueType PDF fonts)."""
    plt.rcParams.update({
        # Type 42 embeds scalable TrueType outlines rather than Type 3 bitmaps rejected by venue validators
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "figure.titlesize": 11.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_model_bundle(model_dir: pathlib.Path) -> dict[str, Any] | None:
    """Loads matrix summary + reconstructed per-item runs + strict341/diag491 reports for one model."""
    # Prefer the full calibration-disjoint diag491 summary when present, while keeping strict341 separately
    summary_path = None
    for cand in ("matrix_summary_diag491.json", "matrix_summary_strict341.json", "matrix_summary.json"):
        p = model_dir / cand
        if p.exists():
            summary_path = p
            break
    if summary_path is None:
        return None

    report = json.loads(summary_path.read_text(encoding="utf-8"))
    strict_diag = None
    full_diag = None
    if (model_dir / "matrix_summary_strict341.json").exists():
        strict_diag = json.loads(
            (model_dir / "matrix_summary_strict341.json").read_text(encoding="utf-8")
        ).get("section6_diagnostics")
    if (model_dir / "matrix_summary_diag491.json").exists():
        full_diag = json.loads(
            (model_dir / "matrix_summary_diag491.json").read_text(encoding="utf-8")
        ).get("section6_diagnostics")
    if full_diag is None and strict_diag is None:
        full_diag = report.get("section6_diagnostics")

    calib_summary = None
    if (model_dir / "calibration_summary.json").exists():
        calib_summary = json.loads((model_dir / "calibration_summary.json").read_text(encoding="utf-8"))

    run_manifest = None
    if (model_dir / "run_manifest.json").exists():
        run_manifest = json.loads((model_dir / "run_manifest.json").read_text(encoding="utf-8"))

    tp = report.get("training_params") if isinstance(report.get("training_params"), dict) else {}
    code_ver = str(
        report.get("code_version")
        or tp.get("code_version")
        or (run_manifest or {}).get("code_version")
        or "unknown"
    )
    git_rev = str(
        report.get("git_rev")
        or tp.get("git_rev")
        or (run_manifest or {}).get("git_rev")
        or "unknown"
    )
    provenance = {
        "code_version": code_ver,
        "git_rev": git_rev,
        "run_id": str(report.get("run_id") or tp.get("run_id") or (run_manifest or {}).get("run_id") or ""),
        "sync_seq": (run_manifest or {}).get("sync_seq"),
        "result_version": (run_manifest or {}).get("result_version") or (f"v{code_ver}" if code_ver != "unknown" else "unknown"),
    }

    ext_deltanet_blend = None
    if (model_dir / "ext_deltanet_blend.json").exists():
        ext_deltanet_blend = json.loads(
            (model_dir / "ext_deltanet_blend.json").read_text(encoding="utf-8")
        )
    ext_deltanet_blend_conv_kv = None
    if (model_dir / "ext_deltanet_blend_conv_kv.json").exists():
        ext_deltanet_blend_conv_kv = json.loads(
            (model_dir / "ext_deltanet_blend_conv_kv.json").read_text(encoding="utf-8")
        )
    ext_deltanet_blend_evict_kv = None
    if (model_dir / "ext_deltanet_blend_evict_kv.json").exists():
        ext_deltanet_blend_evict_kv = json.loads(
            (model_dir / "ext_deltanet_blend_evict_kv.json").read_text(encoding="utf-8")
        )
    prefill_mlp = None
    if (model_dir / "prefill_mlp_summary.json").exists():
        prefill_mlp = json.loads(
            (model_dir / "prefill_mlp_summary.json").read_text(encoding="utf-8")
        )
    m2v2_comparison = None
    if (model_dir / "eval_450_m2v2_comparison.json").exists():
        m2v2_comparison = json.loads(
            (model_dir / "eval_450_m2v2_comparison.json").read_text(encoding="utf-8")
        )
    ext_ifbench_ood = None
    if (model_dir / "ext_ifbench_ood.json").exists():
        ext_ifbench_ood = json.loads(
            (model_dir / "ext_ifbench_ood.json").read_text(encoding="utf-8")
        )

    eval_cache_path = model_dir / "eval_450_cache.jsonl"
    items: list[dict[str, Any]] = []
    per_system_runs: dict[str, list[dict[str, Any]]] = {k: [] for k in VARIANT_ORDER}
    if eval_cache_path.exists():
        rows = _load_jsonl(eval_cache_path)
        params = CalibratedRouterParams(**report["calibrated_params"])
        tp_rep = report.get("training_params") if isinstance(report.get("training_params"), dict) else {}
        lam_u = float(tp_rep.get("lam_utility", 0.05))
        norm_t = float(tp_rep.get("norm_tokens", 250.0))
        rng_b2b = np.random.default_rng(42)
        regraded_any = False
        for row in rows:
            if _regrade_cached_gsm8k_row(row):
                regraded_any = True
            items.append(row["item"])
            resolved = resolve_all_variants_for_item(
                row, params, rng_b2b, target_esc_prob=params.m2_escalation_rate_dev
            )
            for k in VARIANT_ORDER:
                per_system_runs[k].append(resolved[k])
        if regraded_any and items and isinstance(report.get("systems"), dict):
            z_b1b = [int(r["z"]) for r in per_system_runs["B1b"]]
            comp_keys = [k for k in VARIANT_ORDER if k != "B1b"]
            raw_ps = [
                exact_and_midp_mcnemar(z_b1b, [int(r["z"]) for r in per_system_runs[k]])["p_exact"]
                for k in comp_keys
            ]
            p_map = dict(zip(comp_keys, holm_bonferroni(raw_ps)))
            for k in VARIANT_ORDER:
                re_tab = evaluate_system_table(
                    items, per_system_runs[k], lam_utility=lam_u, norm_tokens=norm_t
                )
                re_clean = {tk: tv for tk, tv in re_tab.items() if tk not in ("z_vec", "u_vec")}
                if k != "B1b" and k in p_map:
                    re_clean["p_holm_vs_b1b"] = float(p_map[k])
                report["systems"][k] = re_clean

    return {
        "dir": model_dir,
        "report": report,
        "calib_summary": calib_summary,
        "run_manifest": run_manifest,
        "provenance": provenance,
        "strict_diag": strict_diag,
        "full_diag": full_diag,
        "ext_deltanet_blend": ext_deltanet_blend,
        "ext_deltanet_blend_conv_kv": ext_deltanet_blend_conv_kv,
        "ext_deltanet_blend_evict_kv": ext_deltanet_blend_evict_kv,
        "prefill_mlp": prefill_mlp,
        "m2v2_comparison": m2v2_comparison,
        "ext_ifbench_ood": ext_ifbench_ood,
        "items": items,
        "per_system_runs": per_system_runs,
    }


def _vectorize_runs(
    items: list[dict[str, Any]], runs: list[dict[str, Any]], lam_utility: float = 0.05, norm_tokens: float = 250.0
) -> dict[str, np.ndarray]:
    z = np.array([int(r["z"]) for r in runs], dtype=np.float64)
    t_tot = np.array([int(r["T_total"]) for r in runs], dtype=np.float64)
    t_delib = np.array([int(r.get("T_probe", 0)) + int(r.get("T_think", 0)) for r in runs], dtype=np.float64)
    oaa_250 = ((z > 0.5) & (t_delib <= 250.0)).astype(np.float64)
    u_005 = z - lam_utility * (t_tot / norm_tokens)
    slices = np.array([SLICE_ORDER.index(it["slice"]) for it in items], dtype=np.int64)
    return {"z": z, "t_tot": t_tot, "oaa_250": oaa_250, "u_005": u_005, "slices": slices}


def paired_metric_bootstrap_ci(
    items: list[dict[str, Any]],
    runs_a: list[dict[str, Any]],
    runs_b: list[dict[str, Any]],
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Computes paired 10k-resample 95% bootstrap CIs for (B - A) on MacroAcc, OAA_250, and U_0.05."""
    va = _vectorize_runs(items, runs_a)
    vb = _vectorize_runs(items, runs_b)
    n = len(items)
    if n == 0:
        empty = {"diff_b_minus_a": 0.0, "ci95_lo": 0.0, "ci95_hi": 0.0}
        return {"macro_acc": dict(empty), "oaa_250": dict(empty), "u_005": dict(empty)}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))

    # Point estimates
    diff_oaa_vec = vb["oaa_250"] - va["oaa_250"]
    diff_u_vec = vb["u_005"] - va["u_005"]
    diff_z_vec = vb["z"] - va["z"]

    boot_oaa = diff_oaa_vec[idx].mean(axis=1)
    boot_u = diff_u_vec[idx].mean(axis=1)

    # Slice-balanced MacroAcc recomputed per resample so unequal slice counts remain balanced
    slice_masks = [va["slices"] == s_idx for s_idx in range(len(SLICE_ORDER))]
    active_masks = [m for m in slice_masks if np.any(m)]
    point_macro = float(np.mean([diff_z_vec[m].mean() for m in active_masks]))

    slice_boot_means = []
    boot_slices = va["slices"][idx]
    boot_dz = diff_z_vec[idx]
    for s_idx, m in enumerate(slice_masks):
        if not np.any(m):
            continue
        in_slice = boot_slices == s_idx
        counts = np.maximum(1, in_slice.sum(axis=1))
        sums = (boot_dz * in_slice).sum(axis=1)
        slice_boot_means.append(sums / counts)
    boot_macro = np.mean(np.stack(slice_boot_means, axis=0), axis=0)

    def _summarize(pt: float, arr: np.ndarray) -> dict[str, float]:
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return {"diff_b_minus_a": float(pt), "ci95_lo": float(lo), "ci95_hi": float(hi)}

    return {
        "macro_acc": _summarize(point_macro, boot_macro),
        "oaa_250": _summarize(float(diff_oaa_vec.mean()), boot_oaa),
        "u_005": _summarize(float(diff_u_vec.mean()), boot_u),
    }


def _extract_causal_shares(diag: dict[str, Any]) -> tuple[float, float]:
    """Extracts (rho_recency, rho_interference) fractions, supporting percentage fallback keys."""
    ca = diag.get("causal_attribution", {})
    if "rho_recency" in ca and "rho_interference" in ca:
        return float(ca["rho_recency"]), float(ca["rho_interference"])
    return (
        float(ca.get("recency_recovery_share_pct", 0.0)) / 100.0,
        float(ca.get("residual_interference_share_pct", 0.0)) / 100.0,
    )


def _probe_horizon_ablation(
    model_dir: pathlib.Path,
    lam_utility: float = 0.05,
    norm_tokens: float = 250.0,
    reg_c: float = 1.0,
    tau_grid: tuple[float, float, int] = (0.15, 0.85, 29),
    seed: int = 42,
) -> dict[str, Any]:
    """Counterfactual GlimpRouter probe-horizon sweep K in {1, 4, 8, 16, 32} with zero GPU calls."""
    seed_p = model_dir / "seed_60_cache.jsonl"
    dev_p = model_dir / "dev_60_cache.jsonl"
    eval_p = model_dir / "eval_450_cache.jsonl"
    if not (seed_p.exists() and dev_p.exists() and eval_p.exists()):
        return {}
    seed_rows = _load_jsonl(seed_p)
    dev_rows = _load_jsonl(dev_p)
    eval_rows = _load_jsonl(eval_p)
    if not seed_rows or not dev_rows or not eval_rows:
        return {}
    for row in seed_rows + dev_rows + eval_rows:
        _regrade_cached_gsm8k_row(row)

    sample_probe = seed_rows[0].get("probe", {})
    horizons: list[tuple[int, str, str, str]] = [
        (1, "H_first", "H_first", "I_inertia_4" if "I_inertia_4" in sample_probe else "I_inertia"),
    ]
    for k_val in (4, 8, 16):
        if f"H_{k_val}_mean" in sample_probe and f"H_{k_val}_max" in sample_probe:
            i_key = f"I_inertia_{k_val}" if f"I_inertia_{k_val}" in sample_probe else "I_inertia"
            horizons.append((k_val, f"H_{k_val}_mean", f"H_{k_val}_max", i_key))
    horizons.append((32, "H_mean", "H_max", "I_inertia"))

    def _feat(row: dict[str, Any], g1: Any, h_mean_k: str, h_max_k: str, i_k: str) -> np.ndarray:
        pr = row["probe"]
        return np.array(
            [
                _finite_float(pr.get("H_first"), 0.0),
                _finite_float(pr.get(h_mean_k, pr.get("H_mean")), 0.0),
                _finite_float(pr.get(h_max_k, pr.get("H_max")), 0.0),
                _finite_float(pr.get(i_k, pr.get("I_inertia")), 0.0),
                _finite_float(g1.phi_sym, 0.0),
            ],
            dtype=np.float64,
        )

    def _route_row(
        row: dict[str, Any],
        coef: np.ndarray,
        intercept: float,
        tau: float,
        tau_h_high_k: float,
        k_val: int,
        hm_k: str,
        hx_k: str,
        in_k: str,
    ) -> tuple[dict[str, Any], int]:
        prompt_str = row["item"]["prompt"]
        g1 = run_gate1_lexical(prompt_str)
        pr = row["probe"]
        eff_probe = min(int(pr.get("T_probe", 32)), k_val)
        if g1.phi_prec == 1 and g1.phi_plan == 0 and g1.phi_sym == 0:
            return row["b0"], 0
        if g1.phi_plan == 1:
            b_dir = select_tiered_budget(prompt_str, g1, None, tau_h_high_k)
            return _get_slow_rec(row, b_dir, direct=True, prefix="m2"), 1
        fv = _feat(row, g1, hm_k, hx_k, in_k)
        prob = 1.0 / (1.0 + np.exp(-float(np.clip(np.dot(coef, fv) + intercept, -60.0, 60.0))))
        if prob >= tau:
            probe_k = {
                **pr,
                "H_first": _finite_float(pr.get("H_first"), 0.0),
                "H_mean": _finite_float(pr.get(hm_k, pr.get("H_mean")), 0.0),
                "H_max": _finite_float(pr.get(hx_k, pr.get("H_max")), -1.0),
                "I_inertia": int(_finite_float(pr.get(in_k, pr.get("I_inertia")), 0.0)),
                "T_probe": eff_probe,
            }
            b_esc = select_tiered_budget(prompt_str, g1, probe_k, tau_h_high_k)
            slow = _get_slow_rec(row, b_esc, direct=False, prefix="m2")
            merged = dict(slow)
            merged["T_probe"] = eff_probe
            merged["T_total"] = eff_probe + int(slow["T_total"])
            merged["calls"] = 2
            merged["latency_ms"] = float(pr.get("latency_ms", 0.0)) + float(slow.get("latency_ms", 0.0))
            return merged, 1
        return row["fast_done"], 0

    t_min, t_max, t_steps = float(tau_grid[0]), float(tau_grid[1]), max(2, int(tau_grid[2]))
    out: dict[str, Any] = {}
    eval_items = [r["item"] for r in eval_rows]
    for k_val, hm_k, hx_k, in_k in horizons:
        x_s = np.vstack([_feat(r, run_gate1_lexical(r["item"]["prompt"]), hm_k, hx_k, in_k) for r in seed_rows])
        y_s = np.asarray([_gate2_target_label(r) for r in seed_rows], dtype=int)
        coef_list, intercept = fit_gate2_logistic_weights(x_s, y_s, seed=seed, reg_c=reg_c)
        coef = np.asarray(coef_list, dtype=np.float64)
        h_max_helped = [
            hv
            for r in seed_rows
            if int(r["m2_b128"]["z"]) > int(r["fast_done"]["z"])
            for hv in [_finite_float(r["probe"].get(hx_k, r["probe"].get("H_max")), float("nan"))]
            if np.isfinite(hv)
        ]
        tau_h_high_k = float(np.median(h_max_helped)) if h_max_helped else 0.85

        tau_sweep: list[dict[str, float]] = []
        for tau in np.linspace(t_min, t_max, t_steps):
            u_list = []
            esc_list = []
            for r in dev_rows:
                chosen, is_esc = _route_row(r, coef, intercept, float(tau), tau_h_high_k, k_val, hm_k, hx_k, in_k)
                esc_list.append(is_esc)
                u_list.append(float(chosen["z"]) - lam_utility * (float(chosen["T_total"]) / norm_tokens))
            tau_sweep.append(
                {
                    "tau": float(tau),
                    "utility_u005": float(np.mean(u_list)),
                    "esc_rate": float(np.mean(esc_list)),
                }
            )
        best_tau = float(select_cost_aware_tau(tau_sweep)["tau"])

        eval_runs = [
            _route_row(r, coef, intercept, best_tau, tau_h_high_k, k_val, hm_k, hx_k, in_k)[0]
            for r in eval_rows
        ]
        tab = evaluate_system_table(eval_items, eval_runs, lam_utility=lam_utility, norm_tokens=norm_tokens)
        out[f"K_{k_val}"] = {
            "horizon_tokens": k_val,
            "tau_prob": best_tau,
            "tau_h_high": tau_h_high_k,
            "macro_acc": float(tab["macro_acc"]),
            "oaa_250": float(tab["oaa_250"]),
            "u_005": float(tab["u_005"]),
            "mean_t_tot": float(tab["mean_t_tot"]),
        }
    return out


def _l2_regularization_sensitivity(
    model_dir: pathlib.Path,
    lam_utility: float = 0.05,
    norm_tokens: float = 250.0,
    tau_grid: tuple[float, float, int] = (0.15, 0.85, 29),
    seed: int = 42,
    c_values: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0),
) -> list[dict[str, Any]]:
    """CPU-only sensitivity check over uncapped L2 inverse strength C in {0.1, 0.25, 0.5, 1.0}."""
    seed_p = model_dir / "seed_60_cache.jsonl"
    dev_p = model_dir / "dev_60_cache.jsonl"
    eval_p = model_dir / "eval_450_cache.jsonl"
    if not (seed_p.exists() and dev_p.exists() and eval_p.exists()):
        return []
    seed_rows = _load_jsonl(seed_p)
    dev_rows = _load_jsonl(dev_p)
    eval_rows = _load_jsonl(eval_p)
    if not seed_rows or not dev_rows or not eval_rows:
        return []
    for row in seed_rows + dev_rows + eval_rows:
        _regrade_cached_gsm8k_row(row)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x_seed_l, y_seed_l, h_help = [], [], []
    for rec in seed_rows:
        g1 = rec.get("g1") or run_gate1_lexical(rec["item"]["prompt"])
        rec["g1"] = g1
        x_seed_l.append(_build_gate2_feature_vec(g1, rec["probe"]))
        y_seed_l.append(_gate2_target_label(rec))
        if int(rec["m2_b128"]["z"]) > int(rec["fast_done"]["z"]):
            hv = _finite_float(rec["probe"].get("H_max"), float("nan"))
            if np.isfinite(hv):
                h_help.append(hv)
    for rec in dev_rows:
        rec["g1"] = rec.get("g1") or run_gate1_lexical(rec["item"]["prompt"])

    x_arr = np.nan_to_num(np.vstack(x_seed_l), nan=0.0, posinf=0.0, neginf=0.0)
    y_arr = np.asarray(y_seed_l, dtype=int)
    tau_h_high = float(np.median(h_help)) if h_help else 0.85
    eval_items = [r["item"] for r in eval_rows]

    out: list[dict[str, Any]] = []
    for c_val in c_values:
        if len(np.unique(y_arr)) < 2 or not np.any(np.std(x_arr, axis=0) > 1e-12):
            coef_list, intercept = [0.6, 0.8, 0.4, 1.0, 0.5], -1.0
        else:
            scaler = StandardScaler()
            x_scaled = np.nan_to_num(scaler.fit_transform(x_arr), nan=0.0, posinf=0.0, neginf=0.0)
            clf = LogisticRegression(C=float(c_val), class_weight="balanced", random_state=seed)
            clf.fit(x_scaled, y_arr)
            scales = np.where(np.abs(scaler.scale_) > 1e-9, scaler.scale_, 1.0)
            coef_raw = clf.coef_[0] / scales
            intercept = float(clf.intercept_[0] - np.dot(coef_raw, scaler.mean_))
            coef_list = [float(x) for x in coef_raw]

        w_vec = np.asarray(coef_list, dtype=np.float64)
        tau_sweep: list[dict[str, float]] = []
        for tau in np.linspace(tau_grid[0], tau_grid[1], tau_grid[2]):
            utilities: list[float] = []
            escs: list[int] = []
            for rec in dev_rows:
                g1 = rec["g1"]
                prompt_str = rec["item"]["prompt"]
                if g1.phi_prec == 1 and g1.phi_plan == 0 and g1.phi_sym == 0:
                    z_sel, t_sel, e_f = rec["b0"]["z"], rec["b0"]["T_total"], 0
                elif g1.phi_plan == 1:
                    b_dir = select_tiered_budget(prompt_str, g1, None, tau_h_high)
                    sr = _get_slow_rec(rec, b_dir, direct=True)
                    z_sel, t_sel, e_f = sr["z"], sr["T_total"], 1
                else:
                    fv = _build_gate2_feature_vec(g1, rec["probe"])
                    prob = 1.0 / (1.0 + np.exp(-float(np.dot(w_vec, fv) + intercept)))
                    if prob >= float(tau):
                        b_esc = select_tiered_budget(prompt_str, g1, rec["probe"], tau_h_high)
                        sr = _get_slow_rec(rec, b_esc, direct=False)
                        z_sel = sr["z"]
                        t_sel = rec["probe"]["T_probe"] + sr["T_total"]
                        e_f = 1
                    else:
                        z_sel, t_sel, e_f = rec["fast_done"]["z"], rec["fast_done"]["T_total"], 0
                utilities.append(float(z_sel) - lam_utility * (float(t_sel) / norm_tokens))
                escs.append(e_f)
            tau_sweep.append(
                {"tau": float(tau), "utility_u005": float(np.mean(utilities)), "esc_rate": float(np.mean(escs))}
            )

        best = select_cost_aware_tau(tau_sweep)
        params_c = CalibratedRouterParams(
            coef=coef_list,
            intercept=intercept,
            tau_prob=float(best["tau"]),
            tau_h_high=tau_h_high,
            m2_escalation_rate_dev=float(best["esc_rate"]),
        )
        rng_b2b = np.random.default_rng(seed)
        m2_runs = [
            resolve_all_variants_for_item(r, params_c, rng_b2b, float(best["esc_rate"]))["M2"]
            for r in eval_rows
        ]
        m2_tab = evaluate_system_table(eval_items, m2_runs, lam_utility=lam_utility, norm_tokens=norm_tokens)
        out.append(
            {
                "reg_c": float(c_val),
                "is_prod_effective_c": bool(abs(float(c_val) - 0.25) < 1e-9),
                "tau_prob": float(params_c.tau_prob),
                "tau_h_high": float(tau_h_high),
                "coef_l2_norm": float(np.linalg.norm(w_vec)),
                "dev_esc_rate": float(best["esc_rate"]),
                "eval_m2_macro": float(m2_tab["macro_acc"]),
                "eval_m2_oaa250": float(m2_tab["oaa_250"]),
                "eval_m2_u005": float(m2_tab["u_005"]),
                "eval_m2_tok": float(m2_tab["mean_t_tot"]),
            }
        )
    return out


def _collect_diag_k_over_n(
    model_dir: pathlib.Path,
    diag_dict: dict[str, Any] | None,
    *,
    strict_pool: bool,
) -> dict[str, Any]:
    """Extracts exact k/n counts and, when diag_controls_cache.jsonl is present, paired vectors."""
    if not diag_dict:
        return {}
    modes = ("A1_strip", "B2a_post_anchor", "B1_pre_anchor", "B2b_post_raw", "A2_math_swap", "B3_early_cut_no_anchor")
    out: dict[str, Any] = {"modes": {}}
    n_expected = 0
    for m in modes:
        md = diag_dict.get(m)
        if not isinstance(md, dict):
            continue
        n_val = int(md.get("n_ehurt", 0))
        if n_val > 0:
            n_expected = n_val
        k_val = md.get("overall_pass_count")
        if k_val is None and n_val > 0 and "overall_recovery_pct" in md:
            k_val = int(round((float(md["overall_recovery_pct"]) / 100.0) * n_val))
        k_int = int(k_val or 0)
        rate, w_lo, w_hi = wilson_ci(k_int, n_val)
        out["modes"][m] = {
            "k": k_int,
            "n": n_val,
            "pct": 100.0 * rate,
            "wilson_ci95_pct": [100.0 * w_lo, 100.0 * w_hi],
        }

    if n_expected > 0 and "B2a_post_anchor" in out["modes"]:
        k_post = out["modes"]["B2a_post_anchor"]["k"]
        mc_vs_b1b = exact_and_midp_mcnemar([0] * n_expected, [1] * k_post + [0] * (n_expected - k_post))
        out["b2a_vs_b1b_all_fail"] = mc_vs_b1b

    ctrl_path = model_dir / "diag_controls_cache.jsonl"
    ehurt_path = model_dir / "diag_ehurt_cache.jsonl"
    if ctrl_path.is_file() and n_expected > 0:
        eval_uids: set[str] = set()
        eval_p = model_dir / "eval_450_cache.jsonl"
        if strict_pool and eval_p.is_file():
            eval_uids = {str(r["item"]["uid"]) for r in _load_jsonl(eval_p)}
        pool_fname = "ifeval_diag_341_strict.jsonl" if strict_pool else "ifeval_diag_491.jsonl"
        pool_file_uids: set[str] | None = None
        for cand_sf in (ROOT / "data" / pool_fname, ROOT / "data" / "frozen_splits" / pool_fname):
            if cand_sf.is_file():
                pool_file_uids = {str(r["uid"]) for r in _load_jsonl(cand_sf)}
                break

        active_ehurt_uids: set[str] | None = None
        if ehurt_path.is_file():
            active_ehurt_uids = set()
            for erow in _load_jsonl(ehurt_path):
                e_uid = str((erow.get("item") or {}).get("uid") or erow.get("uid") or "")
                if not e_uid:
                    continue
                if pool_file_uids is not None and e_uid not in pool_file_uids:
                    continue
                if strict_pool and pool_file_uids is None and e_uid in eval_uids:
                    continue
                if int((erow.get("b0") or {}).get("z", 0)) == 1 and int((erow.get("b1b") or {}).get("z", 1)) == 0:
                    active_ehurt_uids.add(e_uid)

        by_mode_uid: dict[str, dict[str, int]] = {m: {} for m in modes}
        for row in _load_jsonl(ctrl_path):
            m = str(row.get("mode", ""))
            uid = str(row.get("uid", ""))
            if m not in by_mode_uid:
                continue
            if active_ehurt_uids is not None and len(active_ehurt_uids) == n_expected:
                if uid not in active_ehurt_uids:
                    continue
            else:
                if pool_file_uids is not None and uid not in pool_file_uids:
                    continue
                if strict_pool and pool_file_uids is None and uid in eval_uids:
                    continue
            by_mode_uid[m][uid] = int(row.get("grade", {}).get("z", 0))

        common_uids = sorted(
            set(by_mode_uid["A1_strip"].keys()) & set(by_mode_uid["B2a_post_anchor"].keys())
        )
        if len(common_uids) == n_expected:
            z_a1 = [by_mode_uid["A1_strip"][u] for u in common_uids]
            z_b2a = [by_mode_uid["B2a_post_anchor"][u] for u in common_uids]
            out["paired_uids"] = common_uids
            out["paired_z"] = {"A1_strip": z_a1, "B2a_post_anchor": z_b2a}
            out["b2a_vs_a1_strip"] = exact_and_midp_mcnemar(z_a1, z_b2a)

    # Exact mathematical fallback when A1_strip passes 100% (k_a1 == n_expected => z_a1 is identically all-1s)
    if (
        "b2a_vs_a1_strip" not in out
        and n_expected > 0
        and "A1_strip" in out["modes"]
        and "B2a_post_anchor" in out["modes"]
        and out["modes"]["A1_strip"]["k"] == n_expected
    ):
        k_b2a = int(out["modes"]["B2a_post_anchor"]["k"])
        z_a1_det = [1] * n_expected
        z_b2a_det = [1] * k_b2a + [0] * (n_expected - k_b2a)
        out["paired_z"] = {"A1_strip": z_a1_det, "B2a_post_anchor": z_b2a_det}
        out["b2a_vs_a1_strip"] = exact_and_midp_mcnemar(z_a1_det, z_b2a_det)
    return out


def analyze_model_bundle(bundle: dict[str, Any], n_boot: int, seed: int) -> dict[str, Any]:
    """Runs statistical inference, Clean-446 audit, and Gate checks on a single model bundle."""
    report = bundle["report"]
    items = bundle["items"]
    runs = bundle["per_system_runs"]
    systems = report["systems"]

    clean_sensitivity: dict[str, Any] = {}
    paired_bootstrap: dict[str, Any] = {}
    targeted_mcnemar: dict[str, Any] = {}

    if items and len(runs["M2"]) == len(items):
        # Clean-446 upstream-defect sensitivity check (keys {30, 337, 2078, 3311})
        c_items_b0, c_runs_b0 = filter_clean_446(items, runs["B0"])
        max_macro_delta = 0.0
        for var in ("B0", "B1b", "M1c", "M2"):
            c_items, c_runs = filter_clean_446(items, runs[var])
            c_tab = evaluate_system_table(c_items, c_runs)
            delta_macro = abs(float(c_tab["macro_acc"]) - float(systems[var]["macro_acc"]))
            delta_oaa = abs(float(c_tab["oaa_250"]) - float(systems[var]["oaa_250"]))
            max_macro_delta = max(max_macro_delta, delta_macro)
            clean_sensitivity[var] = {
                "n_clean": len(c_items),
                "n_excluded": len(items) - len(c_items),
                "clean_macro_acc": float(c_tab["macro_acc"]),
                "full_macro_acc": float(systems[var]["macro_acc"]),
                "abs_macro_delta_pp": 100.0 * delta_macro,
                "abs_oaa250_delta_pp": 100.0 * delta_oaa,
            }
        clean_sensitivity["n_clean_items"] = len(c_items_b0)
        clean_sensitivity["max_abs_macro_delta_pp"] = 100.0 * max_macro_delta

        # 10k Paired bootstrap 95% CIs for key comparisons
        for base_k in ("B1b", "B0", "B2a", "M1c", "M2_Pre"):
            paired_bootstrap[f"M2_minus_{base_k}"] = paired_metric_bootstrap_ci(
                items, runs[base_k], runs["M2"], n_boot=n_boot, seed=seed
            )

        # Recompute McNemar exact + mid-p with Holm across all 12 vs B1b + targeted vs M2
        comp_keys = [k for k in VARIANT_ORDER if k != "B1b"]
        z_b1b = [int(r["z"]) for r in runs["B1b"]]
        z_m2 = [int(r["z"]) for r in runs["M2"]]
        mcnemar_raw = [exact_and_midp_mcnemar(z_b1b, [int(r["z"]) for r in runs[k]]) for k in comp_keys]
        p_holm = holm_bonferroni([m["p_exact"] for m in mcnemar_raw])
        for k, mc, pa in zip(comp_keys, mcnemar_raw, p_holm):
            targeted_mcnemar[f"{k}_vs_B1b"] = {**mc, "p_holm": float(pa)}
        for base_k in ("B0", "B2a", "M1c", "M2_Pre"):
            targeted_mcnemar[f"M2_vs_{base_k}"] = exact_and_midp_mcnemar(
                [int(r["z"]) for r in runs[base_k]], z_m2
            )

    # Pre-registered 4-pair Confirmatory Holm Family (avoids 12-row FWER alpha=0.00417 dilution)
    confirmatory_holm_family: dict[str, Any] = {
        "family_size": len(CONFIRMATORY_HOLM_PAIRS),
        "alpha_bonferroni_worst": round(0.05 / max(1, len(CONFIRMATORY_HOLM_PAIRS)), 6),
        "pairs": {},
    }
    if items and len(runs["M2"]) == len(items):
        z_m2_c = [int(r["z"]) for r in runs["M2"]]
        t_m2_tok = float(systems["M2"]["mean_t_tot"])
        raw_c_stats: list[tuple[str, str, str, dict[str, float]]] = []
        for base_k, cand_k in CONFIRMATORY_HOLM_PAIRS:
            pk = f"{cand_k}_vs_{base_k}"
            mc = exact_and_midp_mcnemar([int(r["z"]) for r in runs[base_k]], z_m2_c)
            raw_c_stats.append((pk, base_k, cand_k, mc))
        adj_exact_c = holm_bonferroni([m[3]["p_exact"] for m in raw_c_stats])
        adj_midp_c = holm_bonferroni([m[3]["p_midp"] for m in raw_c_stats])
        for (pk, base_k, cand_k, mc), p_he, p_hm in zip(raw_c_stats, adj_exact_c, adj_midp_c):
            b_tok = max(1e-6, float(systems[base_k]["mean_t_tot"]))
            confirmatory_holm_family["pairs"][pk] = {
                "base": base_k,
                "method": cand_k,
                **mc,
                "p_holm_exact": float(p_he),
                "p_holm_midp": float(p_hm),
                "tok_savings_vs_base_pct": 100.0 * (1.0 - (t_m2_tok / b_tok)),
                "paired_bootstrap": paired_bootstrap.get(f"M2_minus_{base_k}", {}),
            }

    # Gate 4 + Gate 6 + Contingencies R1-R4 audit
    b1b_s = systems["B1b"]
    m2_s = systems["M2"]
    b1b_tok = max(1e-6, float(b1b_s["mean_t_tot"]))
    m2_tok = float(m2_s["mean_t_tot"])
    tok_savings_frac = 1.0 - (m2_tok / b1b_tok)

    esc_rate = float(report["calibrated_params"]["m2_escalation_rate_dev"])
    r1_ok = 0.20 <= esc_rate <= 0.85

    diag_pref = bundle["full_diag"] or bundle["strict_diag"] or {}
    strict_diag = bundle["strict_diag"] or {}
    strict_n_ehurt = int(strict_diag.get("A1_strip", {}).get("n_ehurt", 0))
    pref_n_ehurt = int(diag_pref.get("A1_strip", {}).get("n_ehurt", 0))
    rho_recency, rho_interf = _extract_causal_shares(diag_pref)

    # Contingency R3 checks whether unbudgeted B1b still plateaus below 70% on hard/planning slices

    r3_plateau = (
        float(b1b_s["slice_acc"].get("gsm8k_hard", 0.0)) < 0.70
        or float(b1b_s["slice_acc"].get("ifeval_planning", 0.0)) < 0.70
    )
    r3_trails_hard = (
        float(b1b_s["slice_acc"].get("gsm8k_hard", 0.0))
        - float(m2_s["slice_acc"].get("gsm8k_hard", 0.0))
    ) > 0.03

    gate_audit = {
        "gate4_primary": {
            "m2_macro_ge_b1b": bool(m2_s["macro_acc"] >= b1b_s["macro_acc"]),
            "m2_oaa250_gt_b1b": bool(m2_s["oaa_250"] > b1b_s["oaa_250"]),
            "m2_u005_gt_b1b": bool(m2_s["u_005"] > b1b_s["u_005"]),
            "token_reduction_pct": 100.0 * tok_savings_frac,
            "meets_40pct_token_savings": bool(tok_savings_frac >= 0.40),
            "m2_p_holm_vs_b1b": float(m2_s.get("p_holm_vs_b1b", 1.0)),
        },
        "contingencies": {
            "R1_dev_escalation_rate": esc_rate,
            "R1_in_healthy_range_0.20_to_0.85": r1_ok,
            "R2_rho_recency": rho_recency,
            "R2_rho_interference": rho_interf,
            "R2_recency_dominant": bool(rho_recency >= 0.50),
            "R3_b1b_low_hard_or_plan_flag": r3_plateau,
            "R3_m2_trails_b1b_gsm8k_hard_gt_3pp": r3_trails_hard,
            "R3_m2_b512_evaluated": "M2_B512" in report.get("auxiliary_systems", {}),
            "R4_strict341_n_ehurt": strict_n_ehurt,
            "R4_diag_preferred_n_ehurt": pref_n_ehurt,
            "R4_strict_underpowered_use_diag491": bool(0 < strict_n_ehurt < 30),
        },
    }

    oracle_eff = compute_oracle_efficiency_audit(
        systems, report["oracle_and_4regime"]["oracle_summary"]
    )
    conformal_audit: dict[str, Any] = {}
    if bundle.get("calib_summary") and "conformal_risk_dev" in bundle["calib_summary"]:
        conformal_audit["dev_calibration"] = bundle["calib_summary"]["conformal_risk_dev"]
    if items and len(runs["M2"]) == len(items):
        z_eval_b1b = [int(r["z"]) for r in runs["B1b"]]
        z_eval_b0 = [int(r["z"]) for r in runs["B0"]]
        z_eval_m2 = [int(r["z"]) for r in runs["M2"]]
        eval_deg = compute_conformal_risk_bounds(z_eval_b1b, z_eval_m2, alpha_delta=0.05, n_grid=1)
        eval_deg_b0 = compute_conformal_risk_bounds(z_eval_b0, z_eval_m2, alpha_delta=0.05, n_grid=1)
        conformal_audit["eval_observed_degradation_risk"] = eval_deg["empirical_degradation_risk"]
        conformal_audit["eval_degradation_events"] = eval_deg["degradation_events"]
        conformal_audit["eval_vs_b0_degradation_risk"] = eval_deg_b0["empirical_degradation_risk"]
        conformal_audit["eval_vs_b0_degradation_events"] = eval_deg_b0["degradation_events"]
        if "dev_calibration" in conformal_audit:
            dev_crc = conformal_audit["dev_calibration"]
            conformal_audit["within_hoeffding_ucb"] = bool(
                eval_deg["empirical_degradation_risk"] <= dev_crc["hoeffding_ucb"]
            )
            conformal_audit["within_crc_expectation_bound"] = bool(
                eval_deg["empirical_degradation_risk"] <= dev_crc["crc_expectation_bound"]
            )

    tp = report.get("training_params") if isinstance(report.get("training_params"), dict) else {}
    tg_raw = tp.get("tau_grid")
    tau_grid = (
        (float(tg_raw[0]), float(tg_raw[1]), int(tg_raw[2]))
        if isinstance(tg_raw, (list, tuple)) and len(tg_raw) >= 3
        else (0.15, 0.85, 29)
    )
    probe_horizons = _probe_horizon_ablation(
        bundle["dir"],
        lam_utility=float(tp.get("lam_utility", 0.05)),
        norm_tokens=float(tp.get("norm_tokens", 250.0)),
        reg_c=float(tp.get("logreg_C", 1.0)),
        tau_grid=tau_grid,
        seed=int(tp.get("seed", 42)),
    )
    l2_sensitivity = _l2_regularization_sensitivity(
        bundle["dir"],
        lam_utility=float(tp.get("lam_utility", 0.05)),
        norm_tokens=float(tp.get("norm_tokens", 250.0)),
        tau_grid=tau_grid,
        seed=int(tp.get("seed", 42)),
    )
    section6_k_over_n = {
        "strict341": _collect_diag_k_over_n(bundle["dir"], bundle["strict_diag"], strict_pool=True),
        "diag491": _collect_diag_k_over_n(bundle["dir"], bundle["full_diag"], strict_pool=False),
    }

    return {
        "model": report["model"],
        "diag_pool_loaded": report.get("diag_pool", "unknown"),
        "calibrated_params": report["calibrated_params"],
        "clean_446_sensitivity": clean_sensitivity,
        "paired_bootstrap_95ci": paired_bootstrap,
        "targeted_mcnemar": targeted_mcnemar,
        "confirmatory_holm_family": confirmatory_holm_family,
        "gate_audit": gate_audit,
        "conformal_risk_audit": conformal_audit,
        "oracle_efficiency_audit": oracle_eff,
        "probe_horizon_ablation": probe_horizons,
        "l2_regularization_sensitivity": l2_sensitivity,
        "oracle_and_4regime": report["oracle_and_4regime"],
        "auxiliary_systems": report.get("auxiliary_systems", {}),
        "section6_strict341": bundle["strict_diag"],
        "section6_diag491": bundle["full_diag"],
        "section6_k_over_n": section6_k_over_n,
        "ext_ifbench_ood": bundle.get("ext_ifbench_ood"),
    }


def _fmt_pct(val: float) -> str:
    return f"{100.0 * float(val):.1f}"


def _tex_escape_tt(val: str) -> str:
    return str(val).replace("_", r"\_")


def generate_latex_table_parts(
    bundles: dict[str, dict[str, Any]],
    primary_model: str,
    dense_model: str,
    stats_models: dict[str, Any] | None = None,
    r6_sampled: list[dict[str, Any]] | None = None,
    pooled_diag: dict[str, Any] | None = None,
    r6_summary: list[dict[str, Any]] | None = None,
    ifbench_ood: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Builds modular LaTeX tables (tab1..tab5) populated from observed bundle & W7/Appendix B metrics."""
    header = "% Auto-generated by scripts/analyze_and_plot.py from frozen matrix outputs\n"

    # Table 1: Primary (or first available) model
    t1_key = primary_model if primary_model in bundles else next(iter(bundles))
    t1_rep = bundles[t1_key]["report"]
    t1_sys = t1_rep["systems"]
    t1_orc = t1_rep["oracle_and_4regime"]["oracle_summary"]
    t1_conf = (stats_models or {}).get(t1_key, {}).get("confirmatory_holm_family", {}).get("pairs", {})
    m2_conf_p = float(t1_conf.get("M2_vs_B1b", {}).get("p_holm_exact", 1.0)) if t1_conf else 1.0

    t1_lines = [
        rf"% Table 1: Main C-BARR Comparison ({t1_key})",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Main results across the frozen $N=450$ suite (\texttt{{{_tex_escape_tt(t1_key)}}}). Headline evaluation leads with MacroAcc, $U_{{0.05}}$, $\text{{AUC}}_{{\text{{OAA}}}}$, and Token Savings; $\text{{OAA}}_{{250}}$ equals MicroAcc on \textbf{{M2}} by D13 cap design ($\le 250$ SLA). $^\dagger p_{{\text{{Holm}},12}} < 0.05$ vs.\ \textbf{{B1b}} (exploratory 12-way exact McNemar); $^\ddagger p_{{\text{{Holm}},4}} < 0.05$ (pre-registered 4-pair confirmatory family).}}",
        r"\label{tab:main_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccccccccc}",
        r"\toprule",
        r"\textbf{System Variant} & \textbf{GSM-E} & \textbf{GSM-H} & \textbf{CSQA} & \textbf{IFE-Plan} & \textbf{IFE-Prec} & \textbf{IFE-Conf} & \textbf{MacroAcc} $\uparrow$ & $\textbf{OAA}_{250}$ (SLA) $\uparrow$ & $\textbf{AUC}_{\text{OAA}}$ $\uparrow$ & ${U}_{0.05}$ $\uparrow$ & ${\overline{T}}_{\text{tot}}$ / \textbf{Calls} $\downarrow$ \\",
        r"\midrule",
    ]

    def _row_tex(label: str, row_d: dict[str, Any], sig: str = "") -> str:
        sc = " & ".join(_fmt_pct(row_d["slice_acc"][s]) for s in SLICE_ORDER)
        return (
            f"{label} & {sc} & {_fmt_pct(row_d['macro_acc'])}{sig} & "
            f"{_fmt_pct(row_d['oaa_250'])} & {_fmt_pct(row_d['auc_oaa_2048'])} & "
            f"{_fmt_pct(row_d['u_005'])} & {row_d['mean_t_tot']:.0f} / {row_d['mean_calls']:.2f} \\\\"
        )

    for k in VARIANT_ORDER:
        if k in ("M1a", "M2_B64"):
            t1_lines.append(r"\midrule")
        has_dag = float(t1_sys[k].get("p_holm_vs_b1b", 1.0)) < 0.05
        has_ddag = k == "M2" and m2_conf_p < 0.05
        sig = (
            r"$^{\dagger\ddagger}$"
            if (has_dag and has_ddag)
            else (r"$^\dagger$" if has_dag else (r"$^\ddagger$" if has_ddag else ""))
        )
        t1_lines.append(_row_tex(VARIANT_LATEX_LABELS[k], t1_sys[k], sig=sig))
    if "M2_Prec" in t1_rep.get("auxiliary_systems", {}):
        t1_lines.append(_row_tex(r"M2-Prec: C-BARR Aux.\ ($\phi_{\text{prec}} \to \text{M2-128}$)", t1_rep["auxiliary_systems"]["M2_Prec"]))
    if "M2_B512" in t1_rep.get("auxiliary_systems", {}):
        t1_lines.append(_row_tex(r"M2-512: C-BARR Aux.\ Cap ($B=512$, R3)", t1_rep["auxiliary_systems"]["M2_B512"]))
    t1_lines.append(r"\midrule")
    t1_lines.append(_row_tex(r"\textit{Instance-Level Oracle\_UB}", t1_orc))
    t1_lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}"])

    # Table 2: Cross-Scale 4-Regime Flip Audit & Loop Pathology
    t2_lines = [
        "% Table 2: Cross-Scale 4-Regime Flip Audit, Loop Pathology, and C-BARR Recovery",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-scale 4-regime disagreement audit (\textbf{B0} vs.\ \textbf{B1b} on $N=450$), loop rate, budget exhaustion rate ($B=2048$), and \textbf{M2} token savings.}",
        r"\label{tab:scale_and_pathology}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"\textbf{Model Backbone} & \textbf{Mixer Architecture} & \textbf{\% Helped} & \textbf{\% Hurt} & \textbf{B1b Loop\%} $\downarrow$ & \textbf{B1b Exh.\%} $\downarrow$ & \textbf{M2 Loop\%} $\downarrow$ & \textbf{B1b Macro} & \textbf{M2 Macro} $\uparrow$ & \textbf{Token Sav.\%} $\uparrow$ \\",
        r"\midrule",
    ]
    known_ids = {m for m, _, _ in DEFAULT_SCALE_ORDER}
    scale_rows = list(DEFAULT_SCALE_ORDER)
    for extra_m in bundles:
        if extra_m not in known_ids:
            scale_rows.append((extra_m, rf"\texttt{{{_tex_escape_tt(extra_m)}}}", "Evaluated Backbone"))
    for model_id, tex_name, mixer in scale_rows:
        if model_id == dense_model:
            t2_lines.append(r"\midrule")
        if model_id not in bundles:
            t2_lines.append(f"{tex_name} & {mixer} & -- & -- & -- & -- & -- & -- & -- & -- \\\\")
            continue
        rep = bundles[model_id]["report"]
        o4 = rep["oracle_and_4regime"]
        b1b_s = rep["systems"]["B1b"]
        m2_s = rep["systems"]["M2"]
        sav = 100.0 * (1.0 - float(m2_s["mean_t_tot"]) / max(1e-6, float(b1b_s["mean_t_tot"])))
        t2_lines.append(
            f"{tex_name} & {mixer} & {_fmt_pct(o4['helped_rate'])} & {_fmt_pct(o4['hurt_rate'])} & "
            f"{_fmt_pct(b1b_s['loop_rate'])} & {_fmt_pct(b1b_s['exhaust_rate'])} & "
            f"{_fmt_pct(m2_s['loop_rate'])} & {_fmt_pct(b1b_s['macro_acc'])} & "
            f"{_fmt_pct(m2_s['macro_acc'])} & {sav:.1f} \\\\"
        )
    t2_lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}"])

    # Table 3: Section 6 Causal Diagnostic Control on E_hurt
    p_diag = (
        bundles[primary_model]["full_diag"] or bundles[primary_model]["strict_diag"]
        if primary_model in bundles
        else bundles[t1_key]["full_diag"] or bundles[t1_key]["strict_diag"] or {}
    )
    d_diag = (
        (bundles[dense_model]["full_diag"] or bundles[dense_model]["strict_diag"] or {})
        if dense_model in bundles
        else {}
    )

    def _d_val(d: dict[str, Any], mode: str, field: str) -> str:
        if not d or mode not in d:
            return "--"
        return f"{float(d[mode][field]):.1f}"

    def _rho_val(d: dict[str, Any], which: str) -> str:
        if not d or "causal_attribution" not in d:
            return "--"
        rho_r, rho_i = _extract_causal_shares(d)
        val = rho_r if which == "rho_recency" else rho_i
        return f"{100.0 * val:.1f}"

    def _d_all_kn(d: dict[str, Any], mode: str) -> str:
        if not d or mode not in d:
            return "--"
        md = d[mode]
        n_v = int(md.get("n_ehurt", 0))
        if n_v <= 0:
            return f"{float(md.get('overall_recovery_pct', 0.0)):.1f}"
        k_v = md.get("overall_pass_count")
        if k_v is None:
            k_v = int(round((float(md.get("overall_recovery_pct", 0.0)) / 100.0) * n_v))
        return f"{float(md['overall_recovery_pct']):.1f} ({int(k_v)}/{n_v})"

    p_ehurt_n = int((p_diag or {}).get("A1_strip", {}).get("n_ehurt", 0))
    d_ehurt_n = int((d_diag or {}).get("A1_strip", {}).get("n_ehurt", 0))
    pooled_all_341 = (pooled_diag or {}).get("diag_491_pooled", {}).get("all_models") or (pooled_diag or {}).get("strict341_pooled", {}).get("all_models")
    pooled_footer = ""
    if isinstance(pooled_all_341, dict) and pooled_all_341.get("B2a_post_anchor"):
        pb2a = pooled_all_341["B2a_post_anchor"]
        pa1 = pooled_all_341.get("A1_strip", {})
        mc_p = (pooled_all_341.get("b2a_vs_a1_strip") or {}).get("p_midp")
        mc_b = (pooled_all_341.get("b2a_vs_b1b_all_fail") or {}).get("p_midp")
        pooled_footer = (
            f" Pooled (\\texttt{{{_tex_escape_tt(str(pooled_all_341.get('pool_label', 'diag')))}}}): "
            f"A1={pa1.get('pct', 0.0):.1f}\\% ({pa1.get('k', 0)}/{pa1.get('n', 0)}), "
            f"B2a={pb2a.get('pct', 0.0):.1f}\\% ({pb2a.get('k', 0)}/{pb2a.get('n', 0)}), "
            + (f"$p_{{\\text{{mid}},\\text{{B2a vs A1}}}}={mc_p:.3g}$, " if mc_p is not None else "")
            + (f"$p_{{\\text{{mid}},\\text{{B2a vs B1b}}}}={mc_b:.3g}$." if mc_b is not None else "")
        )

    t3_lines = [
        r"% Table 3: Section 6 Causal Diagnostic Control on E_hurt",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Strict prompt-level recovery (\% and exact $k/n$) on held-out IFEval ``Hurt'' prompts comparing Trace Stripping vs.\ Post-Think Constraint Re-Anchoring.{pooled_footer}}}",
        r"\label{tab:diagnostic_control}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        rf"\textbf{{Intervention Condition (Fixed $t_{{\text{{think}}}}$)}} & \textbf{{Causal Factor Isolated}} & \multicolumn{{3}}{{c}}{{\textbf{{Qwen3.5-4B (DeltaNet, $n={p_ehurt_n}$)}}}} & \multicolumn{{3}}{{c}}{{\textbf{{SmolLM3-3B (Dense, $n={d_ehurt_n}$)}}}} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
        r" & & \textbf{All ($k/n$)} & \textbf{Inhib.} & \textbf{Struct.} & \textbf{All ($k/n$)} & \textbf{Inhib.} & \textbf{Struct.} \\",
        r"\midrule",
        rf"B1b: Unmodified \texttt{{<think>}} ($|t_{{\text{{think}}}}|$) & Baseline failure state (0\%) & 0.0 (0/{p_ehurt_n}) & 0.0 & 0.0 & 0.0 (0/{d_ehurt_n}) & 0.0 & 0.0 \\",
    ]
    for m_key, m_lbl, m_desc in DIAG_MODES:
        if m_key == "B1_pre_anchor":
            t3_lines.append(r"\midrule")
        t3_lines.append(
            f"{m_lbl} & {m_desc} & "
            f"{_d_all_kn(p_diag, m_key)} & {_d_val(p_diag, m_key, 'inhibition_recovery_pct')} & {_d_val(p_diag, m_key, 'structural_recovery_pct')} & "
            f"{_d_all_kn(d_diag, m_key)} & {_d_val(d_diag, m_key, 'inhibition_recovery_pct')} & {_d_val(d_diag, m_key, 'structural_recovery_pct')} \\\\"
        )
    t3_lines.extend([
        r"\midrule",
        rf"\textit{{Derived Recency-Decay Share ($\rho_{{\text{{recency}}}}$, \%)}} & $R_{{\text{{post}}}}/R_{{\text{{strip}}}}$ & \multicolumn{{3}}{{c}}{{{_rho_val(p_diag, 'rho_recency')}}} & \multicolumn{{3}}{{c}}{{{_rho_val(d_diag, 'rho_recency')}}} \\",
        rf"\textit{{Residual Semantic Interference ($\rho_{{\text{{interf}}}}$, \%)}} & $(R_{{\text{{strip}}}}-\max(R_{{\text{{post}}}},R_{{\text{{pre}}}}))/R_{{\text{{strip}}}}$ & \multicolumn{{3}}{{c}}{{{_rho_val(p_diag, 'rho_interference')}}} & \multicolumn{{3}}{{c}}{{{_rho_val(d_diag, 'rho_interference')}}} \\",
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])

    # Table 4: Probe-Horizon Ablation (K=1..32) + Optional W7 Extensions (DeltaNet Blend & Prefill MLP)
    t4_lines = [
        rf"% Table 4: Probe-Horizon Ablation and Mechanistic W7 Extensions ({t1_key})",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Gate-2 probe-horizon ablation ($K \in \{{1,4,8,16,32\}}$) and optional mechanistic extensions (\texttt{{{_tex_escape_tt(t1_key.split('/')[-1])}}}): DeltaNet recurrent-state blending on \texttt{{E\_hurt}} and multi-head prefill probing under \texttt{{IFEval++}} paraphrase shift.}}",
        r"\label{tab:extensions_w7}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Routing / Probe Variant} & \textbf{Configuration} & \textbf{MacroAcc (\%)} $\uparrow$ & $\textbf{OAA}_{250}$ (\%) $\uparrow$ & ${U}_{0.05}$ (\%) $\uparrow$ & ${\overline{T}}_{\text{tot}}$ $\downarrow$ \\",
        r"\midrule",
    ]
    ph_map = (stats_models or {}).get(t1_key, {}).get("probe_horizon_ablation", {})
    if ph_map:
        for k_val in (1, 4, 8, 16, 32):
            kr = ph_map.get(f"K_{k_val}")
            if not kr:
                continue
            k_lbl = r"Gate-2 Probe ($K=32$, full)" if k_val == 32 else f"Gate-2 Probe ($K={k_val}$)"
            t4_lines.append(
                f"{k_lbl} & $\\tau^*={float(kr['tau_prob']):.3f}$ & "
                f"{_fmt_pct(kr['macro_acc'])} & {_fmt_pct(kr['oaa_250'])} & "
                f"{_fmt_pct(kr['u_005'])} & {float(kr['mean_t_tot']):.0f} \\\\"
            )
    else:
        t4_lines.append(r"Gate-2 Probe Horizon Sweep & $K \in \{1,4,8,16,32\}$ & -- & -- & -- & -- \\")

    mlp_models = [m for m in bundles if isinstance(bundles[m].get("prefill_mlp"), dict)]
    for m_id in mlp_models:
        mlp = bundles[m_id]["prefill_mlp"]
        et = mlp.get("eval_table", {})
        if et:
            m_short = _tex_escape_tt(m_id.split("/")[-1])
            t4_lines.append(
                f"Prefill MLP (\\texttt{{{m_short}}}) & Zero-Gen ($\\tau^*={float(mlp.get('best_tau', 0.0)):.3f}$) & "
                f"{_fmt_pct(et.get('macro_acc', 0.0))} & {_fmt_pct(et.get('oaa_250', 0.0))} & "
                f"{_fmt_pct(et.get('u_005', 0.0))} & {float(et.get('mean_t_tot', 0.0)):.0f} \\\\"
            )

    blend_models = [
        m for m in bundles
        if isinstance(bundles[m].get("ext_deltanet_blend"), dict)
        and ("alphas" in bundles[m]["ext_deltanet_blend"] or "sweep" in bundles[m]["ext_deltanet_blend"])
    ]
    blend_conv_kv_models = [
        m for m in bundles
        if isinstance(bundles[m].get("ext_deltanet_blend_conv_kv"), dict)
        and ("alphas" in bundles[m]["ext_deltanet_blend_conv_kv"] or "sweep" in bundles[m]["ext_deltanet_blend_conv_kv"])
    ]
    blend_evict_kv_models = [
        m for m in bundles
        if isinstance(bundles[m].get("ext_deltanet_blend_evict_kv"), dict)
        and ("alphas" in bundles[m]["ext_deltanet_blend_evict_kv"] or "sweep" in bundles[m]["ext_deltanet_blend_evict_kv"])
    ]
    m2v2_models = [
        m for m in bundles
        if isinstance(bundles[m].get("m2v2_comparison"), dict)
        and isinstance(bundles[m]["m2v2_comparison"].get("systems"), dict)
    ]
    has_para = any(
        isinstance((bundles[m].get("prefill_mlp") or {}).get("ifevalpp_paraphrase_stress"), dict)
        for m in bundles
    )
    if (
        blend_models
        or blend_conv_kv_models
        or blend_evict_kv_models
        or m2v2_models
        or has_para
        or (r6_sampled or [])
    ):
        t4_lines.extend([
            r"\midrule",
            r"\textbf{Mechanistic / Robustness Test} & \textbf{Condition} & \textbf{Overall / Prec F1} & \textbf{Inhib.\ / Plan F1} & \textbf{Struct.\ / Scope} & \textbf{$N$} \\",
            r"\midrule",
        ])
        for label_pref, b_key in (
            ("DeltaNet Blend", "ext_deltanet_blend"),
            ("DeltaNet Conv+KV", "ext_deltanet_blend_conv_kv"),
            ("Dense KV Evict", "ext_deltanet_blend_evict_kv"),
        ):
            for m_id in bundles:
                bl = bundles[m_id].get(b_key)
                if not isinstance(bl, dict):
                    continue
                m_short = _tex_escape_tt(m_id.split("/")[-1])
                if "alphas" in bl:
                    for a_k in sorted(bl["alphas"], key=lambda x: float(x)):
                        ad = bl["alphas"][a_k]
                        t4_lines.append(
                            f"{label_pref} (\\texttt{{{m_short}}}) & $\\alpha = {float(ad['alpha']):.2f}$ & "
                            f"{float(ad['overall_recovery_pct']):.1f} & "
                            f"{float(ad['inhibition_recovery_pct']):.1f} & "
                            f"{float(ad['structural_recovery_pct']):.1f} & "
                            f"{int(ad['n_ehurt'])} \\\\"
                        )
                elif "sweep" in bl and bl["sweep"]:
                    t4_lines.append(
                        f"{label_pref} (\\texttt{{{m_short}}}) & mock sweep ({len(bl['sweep'])} $\\alpha$) & "
                        f"-- & -- & conv={bool(bl.get('blend_conv_state'))}, kv={bool(bl.get('evict_think_kv'))} & -- \\\\"
                    )
        for m_id in mlp_models:
            mlp = bundles[m_id]["prefill_mlp"]
            sem = mlp.get("gate1_semantic_probe") or {}
            para = mlp.get("ifevalpp_paraphrase_stress") or {}
            if sem and para:
                r_orig = sem.get("regex_orig", {})
                r_para = para.get("regex_para", {})
                m_orig = sem.get("mlp_orig", {})
                m_para = para.get("mlp_para", {})
                m_short = _tex_escape_tt(m_id.split("/")[-1])
                t4_lines.append(
                    f"Gate-1 Regex Screen (\\texttt{{{m_short}}}) & Orig.\\ $\\to$ \\texttt{{IFEval++}} & "
                    f"{float(r_orig.get('prec', {}).get('f1', 0.0)):.1f} $\\to$ {float(r_para.get('prec', {}).get('f1', 0.0)):.1f} & "
                    f"{float(r_orig.get('plan', {}).get('f1', 0.0)):.1f} $\\to$ {float(r_para.get('plan', {}).get('f1', 0.0)):.1f} & "
                    f"Lexical Rules & {len(bundles[m_id].get('items') or []) or 450} \\\\"
                )
                if m_orig and m_para:
                    t4_lines.append(
                        f"Prefill MLP Probe (\\texttt{{{m_short}}}) & Orig.\\ $\\to$ \\texttt{{IFEval++}} & "
                        f"{float(m_orig.get('prec', {}).get('f1', 0.0)):.1f} $\\to$ {float(m_para.get('prec', {}).get('f1', 0.0)):.1f} & "
                        f"{float(m_orig.get('plan', {}).get('f1', 0.0)):.1f} $\\to$ {float(m_para.get('plan', {}).get('f1', 0.0)):.1f} & "
                        f"Hidden $L/2$ & {len(bundles[m_id].get('items') or []) or 450} \\\\"
                    )
        for m_id in m2v2_models:
            comp = bundles[m_id]["m2v2_comparison"]
            sy = comp["systems"]
            c_s = sy.get("M2_classic", {})
            s_s = sy.get("M2_solve_first", {})
            m_short = _tex_escape_tt(m_id.split("/")[-1])
            p_mcn = (comp.get("paired_mcnemar", {}).get("M2_solve_first_vs_M2_classic", {}) or {}).get("p_exact", 1.0)
            t4_lines.append(
                f"Stage-2 Bridge (\\texttt{{{m_short}}}) & Classic $\\to$ \\texttt{{solve\\_first}} & "
                f"{_fmt_pct(c_s.get('macro_acc', 0.0))} $\\to$ {_fmt_pct(s_s.get('macro_acc', 0.0))} & "
                f"OAA {_fmt_pct(c_s.get('oaa_250', 0.0))} $\\to$ {_fmt_pct(s_s.get('oaa_250', 0.0))} & "
                f"$p={float(p_mcn):.3g}$ ({int(comp.get('stage2_items_replayed', 0))} rep.) & "
                f"{int(comp.get('n_eval', 450))} \\\\"
            )
        for r6 in r6_sampled or []:
            m_short = _tex_escape_tt(str(r6["model"]).split("/")[-1])
            n_eval_r6 = int(r6.get("n_eval") or 450)
            seed_r6 = int(r6.get("seed", 42))
            t4_lines.append(
                f"R6 Sampled M2 (\\texttt{{{m_short}}}) & $T={float(r6['temperature']):g}, p={float(r6['top_p']):g}, \\text{{pp}}={float(r6['presence_penalty']):g}, s={seed_r6}$ & "
                f"Macro {_fmt_pct(r6['sampled_m2_macro'])} & OAA {_fmt_pct(r6['sampled_m2_oaa250'])} & "
                f"Tok {float(r6['sampled_m2_tok']):.0f} (vs B1b {_fmt_pct(r6['sampled_b1b_macro'])}) & {n_eval_r6} \\\\"
            )

    t4_lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}"])

    # Table 5 (Appendix B): OOD IFBench_test, 3-Seed Sampled Validation (T=0.6), and L2 C Sensitivity
    ood_map = ifbench_ood or {
        m: bundles[m]["ext_ifbench_ood"]
        for m in bundles
        if isinstance(bundles[m].get("ext_ifbench_ood"), dict)
    }
    l2_models = [
        m for m in bundles
        if (stats_models or {}).get(m, {}).get("l2_regularization_sensitivity")
    ]
    t5_caption = (
        r"\caption{Appendix B Robustness Checks: (A) Zero-refit OOD generalization on \texttt{allenai/IFBench\_test} ($N=300$ prompts, 58 unseen constraints); (B) 3-seed sampled validation ($T=0.6, p=0.95, \text{pp}=1.5$, mean $\pm$ SD); (C) Gate-2 L2 inverse regularization sensitivity $C \in \{0.1, 0.25\text{ (prod)}, 0.5, 1.0\}$.}"
        if (r6_summary or r6_sampled)
        else r"\caption{Appendix B Robustness Checks: (A) Zero-refit OOD generalization on \texttt{allenai/IFBench\_test} ($N=300$ prompts, 58 unseen constraints); (B) Gate-2 L2 inverse regularization sensitivity $C \in \{0.1, 0.25\text{ (prod)}, 0.5, 1.0\}$ across all five backbones.}"
    )
    t5_lines = [
        r"% Table 5 (Appendix B): OOD IFBench_test, Optional Sampled Regime, and L2 C Sensitivity",
        r"\begin{table}[t]",
        r"\centering",
        t5_caption,
        r"\label{tab:appendix_b_robustness}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Robustness Suite} & \textbf{Backbone / Variant} & \textbf{Primary Acc.\ / Macro} & \textbf{Instr.\ / OAA$_{250}$} & \textbf{$U_{0.05}$ / Mean Tok} & \textbf{Stat.\ / Delta} \\",
        r"\midrule",
    ]
    if ood_map:
        for m_id, ood in ood_map.items():
            m_short = _tex_escape_tt(str(m_id).split("/")[-1])
            syss = ood.get("systems", {})
            mc_b1b = ood.get("mcnemar", {}).get("M2_vs_B1b", {})
            for v_k in ("B0", "B1b", "M2"):
                vs = syss.get(v_k)
                if not isinstance(vs, dict) or not vs.get("n"):
                    continue
                stat_col = (
                    f"$p_{{\\text{{exact}}}}={float(mc_b1b.get('p_exact', 1.0)):.3g}$ (vs B1b)"
                    if v_k == "M2" and mc_b1b
                    else f"$N={int(vs.get('n', 0))}$"
                )
                t5_lines.append(
                    f"A: IFBench OOD (\\texttt{{{m_short}}}) & \\textbf{{{v_k}}} & "
                    f"{float(vs.get('prompt_strict_pct', 0.0)):.1f}\\% ({int(vs.get('prompt_pass', 0))}/{int(vs.get('n', 0))}) & "
                    f"{float(vs.get('inst_strict_pct', 0.0)):.1f}\\% / {float(vs.get('oaa_250_pct', 0.0)):.1f}\\% & "
                    f"{float(vs.get('u_005_pct', 0.0)):.1f}\\% / {float(vs.get('mean_t_tot', 0.0)):.0f} & {stat_col} \\\\"
                )

    if r6_summary:
        t5_lines.append(r"\midrule")
        for rs in r6_summary:
            m_short = _tex_escape_tt(str(rs["model"]).split("/")[-1])
            seeds_str = ",".join(str(s) for s in rs["seeds"])
            t5_lines.append(
                f"B: Sampled ($k={rs['n_seeds']}$: {seeds_str}) & \\texttt{{{m_short}}} ($T={rs['temperature']:g}, pp={rs['presence_penalty']:g}$) & "
                f"M2 ${100*rs['m2_macro_mean']:.1f} \\pm {100*rs['m2_macro_std']:.1f}$\\% & "
                f"OAA ${100*rs['m2_oaa250_mean']:.1f} \\pm {100*rs['m2_oaa250_std']:.1f}$\\% & "
                f"${100*rs['m2_u005_mean']:.1f}$ / ${rs['m2_tok_mean']:.0f} \\pm {rs['m2_tok_std']:.0f}$ & "
                f"$\\Delta$B1b ${rs['delta_m2_minus_b1b_pp']:+.1f}$ pp ({rs['delta_tok_savings_pct']:.1f}\\% saved) \\\\"
            )
    elif r6_sampled:
        t5_lines.append(r"\midrule")
        for r6 in r6_sampled:
            m_short = _tex_escape_tt(str(r6["model"]).split("/")[-1])
            t5_lines.append(
                f"B: Sampled (seed {int(r6.get('seed', 42))}) & \\texttt{{{m_short}}} ($T={float(r6['temperature']):g}$) & "
                f"M2 {_fmt_pct(r6['sampled_m2_macro'])}\\% & OAA {_fmt_pct(r6['sampled_m2_oaa250'])}\\% & "
                f"Tok {float(r6['sampled_m2_tok']):.0f} & B1b {_fmt_pct(r6['sampled_b1b_macro'])}\\% \\\\"
            )

    l2_prefix = "C" if (r6_summary or r6_sampled) else "B"
    if ood_map or r6_summary or r6_sampled:
        t5_lines.append(r"\midrule")
    if l2_models:
        for m_id in l2_models:
            m_short = _tex_escape_tt(str(m_id).split("/")[-1])
            for crow in (stats_models or {})[m_id]["l2_regularization_sensitivity"]:
                tag = " (prod)" if crow.get("is_prod_effective_c") else ""
                t5_lines.append(
                    f"{l2_prefix}: Gate-2 L2 (\\texttt{{{m_short}}}) & $C={float(crow['reg_c']):g}${tag} & "
                    f"{_fmt_pct(crow['eval_m2_macro'])}\\% & {_fmt_pct(crow['eval_m2_oaa250'])}\\% & "
                    f"{_fmt_pct(crow['eval_m2_u005'])}\\% / {float(crow['eval_m2_tok']):.0f} & "
                    f"$\\tau^*={float(crow['tau_prob']):.3f}$, $\\|w\\|_2={float(crow['coef_l2_norm']):.2f}$ \\\\"
                )
    else:
        t5_lines.append(r"C: Gate-2 L2 ($C \in \{0.1,0.25,0.5,1.0\}$) & Requires cached shards & -- & -- & -- & -- \\")
    t5_lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}"])

    return {
        "tab1_main_results.tex": header + "\n".join(t1_lines) + "\n",
        "tab2_scale_pathology.tex": header + "\n".join(t2_lines) + "\n",
        "tab3_diagnostic_control.tex": header + "\n".join(t3_lines) + "\n",
        "tab4_extensions_w7.tex": header + "\n".join(t4_lines) + "\n",
        "tab5_appendix_ood_r6_l2.tex": header + "\n".join(t5_lines) + "\n",
    }


def generate_latex_tables(
    bundles: dict[str, dict[str, Any]],
    primary_model: str,
    dense_model: str,
    stats_models: dict[str, Any] | None = None,
    r6_sampled: list[dict[str, Any]] | None = None,
    pooled_diag: dict[str, Any] | None = None,
    r6_summary: list[dict[str, Any]] | None = None,
    ifbench_ood: dict[str, Any] | None = None,
) -> str:
    """Emits complete LaTeX Tables 1-5 populated from observed bundle & W7/Appendix B metrics."""
    parts = generate_latex_table_parts(
        bundles,
        primary_model,
        dense_model,
        stats_models=stats_models,
        r6_sampled=r6_sampled,
        pooled_diag=pooled_diag,
        r6_summary=r6_summary,
        ifbench_ood=ifbench_ood,
    )
    blocks = [
        "% Auto-generated by scripts/analyze_and_plot.py from frozen matrix outputs",
        *[
            txt.split("\n", 1)[1].rstrip() if txt.startswith("% Auto-generated") else txt.rstrip()
            for txt in parts.values()
        ],
    ]
    return "\n\n".join(blocks) + "\n"


def plot_fig0_architecture_overview(fig_path: pathlib.Path) -> None:
    """Generates Figure 0: Vector schematic of the 3-stage C-BARR inference & state-bridging pipeline."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.8, 2.95))
    ax.set_xlim(0.0, 11.8)
    ax.set_ylim(0.0, 3.05)
    ax.axis("off")

    def _box(x: float, y: float, w: float, h: float, title: str, body: str, fc: str, ec: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.10",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.35,
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h - 0.16,
            title,
            ha="center",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color="#111111",
            zorder=3,
        )
        ax.text(
            x + w / 2,
            y + (h - 0.24) / 2,
            body,
            ha="center",
            va="center",
            fontsize=7.4,
            color="#222222",
            zorder=3,
        )

    def _arrow(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        label: str = "",
        col: str = "#333333",
        lx: float | None = None,
        ly: float | None = None,
    ) -> None:
        arr = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>,head_length=5.5,head_width=3.5",
            linewidth=1.3,
            color=col,
            zorder=4,
        )
        ax.add_patch(arr)
        if label:
            tx = (x0 + x1) / 2 if lx is None else lx
            ty = (y0 + y1) / 2 if ly is None else ly
            ax.text(
                tx,
                ty,
                label,
                ha="center",
                va="center",
                fontsize=7.1,
                color=col,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.13", fc="white", ec="none", alpha=0.92),
                zorder=5,
            )

    # Column 1: User Prompt
    _box(0.08, 1.00, 1.38, 1.02, "User Prompt $x$", "Frozen weights\n$\\mathcal{M}_\\theta$ (no FT)", "#F4F6F8", "#4A5568")
    _arrow(1.52, 1.51, 1.92, 1.51)

    # Column 2: Gate 1
    _box(
        1.98,
        0.88,
        2.22,
        1.26,
        "Gate 1: Lexical Screen",
        "25 numeral-masked rules\n$\\phi_{\\mathrm{plan}}, \\phi_{\\mathrm{prec}}, \\phi_{\\mathrm{sym}}$\nConflict clause $c_{\\mathrm{anc}}(x)$",
        "#EBF8FF",
        OKABE_ITO["blue"],
    )

    # Branch A (Top): Pure precision -> Direct Non-Thinking
    _arrow(4.26, 1.82, 5.22, 2.38, "Pure $\\phi_{\\mathrm{prec}}$", OKABE_ITO["green"], lx=4.72, ly=2.24)
    _box(
        5.28,
        2.06,
        2.68,
        0.82,
        "Direct Non-Thinking ($B=0$)",
        "Bypass <think> drift & loops\nZero deliberation overhead",
        "#E6FFFA",
        OKABE_ITO["green"],
    )

    # Branch B (Middle): Planning hit -> straight to Method M2
    _arrow(4.26, 1.51, 8.88, 1.51, "$\\phi_{\\mathrm{plan}}=1$ (planning tier)", OKABE_ITO["purple"], lx=6.62, ly=1.67)

    # Branch C (Bottom): Unmatched -> Gate 2 Fast Probe
    _arrow(4.26, 1.16, 5.22, 0.68, "Unmatched", OKABE_ITO["sky"], lx=4.72, ly=0.76)
    _box(
        5.28,
        0.14,
        2.68,
        1.04,
        "Gate 2: $K\\leq 32$ Fast Probe",
        "Non-thinking $(H_1, \\bar{H}, H_{\\max}, I_{\\mathrm{inertia}})$\nL2 scorer $p_{\\mathrm{slow}}(x) \\lessgtr \\tau^*$ ($6$ scalars)",
        "#FFF5EB",
        OKABE_ITO["orange"],
    )

    # Gate 2 -> Escalate to Method M2 vs Fast Continuation
    _arrow(8.02, 0.92, 8.88, 1.24, "$p_{\\mathrm{slow}} \\geq \\tau^*$", OKABE_ITO["orange"], lx=8.43, ly=1.18)
    _arrow(8.02, 0.44, 8.88, 0.44, "$p_{\\mathrm{slow}} < \\tau^*$", OKABE_ITO["green"], lx=8.45, ly=0.60)

    # Column 4 Top: Compliant Answer
    _arrow(8.02, 2.47, 8.88, 2.47, "", OKABE_ITO["green"])
    _box(
        8.94,
        2.16,
        2.74,
        0.68,
        "Compliant Answer $a_{\\mathrm{final}}$",
        "$T_{\\mathrm{delib}} \\leq 250$ (100% $\\mathrm{OAA}_{250}$ SLA)",
        "#F0FFF4",
        OKABE_ITO["green"],
    )

    # Column 4 Middle: Method M2
    _box(
        8.94,
        0.86,
        2.74,
        1.08,
        "Method M2: Capped Bridge",
        "Cap $B \\in \\{64,128,250-T_{\\mathrm{probe}}\\}$\nSplice P_TERM + </think>\nTurn-delimited user $c_{\\mathrm{anc}}(x)$",
        "#FAF5FF",
        OKABE_ITO["purple"],
    )
    _arrow(10.31, 1.98, 10.31, 2.14, "", OKABE_ITO["purple"])

    # Column 4 Bottom: Stateful Fast Continuation
    _box(
        8.94,
        0.12,
        2.74,
        0.58,
        "Stateful Continuation ($\\mathcal{C}_1$)",
        "Reuse KV/DeltaNet cache (no re-prefill)",
        "#F0FFF4",
        OKABE_ITO["green"],
    )

    fig.tight_layout(pad=0.12)
    fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig1_pareto_frontier(
    bundles: dict[str, dict[str, Any]],
    primary_model: str,
    fig_path: pathlib.Path,
) -> None:
    """Generates Figure 1: (a) Tokens vs. MacroAcc Pareto Frontier + (b) Per-Slice Accuracy."""
    t_key = primary_model if primary_model in bundles else next(iter(bundles))
    rep = bundles[t_key]["report"]
    sys_d = rep["systems"]
    oracle_s = rep["oracle_and_4regime"]["oracle_summary"]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    style_groups = [
        (["B0", "B1a", "B1b", "B2a", "B2b"], OKABE_ITO["grey"], "o", "Baselines (B0-B2b)"),
        (["M1a", "M1b", "M1c"], OKABE_ITO["sky"], "s", "Unbudgeted Routers (M1a-M1c)"),
        (["M2_B64", "M2_B128", "M2_B256", "M2_Pre"], OKABE_ITO["orange"], "^", "C-BARR Caps / Pre-Anc."),
    ]
    custom_offsets: dict[str, tuple[int, int]] = {
        "B0": (-20, 7),
        "B2a": (-24, -12),
        "B1a": (9, -2),
        "B2b": (6, 4),
        "B1b": (-26, 6),
        "M1a": (6, 4),
        "M1b": (9, -8),
        "M1c": (6, 4),
        "M2_Pre": (-44, -3),
    }
    for keys, col, mk, grp_lbl in style_groups:
        xs = [float(sys_d[k]["mean_t_tot"]) for k in keys]
        ys = [100.0 * float(sys_d[k]["macro_acc"]) for k in keys]
        ax_l.scatter(xs, ys, color=col, marker=mk, s=55, label=grp_lbl, zorder=3)
        for k, x, y in zip(keys, xs, ys):
            if k in ("M2_B64", "M2_B128", "M2_B256"):
                continue
            ann_k = "M2-Pre" if k == "M2_Pre" else k
            dx, dy = custom_offsets.get(k, (5, 4))
            ax_l.annotate(ann_k, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.6)

    ax_l.scatter(
        [float(sys_d["M2"]["mean_t_tot"])],
        [100.0 * float(sys_d["M2"]["macro_acc"])],
        color=OKABE_ITO["green"],
        marker="D",
        s=90,
        edgecolor=OKABE_ITO["black"],
        linewidth=1.0,
        label="M2 (Full C-BARR)",
        zorder=5,
    )
    ax_l.annotate(
        "M2",
        (float(sys_d["M2"]["mean_t_tot"]), 100.0 * float(sys_d["M2"]["macro_acc"])),
        xytext=(-14, 10),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
    )
    ax_l.scatter(
        [float(oracle_s["mean_t_tot"])],
        [100.0 * float(oracle_s["macro_acc"])],
        color=OKABE_ITO["purple"],
        marker="*",
        s=120,
        label="Oracle_UB",
        zorder=4,
    )
    ax_l.annotate(
        "Oracle_UB",
        (float(oracle_s["mean_t_tot"]), 100.0 * float(oracle_s["macro_acc"])),
        xytext=(7, -3),
        textcoords="offset points",
        fontsize=7.6,
    )
    aux_d = rep.get("auxiliary_systems") if isinstance(rep.get("auxiliary_systems"), dict) else {}
    for aux_k, aux_lbl, aux_mk, aux_col, aux_off in (
        ("M2_Prec", "M2-Prec", "X", OKABE_ITO["blue"], (-4, 12)),
        ("M2_B512", "M2-512 (R3)", "P", OKABE_ITO["yellow"], (9, 2)),
    ):
        if aux_k in aux_d:
            ax_l.scatter(
                [float(aux_d[aux_k]["mean_t_tot"])],
                [100.0 * float(aux_d[aux_k]["macro_acc"])],
                color=aux_col,
                marker=aux_mk,
                s=75,
                edgecolor=OKABE_ITO["black"],
                linewidth=0.8,
                label=aux_lbl,
                zorder=5,
            )
            ax_l.annotate(
                aux_lbl,
                (float(aux_d[aux_k]["mean_t_tot"]), 100.0 * float(aux_d[aux_k]["macro_acc"])),
                xytext=aux_off,
                textcoords="offset points",
                fontsize=7.5,
            )
    ax_l.set_xlim(175, 1840)
    ax_l.set_ylim(25, 99)
    ax_l.set_xlabel(r"Mean Total Tokens ($\overline{T}_{\mathrm{tot}}$)")
    ax_l.set_ylabel("Macro Accuracy (%)")
    ax_l.set_title(f"(a) Accuracy vs. Token Budget ({t_key.split('/')[-1]})")
    ax_l.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.6)

    slice_short = ["GSM-E", "GSM-H", "CSQA", "IFE-Plan", "IFE-Prec", "IFE-Conf"]
    x_pos = np.arange(len(SLICE_ORDER))
    w = 0.26
    bars_spec = [
        ("B0", OKABE_ITO["grey"], "B0 (No Think)"),
        ("B1b", OKABE_ITO["blue"], "B1b (Think 2048)"),
        ("M2", OKABE_ITO["green"], "M2 (Full C-BARR)"),
    ]
    for offset_idx, (vk, col, lbl) in enumerate(bars_spec):
        vals = [100.0 * float(sys_d[vk]["slice_acc"][s]) for s in SLICE_ORDER]
        ax_r.bar(x_pos + (offset_idx - 1) * w, vals, width=w, color=col, label=lbl)

    ax_r.set_xticks(x_pos)
    ax_r.set_xticklabels(slice_short, rotation=18, ha="right")
    ax_r.set_ylabel("Slice Accuracy (%)")
    ax_r.set_ylim(0, 108)
    ax_r.set_title("(b) Per-Slice Accuracy Breakdown")
    ax_r.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig2_bsr_sla(
    bundles: dict[str, dict[str, Any]],
    primary_model: str,
    dense_model: str,
    fig_path: pathlib.Path,
) -> None:
    """Generates Figure 2: Budget Survival Rate BSR(B) SLA curves across B in {64..2048}."""
    panel_models = [m for m in (primary_model, dense_model) if m in bundles]
    if not panel_models:
        panel_models = [next(iter(bundles))]

    fig, axes = plt.subplots(1, len(panel_models), figsize=(5.2 * len(panel_models), 3.8), squeeze=False)
    budgets = ["64", "128", "256", "512", "1024", "2048"]
    x_idx = np.arange(len(budgets))
    curve_specs = [
        ("B0", OKABE_ITO["grey"], "--", "o", "B0"),
        ("B1b", OKABE_ITO["blue"], "-.", "s", "B1b (Always Think)"),
        ("M1c", OKABE_ITO["sky"], ":", "^", "M1c (Unbudgeted)"),
        ("M2_Pre", OKABE_ITO["orange"], "--", "v", "M2_Pre (Pre-Think Anc.)"),
        ("M2", OKABE_ITO["green"], "-", "D", "M2 (Full C-BARR)"),
    ]

    for ax, m_key in zip(axes[0], panel_models):
        sys_d = bundles[m_key]["report"]["systems"]
        for vk, col, ls, mk, lbl in curve_specs:
            bsr_dict = sys_d[vk]["bsr"]
            ys = [100.0 * float(bsr_dict.get(b, 0.0)) for b in budgets]
            ax.plot(x_idx, ys, color=col, linestyle=ls, marker=mk, linewidth=1.8, markersize=5, label=lbl)
        # Mark index 2 (B=256), highlighting C-BARR's B<=250 SLA operating point
        ax.axvline(x=2.0, color=OKABE_ITO["black"], linestyle=":", linewidth=1.1, alpha=0.8)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(budgets)
        ax.set_xlabel("Budget SLA Threshold $B$ (Tokens)")
        ax.set_ylabel("Budget Survival Rate BSR($B$) (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"{m_key.split('/')[-1]}")
        ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig3_ehurt_recovery(
    bundles: dict[str, dict[str, Any]],
    primary_model: str,
    dense_model: str,
    fig_path: pathlib.Path,
) -> None:
    """Generates Figure 3: Section 6 Causal Diagnostic recovery (%) with Wilson 95% CIs on E_hurt."""
    panel_models = [m for m in (primary_model, dense_model) if m in bundles]
    if not panel_models:
        panel_models = [next(iter(bundles))]

    mode_order = [
        ("A1_strip", "A1\nStrip"),
        ("B2a_post_anchor", "B2a\nPost-Anc"),
        ("B1_pre_anchor", "B1\nPre-Anc"),
        ("B2b_post_raw", "B2b\nRaw"),
        ("A2_math_swap", "A2\nMathSwap"),
        ("B3_early_cut_no_anchor", "B3\nEarlyCut"),
    ]

    fig, axes = plt.subplots(1, len(panel_models), figsize=(5.5 * len(panel_models), 3.9), squeeze=False)
    for ax, m_key in zip(axes[0], panel_models):
        b = bundles[m_key]
        diag = b["full_diag"] or b["strict_diag"] or {}
        x_pos = np.arange(len(mode_order))
        w = 0.25

        overalls, err_lo, err_hi, inh_vals, str_vals = [], [], [], [], []
        n_ehurt = int(diag.get("A1_strip", {}).get("n_ehurt", 0))
        for mk, _ in mode_order:
            md = diag.get(mk, {})
            ov = float(md.get("overall_recovery_pct", 0.0))
            if "overall_wilson_ci95" in md:
                ci_lo, ci_hi = md["overall_wilson_ci95"]
            else:
                k_pass = int(round(ov / 100.0 * n_ehurt))
                _, lo_f, hi_f = wilson_ci(k_pass, n_ehurt)
                ci_lo, ci_hi = 100.0 * lo_f, 100.0 * hi_f
            overalls.append(ov)
            err_lo.append(max(0.0, ov - float(ci_lo)))
            err_hi.append(max(0.0, float(ci_hi) - ov))
            inh_vals.append(float(md.get("inhibition_recovery_pct", 0.0)))
            str_vals.append(float(md.get("structural_recovery_pct", 0.0)))

        ax.bar(
            x_pos - w,
            overalls,
            width=w,
            yerr=[err_lo, err_hi],
            capsize=3,
            color=OKABE_ITO["blue"],
            label="Overall (95% Wilson CI)",
        )
        ax.bar(x_pos, inh_vals, width=w, color=OKABE_ITO["green"], label="Inhibition")
        ax.bar(x_pos + w, str_vals, width=w, color=OKABE_ITO["orange"], label="Structural")

        rho_r_frac, rho_i_frac = _extract_causal_shares(diag)
        rho_r = 100.0 * rho_r_frac
        rho_i = 100.0 * rho_i_frac
        ax.set_xticks(x_pos)
        ax.set_xticklabels([lbl for _, lbl in mode_order])
        ax.set_ylim(0, 112)
        ax.set_ylabel("Strict Recovery Rate (%)")
        ax.set_title(
            f"{m_key.split('/')[-1]} ($|E_{{hurt}}|={n_ehurt}$; "
            f"$\\rho_{{rec}}={rho_r:.1f}\\%$, $\\rho_{{int}}={rho_i:.1f}\\%$)"
        )
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig4_horizons_and_w7(
    bundles: dict[str, dict[str, Any]],
    primary_model: str,
    dense_model: str,
    stats_models: dict[str, Any],
    fig_path: pathlib.Path,
) -> None:
    """Generates Figure 4: (a) Gate-2 Probe Horizon K=1..32 vs Prefill MLP & (b) W7 Blending / Loop Pathology."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.5, 3.9))

    # Panel (a): Multi-backbone Probe-Horizon Ablation K in {1,4,8,16,32} + Zero-Generation Prefill MLP
    t_key = primary_model if primary_model in bundles else next(iter(bundles))
    ks = [1, 4, 8, 16, 32]
    horizon_specs = [
        ("Qwen/Qwen3.5-9B", OKABE_ITO["purple"], "-", "s", "Qwen3.5-9B"),
        (t_key, OKABE_ITO["blue"], "-", "o", f"{t_key.split('/')[-1]} (Primary)"),
        ("Qwen/Qwen3.5-2B", OKABE_ITO["green"], "--", "^", "Qwen3.5-2B"),
        (dense_model, OKABE_ITO["orange"], "-.", "D", f"{dense_model.split('/')[-1]} (Dense)"),
    ]
    plotted_vals: list[float] = []
    seen_h: set[str] = set()
    for m_h, col_h, ls_h, mk_h, lbl_h in horizon_specs:
        if m_h in seen_h or m_h not in bundles:
            continue
        seen_h.add(m_h)
        ph_m = (stats_models.get(m_h, {}) or {}).get("probe_horizon_ablation", {})
        xs_k = [k for k in ks if f"K_{k}" in ph_m]
        if xs_k:
            ys_k = [100.0 * float(ph_m[f"K_{k}"]["macro_acc"]) for k in xs_k]
            plotted_vals.extend(ys_k)
            ax_l.plot(xs_k, ys_k, color=col_h, linestyle=ls_h, marker=mk_h, linewidth=1.9, markersize=5, label=lbl_h)
        mlp_m = bundles[m_h].get("prefill_mlp")
        if isinstance(mlp_m, dict) and isinstance(mlp_m.get("eval_table"), dict):
            mv = 100.0 * float(mlp_m["eval_table"].get("macro_acc", 0.0))
            plotted_vals.append(mv)
            ax_l.scatter([0], [mv], color=col_h, marker="*", s=105, edgecolor="#111111", linewidth=0.6, zorder=5)
    if any(isinstance(bundles[m].get("prefill_mlp"), dict) for m in seen_h):
        ax_l.scatter([], [], color="#555555", marker="*", s=95, label="$K=0$ Prefill MLP")
    ax_l.set_xticks([0, 1, 4, 8, 16, 32])
    ax_l.set_xticklabels(["0\n(MLP)", "1", "4", "8", "16", "32"])
    ax_l.set_xlabel("Gate-2 Non-Thinking Probe Horizon $K$ (Tokens)")
    ax_l.set_ylabel("Eval-450 Macro Accuracy (%)")
    if plotted_vals:
        ax_l.set_ylim(max(0.0, min(plotted_vals) - 6.0), min(100.0, max(plotted_vals) + 5.0))
    else:
        ax_l.set_ylim(0, 105)
    ax_l.set_title("(a) Probe Horizon vs. Zero-Gen Prefill ($K=0$)")
    if ax_l.get_legend_handles_labels()[0]:
        ax_l.legend(loc="center right", frameon=True, framealpha=0.92, fontsize=7.5)

    # Panel (b): W7 State Restoration (DeltaNet alpha-blend / conv+KV & Dense KV eviction)
    bl = bundles[t_key].get("ext_deltanet_blend")
    if isinstance(bl, dict) and isinstance(bl.get("alphas"), dict) and bl["alphas"]:
        a_keys = sorted(bl["alphas"].keys(), key=lambda x: float(x))
        alphas = [float(bl["alphas"][k]["alpha"]) for k in a_keys]
        ov_pct = [float(bl["alphas"][k]["overall_recovery_pct"]) for k in a_keys]
        ax_r.plot(alphas, ov_pct, color=OKABE_ITO["purple"], marker="o", linewidth=2.0, label="Qwen3.5-4B Rec. Blend")
        bl_ck = bundles[t_key].get("ext_deltanet_blend_conv_kv")
        if isinstance(bl_ck, dict) and isinstance(bl_ck.get("alphas"), dict) and bl_ck["alphas"]:
            ck_keys = sorted(bl_ck["alphas"].keys(), key=lambda x: float(x))
            ax_r.plot(
                [float(bl_ck["alphas"][k]["alpha"]) for k in ck_keys],
                [float(bl_ck["alphas"][k]["overall_recovery_pct"]) for k in ck_keys],
                color=OKABE_ITO["sky"],
                marker="^",
                linestyle="-.",
                linewidth=1.9,
                label="Qwen3.5-4B Conv+Rec+KV Evict",
            )
        diag_d = bundles[t_key].get("full_diag") or bundles[t_key].get("strict_diag") or {}
        if "B2a_post_anchor" in diag_d:
            ax_r.axhline(
                y=float(diag_d["B2a_post_anchor"]["overall_recovery_pct"]),
                color=OKABE_ITO["green"],
                linestyle="--",
                linewidth=1.3,
                label="Qwen3.5-4B B2a Text Anchor (96.1%)",
            )
        if dense_model in bundles:
            d_diag = bundles[dense_model].get("full_diag") or bundles[dense_model].get("strict_diag") or {}
            if "B2a_post_anchor" in d_diag:
                ax_r.axhline(
                    y=float(d_diag["B2a_post_anchor"]["overall_recovery_pct"]),
                    color=OKABE_ITO["grey"],
                    linestyle=":",
                    linewidth=1.3,
                    label=f"{dense_model.split('/')[-1]} B2a Text Anchor (62.5%)",
                )
            d_ev = bundles[dense_model].get("ext_deltanet_blend_evict_kv")
            if isinstance(d_ev, dict) and isinstance(d_ev.get("alphas"), dict) and d_ev["alphas"]:
                first_k = next(iter(d_ev["alphas"]))
                ax_r.axhline(
                    y=float(d_ev["alphas"][first_k]["overall_recovery_pct"]),
                    color=OKABE_ITO["orange"],
                    linestyle="-",
                    linewidth=1.6,
                    label=f"{dense_model.split('/')[-1]} <think> KV Evict (86.2%)",
                )
        ax_r.set_xlabel(r"State Blend Weight $\alpha$ (Prompt State Retention)")
        ax_r.set_ylabel(r"Strict $E_{\mathrm{hurt}}$ Recovery (%)")
        ax_r.set_ylim(54, 102)
        ax_r.set_title(r"(b) Zero-Token State Restoration on $E_{\mathrm{hurt}}$")
        ax_r.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.3)
    else:
        m_list = list(bundles.keys())
        x_m = np.arange(len(m_list))
        w = 0.25
        ax_r.bar(
            x_m - w,
            [100.0 * float(bundles[m]["report"]["systems"]["B1b"]["loop_rate"]) for m in m_list],
            width=w,
            color=OKABE_ITO["blue"],
            label="B1b Loop%",
        )
        ax_r.bar(
            x_m,
            [100.0 * float(bundles[m]["report"]["systems"]["B1b"]["exhaust_rate"]) for m in m_list],
            width=w,
            color=OKABE_ITO["orange"],
            label="B1b Exhaust%",
        )
        ax_r.bar(
            x_m + w,
            [100.0 * float(bundles[m]["report"]["systems"]["M2"]["loop_rate"]) for m in m_list],
            width=w,
            color=OKABE_ITO["green"],
            label="M2 Loop%",
        )
        ax_r.set_xticks(x_m)
        ax_r.set_xticklabels([m.split("/")[-1] for m in m_list], rotation=18, ha="right")
        ax_r.set_ylabel("Pathology Rate (%)")
        ax_r.set_ylim(0, 105)
        ax_r.set_title("(b) Repetition Loop & Budget Exhaustion")
        ax_r.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def discover_bundles(
    out_dir: pathlib.Path, include_sampled: bool = False
) -> dict[str, dict[str, Any]]:
    bundles: dict[str, dict[str, Any]] = {}
    if not out_dir.exists():
        return bundles
    for sub in sorted(out_dir.iterdir()):
        if not sub.is_dir():
            continue
        bundle = load_model_bundle(sub)
        if bundle is None:
            continue
        tp = bundle["report"].get("training_params", {})
        if not include_sampled and float(tp.get("temperature", 0.0) or 0.0) > 0.0:
            # Keep greedy (do_sample=False) bundles canonical in primary Phase 7 tables/figures.
            continue
        model_name = bundle["report"]["model"]
        bundles[model_name] = bundle
    return bundles


def _build_confirmatory_pooled_diag(
    stats_models: dict[str, Any],
) -> dict[str, Any]:
    """Pools E_hurt k/n recovery counts and mid-p McNemar tests across backbones (strict341 & diag491)."""
    out: dict[str, Any] = {}
    modes = ("A1_strip", "B2a_post_anchor", "B1_pre_anchor", "B2b_post_raw", "A2_math_swap", "B3_early_cut_no_anchor")
    groups = {
        "deltanet_models": lambda m: "qwen" in m.lower(),
        "dense_models": lambda m: "smol" in m.lower(),
        "all_models": lambda _m: True,
    }
    for pool_key, out_key, label in (
        ("strict341", "strict341_pooled", "diag_341_strict"),
        ("diag491", "diag_491_pooled", "diag_491"),
    ):
        pool_block: dict[str, Any] = {}
        for grp_name, pred in groups.items():
            matched = [m for m in stats_models if pred(m)]
            mode_totals: dict[str, list[int]] = {m_k: [0, 0] for m_k in modes}
            pooled_a1_z: list[int] = []
            pooled_b2a_z: list[int] = []
            all_have_paired = bool(matched)
            models_included: list[str] = []

            for m_id in matched:
                kn = (stats_models[m_id].get("section6_k_over_n") or {}).get(pool_key, {})
                md = kn.get("modes", {})
                if not md:
                    all_have_paired = False
                    continue
                models_included.append(m_id)
                for mk in modes:
                    if mk in md:
                        mode_totals[mk][0] += int(md[mk]["k"])
                        mode_totals[mk][1] += int(md[mk]["n"])
                pz = kn.get("paired_z")
                if isinstance(pz, dict) and "A1_strip" in pz and "B2a_post_anchor" in pz:
                    pooled_a1_z.extend(int(v) for v in pz["A1_strip"])
                    pooled_b2a_z.extend(int(v) for v in pz["B2a_post_anchor"])
                else:
                    all_have_paired = False

            if not models_included:
                continue
            grp_entry: dict[str, Any] = {
                "pool_label": label,
                "models": models_included,
            }
            for mk in modes:
                k_sum, n_sum = mode_totals[mk]
                if n_sum > 0:
                    rate, lo, hi = wilson_ci(k_sum, n_sum)
                    grp_entry[mk] = {
                        "k": k_sum,
                        "n": n_sum,
                        "pct": 100.0 * rate,
                        "wilson_ci95_pct": [100.0 * lo, 100.0 * hi],
                    }
            if "B2a_post_anchor" in grp_entry:
                kb = grp_entry["B2a_post_anchor"]["k"]
                nb = grp_entry["B2a_post_anchor"]["n"]
                grp_entry["b2a_vs_b1b_all_fail"] = exact_and_midp_mcnemar(
                    [0] * nb, [1] * kb + [0] * (nb - kb)
                )
            if all_have_paired and pooled_a1_z and len(pooled_a1_z) == len(pooled_b2a_z):
                grp_entry["b2a_vs_a1_strip"] = {
                    **exact_and_midp_mcnemar(pooled_a1_z, pooled_b2a_z),
                    "paired_n": len(pooled_a1_z),
                }
            pool_block[grp_name] = grp_entry
        if pool_block:
            out[out_key] = pool_block

    # Strip large per-UID vectors from per-model k/n dicts before writing stats_summary.json
    for m_stats in stats_models.values():
        k_n = m_stats.get("section6_k_over_n") or {}
        for pk in ("strict341", "diag491"):
            if isinstance(k_n.get(pk), dict):
                k_n[pk].pop("paired_z", None)
                k_n[pk].pop("paired_uids", None)
    return out


def _aggregate_r6_sampled_summary(r6_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregates multi-seed Contingency R6 sampled validation runs into mean +/- std per model/regime."""
    if not r6_rows:
        return []
    grouped: dict[tuple[str, float, float, float], list[dict[str, Any]]] = {}
    for row in r6_rows:
        key = (
            str(row["model"]),
            round(float(row["temperature"]), 4),
            round(float(row["top_p"]), 4),
            round(float(row["presence_penalty"]), 4),
        )
        grouped.setdefault(key, []).append(row)

    def _mean_std(vals: list[float]) -> tuple[float, float]:
        arr = np.asarray(vals, dtype=np.float64)
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0
        return mean_v, std_v

    summary: list[dict[str, Any]] = []
    for (model_id, temp, top_p, pp), grp in sorted(grouped.items()):
        grp_sorted = sorted(grp, key=lambda x: int(x.get("seed", 42)))
        m2_m_mu, m2_m_sd = _mean_std([float(x["sampled_m2_macro"]) for x in grp_sorted])
        m2_o_mu, m2_o_sd = _mean_std([float(x["sampled_m2_oaa250"]) for x in grp_sorted])
        m2_u_mu, m2_u_sd = _mean_std([float(x.get("sampled_m2_u005", 0.0)) for x in grp_sorted])
        m2_t_mu, m2_t_sd = _mean_std([float(x["sampled_m2_tok"]) for x in grp_sorted])
        b1b_m_mu, b1b_m_sd = _mean_std([float(x["sampled_b1b_macro"]) for x in grp_sorted])
        b1b_t_mu, _ = _mean_std([float(x.get("sampled_b1b_tok", 0.0)) for x in grp_sorted])
        b0_m_mu, b0_m_sd = _mean_std([float(x["sampled_b0_macro"]) for x in grp_sorted])
        tok_sav = (100.0 * (b1b_t_mu - m2_t_mu) / b1b_t_mu) if b1b_t_mu > 1e-6 else 0.0
        summary.append(
            {
                "model": model_id,
                "temperature": temp,
                "top_p": top_p,
                "presence_penalty": pp,
                "n_seeds": len(grp_sorted),
                "seeds": [int(x.get("seed", 42)) for x in grp_sorted],
                "b0_macro_mean": b0_m_mu,
                "b0_macro_std": b0_m_sd,
                "b1b_macro_mean": b1b_m_mu,
                "b1b_macro_std": b1b_m_sd,
                "m2_macro_mean": m2_m_mu,
                "m2_macro_std": m2_m_sd,
                "m2_oaa250_mean": m2_o_mu,
                "m2_oaa250_std": m2_o_sd,
                "m2_u005_mean": m2_u_mu,
                "m2_u005_std": m2_u_sd,
                "m2_tok_mean": m2_t_mu,
                "m2_tok_std": m2_t_sd,
                "delta_m2_minus_b1b_pp": 100.0 * (m2_m_mu - b1b_m_mu),
                "delta_tok_savings_pct": tok_sav,
            }
        )
    return summary


def discover_sampled_bundles(out_dir: pathlib.Path) -> list[dict[str, Any]]:
    """Discovers Contingency R6 sampled bundles (temperature > 0) without clobbering greedy keys."""
    out: list[dict[str, Any]] = []
    if not out_dir.exists():
        return out
    for sub in sorted(out_dir.iterdir()):
        if not sub.is_dir():
            continue
        bundle = load_model_bundle(sub)
        if bundle is None:
            continue
        tp = bundle["report"].get("training_params", {})
        if float(tp.get("temperature", 0.0) or 0.0) > 0.0:
            out.append(bundle)
    return out


def build_version_audit(bundles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Audits code_version / result_version consistency across separately trained model outputs."""
    prov = get_version_provenance(ROOT)
    analyzer_ver = prov["code_version"]
    model_entries: dict[str, dict[str, Any]] = {}
    unique_known_vers: set[str] = set()
    mismatched_models: list[str] = []

    for m_name, bundle in bundles.items():
        mp = dict(bundle.get("provenance") or {})
        cver = str(mp.get("code_version") or "unknown")
        if cver != "unknown":
            unique_known_vers.add(cver)
        matches_analyzer = (
            True if analyzer_ver == "0.0.0" or cver == "unknown" else cver == analyzer_ver
        )
        if not matches_analyzer:
            mismatched_models.append(m_name)
        model_entries[m_name] = {
            **mp,
            "matches_analyzer_version": matches_analyzer,
        }

    consistent_across_models = len(unique_known_vers) <= 1
    matches_analyzer_version = len(mismatched_models) == 0 and (
        consistent_across_models or len(unique_known_vers) == 0
    )
    return {
        "analyzer_code_version": analyzer_ver,
        "analyzer_git_rev": prov["git_rev"],
        "consistent_across_models": consistent_across_models,
        "matches_analyzer_version": matches_analyzer_version,
        "versions_seen": sorted(unique_known_vers),
        "mismatched_models": mismatched_models,
        "models": model_entries,
    }


def generate_paper_writing_digest(
    bundles: dict[str, dict[str, Any]],
    stats_out: dict[str, Any],
) -> str:
    """Synthesizes a complete, self-contained Markdown reference digest of all paper numbers, CIs, and p-values."""
    va = stats_out.get("version_audit", {})
    lines: list[str] = [
        "# C-BARR Final Paper-Writing Statistical & Mechanistic Digest",
        "",
        f"- **Analyzer Code Version**: `v{va.get('analyzer_code_version', 'unknown')}` (`{va.get('analyzer_git_rev', 'unknown')}`)",
        f"- **Evaluated Backbones ({len(bundles)})**: " + ", ".join(f"`{m}`" for m in bundles),
        f"- **Bootstrap Resamples**: `n_boot={stats_out.get('n_boot', 10000)}` (`seed={stats_out.get('seed', 42)}`)",
        "",
        "## 1. Cross-Backbone Headline Accuracy, Efficiency & Pathology (`N=450`)",
        "",
        "| Model | B0 Macro (Tok) | B1b Macro (Tok, Loop%, Exh%) | M2 Macro (Tok, Loop%, Spill%) | M2_Prec Macro (IF-Prec) | M2_B512 Macro | Oracle_UB Macro | Tok Sav.% | Delta Macro vs B1b [95% CI] | Clean-446 Max Delta | Gate 4 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    ordered_m = [m for m, _, _ in DEFAULT_SCALE_ORDER if m in bundles] + [
        m for m in bundles if m not in {x for x, _, _ in DEFAULT_SCALE_ORDER}
    ]
    for m in ordered_m:
        rep = bundles[m]["report"]
        ms = stats_out["models"][m]
        sys_d = rep["systems"]
        aux_d = rep.get("auxiliary_systems", {})
        orc_d = rep["oracle_and_4regime"]["oracle_summary"]
        b0, b1b, m2 = sys_d["B0"], sys_d["B1b"], sys_d["M2"]
        g4 = ms["gate_audit"]["gate4_primary"]
        b_ci = (ms.get("paired_bootstrap_95ci", {}).get("M2_minus_B1b") or {}).get("macro_acc") or {}
        cl_d = ms.get("clean_446_sensitivity", {}).get("max_abs_macro_delta_pp", 0.0)
        m2p = aux_d.get("M2_Prec")
        m2p_s = f"{100*m2p['macro_acc']:.1f}% ({100*m2p['slice_acc']['ifeval_precision']:.1f}%)" if m2p else "--"
        m512 = aux_d.get("M2_B512")
        m512_s = f"{100*m512['macro_acc']:.1f}%" if m512 else "--"
        ci_s = (
            f"{100*b_ci.get('diff_b_minus_a', 0):+.1f} [{100*b_ci.get('ci95_lo', 0):+.1f}, {100*b_ci.get('ci95_hi', 0):+.1f}] pp"
            if b_ci
            else "--"
        )
        g4_ok = "PASS" if (g4["m2_macro_ge_b1b"] and g4["token_reduction_pct"] > 0) else "PARITY/CHECK"
        lines.append(
            f"| `{m}` | {100*b0['macro_acc']:.1f}% ({b0['mean_t_tot']:.0f}t) | "
            f"{100*b1b['macro_acc']:.1f}% ({b1b['mean_t_tot']:.0f}t, {100*b1b.get('loop_rate',0):.1f}%, {100*b1b.get('exhaust_rate',0):.1f}%) | "
            f"**{100*m2['macro_acc']:.1f}%** ({m2['mean_t_tot']:.0f}t, {100*m2.get('loop_rate',0):.1f}%, {100*m2.get('spillover_rate',0):.2f}%) | "
            f"{m2p_s} | {m512_s} | {100*orc_d['macro_acc']:.1f}% | {g4['token_reduction_pct']:.1f}% | "
            f"{ci_s} | {cl_d:.2f} pp | {g4_ok} |"
        )

    lines.extend(
        [
            "",
            "## 2. Pre-Registered 4-Pair Confirmatory Holm-Bonferroni (`M2` vs. `B1b, B0, B2a, M1c`)",
            "",
            "| Model | Comparison | Wins (`M2` / Base) | Delta Macro [95% CI] | Delta U_0.05 [95% CI] | Tok Sav.% | `p_exact` | `p_holm_4` (exact / mid-p) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for m in ordered_m:
        pairs = (stats_out["models"][m].get("confirmatory_holm_family") or {}).get("pairs") or {}
        for pk, pv in pairs.items():
            pb_m = (pv.get("paired_bootstrap") or {}).get("macro_acc") or {}
            pb_u = (pv.get("paired_bootstrap") or {}).get("u_005") or {}
            lines.append(
                f"| `{m}` | `{pk}` | {pv.get('method_wins',0)} / {pv.get('base_wins',0)} | "
                f"{100*pb_m.get('diff_b_minus_a',0):+.1f} [{100*pb_m.get('ci95_lo',0):+.1f}, {100*pb_m.get('ci95_hi',0):+.1f}] pp | "
                f"{100*pb_u.get('diff_b_minus_a',0):+.1f} [{100*pb_u.get('ci95_lo',0):+.1f}, {100*pb_u.get('ci95_hi',0):+.1f}] pp | "
                f"{pv.get('tok_savings_vs_base_pct',0):+.1f}% | {pv.get('p_exact',1.0):.3g} | "
                f"{pv.get('p_holm_exact',1.0):.3g} / {pv.get('p_holm_midp',1.0):.3g} |"
            )

    lines.extend(
        [
            "",
            "## 3. Proposition 1 Conformal Degradation Risk & Hindsight `Oracle_UB` Retention",
            "",
            "| Model | Dev Deg. vs B1b | CRC Bound | Hoeffding UCB (`M=29`) | Eval Deg. vs B1b (`k/450`) | Eval Deg. vs B0 (`k/450`) | Within Bounds | Oracle Macro Ret.% | Tok Ratio vs Oracle |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for m in ordered_m:
        ms = stats_out["models"][m]
        ca = ms.get("conformal_risk_audit") or {}
        dc = ca.get("dev_calibration") or {}
        oe = ms.get("oracle_efficiency_audit") or {}
        dev_r = f"{100*dc['empirical_degradation_risk']:.1f}%" if "empirical_degradation_risk" in dc else "--"
        crc_b = f"{100*dc['crc_expectation_bound']:.1f}%" if "crc_expectation_bound" in dc else "--"
        hoef_b = f"{100*dc['hoeffding_ucb']:.1f}%" if "hoeffding_ucb" in dc else "--"
        ev_b1 = f"{100*ca.get('eval_observed_degradation_risk',0):.2f}% ({ca.get('eval_degradation_events',0)}/450)"
        ev_b0 = f"{100*ca.get('eval_vs_b0_degradation_risk',0):.2f}% ({ca.get('eval_vs_b0_degradation_events',0)}/450)"
        wb = f"CRC={ca.get('within_crc_expectation_bound', 'n/a')}, Hoef={ca.get('within_hoeffding_ucb', 'n/a')}"
        m2_oe = (oe.get("variants") or {}).get("M2") or {}
        tok_rat = float(m2_oe.get("oracle_token_overhead_ratio", 0.0))
        lines.append(
            f"| `{m}` | {dev_r} | {crc_b} | {hoef_b} | {ev_b1} | {ev_b0} | {wb} | "
            f"{100*float(m2_oe.get('macro_acc_retention',0)):.1f}% | {tok_rat:.2f}x |"
        )

    lines.extend(["", "## 4. Section 6 Counterfactual Controls & Pooled Exact `k/n` Audit", ""])
    pooled = stats_out.get("confirmatory_pooled_diag") or {}
    for pk_key, pk_title in (("strict341_pooled", "Strict Disjoint Pool (`diag_341_strict`)"), ("diag_491_pooled", "Full Calibration-Disjoint Pool (`diag_491`)")):
        pb = pooled.get(pk_key) or {}
        if not pb:
            continue
        lines.append(f"### {pk_title}")
        lines.append("| Stratum | Models | `A1_strip` (`k/n`) | `B2a_post` (`k/n`, 95% CI) | `B1_pre` (`k/n`) | `B2b_raw` (`k/n`) | `A2_swap` (`k/n`) | `B3_cut` (`k/n`) | `p_midp(B2a vs B1b)` | `p_midp(B2a vs A1)` |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for g_key in ("deltanet_models", "dense_models", "all_models"):
            gd = pb.get(g_key)
            if not isinstance(gd, dict) or "A1_strip" not in gd:
                continue
            a1, b2a, b1, b2b, a2, b3 = (
                gd["A1_strip"],
                gd["B2a_post_anchor"],
                gd["B1_pre_anchor"],
                gd["B2b_post_raw"],
                gd["A2_math_swap"],
                gd["B3_early_cut_no_anchor"],
            )
            ci_b2a = b2a.get("wilson_ci95_pct", [0.0, 0.0])
            p_b1b = (gd.get("b2a_vs_b1b_all_fail") or {}).get("p_midp")
            p_a1 = (gd.get("b2a_vs_a1_strip") or {}).get("p_midp")
            lines.append(
                f"| `{g_key}` | {len(gd.get('models', []))} | {a1['pct']:.1f}% ({a1['k']}/{a1['n']}) | "
                f"**{b2a['pct']:.1f}%** ({b2a['k']}/{b2a['n']}, [{ci_b2a[0]:.1f}, {ci_b2a[1]:.1f}]) | "
                f"{b1['pct']:.1f}% ({b1['k']}/{b1['n']}) | {b2b['pct']:.1f}% ({b2b['k']}/{b2b['n']}) | "
                f"{a2['pct']:.1f}% ({a2['k']}/{a2['n']}) | {b3['pct']:.1f}% ({b3['k']}/{b3['n']}) | "
                f"{f'{p_b1b:.3g}' if p_b1b is not None else '--'} | {f'{p_a1:.3g}' if p_a1 is not None else '--'} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 5. Gate-2 Probe Horizon Ablation (`K in {1, 4, 8, 16, 32}`)",
            "",
            "| Model | `K=1` Macro / U_0.05 (Tok) | `K=4` Macro / U_0.05 (Tok) | `K=8` Macro / U_0.05 (Tok) | `K=16` Macro / U_0.05 (Tok) | `K=32` Macro / U_0.05 (Tok) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for m in ordered_m:
        ph = stats_out["models"][m].get("probe_horizon_ablation") or {}
        if not ph:
            continue
        cells = []
        for k_v in (1, 4, 8, 16, 32):
            kd = ph.get(f"K_{k_v}")
            cells.append(
                f"{100*kd['macro_acc']:.1f}% / {100*kd['u_005']:.1f}% ({kd['mean_t_tot']:.0f}t)" if kd else "--"
            )
        lines.append(f"| `{m}` | " + " | ".join(cells) + " |")

    w7_ext = stats_out.get("w7_extensions") or {}
    if w7_ext:
        lines.extend(["", "## 6. Workstream W7 Mechanistic Extensions & Paraphrase Probe", ""])
        for m, wd in w7_ext.items():
            lines.append(f"### `{m}`")
            for b_key, b_lbl in (
                ("ext_deltanet_blend", "DeltaNet Recurrent State Blend (`conv=False, evict_kv=False`)"),
                ("ext_deltanet_blend_conv_kv", "DeltaNet Joint Recurrent + 1D Conv + `<think>` KV Eviction"),
                ("ext_deltanet_blend_evict_kv", "Dense Softmax `<think>` KV Cache Eviction (`alpha=0`)"),
            ):
                bd = wd.get(b_key)
                if isinstance(bd, dict) and "alphas" in bd:
                    a_parts = [
                        f"`alpha={ak}`: **{av['overall_recovery_pct']:.1f}%** (Inhib={av.get('inhibition_recovery_pct',0):.1f}%, Struct={av.get('structural_recovery_pct',0):.1f}%)"
                        for ak, av in bd["alphas"].items()
                    ]
                    lines.append(f"- **{b_lbl}** (`n={bd.get('n_ehurt', 0)}`): " + "; ".join(a_parts))
            mlp = wd.get("prefill_mlp")
            if isinstance(mlp, dict) and "eval_table" in mlp:
                et = mlp["eval_table"]
                g1p = mlp.get("gate1_semantic_probe") or {}
                para = mlp.get("ifevalpp_paraphrase_stress") or {}
                ro = (g1p.get("regex_orig") or {}).get("prec") or {}
                mo = (g1p.get("mlp_orig") or {}).get("prec") or {}
                rp = (para.get("regex_para") or {}).get("prec") or {}
                mp = (para.get("mlp_para") or {}).get("prec") or {}
                lines.append(
                    f"- **Prefill MLP Probe (`{mlp.get('mode')}`, `tau={mlp.get('best_tau',0):.3f}`)**: "
                    f"Eval Macro={100*et['macro_acc']:.1f}%, OAA_250={100*et['oaa_250']:.1f}%, U_0.05={100*et['u_005']:.1f}%, Tok={et['mean_t_tot']:.0f} | "
                    f"Canonical Prec F1: Regex={ro.get('f1',0):.1f}% vs MLP={mo.get('f1',0):.1f}% | "
                    f"`IFEval++` Paraphrase Prec F1 (Rec): Regex={rp.get('f1',0):.1f}% ({rp.get('recall',0):.1f}%) vs **MLP={mp.get('f1',0):.1f}% ({mp.get('recall',0):.1f}%)**"
                )
            m2v2 = wd.get("m2v2_comparison")
            if isinstance(m2v2, dict) and isinstance(m2v2.get("systems"), dict):
                cs = m2v2["systems"].get("M2_classic", {})
                ss = m2v2["systems"].get("M2_solve_first", {})
                lines.append(
                    f"- **Stage-2 Completion Bridge Replay (`replayed={m2v2.get('stage2_items_replayed',0)}`)**: "
                    f"Classic Macro={100*cs.get('macro_acc',0):.1f}% (OAA_250={100*cs.get('oaa_250',0):.1f}%) vs "
                    f"SolveFirst Macro={100*ss.get('macro_acc',0):.1f}% (OAA_250={100*ss.get('oaa_250',0):.1f}%)"
                )
            lines.append("")

    lines.extend(["", "## 7. Appendix B: OOD `IFBench_test`, 3-Seed `R6`, and Cross-Model L2 Sensitivity", ""])
    ifb_map = stats_out.get("ifbench_ood") or {}
    if ifb_map:
        for m, ib in ifb_map.items():
            sy = ib.get("systems", {})
            b0i, b1i, m2i = sy.get("B0", {}), sy.get("B1b", {}), sy.get("M2", {})
            mc_b1 = (ib.get("mcnemar") or {}).get("M2_vs_B1b") or {}
            lines.append(
                f"- **IFBench_test OOD (`{m}`, `N={ib.get('n_prompts',0)}`)**: "
                f"B0={b0i.get('prompt_strict_pct',0):.1f}% ({b0i.get('mean_t_tot',0):.0f}t) | "
                f"B1b={b1i.get('prompt_strict_pct',0):.1f}% ({b1i.get('mean_t_tot',0):.0f}t) | "
                f"**M2={m2i.get('prompt_strict_pct',0):.1f}%** (OAA_250={m2i.get('oaa_250_pct',0):.1f}%, {m2i.get('mean_t_tot',0):.0f}t, `p_exact={mc_b1.get('p_exact',1.0):.3g}`)"
            )
    else:
        lines.append("- **IFBench_test OOD**: *Pending execution (`ext_ifbench_ood.py`).*")

    r6_sum = stats_out.get("r6_sampled_summary") or []
    if r6_sum:
        for rs in r6_sum:
            lines.append(
                f"- **3-Seed Sampled R6 (`{rs['model']}`, seeds `{rs['seeds']}`, `T={rs['temperature']:g}`)**: "
                f"B0={100*rs['b0_macro_mean']:.1f}+/-{100*rs['b0_macro_std']:.1f}% | "
                f"B1b={100*rs['b1b_macro_mean']:.1f}+/-{100*rs['b1b_macro_std']:.1f}% | "
                f"**M2={100*rs['m2_macro_mean']:.1f}+/-{100*rs['m2_macro_std']:.1f}%** "
                f"(OAA_250={100*rs['m2_oaa250_mean']:.1f}+/-{100*rs['m2_oaa250_std']:.1f}%, Tok={rs['m2_tok_mean']:.0f}+/-{rs['m2_tok_std']:.0f}, TokSav={rs['delta_tok_savings_pct']:.1f}%)"
            )
    else:
        lines.append("- **3-Seed Sampled R6**: *Pending execution (`--part r6`).*")

    lines.extend(
        [
            "",
            "### Gate-2 L2 Regularization Sensitivity (`C in {0.1, 0.25 (prod), 0.5, 1.0}`)",
            "| Model | `C=0.10` Macro (Esc, Tok) | `C=0.25 (prod)` Macro (Esc, Tok) | `C=0.50` Macro (Esc, Tok) | `C=1.00` Macro (Esc, Tok) |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for m in ordered_m:
        l2_rows = stats_out["models"][m].get("l2_regularization_sensitivity") or []
        if not l2_rows:
            continue
        by_c = {round(float(r["reg_c"]), 2): r for r in l2_rows}
        c_cells = []
        for cv in (0.10, 0.25, 0.50, 1.00):
            rc = by_c.get(cv)
            c_cells.append(
                f"{100*rc['eval_m2_macro']:.1f}% ({100*rc['dev_esc_rate']:.1f}%, {rc['eval_m2_tok']:.0f}t)" if rc else "--"
            )
        lines.append(f"| `{m}` | " + " | ".join(c_cells) + " |")

    lines.extend(
        [
            "",
            "## 8. End-to-End Wall-Clock Latency, Compute Footprint & Reproducibility Checklist",
            "",
            "| Model | Dtype | Seed | `B0` Mean / P95 | `B1b` Mean / P95 (`Sum min`) | `M2` Mean / P95 (`Sum min`) | Speedup (`B1b / M2`) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for m in ordered_m:
        rep = bundles[m]["report"]
        runs_m = bundles[m].get("per_system_runs") or {}
        rt_dtype = (rep.get("runtime") or {}).get("dtype", "n/a")
        seed_v = (rep.get("training_params") or {}).get("seed", 42)

        def _lat_cell(k: str, with_sum: bool = False) -> tuple[str, float]:
            arr = [float(r.get("latency_ms", 0.0)) / 1000.0 for r in runs_m.get(k, []) if float(r.get("latency_ms", 0.0)) > 0]
            if not arr:
                return "--", 0.0
            mu = float(np.mean(arr))
            p95 = float(np.percentile(arr, 95))
            tot_min = float(np.sum(arr)) / 60.0
            base = f"{mu:.2f}s / {p95:.2f}s"
            return (f"{base} ({tot_min:.1f}m)" if with_sum else base), mu

        b0_l, _ = _lat_cell("B0", with_sum=False)
        b1_l, b1_mu = _lat_cell("B1b", with_sum=True)
        m2_l, m2_mu = _lat_cell("M2", with_sum=True)
        spd = f"**{b1_mu / m2_mu:.2f}x**" if (b1_mu > 0 and m2_mu > 0) else "--"
        lines.append(f"| `{m}` | `{rt_dtype}` | `{seed_v}` | {b0_l} | {b1_l} | {m2_l} | {spd} |")

    return "\n".join(lines) + "\n"


def generate_paper_qualitative_examples(bundles: dict[str, dict[str, Any]]) -> str:
    """Mines verbatim, publication-ready qualitative case studies directly from cached model runs."""

    def _cspan(txt: str, max_chars: int = 420) -> str:
        s = " ".join(str(txt or "").strip().split()).replace("````", "'''")
        if len(s) > max_chars:
            s = s[: max_chars - 3] + "..."
        return f"```` {s} ````" if s else "`(empty)`"

    lines: list[str] = [
        "# C-BARR Verbatim Qualitative Case Studies (Auto-Mined from Frozen Caches)",
        "",
        "Curated verbatim examples for Section 5/6 figures and Appendix qualitative analysis.",
        "",
    ]

    # Case A & B from primary (or first available) model
    prim_m = "Qwen/Qwen3.5-4B" if "Qwen/Qwen3.5-4B" in bundles else next(iter(bundles))
    p_dir = bundles[prim_m]["dir"]
    p_runs = bundles[prim_m].get("per_system_runs") or {}
    p_items = bundles[prim_m].get("items") or []
    eval_p = p_dir / "eval_450_cache.jsonl"
    raw_by_uid = {str(r["item"]["uid"]): r for r in _load_jsonl(eval_p)} if eval_p.is_file() else {}

    # Case A: Overthinking Hurt (B0=1, B1b=0, M2=1)
    for idx, it in enumerate(p_items):
        if idx >= len(p_runs.get("M2", [])):
            break
        b0r, b1r, m2r = p_runs["B0"][idx], p_runs["B1b"][idx], p_runs["M2"][idx]
        if int(b0r["z"]) == 1 and int(b1r["z"]) == 0 and int(m2r["z"]) == 1 and str(it.get("slice", "")).startswith("ifeval"):
            uid = str(it["uid"])
            g1 = run_gate1_lexical(str(it["prompt"]))
            lines.extend(
                [
                    f"## Case A: Overthinking Constraint Drift & C-BARR Recovery (`{prim_m}`, `{uid}`, slice=`{it.get('slice')}`)",
                    f"- **Prompt**: {_cspan(it['prompt'], 360)}",
                    f"- **Gate-1 Extracted Clause**: `phi_plan={g1.phi_plan}, phi_prec={g1.phi_prec}` | {_cspan(g1.anchor_clause, 260)}",
                    f"- **B0 (`z=1`, `T={b0r.get('T_total',0)}`)**: {_cspan(b0r.get('a_final',''), 320)}",
                    f"- **B1b (`z=0`, `T={b1r.get('T_total',0)}`, `loop={b1r.get('has_4gram_loop')}`, `exhausted={b1r.get('think_exhausted')}`)**:",
                    f"  - `<think>` excerpt: {_cspan(b1r.get('t_think',''), 320)}",
                    f"  - Final answer: {_cspan(b1r.get('a_final',''), 320)}",
                    f"- **C-BARR M2 (`z=1`, `T={m2r.get('T_total',0)}`, `calls={m2r.get('calls',1)}`)**: {_cspan(m2r.get('a_final',''), 320)}",
                    "",
                ]
            )
            break

    # Case B: Deliberation Helped (B0=0, M2=1)
    for idx, it in enumerate(p_items):
        if idx >= len(p_runs.get("M2", [])):
            break
        b0r, b1r, m2r = p_runs["B0"][idx], p_runs["B1b"][idx], p_runs["M2"][idx]
        if int(b0r["z"]) == 0 and int(m2r["z"]) == 1:
            uid = str(it["uid"])
            raw_r = raw_by_uid.get(uid, {})
            pr = raw_r.get("probe", {})
            lines.extend(
                [
                    f"## Case B: Budgeted Deliberation Gain Over Non-Thinking (`{prim_m}`, `{uid}`, slice=`{it.get('slice')}`)",
                    f"- **Prompt**: {_cspan(it['prompt'], 360)}",
                    f"- **Gate-2 Fast Probe**: `H_first={pr.get('H_first',0):.2f}, H_mean={pr.get('H_mean',0):.2f}, H_max={pr.get('H_max',0):.2f}, I_inertia={pr.get('I_inertia',0)}`",
                    f"- **B0 (`z=0`, `T={b0r.get('T_total',0)}`)**: {_cspan(b0r.get('a_final',''), 300)}",
                    f"- **C-BARR M2 (`z=1`, `T={m2r.get('T_total',0)}` vs `B1b T={b1r.get('T_total',0)}`)**:",
                    f"  - Capped `<think>`: {_cspan(m2r.get('t_think',''), 300)}",
                    f"  - Final answer: {_cspan(m2r.get('a_final',''), 300)}",
                    "",
                ]
            )
            break

    # Case C: Section 6 Raw Assistant Wrapper Corruption (B2b=0 vs B2a=1)
    ctrl_p = p_dir / "diag_controls_cache.jsonl"
    ehurt_p = p_dir / "diag_ehurt_cache.jsonl"
    if ctrl_p.is_file() and ehurt_p.is_file():
        eh_map = {str(r["item"]["uid"]): r for r in _load_jsonl(ehurt_p)}
        by_u: dict[str, dict[str, Any]] = {}
        for cr in _load_jsonl(ctrl_p):
            by_u.setdefault(str(cr.get("uid", "")), {})[str(cr.get("mode", ""))] = cr
        for uid, md in sorted(by_u.items()):
            b2a_c = md.get("B2a_post_anchor")
            b2b_c = md.get("B2b_post_raw")
            if b2a_c and b2b_c and int(b2a_c["grade"]["z"]) == 1 and int(b2b_c["grade"]["z"]) == 0 and uid in eh_map:
                prompt_c = eh_map[uid]["item"]["prompt"]
                lines.extend(
                    [
                        f"## Case C: Section 6 Unstripped Echo Corruption (`B2b=0`) vs. Turn-Delimited Bridge (`B2a=1`) (`{prim_m}`, `{uid}`)",
                        f"- **Prompt**: {_cspan(prompt_c, 340)}",
                        f"- **Control B2b (Unstripped Assistant Prefill, `z=0`)**: {_cspan(b2b_c.get('a_final',''), 320)}",
                        f"- **Control B2a (Turn-Delimited Post-Think Anchor, `z=1`)**: {_cspan(b2a_c.get('a_final',''), 320)}",
                        "",
                    ]
                )
                break

    # Case D: W7 Dense KV Eviction Recovery on SmolLM3-3B (B2a=0 -> DenseKVEvict=1)
    dense_m = "HuggingFaceTB/SmolLM3-3B"
    if dense_m in bundles:
        d_dir = bundles[dense_m]["dir"]
        d_ctrl = d_dir / "diag_controls_cache.jsonl"
        d_blend = d_dir / "ext_deltanet_blend_cache.jsonl"
        d_eh = d_dir / "diag_ehurt_cache.jsonl"
        if d_ctrl.is_file() and d_blend.is_file() and d_eh.is_file():
            deh_map = {str(r["item"]["uid"]): r for r in _load_jsonl(d_eh)}
            b2a_fail = {
                str(r["uid"]): r
                for r in _load_jsonl(d_ctrl)
                if r.get("mode") == "B2a_post_anchor" and int(r["grade"]["z"]) == 0
            }
            for br in _load_jsonl(d_blend):
                uid = str(br.get("uid", ""))
                if br.get("evict_think_kv") and int(br.get("grade", {}).get("z", 0)) == 1 and uid in b2a_fail and uid in deh_map:
                    lines.extend(
                        [
                            f"## Case D: Dense Softmax `<think>` KV Interference (`B2a=0`) vs. Zero-Token `<think>` KV Eviction (`z=1`) (`{dense_m}`, `{uid}`)",
                            f"- **Prompt**: {_cspan(deh_map[uid]['item']['prompt'], 340)}",
                            f"- **Text Re-Anchor Only (`B2a`, `z=0` due to `<think>` KV attention)**: {_cspan(b2a_fail[uid].get('a_final',''), 300)}",
                            f"- **Zero-Token `<think>` KV Cache Eviction (`evict_kv=True`, `z=1`)**: {_cspan(br.get('a_final',''), 300)}",
                            "",
                        ]
                    )
                    break

    # Case E: OOD Unseen Rule Recovery on IFBench_test (when present)
    for m_id, bnd in bundles.items():
        ifb_c = bnd["dir"] / "ext_ifbench_ood_cache.jsonl"
        if ifb_c.is_file():
            for ir in _load_jsonl(ifb_c):
                if int((ir.get("b1b") or {}).get("z", 1)) == 0 and int((ir.get("m2") or {}).get("z", 0)) == 1:
                    it_i = ir.get("item", {})
                    lines.extend(
                        [
                            f"## Case E: Out-of-Distribution `IFBench_test` Unseen Constraint Recovery (`{m_id}`, `{it_i.get('uid')}`)",
                            f"- **Instruction IDs**: `{it_i.get('instruction_id_list')}`",
                            f"- **Prompt**: {_cspan(it_i.get('prompt',''), 340)}",
                            f"- **B1b (`z=0`, `T={ir['b1b'].get('T_total',0)}`)**: {_cspan(ir['b1b'].get('a_final',''), 300)}",
                            f"- **Frozen C-BARR M2 (`z=1`, `T={ir['m2'].get('T_total',0)}`)**: {_cspan(ir['m2'].get('a_final',''), 300)}",
                            "",
                        ]
                    )
                    break
            break

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze C-BARR outputs and generate tables/figures.")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--fig-dir", type=str, default="paper/figures")
    parser.add_argument(
        "--table-dir",
        type=str,
        default="",
        help="Output directory for modular LaTeX tables tab1..tab4 (defaults to paper/tables when --out-dir is outputs, else <out-dir>/tables).",
    )
    parser.add_argument("--primary-model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--dense-model", type=str, default="HuggingFaceTB/SmolLM3-3B")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict-version-match",
        action="store_true",
        help="Fail if analyzed model outputs come from mismatched or stale VERSIONs.",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    fig_dir = pathlib.Path(args.fig_dir)
    table_dir = (
        pathlib.Path(args.table_dir)
        if str(args.table_dir).strip()
        else (
            ROOT / "paper" / "tables"
            if out_dir.resolve() == (ROOT / "outputs").resolve()
            else out_dir / "tables"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    bundles = discover_bundles(out_dir)
    if not bundles:
        raise SystemExit(f"No model matrix summaries found under {out_dir}")

    _configure_matplotlib()

    version_audit = build_version_audit(bundles)
    stats_out: dict[str, Any] = {
        "models": {},
        "version_audit": version_audit,
        "n_boot": args.n_boot,
        "seed": args.seed,
    }
    for m_name, bundle in bundles.items():
        stats_out["models"][m_name] = analyze_model_bundle(bundle, n_boot=args.n_boot, seed=args.seed)

    # Gate 5 cross-scale + dense parity summary if >=2 models present
    qwen_scales = [m for m, _, _ in DEFAULT_SCALE_ORDER[:4] if m in bundles]
    if qwen_scales:
        stats_out["gate5_scale_summary"] = {
            "qwen_scales_present": qwen_scales,
            "dense_control_present": args.dense_model in bundles,
            "m2_macro_by_model": {
                m: bundles[m]["report"]["systems"]["M2"]["macro_acc"] for m in bundles
            },
            "b1b_macro_by_model": {
                m: bundles[m]["report"]["systems"]["B1b"]["macro_acc"] for m in bundles
            },
        }

    sampled_bundles = discover_sampled_bundles(out_dir)
    r6_sampled_rows: list[dict[str, Any]] = []
    for sb in sampled_bundles:
        tp = sb["report"].get("training_params") if isinstance(sb["report"].get("training_params"), dict) else {}
        m_key = sb["report"].get("model", sb["dir"].name)
        sys_d = sb["report"].get("systems", {})
        m2_s = sys_d.get("M2", {})
        b1b_s = sys_d.get("B1b", {})
        b0_s = sys_d.get("B0", {})
        r6_sampled_rows.append(
            {
                "model": m_key,
                "dir": sb["dir"].name,
                "temperature": float(tp.get("temperature", 0.0)),
                "top_p": float(tp.get("top_p", 1.0)),
                "presence_penalty": float(tp.get("presence_penalty", 0.0)),
                "seed": int(tp.get("seed", 42)),
                "n_eval": len(sb.get("items") or []) or 450,
                "sampled_b0_macro": float(b0_s.get("macro_acc", 0.0)),
                "sampled_b1b_macro": float(b1b_s.get("macro_acc", 0.0)),
                "sampled_b1b_tok": float(b1b_s.get("mean_t_tot", 0.0)),
                "sampled_m2_macro": float(m2_s.get("macro_acc", 0.0)),
                "sampled_m2_oaa250": float(m2_s.get("oaa_250", 0.0)),
                "sampled_m2_u005": float(m2_s.get("u_005", 0.0)),
                "sampled_m2_tok": float(m2_s.get("mean_t_tot", 0.0)),
                "sampled_m2_loop_rate": float(m2_s.get("loop_rate", 0.0)),
            }
        )
    if r6_sampled_rows:
        stats_out["r6_sampled_validation"] = r6_sampled_rows
    r6_sampled_summary = _aggregate_r6_sampled_summary(r6_sampled_rows)
    if r6_sampled_summary:
        stats_out["r6_sampled_summary"] = r6_sampled_summary

    confirmatory_pooled_diag = _build_confirmatory_pooled_diag(stats_out["models"])
    if confirmatory_pooled_diag:
        stats_out["confirmatory_pooled_diag"] = confirmatory_pooled_diag

    ifbench_ood_out: dict[str, Any] = {
        m_name: bundle["ext_ifbench_ood"]
        for m_name, bundle in bundles.items()
        if isinstance(bundle.get("ext_ifbench_ood"), dict)
    }
    if ifbench_ood_out:
        stats_out["ifbench_ood"] = ifbench_ood_out

    w7_summary_out: dict[str, Any] = {}
    for m_name, bundle in bundles.items():
        w7_entry: dict[str, Any] = {}
        for wk in (
            "ext_deltanet_blend",
            "ext_deltanet_blend_conv_kv",
            "ext_deltanet_blend_evict_kv",
            "prefill_mlp",
            "m2v2_comparison",
        ):
            if bundle.get(wk) is not None:
                w7_entry[wk] = bundle[wk]
        if w7_entry:
            w7_summary_out[m_name] = w7_entry
    if w7_summary_out:
        stats_out["w7_extensions"] = w7_summary_out

    (out_dir / "stats_summary.json").write_text(
        json.dumps(stats_out, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "PAPER_WRITING_DIGEST.md").write_text(
        generate_paper_writing_digest(bundles, stats_out), encoding="utf-8"
    )
    (out_dir / "PAPER_QUALITATIVE_EXAMPLES.md").write_text(
        generate_paper_qualitative_examples(bundles), encoding="utf-8"
    )
    table_parts = generate_latex_table_parts(
        bundles,
        args.primary_model,
        args.dense_model,
        stats_models=stats_out["models"],
        r6_sampled=r6_sampled_rows,
        pooled_diag=confirmatory_pooled_diag,
        r6_summary=r6_sampled_summary,
        ifbench_ood=ifbench_ood_out,
    )
    for t_name, t_tex in table_parts.items():
        (table_dir / t_name).write_text(t_tex, encoding="utf-8")
    (out_dir / "paper_tables.tex").write_text(
        generate_latex_tables(
            bundles,
            args.primary_model,
            args.dense_model,
            stats_models=stats_out["models"],
            r6_sampled=r6_sampled_rows,
            pooled_diag=confirmatory_pooled_diag,
            r6_summary=r6_sampled_summary,
            ifbench_ood=ifbench_ood_out,
        ),
        encoding="utf-8",
    )

    plot_fig0_architecture_overview(fig_dir / "fig0_cbarr_overview.pdf")
    plot_fig1_pareto_frontier(bundles, args.primary_model, fig_dir / "fig1_pareto_frontier.pdf")
    plot_fig2_bsr_sla(bundles, args.primary_model, args.dense_model, fig_dir / "fig2_bsr_sla_curve.pdf")
    plot_fig3_ehurt_recovery(
        bundles, args.primary_model, args.dense_model, fig_dir / "fig3_ehurt_causal_recovery.pdf"
    )
    plot_fig4_horizons_and_w7(
        bundles,
        args.primary_model,
        args.dense_model,
        stats_out["models"],
        fig_dir / "fig4_horizons_and_w7.pdf",
    )

    print(banner(f"PHASE 7 ANALYSIS COMPLETE ({len(bundles)} models)", "magenta"))
    print(f"Analyzed {len(bundles)} model(s): {', '.join(bundles.keys())}")
    ver_tag_str = ", ".join(
        f"{m}={d.get('result_version', 'unknown')}" for m, d in version_audit["models"].items()
    )
    print(
        f"[VERSION AUDIT] analyzer=v{version_audit['analyzer_code_version']} | "
        f"consistent={version_audit['consistent_across_models']} | "
        f"matches_repo={version_audit['matches_analyzer_version']} | {ver_tag_str}"
    )
    if not version_audit["matches_analyzer_version"]:
        print(
            hl(
                f"[WARN] Version mismatch across model results: versions_seen={version_audit['versions_seen']}, "
                f"mismatched={version_audit['mismatched_models']} (repo=v{version_audit['analyzer_code_version']})",
                "yellow",
            )
        )
        if args.strict_version_match:
            raise SystemExit(1)
    print(f"Saved stats to {out_dir / 'stats_summary.json'}")
    print(f"Saved LaTeX tables to {out_dir / 'paper_tables.tex'} & {table_dir}/")
    print(f"Saved 4 vector PDF figures to {fig_dir}/")
    for m_name, m_stats in stats_out["models"].items():
        g4 = m_stats["gate_audit"]["gate4_primary"]
        cont = m_stats["gate_audit"]["contingencies"]
        rep_sys = bundles[m_name]["report"]["systems"]
        m2_s = rep_sys["M2"]
        b1b_s = rep_sys["B1b"]
        boot = m_stats.get("paired_bootstrap_95ci", {}).get("M2_minus_B1b", {}).get("oaa_250", {})
        gate4_ok = g4["m2_macro_ge_b1b"] and g4["token_reduction_pct"] > 0
        g4_tag = hl("PASS", "green") if gate4_ok else hl("CHECK", "yellow")
        m2_macro_str = hl(f"{100 * m2_s['macro_acc']:.1f}%", "green")
        print(
            hl(f"[SUMMARY {m_name}]", "cyan")
            + f" M2 Macro={m2_macro_str} vs B1b={100*b1b_s['macro_acc']:.1f}% | "
            + f"OAA_250={100*m2_s['oaa_250']:.1f}% (Delta={100*boot.get('diff_b_minus_a', 0):.1f} pp) | "
            + f"TokRed={g4['token_reduction_pct']:.1f}% | p_holm={g4['m2_p_holm_vs_b1b']:.4g} | "
            + f"Gate4={g4_tag} | "
            + f"R2(rec/int)={100*cont['R2_rho_recency']:.1f}%/{100*cont['R2_rho_interference']:.1f}%"
        )


if __name__ == "__main__":
    main()
