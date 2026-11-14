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

from sklearn.linear_model import LogisticRegression

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
from src.routers import CalibratedRouterParams, _gate2_target_label, resolve_all_variants_for_item

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
    for cand in ("matrix_summary_diag491.json", "matrix_summary.json", "matrix_summary_strict341.json"):
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

    eval_cache_path = model_dir / "eval_450_cache.jsonl"
    items: list[dict[str, Any]] = []
    per_system_runs: dict[str, list[dict[str, Any]]] = {k: [] for k in VARIANT_ORDER}
    if eval_cache_path.exists():
        rows = _load_jsonl(eval_cache_path)
        params = CalibratedRouterParams(**report["calibrated_params"])
        rng_b2b = np.random.default_rng(42)
        for row in rows:
            items.append(row["item"])
            resolved = resolve_all_variants_for_item(
                row, params, rng_b2b, target_esc_prob=params.m2_escalation_rate_dev
            )
            for k in VARIANT_ORDER:
                per_system_runs[k].append(resolved[k])

    return {
        "dir": model_dir,
        "report": report,
        "calib_summary": calib_summary,
        "strict_diag": strict_diag,
        "full_diag": full_diag,
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
                float(pr.get("H_first", 0.0)),
                float(pr.get(h_mean_k, pr.get("H_mean", 0.0))),
                float(pr.get(h_max_k, pr.get("H_max", 0.0))),
                float(pr.get(i_k, pr.get("I_inertia", 0))),
                float(g1.phi_sym),
            ],
            dtype=np.float64,
        )

    out: dict[str, Any] = {}
    eval_items = [r["item"] for r in eval_rows]
    for k_val, hm_k, hx_k, in_k in horizons:
        x_s = np.vstack([_feat(r, run_gate1_lexical(r["item"]["prompt"]), hm_k, hx_k, in_k) for r in seed_rows])
        y_s = np.asarray([_gate2_target_label(r) for r in seed_rows], dtype=int)
        if len(np.unique(y_s)) < 2:
            coef = np.array([0.6, 0.8, 0.4, 1.0, 0.5], dtype=np.float64)
            intercept = -1.0
        else:
            clf = LogisticRegression(C=1.0, class_weight="balanced", random_state=42)
            clf.fit(x_s, y_s)
            coef = clf.coef_[0].astype(np.float64)
            intercept = float(clf.intercept_[0])

        # Calibrate tau on dev_60 matching CBARRCalibrator (adjusting effective Gate-2 probe tax to <= k_val)
        best_tau, best_u = 0.50, -1e9
        for tau in np.linspace(0.15, 0.85, 29):
            u_list = []
            for r in dev_rows:
                g1 = run_gate1_lexical(r["item"]["prompt"])
                if g1.phi_prec == 1 and g1.phi_plan == 0 and g1.phi_sym == 0:
                    chosen = r["b0"]
                elif g1.phi_plan == 1:
                    chosen = r.get("m2_b256_direct", r["m2_b256"])
                else:
                    fv = _feat(r, g1, hm_k, hx_k, in_k)
                    prob = 1.0 / (1.0 + np.exp(-float(np.dot(coef, fv) + intercept)))
                    if prob >= tau:
                        slow = r.get("m2_b256", r["m2_b128"])
                        chosen = dict(slow)
                        eff_probe = min(int(r["probe"].get("T_probe", 32)), k_val)
                        chosen["T_total"] = eff_probe + int(slow["T_total"])
                    else:
                        chosen = r["fast_done"]
                u_list.append(float(chosen["z"]) - lam_utility * (float(chosen["T_total"]) / norm_tokens))
            mu = float(np.mean(u_list))
            if mu > best_u:
                best_u = mu
                best_tau = float(tau)

        eval_runs = []
        for r in eval_rows:
            g1 = run_gate1_lexical(r["item"]["prompt"])
            if g1.phi_prec == 1 and g1.phi_plan == 0 and g1.phi_sym == 0:
                eval_runs.append(r["b0"])
            elif g1.phi_plan == 1:
                eval_runs.append(r.get("m2_b256_direct", r["m2_b256"]))
            else:
                fv = _feat(r, g1, hm_k, hx_k, in_k)
                prob = 1.0 / (1.0 + np.exp(-float(np.dot(coef, fv) + intercept)))
                if prob >= best_tau:
                    slow = r.get("m2_b256", r["m2_b128"])
                    eff_probe = min(int(r["probe"].get("T_probe", 32)), k_val)
                    merged = dict(slow)
                    merged["T_probe"] = eff_probe
                    merged["T_total"] = eff_probe + int(slow["T_total"])
                    merged["calls"] = 2
                    eval_runs.append(merged)
                else:
                    eval_runs.append(r["fast_done"])

        tab = evaluate_system_table(eval_items, eval_runs, lam_utility=lam_utility, norm_tokens=norm_tokens)
        out[f"K_{k_val}"] = {
            "horizon_tokens": k_val,
            "tau_prob": best_tau,
            "macro_acc": float(tab["macro_acc"]),
            "oaa_250": float(tab["oaa_250"]),
            "u_005": float(tab["u_005"]),
            "mean_t_tot": float(tab["mean_t_tot"]),
        }
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
        for base_k in ("B1b", "B0", "M1c", "M2_Pre"):
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
        for base_k in ("B0", "M1c", "M2_Pre"):
            targeted_mcnemar[f"M2_vs_{base_k}"] = exact_and_midp_mcnemar(
                [int(r["z"]) for r in runs[base_k]], z_m2
            )

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
        z_eval_m2 = [int(r["z"]) for r in runs["M2"]]
        eval_deg = compute_conformal_risk_bounds(z_eval_b1b, z_eval_m2, alpha_delta=0.05, n_grid=1)
        conformal_audit["eval_observed_degradation_risk"] = eval_deg["empirical_degradation_risk"]
        conformal_audit["eval_degradation_events"] = eval_deg["degradation_events"]
        if "dev_calibration" in conformal_audit:
            dev_crc = conformal_audit["dev_calibration"]
            conformal_audit["within_hoeffding_ucb"] = bool(
                eval_deg["empirical_degradation_risk"] <= dev_crc["hoeffding_ucb"]
            )
            conformal_audit["within_crc_expectation_bound"] = bool(
                eval_deg["empirical_degradation_risk"] <= dev_crc["crc_expectation_bound"]
            )

    probe_horizons = _probe_horizon_ablation(bundle["dir"])

    return {
        "model": report["model"],
        "diag_pool_loaded": report.get("diag_pool", "unknown"),
        "calibrated_params": report["calibrated_params"],
        "clean_446_sensitivity": clean_sensitivity,
        "paired_bootstrap_95ci": paired_bootstrap,
        "targeted_mcnemar": targeted_mcnemar,
        "gate_audit": gate_audit,
        "conformal_risk_audit": conformal_audit,
        "oracle_efficiency_audit": oracle_eff,
        "probe_horizon_ablation": probe_horizons,
        "oracle_and_4regime": report["oracle_and_4regime"],
        "auxiliary_systems": report.get("auxiliary_systems", {}),
        "section6_strict341": bundle["strict_diag"],
        "section6_diag491": bundle["full_diag"],
    }


def _fmt_pct(val: float) -> str:
    return f"{100.0 * float(val):.1f}"


def generate_latex_tables(
    bundles: dict[str, dict[str, Any]],
    primary_model: str,
    dense_model: str,
) -> str:
    """Emits complete LaTeX Tables 1, 2, and 3 populated from observed bundle metrics."""
    lines: list[str] = ["% Auto-generated by scripts/analyze_and_plot.py from frozen matrix outputs"]

    # Table 1: Primary (or first available) model
    t1_key = primary_model if primary_model in bundles else next(iter(bundles))
    t1_rep = bundles[t1_key]["report"]
    t1_sys = t1_rep["systems"]
    t1_orc = t1_rep["oracle_and_4regime"]["oracle_summary"]

    lines.extend([
        "",
        rf"% Table 1: Main C-BARR Comparison ({t1_key})",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Main results across the frozen $N=450$ suite (\texttt{{{t1_key}}}). $^\dagger p < 0.05$ vs.\ \textbf{{B1b}} (Holm-Bonferroni exact McNemar).}}",
        r"\label{tab:main_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccccccccc}",
        r"\toprule",
        r"\textbf{System Variant} & \textbf{GSM-E} & \textbf{GSM-H} & \textbf{CSQA} & \textbf{IFE-Plan} & \textbf{IFE-Prec} & \textbf{IFE-Conf} & \textbf{MacroAcc} $\uparrow$ & $\textbf{OAA}_{250}$ $\uparrow$ & $\textbf{AUC}_{\text{OAA}}$ $\uparrow$ & ${U}_{0.05}$ $\uparrow$ & ${\overline{T}}_{\text{tot}}$ / \textbf{Calls} $\downarrow$ \\",
        r"\midrule",
    ])

    def _row_tex(label: str, row_d: dict[str, Any], sig: str = "") -> str:
        sc = " & ".join(_fmt_pct(row_d["slice_acc"][s]) for s in SLICE_ORDER)
        return (
            f"{label} & {sc} & {_fmt_pct(row_d['macro_acc'])}{sig} & "
            f"{_fmt_pct(row_d['oaa_250'])} & {_fmt_pct(row_d['auc_oaa_2048'])} & "
            f"{_fmt_pct(row_d['u_005'])} & {row_d['mean_t_tot']:.0f} / {row_d['mean_calls']:.2f} \\\\"
        )

    for k in VARIANT_ORDER:
        if k in ("M1a", "M2_B64"):
            lines.append(r"\midrule")
        sig = r"$^\dagger$" if float(t1_sys[k].get("p_holm_vs_b1b", 1.0)) < 0.05 else ""
        lines.append(_row_tex(VARIANT_LATEX_LABELS[k], t1_sys[k], sig=sig))
    if "M2_B512" in t1_rep.get("auxiliary_systems", {}):
        lines.append(_row_tex(r"M2-512: C-BARR Aux.\ Cap ($B=512$, R3)", t1_rep["auxiliary_systems"]["M2_B512"]))
    lines.append(r"\midrule")
    lines.append(_row_tex(r"\textit{Instance-Level Oracle\_UB}", t1_orc))
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}"])

    # Table 2: Cross-Scale 4-Regime Flip Audit & Loop Pathology
    lines.extend([
        "",
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
    ])
    known_ids = {m for m, _, _ in DEFAULT_SCALE_ORDER}
    scale_rows = list(DEFAULT_SCALE_ORDER)
    for extra_m in bundles:
        if extra_m not in known_ids:
            scale_rows.append((extra_m, rf"\texttt{{{extra_m}}}", "Evaluated Backbone"))
    for model_id, tex_name, mixer in scale_rows:
        if model_id == dense_model:
            lines.append(r"\midrule")
        if model_id not in bundles:
            lines.append(f"{tex_name} & {mixer} & -- & -- & -- & -- & -- & -- & -- & -- \\\\")
            continue
        rep = bundles[model_id]["report"]
        o4 = rep["oracle_and_4regime"]
        b1b_s = rep["systems"]["B1b"]
        m2_s = rep["systems"]["M2"]
        sav = 100.0 * (1.0 - float(m2_s["mean_t_tot"]) / max(1e-6, float(b1b_s["mean_t_tot"])))
        lines.append(
            f"{tex_name} & {mixer} & {_fmt_pct(o4['helped_rate'])} & {_fmt_pct(o4['hurt_rate'])} & "
            f"{_fmt_pct(b1b_s['loop_rate'])} & {_fmt_pct(b1b_s['exhaust_rate'])} & "
            f"{_fmt_pct(m2_s['loop_rate'])} & {_fmt_pct(b1b_s['macro_acc'])} & "
            f"{_fmt_pct(m2_s['macro_acc'])} & {sav:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}"])

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

    lines.extend([
        "",
        r"% Table 3: Section 6 Causal Diagnostic Control on E_hurt",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Strict prompt-level recovery (\%) on held-out IFEval ``Hurt'' prompts comparing Trace Stripping vs.\ Post-Think Constraint Re-Anchoring.}",
        r"\label{tab:diagnostic_control}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Intervention Condition (Fixed $t_{\text{think}}$)} & \textbf{Causal Factor Isolated} & \multicolumn{2}{c}{\textbf{Qwen3.5-4B (DeltaNet)}} & \multicolumn{2}{c}{\textbf{SmolLM3-3B (Dense)}} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r" & & \textbf{Inhibition} & \textbf{Structural} & \textbf{Inhibition} & \textbf{Structural} \\",
        r"\midrule",
        r"B1b: Unmodified \texttt{<think>} ($|t_{\text{think}}|$) & Baseline failure state (0\%) & 0.0 & 0.0 & 0.0 & 0.0 \\",
    ])
    for m_key, m_lbl, m_desc in DIAG_MODES:
        if m_key == "B1_pre_anchor":
            lines.append(r"\midrule")
        lines.append(
            f"{m_lbl} & {m_desc} & "
            f"{_d_val(p_diag, m_key, 'inhibition_recovery_pct')} & {_d_val(p_diag, m_key, 'structural_recovery_pct')} & "
            f"{_d_val(d_diag, m_key, 'inhibition_recovery_pct')} & {_d_val(d_diag, m_key, 'structural_recovery_pct')} \\\\"
        )
    lines.extend([
        r"\midrule",
        rf"\textit{{Derived Recency-Decay Share ($\rho_{{\text{{recency}}}}$, \%)}} & $R_{{\text{{post}}}}/R_{{\text{{strip}}}}$ & \multicolumn{{2}}{{c}}{{{_rho_val(p_diag, 'rho_recency')}}} & \multicolumn{{2}}{{c}}{{{_rho_val(d_diag, 'rho_recency')}}} \\",
        rf"\textit{{Residual Semantic Interference ($\rho_{{\text{{interf}}}}$, \%)}} & $(R_{{\text{{strip}}}}-\max(R_{{\text{{post}}}},R_{{\text{{pre}}}}))/R_{{\text{{strip}}}}$ & \multicolumn{{2}}{{c}}{{{_rho_val(p_diag, 'rho_interference')}}} & \multicolumn{{2}}{{c}}{{{_rho_val(d_diag, 'rho_interference')}}} \\",
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


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
    for keys, col, mk, grp_lbl in style_groups:
        xs = [float(sys_d[k]["mean_t_tot"]) for k in keys]
        ys = [100.0 * float(sys_d[k]["macro_acc"]) for k in keys]
        ax_l.scatter(xs, ys, color=col, marker=mk, s=55, label=grp_lbl, zorder=3)
        for k, x, y in zip(keys, xs, ys):
            ann_k = "M2-250" if k == "M2_B256" else k
            ax_l.annotate(ann_k, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7.5)

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
        xytext=(5, 5),
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
    ax_l.set_xlabel(r"Mean Total Tokens ($\overline{T}_{\mathrm{tot}}$)")
    ax_l.set_ylabel("Macro Accuracy (%)")
    ax_l.set_title(f"(a) Accuracy vs. Token Budget ({t_key.split('/')[-1]})")
    ax_l.legend(loc="lower right", frameon=True, framealpha=0.9)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze C-BARR outputs and generate tables/figures.")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--fig-dir", type=str, default="paper/figures")
    parser.add_argument("--primary-model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--dense-model", type=str, default="HuggingFaceTB/SmolLM3-3B")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    fig_dir = pathlib.Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    bundles = discover_bundles(out_dir)
    if not bundles:
        raise SystemExit(f"No model matrix summaries found under {out_dir}")

    _configure_matplotlib()

    stats_out: dict[str, Any] = {"models": {}, "n_boot": args.n_boot, "seed": args.seed}
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

    (out_dir / "stats_summary.json").write_text(
        json.dumps(stats_out, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "paper_tables.tex").write_text(
        generate_latex_tables(bundles, args.primary_model, args.dense_model), encoding="utf-8"
    )

    plot_fig1_pareto_frontier(bundles, args.primary_model, fig_dir / "fig1_pareto_frontier.pdf")
    plot_fig2_bsr_sla(bundles, args.primary_model, args.dense_model, fig_dir / "fig2_bsr_sla_curve.pdf")
    plot_fig3_ehurt_recovery(
        bundles, args.primary_model, args.dense_model, fig_dir / "fig3_ehurt_causal_recovery.pdf"
    )

    print(banner(f"PHASE 7 ANALYSIS COMPLETE ({len(bundles)} models)", "magenta"))
    print(f"Analyzed {len(bundles)} model(s): {', '.join(bundles.keys())}")
    print(f"Saved stats to {out_dir / 'stats_summary.json'}")
    print(f"Saved LaTeX tables to {out_dir / 'paper_tables.tex'}")
    print(f"Saved 3 vector PDF figures to {fig_dir}/")
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
