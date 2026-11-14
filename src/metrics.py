"""Statistical inference, Holm-Bonferroni, OAA_250 / AUC_OAA, BSR(B), Spillover, and Causal Attribution Audit."""
from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any
import numpy as np

from src.graders import has_cot_spillover

SLICE_ORDER: Sequence[str] = (
    "gsm8k_easy",
    "gsm8k_hard",
    "csqa",
    "ifeval_planning",
    "ifeval_precision",
    "ifeval_conflict",
)

# Upstream google/IFEval prompt-kwargs mismatch keys documented in google-research Issue #3103
KNOWN_UPSTREAM_DEFECT_KEYS: frozenset[int] = frozenset({30, 337, 2078, 3311})


def wilson_ci(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float, float]:
    """Computes (rate, lo, hi) Wilson score confidence interval."""
    if total <= 0:
        return 0.0, 0.0, 0.0
    p = float(successes) / float(total)
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def is_clean_item(item: dict[str, Any]) -> bool:
    """Returns False iff item matches upstream google/IFEval defect keys (Issue #3103)."""
    if item.get("dataset") != "ifeval":
        return True
    if bool(item.get("has_upstream_defect", False)):
        return False
    raw_key = item.get("key")
    if raw_key is not None and int(raw_key) in KNOWN_UPSTREAM_DEFECT_KEYS:
        return False
    return True


def filter_clean_446(
    items: list[dict[str, Any]], runs: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filters aligned (items, runs) pairs to Clean-446 (excluding upstream IFEval defects)."""
    clean_items: list[dict[str, Any]] = []
    clean_runs: list[dict[str, Any]] = []
    for item, run in zip(items, runs):
        if is_clean_item(item):
            clean_items.append(item)
            clean_runs.append(run)
    return clean_items, clean_runs


def exact_and_midp_mcnemar(z_base: list[int], z_method: list[int]) -> dict[str, float]:
    """Paired two-sided exact binomial & Lancaster mid-p McNemar tests over discordant pairs."""
    b_only = sum(1 for a, b in zip(z_base, z_method) if a == 1 and b == 0)
    m_only = sum(1 for a, b in zip(z_base, z_method) if a == 0 and b == 1)
    n = b_only + m_only
    if n == 0:
        return {"base_wins": 0, "method_wins": 0, "discordant_n": 0, "p_exact": 1.0, "p_midp": 1.0}
    k = min(b_only, m_only)
    tail_le_k = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    prob_eq_k = math.comb(n, k) / (2**n)
    p_exact = min(1.0, 2.0 * tail_le_k)
    p_midp = min(1.0, 2.0 * tail_le_k - prob_eq_k)
    return {
        "base_wins": b_only,
        "method_wins": m_only,
        "discordant_n": n,
        "p_exact": p_exact,
        "p_midp": p_midp,
    }


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Step-down Holm-Bonferroni FWER adjustment with monotonicity enforcement."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, p_values[idx] * (m - rank))
        # Step-down adjusted p-values must be non-decreasing across sorted hypotheses
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def paired_bootstrap_diff_ci(
    metric_a: np.ndarray, metric_b: np.ndarray, n_boot: int = 10000, seed: int = 42
) -> tuple[float, float, float]:
    """Returns (mean_diff_b_minus_a, ci_2.5, ci_97.5)."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(metric_b, dtype=np.float64) - np.asarray(metric_a, dtype=np.float64)
    boot_means = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def compute_oaa_t(z: list[int], t_delib: list[int], threshold_t: int = 250) -> float:
    """Overthinking-Adjusted Accuracy at thinking-token threshold t (Aggarwal et al., 2025)."""
    return float(np.mean([zi * int(ti <= threshold_t) for zi, ti in zip(z, t_delib)]))


def compute_auc_oaa(z: list[int], t_delib: list[int], t_max: int = 2048) -> float:
    """Exact closed-form Area Under the Overthinking-Adjusted Accuracy curve on [0, t_max]."""
    arr_z = np.asarray(z, dtype=np.float64)
    arr_t = np.asarray(t_delib, dtype=np.float64)
    # Closed form of (1/t_max) * integral_0^t_max 1[T_delib <= t] dt = max(0, (t_max - T_delib) / t_max)
    weights = np.clip((t_max - arr_t) / max(1, t_max), 0.0, 1.0)
    return float(np.mean(arr_z * weights))


def compute_bsr_curve(
    z: list[int],
    t_tot: list[int],
    budgets: Sequence[int] = (64, 128, 256, 512, 1024, 2048),
) -> dict[int, float]:
    """Budget-Constrained Success Rate BSR(B) across total-token SLA ceilings."""
    arr_z = np.asarray(z, dtype=np.float64)
    arr_t = np.asarray(t_tot, dtype=np.float64)
    return {b: float(np.mean(arr_z * (arr_t <= b))) for b in budgets}


def compute_causal_decomposition(
    z_strip: list[int],
    z_post_anchor: list[int],
    z_pre_anchor: list[int],
    z_post_raw: list[int],
) -> dict[str, float]:
    """Quantifies Recency Decay vs. Semantic Interference shares on E_hurt (Causal Attribution)."""
    if not z_strip:
        return {
            "rho_recency": 0.0,
            "rho_interference": 0.0,
            "recency_recovery_share_pct": 0.0,
            "residual_interference_share_pct": 0.0,
            "locus_gap_pre_minus_post_pp": 0.0,
            "wrapper_corruption_penalty_pp": 0.0,
        }
    r_strip = float(np.mean(z_strip))
    r_post = float(np.mean(z_post_anchor))
    r_pre = float(np.mean(z_pre_anchor))
    r_raw = float(np.mean(z_post_raw))
    denom = max(1e-9, r_strip)
    # Recency share = fraction of Trace-Stripping ceiling recovered merely by post-</think> reminder
    rho_recency = r_post / denom
    # Residual semantic interference = ceiling gap not recovered by either pre- or post-think anchoring
    rho_interference = max(0.0, r_strip - max(r_post, r_pre)) / denom
    return {
        "rho_recency": rho_recency,
        "rho_interference": rho_interference,
        "recency_recovery_share_pct": 100.0 * rho_recency,
        "residual_interference_share_pct": 100.0 * rho_interference,
        "locus_gap_pre_minus_post_pp": 100.0 * (r_pre - r_post),
        "wrapper_corruption_penalty_pp": 100.0 * (r_post - r_raw),
    }


def evaluate_system_table(
    items: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    lam_utility: float = 0.05,
    norm_tokens: float = 250.0,
) -> dict[str, Any]:
    """Computes slice accuracies, MacroAcc, OAA_250, AUC_OAA, U_0.05, BSR(B), spillover_rate, T_tot, and Calls."""
    slice_scores: dict[str, list[int]] = {s: [] for s in SLICE_ORDER}
    z_all, t_delib_all, t_tot_all, calls_all = [], [], [], []
    loop_flags, exhaust_flags, spillover_flags = [], [], []

    for item, r in zip(items, runs):
        zi = int(r["z"])
        t_delib = int(r.get("T_probe", 0)) + int(r.get("T_think", 0))
        t_tot = int(r["T_total"])
        slice_scores[item["slice"]].append(zi)
        z_all.append(zi)
        t_delib_all.append(t_delib)
        t_tot_all.append(t_tot)
        calls_all.append(int(r.get("calls", 1)))
        loop_flags.append(int(r.get("has_4gram_loop", False)))
        exhaust_flags.append(int(r.get("think_exhausted", False)))
        if "has_cot_spillover" in r:
            spillover_flags.append(int(r["has_cot_spillover"]))
        else:
            spillover_flags.append(
                int(has_cot_spillover(str(r.get("a_final", "")), bool(r.get("truncated", False))))
            )

    per_slice_acc = {
        s: (float(np.mean(vals)) if vals else 0.0) for s, vals in slice_scores.items()
    }
    macro_acc = float(np.mean(list(per_slice_acc.values())))
    u_vec = np.asarray(z_all, dtype=np.float64) - lam_utility * (
        np.asarray(t_tot_all, dtype=np.float64) / norm_tokens
    )

    return {
        "slice_acc": per_slice_acc,
        "micro_acc": float(np.mean(z_all)),
        "macro_acc": macro_acc,
        "oaa_250": compute_oaa_t(z_all, t_delib_all, threshold_t=250),
        "auc_oaa_2048": compute_auc_oaa(z_all, t_delib_all, t_max=2048),
        "u_005": float(np.mean(u_vec)),
        "mean_t_tot": float(np.mean(t_tot_all)),
        "mean_calls": float(np.mean(calls_all)),
        "loop_rate": float(np.mean(loop_flags)),
        "exhaust_rate": float(np.mean(exhaust_flags)),
        "spillover_rate": float(np.mean(spillover_flags)),
        "bsr": compute_bsr_curve(z_all, t_tot_all),
        "z_vec": z_all,
        "u_vec": u_vec.tolist(),
    }


def compute_conformal_risk_bounds(
    z_base: Sequence[int],
    z_method: Sequence[int],
    alpha_delta: float = 0.05,
    n_grid: int = 1,
) -> dict[str, Any]:
    """Finite-sample split Conformal Risk Control & Hoeffding bounds on degradation loss max(0, z_base - z_method)."""
    losses = [
        1.0 if (int(a) == 1 and int(b) == 0) else 0.0
        for a, b in zip(z_base, z_method)
    ]
    n = len(losses)
    if n == 0:
        return {
            "n_calib": 0,
            "degradation_events": 0,
            "empirical_degradation_risk": 0.0,
            "crc_expectation_bound": 1.0,
            "hoeffding_ucb_delta": float(alpha_delta),
            "hoeffding_ucb": 1.0,
            "n_grid": int(max(1, n_grid)),
        }
    k_events = int(sum(losses))
    emp_risk = float(k_events / n)
    # Standard split conformal risk control expectation bound for loss in [0, 1]: (n*R_hat + 1)/(n + 1)
    crc_bound = min(1.0, float((n * emp_risk + 1.0) / (n + 1.0)))
    delta_c = max(1e-12, min(0.999999, float(alpha_delta)))
    eff_grid = max(1, int(n_grid))
    hoeffding_rad = math.sqrt(math.log(eff_grid / delta_c) / (2.0 * n))
    hoeffding_ucb = min(1.0, emp_risk + hoeffding_rad)
    return {
        "n_calib": n,
        "degradation_events": k_events,
        "empirical_degradation_risk": emp_risk,
        "crc_expectation_bound": crc_bound,
        "hoeffding_ucb_delta": delta_c,
        "hoeffding_ucb": hoeffding_ucb,
        "n_grid": eff_grid,
    }


def compute_oracle_efficiency_audit(
    systems: dict[str, dict[str, Any]],
    oracle_summary: dict[str, Any],
) -> dict[str, Any]:
    """Benchmarks M2 and baselines against the discrete-budget hindsight Oracle_UB."""
    o_tok = max(1e-6, float(oracle_summary.get("mean_t_tot", 0.0)))
    o_acc = float(oracle_summary.get("macro_acc", 0.0))
    o_u = float(oracle_summary.get("u_005", 0.0))
    audit: dict[str, Any] = {
        "oracle_macro_acc": o_acc,
        "oracle_mean_t_tot": float(oracle_summary.get("mean_t_tot", 0.0)),
        "oracle_u_005": o_u,
        "variants": {},
    }
    for name in ("B0", "B1b", "M1c", "M2_B64", "M2_B128", "M2_B256", "M2_Pre", "M2"):
        if name not in systems:
            continue
        row = systems[name]
        m_tok = float(row.get("mean_t_tot", 0.0))
        m_acc = float(row.get("macro_acc", 0.0))
        m_u = float(row.get("u_005", 0.0))
        audit["variants"][name] = {
            "macro_acc_retention": (m_acc / o_acc) if o_acc > 1e-9 else 0.0,
            "oracle_token_overhead_ratio": m_tok / o_tok,
            "excess_tokens_over_oracle": m_tok - float(oracle_summary.get("mean_t_tot", 0.0)),
            "u005_gap_to_oracle": m_u - o_u,
        }
    return audit


def compute_4regime_and_oracle(
    items: list[dict[str, Any]],
    b0_runs: list[dict[str, Any]],
    b1b_runs: list[dict[str, Any]],
    candidate_run_lists: list[list[dict[str, Any]]],
    lam_utility: float = 0.05,
    norm_tokens: float = 250.0,
) -> dict[str, Any]:
    """Computes Helped/Hurt/Tie-Pass/Tie-Fail counts and minimal-token Oracle_UB."""
    z_b0 = [int(r["z"]) for r in b0_runs]
    z_b1b = [int(r["z"]) for r in b1b_runs]
    # 4-regime disagreement audit: identifies where thinking flips B0 from fail->pass (Helped) or pass->fail (Hurt)
    helped = sum(1 for a, b in zip(z_b0, z_b1b) if a == 0 and b == 1)
    hurt = sum(1 for a, b in zip(z_b0, z_b1b) if a == 1 and b == 0)
    tie_pass = sum(1 for a, b in zip(z_b0, z_b1b) if a == 1 and b == 1)
    tie_fail = sum(1 for a, b in zip(z_b0, z_b1b) if a == 0 and b == 0)

    oracle_synthesized: list[dict[str, Any]] = []
    for i in range(len(items)):
        # Oracle_UB assumes an omniscient single-call router choosing the cheapest passing config
        passing = [cfg[i] for cfg in candidate_run_lists if int(cfg[i]["z"]) == 1]
        if passing:
            best = dict(min(passing, key=lambda r: int(r["T_total"])))
            best["calls"] = 1
            oracle_synthesized.append(best)
        else:
            fallback = dict(candidate_run_lists[0][i])
            fallback["calls"] = 1
            oracle_synthesized.append(fallback)

    oracle_table = evaluate_system_table(
        items, oracle_synthesized, lam_utility=lam_utility, norm_tokens=norm_tokens
    )
    n = max(1, len(z_b0))
    return {
        "helped_rate": helped / n,
        "hurt_rate": hurt / n,
        "flip_rate": (helped + hurt) / n,
        "tie_pass_rate": tie_pass / n,
        "tie_fail_rate": tie_fail / n,
        "oracle_summary": oracle_table,
    }
