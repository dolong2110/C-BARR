"""Cascaded C-BARR calibration (seed_60 -> dev_60) and unified B0-M2 policy execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.engine import Gate1Output, run_gate1_lexical


@dataclass
class CalibratedRouterParams:
    coef: list[float]
    intercept: float
    tau_prob: float
    tau_h_high: float
    m2_escalation_rate_dev: float


def _build_gate2_feature_vec(g1: Gate1Output, probe: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            float(probe["H_first"]),
            float(probe["H_mean"]),
            float(probe["H_max"]),
            float(probe["I_inertia"]),
            float(g1.phi_sym),
        ],
        dtype=np.float64,
    )


def select_tiered_budget(
    prompt: str, g1: Gate1Output, probe: dict[str, Any] | None, tau_h_high: float
) -> int:
    """Assigns C-BARR Stage-2 thinking cap B(x) in {64, 128, 250} (Patch D13: capped at 250)."""
    is_mc = "(A)" in prompt and "(B)" in prompt
    if is_mc and g1.phi_sym == 0:
        return 64
    if g1.phi_plan == 1 and (g1.phi_prec == 1 or g1.phi_sym == 1):
        return 250
    if probe is not None and (probe["H_max"] >= tau_h_high or probe["I_inertia"] == 1):
        return 250
    if g1.phi_plan == 1:
        return 128
    return 128


def _get_slow_rec(rec: dict[str, Any], b_val: int, direct: bool, prefix: str = "m2") -> dict[str, Any]:
    if b_val in (250, 256):
        if direct and f"{prefix}_b256_direct" in rec:
            return rec[f"{prefix}_b256_direct"]
        return rec[f"{prefix}_b256"]
    return rec[f"{prefix}_b{b_val}"]


class CBARRCalibrator:
    """Two-stage zero-leakage calibrator using seed_60 for weights and dev_60 for tau*."""

    def __init__(self, lam_utility: float = 0.05, norm_tokens: float = 250.0, seed: int = 42):
        self.lam_utility = lam_utility
        self.norm_tokens = norm_tokens
        self.seed = seed

    def fit(
        self,
        seed_records: list[dict[str, Any]],
        dev_records: list[dict[str, Any]],
    ) -> CalibratedRouterParams:
        X_seed, y_seed = [], []
        h_max_thinking_helped: list[float] = []

        for rec in seed_records:
            X_seed.append(_build_gate2_feature_vec(rec["g1"], rec["probe"]))
            z_fast = int(rec["fast_done"]["z"])
            z_slow = int(rec["m2_b128"]["z"])
            need_slow = int((z_slow > z_fast) or (z_fast == 0 and rec["probe"]["I_inertia"] == 1))
            y_seed.append(need_slow)
            if z_slow > z_fast:
                h_max_thinking_helped.append(float(rec["probe"]["H_max"]))

        X_arr = np.vstack(X_seed)
        y_arr = np.asarray(y_seed, dtype=int)

        if len(np.unique(y_arr)) < 2:
            coef = [0.6, 0.8, 0.4, 1.0, 0.5]
            intercept = -1.0
        else:
            clf = LogisticRegression(C=1.0, class_weight="balanced", random_state=self.seed)
            clf.fit(X_arr, y_arr)
            coef = [float(c) for c in clf.coef_[0]]
            intercept = float(clf.intercept_[0])

        tau_h_high = (
            float(np.median(h_max_thinking_helped)) if h_max_thinking_helped else 0.85
        )

        best_tau = 0.50
        best_u = -1e9
        best_esc_rate = 0.35

        w_vec = np.asarray(coef, dtype=np.float64)
        for tau in np.linspace(0.15, 0.85, 29):
            utilities: list[float] = []
            escalations: list[int] = []
            for rec in dev_records:
                g1: Gate1Output = rec["g1"]
                prompt_str = rec["item"]["prompt"]
                if g1.phi_prec == 1 and g1.phi_plan == 0 and g1.phi_sym == 0:
                    z_sel = rec["b0"]["z"]
                    t_sel = rec["b0"]["T_total"]
                    escalations.append(0)
                elif g1.phi_plan == 1:
                    b_dir = select_tiered_budget(prompt_str, g1, None, tau_h_high)
                    slow_rec = _get_slow_rec(rec, b_dir, direct=True)
                    z_sel = slow_rec["z"]
                    t_sel = slow_rec["T_total"]
                    escalations.append(1)
                else:
                    feat = _build_gate2_feature_vec(g1, rec["probe"])
                    logit = float(np.dot(w_vec, feat) + intercept)
                    prob = 1.0 / (1.0 + np.exp(-logit))
                    if prob >= tau:
                        b_esc = select_tiered_budget(prompt_str, g1, rec["probe"], tau_h_high)
                        slow_rec = _get_slow_rec(rec, b_esc, direct=False)
                        z_sel = slow_rec["z"]
                        t_sel = rec["probe"]["T_probe"] + slow_rec["T_total"]
                        escalations.append(1)
                    else:
                        z_sel = rec["fast_done"]["z"]
                        t_sel = rec["fast_done"]["T_total"]
                        escalations.append(0)
                utilities.append(float(z_sel) - self.lam_utility * (float(t_sel) / self.norm_tokens))

            mean_u = float(np.mean(utilities))
            if mean_u > best_u:
                best_u = mean_u
                best_tau = float(tau)
                best_esc_rate = float(np.mean(escalations))

        return CalibratedRouterParams(
            coef=coef,
            intercept=intercept,
            tau_prob=best_tau,
            tau_h_high=tau_h_high,
            m2_escalation_rate_dev=best_esc_rate,
        )


def score_gate2_probability(
    g1: Gate1Output, probe: dict[str, Any], params: CalibratedRouterParams
) -> float:
    feat = _build_gate2_feature_vec(g1, probe)
    logit = float(np.dot(np.asarray(params.coef), feat) + params.intercept)
    return float(1.0 / (1.0 + np.exp(-logit)))


def resolve_all_variants_for_item(
    cache_row: dict[str, Any],
    params: CalibratedRouterParams,
    rng_b2b: np.random.Generator,
    target_esc_prob: float,
) -> dict[str, dict[str, Any]]:
    """Resolves B0-B2, M1a (Lexical), M1b (Entropy/Inertia), M1c, M2_{64,128,256}, M2, and M2_Pre."""
    prompt_str = cache_row["item"]["prompt"]
    g1 = run_gate1_lexical(prompt_str)
    probe = cache_row["probe"]
    p_esc = score_gate2_probability(g1, probe, params)
    gate2_esc = p_esc >= params.tau_prob

    def _merge_probe_and_slow(slow_run: dict[str, Any]) -> dict[str, Any]:
        merged = dict(slow_run)
        merged["T_probe"] = probe["T_probe"]
        merged["T_total"] = probe["T_probe"] + slow_run["T_total"]
        merged["calls"] = 2
        merged["latency_ms"] = probe["latency_ms"] + slow_run["latency_ms"]
        return merged

    def _route_with_slow_target(
        direct_slow: dict[str, Any], esc_slow: dict[str, Any]
    ) -> dict[str, Any]:
        if g1.phi_prec == 1 and g1.phi_plan == 0 and g1.phi_sym == 0:
            return cache_row["b0"]
        if g1.phi_plan == 1:
            return direct_slow
        return _merge_probe_and_slow(esc_slow) if gate2_esc else cache_row["fast_done"]

    # M1a: Gate-1 zero-cost lexical router; M1b: Gate-2 fast-pass entropy+inertia; M1c: Cascade
    m1a = cache_row["b1b"] if (g1.phi_sym == 1 or g1.phi_plan == 1) else cache_row["b0"]
    m1b = _merge_probe_and_slow(cache_row["b1b"]) if gate2_esc else cache_row["fast_done"]
    m1c = _route_with_slow_target(cache_row["b1b"], cache_row["b1b"])

    # Fixed-cap M2 sweeps across B in {64, 128, 250}
    m2_b64 = _route_with_slow_target(cache_row["m2_b64"], cache_row["m2_b64"])
    m2_b128 = _route_with_slow_target(cache_row["m2_b128"], cache_row["m2_b128"])
    m2_b256 = _route_with_slow_target(cache_row["m2_b256_direct"], cache_row["m2_b256"])

    # Adaptive tiered M2 (B in {64, 128, 250} + Post-Think Stripped Re-Anchoring)
    b_dir = select_tiered_budget(prompt_str, g1, None, params.tau_h_high)
    b_esc = select_tiered_budget(prompt_str, g1, probe, params.tau_h_high)
    m2 = _route_with_slow_target(
        _get_slow_rec(cache_row, b_dir, direct=True, prefix="m2"),
        _get_slow_rec(cache_row, b_esc, direct=False, prefix="m2"),
    )

    # Locus ablation M2_Pre (Pre-</think> Re-Anchoring under identical routing & caps)
    m2_pre = _route_with_slow_target(
        _get_slow_rec(cache_row, b_dir, direct=True, prefix="m2_pre"),
        _get_slow_rec(cache_row, b_esc, direct=False, prefix="m2_pre"),
    )

    b2b = (
        _merge_probe_and_slow(cache_row["b1b"])
        if rng_b2b.random() < target_esc_prob
        else cache_row["fast_done"]
    )

    return {
        "B0": cache_row["b0"],
        "B1a": cache_row["b1a"],
        "B1b": cache_row["b1b"],
        "B2a": cache_row["b2a"],
        "B2b": b2b,
        "M1a": m1a,
        "M1b": m1b,
        "M1c": m1c,
        "M2_B64": m2_b64,
        "M2_B128": m2_b128,
        "M2_B256": m2_b256,
        "M2_Pre": m2_pre,
        "M2": m2,
    }
