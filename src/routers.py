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
    """Extracts the 5 scalar Gate-2 logistic regression features in canonical order."""
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
    # Easy non-symbolic multiple choice needs only a brief 64-token sanity check
    if is_mc and g1.phi_sym == 0:
        return 64
    # Conflicting constraints (planning + precision) or planning+math need the full 250-token SLA tier
    if g1.phi_plan == 1 and (g1.phi_prec == 1 or g1.phi_sym == 1):
        return 250
    # High peak probe entropy or Thinking Inertia escalates uncertain items to the top 250 tier
    if probe is not None and (probe["H_max"] >= tau_h_high or probe["I_inertia"] == 1):
        return 250
    if g1.phi_plan == 1:
        return 128
    return 128


def _get_slow_rec(rec: dict[str, Any], b_val: int, direct: bool, prefix: str = "m2") -> dict[str, Any]:
    """Selects direct-pass (no probe tax) vs. Gate-2-escalated (T_probe deducted) cached slow record."""
    if b_val in (250, 256):
        if direct and f"{prefix}_b256_direct" in rec:
            return rec[f"{prefix}_b256_direct"]
        return rec[f"{prefix}_b256"]
    return rec[f"{prefix}_b{b_val}"]


def _gate2_target_label(rec: dict[str, Any]) -> int:
    """Computes binary Gate-2 training/eval label: slow thinking helps or fast probe showed inertia."""
    z_fast = int(rec["fast_done"]["z"])
    z_slow = int(rec["m2_b128"]["z"])
    return int((z_slow > z_fast) or (z_fast == 0 and int(rec["probe"]["I_inertia"]) == 1))


def _binary_clf_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Computes binary classification Accuracy, Precision, Recall, F1, and pairwise ROC-AUC."""
    n = int(len(y_true))
    if n == 0:
        return {
            "n": 0,
            "threshold": float(threshold),
            "pos_count": 0,
            "pos_rate": 0.0,
            "pred_pos_rate": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.5,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }
    y_pred = (y_prob >= float(threshold)).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = float((tp + tn) / n)
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        diff = pos_scores[:, None] - neg_scores[None, :]
        roc_auc = float(np.mean((diff > 0).astype(np.float64) + 0.5 * (diff == 0)))
    else:
        roc_auc = 0.5

    return {
        "n": n,
        "threshold": float(threshold),
        "pos_count": int(np.sum(y_true == 1)),
        "pos_rate": float(np.mean(y_true)),
        "pred_pos_rate": float(np.mean(y_pred)),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


class CBARRCalibrator:
    """Two-stage zero-leakage calibrator using seed_60 for weights and dev_60 for tau*."""

    def __init__(
        self,
        lam_utility: float = 0.05,
        norm_tokens: float = 250.0,
        seed: int = 42,
        reg_c: float = 1.0,
        tau_min: float = 0.15,
        tau_max: float = 0.85,
        tau_steps: int = 29,
    ):
        self.lam_utility = lam_utility
        self.norm_tokens = norm_tokens
        self.seed = seed
        self.reg_c = float(reg_c)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.tau_steps = max(2, int(tau_steps))
        self.last_tau_sweep_: list[dict[str, float]] = []

    def fit(
        self,
        seed_records: list[dict[str, Any]],
        dev_records: list[dict[str, Any]],
    ) -> CalibratedRouterParams:
        """Stage A: fit logistic regression on seed_60; Stage B: tune tau* on dev_60 for U_0.05."""
        X_seed, y_seed = [], []
        h_max_thinking_helped: list[float] = []

        # Positive label: thinking strictly beats fast probe, or fast probe failed with Thinking Inertia
        for rec in seed_records:
            X_seed.append(_build_gate2_feature_vec(rec["g1"], rec["probe"]))
            need_slow = _gate2_target_label(rec)
            y_seed.append(need_slow)
            if int(rec["m2_b128"]["z"]) > int(rec["fast_done"]["z"]):
                h_max_thinking_helped.append(float(rec["probe"]["H_max"]))

        X_arr = np.vstack(X_seed)
        y_arr = np.asarray(y_seed, dtype=int)

        # Deterministic prior weights fallback if synthetic/small seed split happens to be single-class
        if len(np.unique(y_arr)) < 2:
            coef = [0.6, 0.8, 0.4, 1.0, 0.5]
            intercept = -1.0
        else:
            clf = LogisticRegression(C=self.reg_c, class_weight="balanced", random_state=self.seed)
            clf.fit(X_arr, y_arr)
            coef = [float(c) for c in clf.coef_[0]]
            intercept = float(clf.intercept_[0])

        # High-uncertainty entropy threshold is anchored at the median H_max where thinking helped
        tau_h_high = (
            float(np.median(h_max_thinking_helped)) if h_max_thinking_helped else 0.85
        )

        best_tau = 0.50
        best_u = -1e9
        best_esc_rate = 0.35
        tau_sweep: list[dict[str, float]] = []

        # Grid-search Gate-2 escalation threshold tau across configured range to maximize utility on dev_60
        w_vec = np.asarray(coef, dtype=np.float64)
        for tau in np.linspace(self.tau_min, self.tau_max, self.tau_steps):
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
            esc_rate_tau = float(np.mean(escalations))
            tau_sweep.append(
                {
                    "tau": float(tau),
                    "utility_u005": mean_u,
                    "esc_rate": esc_rate_tau,
                }
            )
            if mean_u > best_u:
                best_u = mean_u
                best_tau = float(tau)
                best_esc_rate = esc_rate_tau

        self.last_tau_sweep_ = tau_sweep
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


def compute_calibration_diagnostics(
    seed_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    params: CalibratedRouterParams,
    tau_sweep: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Computes Gate-2 classifier Precision/Recall/F1/ROC-AUC across seed train and dev validation."""
    def _eval_split(records: list[dict[str, Any]], gate2_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
        y_true_l, y_prob_l = [], []
        for rec in records:
            g1 = rec.get("g1") or run_gate1_lexical(rec["item"]["prompt"])
            if gate2_only and ((g1.phi_prec == 1 and g1.phi_plan == 0 and g1.phi_sym == 0) or g1.phi_plan == 1):
                continue
            y_true_l.append(_gate2_target_label(rec))
            y_prob_l.append(score_gate2_probability(g1, rec["probe"], params))
        return np.asarray(y_true_l, dtype=int), np.asarray(y_prob_l, dtype=np.float64)

    y_seed, p_seed = _eval_split(seed_records, gate2_only=False)
    y_dev_all, p_dev_all = _eval_split(dev_records, gate2_only=False)
    y_dev_g2, p_dev_g2 = _eval_split(dev_records, gate2_only=True)

    diag: dict[str, Any] = {
        "seed_train_at_05": _binary_clf_metrics(y_seed, p_seed, threshold=0.5),
        "seed_train_at_tau": _binary_clf_metrics(y_seed, p_seed, threshold=params.tau_prob),
        "dev_all_at_tau": _binary_clf_metrics(y_dev_all, p_dev_all, threshold=params.tau_prob),
        "dev_gate2_at_tau": _binary_clf_metrics(y_dev_g2, p_dev_g2, threshold=params.tau_prob),
    }
    if tau_sweep:
        best_pt = max(tau_sweep, key=lambda x: x["utility_u005"])
        diag["best_dev_utility_u005"] = best_pt["utility_u005"]
        diag["tau_sweep"] = tau_sweep
    return diag


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
        """Accounts for both the initial Gate-2 probe call and the escalated slow-thinking call."""
        merged = dict(slow_run)
        merged["T_probe"] = probe["T_probe"]
        merged["T_total"] = probe["T_probe"] + slow_run["T_total"]
        merged["calls"] = 2
        merged["latency_ms"] = probe["latency_ms"] + slow_run["latency_ms"]
        return merged

    def _route_with_slow_target(
        direct_slow: dict[str, Any], esc_slow: dict[str, Any]
    ) -> dict[str, Any]:
        """Executes C-BARR's 3-branch cascade: Gate-1 bypass -> Gate-1 plan direct -> Gate-2 probe."""
        if g1.phi_prec == 1 and g1.phi_plan == 0 and g1.phi_sym == 0:
            # Pure precision rules regress under thinking; bypass deliberation immediately (1.0 call)
            return cache_row["b0"]
        if g1.phi_plan == 1:
            # Planning rules benefit reliably from deliberation; escalate directly without probe tax
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
