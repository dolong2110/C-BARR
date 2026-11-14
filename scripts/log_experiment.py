"""Records training params, live in-flight progress, and per-step/per-loop results to EXPERIMENT_LOG.md."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import platform
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.logfmt import hl
from src.metrics import SLICE_ORDER

VARIANT_DISPLAY_ORDER = (
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


def _detect_git_rev(root: pathlib.Path) -> str:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def _sanitize_text(text: str, root: pathlib.Path, secret_token: str = "") -> str:
    """Strips tokens, emails, and host-local home directory prefixes for safe public/blind logs."""
    out = str(text or "")
    if secret_token:
        out = out.replace(secret_token, "***")
    root_str = str(root.resolve())
    if root_str in out:
        out = out.replace(root_str + "/", "").replace(root_str, ".")
    out = re.sub(r"/(?:Users|home)/[^/\s]+", "<HOME>", out)
    out = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<EMAIL>", out)
    return out


def _detect_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": "unavailable",
        "transformers": "unavailable",
        "cuda_available": False,
        "mps_available": False,
        "cuda_version": None,
        "gpu_count": 0,
        "gpu_name": "CPU/MPS",
        "vram_gb": 0.0,
        "sm_cap": None,
        "bf16_native": False,
        "fla_available": False,
        "causal_conv1d_available": False,
    }
    try:
        import transformers

        info["transformers"] = getattr(transformers, "__version__", "unknown")
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if info["cuda_available"]:
            info["gpu_count"] = int(torch.cuda.device_count())
            props = torch.cuda.get_device_properties(0)
            sm = torch.cuda.get_device_capability(0)
            info["gpu_name"] = props.name
            info["vram_gb"] = round(props.total_memory / (1024**3), 1)
            info["sm_cap"] = f"{sm[0]}.{sm[1]}"
            info["bf16_native"] = bool(torch.cuda.is_bf16_supported() and sm[0] >= 8)
        elif info["mps_available"]:
            info["gpu_count"] = 1
            chip_label = platform.machine() or "arm64"
            try:
                brand = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                if brand:
                    chip_label = brand
            except Exception:
                pass
            info["gpu_name"] = f"Apple MPS ({chip_label})"
            if hasattr(torch, "mps") and hasattr(torch.mps, "recommended_max_memory"):
                info["vram_gb"] = round(torch.mps.recommended_max_memory() / (1024**3), 1)
            try:
                p = torch.ones((2, 2), device="mps", dtype=torch.bfloat16)
                _ = float((p @ p).sum().item())
                info["bf16_native"] = True
            except Exception:
                info["bf16_native"] = False
    except Exception:
        pass
    for pkg, key in (("fla", "fla_available"), ("causal_conv1d", "causal_conv1d_available")):
        try:
            __import__(pkg)
            info[key] = True
        except Exception:
            info[key] = False
    return info


def _fmt_duration(seconds: float) -> str:
    sec = max(0.0, float(seconds))
    if sec < 60.0:
        return f"{sec:.1f}s"
    mins, rem = divmod(sec, 60.0)
    if mins < 60.0:
        return f"{int(mins)}m {rem:04.1f}s"
    hrs, mins = divmod(mins, 60.0)
    return f"{int(hrs)}h {int(mins):02d}m {int(rem):02d}s"


def _fmt_pct(val: float | None, decimals: int = 1) -> str:
    if val is None:
        return "n/a"
    return f"{100.0 * float(val):.{decimals}f}%"


def _load_json_safe(path: pathlib.Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_params_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}
    if raw.startswith("@"):
        p = pathlib.Path(raw[1:])
        data = _load_json_safe(p)
        return data if isinstance(data, dict) else {}
    p_cand = pathlib.Path(raw)
    if p_cand.exists() and p_cand.is_file():
        data = _load_json_safe(p_cand)
        return data if isinstance(data, dict) else {}
    data = json.loads(raw)
    return data if isinstance(data, dict) else {}


def _load_timings(path: pathlib.Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _load_model_systems(out_dir: pathlib.Path, model_name: str) -> dict[str, Any]:
    model_dir = out_dir / model_name.replace("/", "__")
    for fname in ("matrix_summary_diag491.json", "matrix_summary.json", "matrix_summary_strict341.json"):
        data = _load_json_safe(model_dir / fname)
        if isinstance(data, dict) and "systems" in data:
            return data["systems"]
    return {}


def _extract_model_summaries(out_dir: pathlib.Path) -> list[dict[str, Any]]:
    stats = _load_json_safe(out_dir / "stats_summary.json")
    if not isinstance(stats, dict):
        return []
    rows: list[dict[str, Any]] = []
    for model_name, m_data in stats.get("models", {}).items():
        g4 = m_data.get("gate_audit", {}).get("gate4_primary", {})
        cont = m_data.get("gate_audit", {}).get("contingencies", {})
        boot = m_data.get("paired_bootstrap_95ci", {}).get("M2_minus_B1b", {})
        mcn = m_data.get("targeted_mcnemar", {})
        clean = m_data.get("clean_446_sensitivity", {})
        sys_tbl = _load_model_systems(out_dir, model_name)
        m2 = sys_tbl.get("M2", {})
        b1b = sys_tbl.get("B1b", {})
        oaa_ci = boot.get("oaa_250", {})
        rows.append(
            {
                "model": model_name,
                "n_clean_446": clean.get("n_clean_items"),
                "clean_max_delta_pp": clean.get("max_abs_macro_delta_pp"),
                "m2_macro": m2.get("macro_acc"),
                "b1b_macro": b1b.get("macro_acc"),
                "m2_oaa250": m2.get("oaa_250"),
                "b1b_oaa250": b1b.get("oaa_250"),
                "tok_red_pct": g4.get("token_reduction_pct"),
                "p_holm": g4.get("m2_p_holm_vs_b1b"),
                "p_m2_vs_b0": mcn.get("M2_vs_B0", {}).get("p_exact"),
                "p_m2_vs_m2pre": mcn.get("M2_vs_M2_Pre", {}).get("p_exact"),
                "gate4_pass": bool(g4.get("m2_macro_ge_b1b") and g4.get("token_reduction_pct", 0) > 0),
                "r1_ok": cont.get("R1_in_healthy_range_0.20_to_0.85"),
                "r2_rec_pct": (100.0 * cont["R2_rho_recency"]) if "R2_rho_recency" in cont else None,
                "r2_int_pct": (100.0 * cont["R2_rho_interference"]) if "R2_rho_interference" in cont else None,
                "oaa_diff_pp": (100.0 * oaa_ci["diff_b_minus_a"]) if "diff_b_minus_a" in oaa_ci else None,
                "oaa_ci95_lo_pp": (100.0 * oaa_ci["ci95_lo"]) if "ci95_lo" in oaa_ci else None,
                "oaa_ci95_hi_pp": (100.0 * oaa_ci["ci95_hi"]) if "ci95_hi" in oaa_ci else None,
            }
        )
    return rows


def _extract_smoke_report(
    root: pathlib.Path, out_dir: pathlib.Path, smoke_override: str
) -> list[dict[str, Any]]:
    candidates: list[pathlib.Path] = []
    if smoke_override:
        candidates.append(pathlib.Path(smoke_override))
    candidates.append(out_dir / "smoke_test_gate3_report.json")
    if out_dir.resolve() == (root / "outputs").resolve():
        candidates.append(root / "outputs" / "smoke_test_gate3_report.json")
    for cand in candidates:
        data = _load_json_safe(cand)
        if isinstance(data, list) and data:
            return [x for x in data if isinstance(x, dict)]
    return []


def _extract_calibrations(out_dir: pathlib.Path) -> list[dict[str, Any]]:
    if not out_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for sub in sorted(out_dir.iterdir()):
        if not sub.is_dir():
            continue
        data = _load_json_safe(sub / "calibration_summary.json")
        if isinstance(data, dict) and "calibrated_params" in data:
            out.append(data)
    return out


def _extract_matrix_loops(out_dir: pathlib.Path) -> list[dict[str, Any]]:
    """Loads every completed (model, diag_pool) matrix summary under out_dir."""
    if not out_dir.exists():
        return []
    loops: list[dict[str, Any]] = []
    for sub in sorted(out_dir.iterdir()):
        if not sub.is_dir():
            continue
        found_pool = False
        for fname in ("matrix_summary_strict341.json", "matrix_summary_diag491.json"):
            data = _load_json_safe(sub / fname)
            if isinstance(data, dict) and "systems" in data:
                loops.append(data)
                found_pool = True
        if not found_pool:
            data = _load_json_safe(sub / "matrix_summary.json")
            if isinstance(data, dict) and "systems" in data:
                loops.append(data)
    return loops


def _extract_w7_extensions(out_dir: pathlib.Path) -> list[dict[str, Any]]:
    if not out_dir.exists():
        return []
    items: list[dict[str, Any]] = []
    for sub in sorted(out_dir.iterdir()):
        if not sub.is_dir():
            continue
        blend = _load_json_safe(sub / "ext_deltanet_blend.json")
        mlp = _load_json_safe(sub / "prefill_mlp_summary.json")
        if isinstance(blend, dict) or isinstance(mlp, dict):
            items.append({"dir": sub.name, "blend": blend, "mlp": mlp})
    return items


def _extract_live_progress(out_dir: pathlib.Path) -> list[dict[str, Any]]:
    """Loads any active mid-split progress_checkpoint.json files for in-flight GPU visibility."""
    if not out_dir.exists():
        return []
    items: list[dict[str, Any]] = []
    for sub in sorted(out_dir.iterdir()):
        if not sub.is_dir():
            continue
        data = _load_json_safe(sub / "progress_checkpoint.json")
        if isinstance(data, dict) and "done" in data:
            items.append(data)
    return items


def _collect_training_params(
    cli_params: dict[str, Any],
    calibrations: list[dict[str, Any]],
    matrix_loops: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merges global config params with runtime/hyperparameters written by calibration/matrix runs."""
    merged: dict[str, Any] = dict(cli_params)
    for src in [*calibrations, *matrix_loops]:
        tp = src.get("training_params")
        if isinstance(tp, dict):
            for k, v in tp.items():
                if k in ("model", "diag_pool"):
                    continue
                merged.setdefault(k, v)
    return merged


def _format_variant_table_lines(report: dict[str, Any]) -> list[str]:
    systems = report.get("systems", {})
    aux = report.get("auxiliary_systems", {})
    oracle_sum = report.get("oracle_and_4regime", {}).get("oracle_summary", {})
    lines = [
        "",
        "| Variant | Macro | GSM8K-E | GSM8K-H | CSQA | IF-Plan | IF-Prec | IF-Conf | OAA_250 | AUC_OAA | U_0.05 | MeanTok | Calls | p_holm vs B1b |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    rows_to_render: list[tuple[str, dict[str, Any], str]] = []
    for k in VARIANT_DISPLAY_ORDER:
        if k in systems:
            p_h = f"{systems[k]['p_holm_vs_b1b']:.3g}" if k != "B1b" and "p_holm_vs_b1b" in systems[k] else "-"
            rows_to_render.append((k, systems[k], p_h))
    if "M2_B512" in aux:
        mc_p = aux["M2_B512"].get("mcnemar_vs_b1b", {}).get("p_exact")
        p_s = f"{mc_p:.3g} (raw)" if mc_p is not None else "-"
        rows_to_render.append(("M2_B512", aux["M2_B512"], p_s))
    if isinstance(oracle_sum, dict) and "macro_acc" in oracle_sum:
        rows_to_render.append(("Oracle_UB", oracle_sum, "-"))

    for label, t, p_col in rows_to_render:
        sa = t.get("slice_acc", {})
        slice_cells = " | ".join(_fmt_pct(sa.get(s)) for s in SLICE_ORDER)
        lbl_md = f"**{label}**" if label in ("B1b", "M2", "Oracle_UB") else label
        lines.append(
            f"| {lbl_md} | {_fmt_pct(t.get('macro_acc'))} | {slice_cells} | "
            f"{_fmt_pct(t.get('oaa_250'))} | {_fmt_pct(t.get('auc_oaa_2048'))} | "
            f"{_fmt_pct(t.get('u_005'))} | {t.get('mean_t_tot', 0.0):.0f} | "
            f"{t.get('mean_calls', 1.0):.2f} | {p_col} |"
        )
    return lines


def _render_run_markdown(
    run_id: str,
    status: str,
    loop_name: str,
    ts_utc: str,
    git_rev: str,
    total_sec: float,
    hw: dict[str, Any],
    command_clean: str,
    rel_log: str,
    notes_clean: str,
    training_params: dict[str, Any],
    stage_timings: list[dict[str, Any]],
    smoke_rows: list[dict[str, Any]],
    calibrations: list[dict[str, Any]],
    matrix_loops: list[dict[str, Any]],
    w7_items: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    live_progress: list[dict[str, Any]] | None = None,
) -> str:
    """Renders an upserted Markdown block summarizing params, live progress, timings, calibration, and matrix results."""
    hw_desc = (
        f"{hw['gpu_count']}x {hw['gpu_name']} ({hw['vram_gb']} GB, SM {hw['sm_cap']}, "
        f"CUDA {hw['cuda_version']}, BF16={hw['bf16_native']})"
        if hw["cuda_available"]
        else f"{hw['gpu_name']} (CPU/MPS)"
    )
    kernels = f"fla={hw['fla_available']}, causal_conv1d={hw['causal_conv1d_available']}"
    lines: list[str] = [
        f"<!-- CBARR_RUN_START:{run_id} -->",
        f"### Run `{run_id}` — **{status}** (Updated {ts_utc})",
        f"- **Git Commit**: `{git_rev}` | **Last Loop**: `{loop_name or 'Final'}` | **Wall Time**: `{_fmt_duration(total_sec)}` ({total_sec:.1f}s)",
        f"- **Hardware**: {hw_desc} | PyTorch `{hw['torch']}` | Transformers `{hw['transformers']}` | Kernels: `{kernels}`",
        f"- **Command**: `{command_clean}`",
    ]
    if rel_log:
        lines.append(f"- **Raw Log**: `{rel_log}`")
    if notes_clean:
        lines.append(f"- **Notes**: {notes_clean}")

    if live_progress:
        lines.append("\n#### Live Step Checkpoints (In-Flight Split)")
        for lp in live_progress:
            sl_str = f" | slices: `{lp['slices']}`" if lp.get("slices") else ""
            lines.append(
                f"- **`{lp.get('model', 'unknown')}`** (`{lp.get('diag_pool', '')}`, "
                f"`{lp.get('split', 'eval_450')}` **{lp.get('done', 0)}/{lp.get('total', 0)}**): "
                f"`{lp.get('metrics', '')}`{sl_str}"
            )

    if training_params:
        lines.append("\n#### 1. Training & Experiment Parameters")
        lines.append(f"```json\n{json.dumps(training_params, indent=2, sort_keys=True)}\n```")

    if stage_timings:
        lines.append("\n#### 2. Stage Timings")
        lines.append("| Stage | Status | Wall Time |")
        lines.append("| :--- | :--- | :--- |")
        for st in stage_timings:
            dur = float(st.get("duration_sec", 0.0))
            lines.append(f"| {st['stage']} | {st['status']} | {_fmt_duration(dur)} ({dur:.1f}s) |")

    if smoke_rows:
        lines.append("\n#### 3. Gate-3 Live 12-Item Smoke Check")
        lines.append("| Model | Items | D13 <=250 | No Reminder Leak | SmolLM3 Frozen Date |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for sr in smoke_rows:
            lines.append(
                f"| `{sr.get('model', 'unknown')}` | {sr.get('n_items', 0)} | "
                f"{sr.get('d13_delib_le_250_verified')} | {sr.get('gate3_p_term_and_zero_reminder_leak')} | "
                f"{sr.get('gate3_smollm3_frozen_date')} |"
            )

    if calibrations:
        lines.append("\n#### 4. Calibration & Gate-2 Classifier Results (Per Model)")
        lines.append(
            "| Model | tau_prob | tau_h_high | Dev Esc (R1) | G1 Skip | Intercept | Coef (`H1, Hm, Hmax, I, sym`) | "
            "Train@0.5 (Acc/Prec/Rec/F1/AUC) | Val-All@tau (Acc/Prec/Rec/F1/AUC) | Val-G2@tau (F1/AUC) | Dev `B1b->M2` Macro (Tok) | Time |"
        )
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for cal in calibrations:
            cp = cal["calibrated_params"]
            cm = cal.get("classifier_metrics", {})
            tr = cm.get("seed_train_at_05", {})
            va = cm.get("dev_all_at_tau", {})
            vg = cm.get("dev_gate2_at_tau", {})
            ds = cal.get("dev_summary", {})
            b1_dev = ds.get("B1b", {})
            m2_dev = ds.get("M2", {})
            coef_str = ", ".join(f"{float(c):.2f}" for c in cp.get("coef", []))
            r1_tag = "OK" if cal.get("contingency_r1_ok") else "WARN"
            tr_str = (
                f"{100*tr.get('accuracy',0):.0f}/{100*tr.get('precision',0):.0f}/"
                f"{100*tr.get('recall',0):.0f}/{100*tr.get('f1',0):.0f}/{tr.get('roc_auc',0):.3f}"
                if tr
                else "n/a"
            )
            va_str = (
                f"{100*va.get('accuracy',0):.0f}/{100*va.get('precision',0):.0f}/"
                f"{100*va.get('recall',0):.0f}/{100*va.get('f1',0):.0f}/{va.get('roc_auc',0):.3f}"
                if va
                else "n/a"
            )
            vg_str = (
                f"{100*vg.get('f1',0):.0f}/{vg.get('roc_auc',0):.3f} (n={vg.get('n',0)})"
                if vg and vg.get("n", 0) > 0
                else "n/a"
            )
            dev_cmp = (
                f"{_fmt_pct(b1_dev.get('macro_acc'))}->{_fmt_pct(m2_dev.get('macro_acc'))} "
                f"({b1_dev.get('mean_t_tot',0):.0f}->{m2_dev.get('mean_t_tot',0):.0f}t)"
                if m2_dev and b1_dev
                else "n/a"
            )
            wsec = cal.get("wall_time_sec")
            t_str = _fmt_duration(float(wsec)) if wsec is not None else "n/a"
            lines.append(
                f"| `{cal['model']}` | {cp['tau_prob']:.3f} | {cp['tau_h_high']:.3f} | "
                f"{100*cp['m2_escalation_rate_dev']:.1f}% ({r1_tag}) | "
                f"{_fmt_pct(cal.get('dev_gate1_skip_rate'))} | {cp['intercept']:.2f} | "
                f"[{coef_str}] | {tr_str} | {va_str} | {vg_str} | {dev_cmp} | {t_str} |"
            )

    if matrix_loops:
        lines.append("\n#### 5. Evaluation Matrix & Section 6 Diagnostics (Per Model x Pool)")
        for rep in matrix_loops:
            rt = rep.get("runtime", {})
            tsec = rep.get("timing_sec", {})
            cp = rep.get("calibrated_params", {})
            reg = rep.get("oracle_and_4regime", {})
            s6 = rep.get("section6_diagnostics", {})
            ca = s6.get("causal_attribution", {})
            a1 = s6.get("A1_strip", {})
            b2a = s6.get("B2a_post_anchor", {})
            b1 = s6.get("B1_pre_anchor", {})
            b2b = s6.get("B2b_post_raw", {})
            a2 = s6.get("A2_math_swap", {})
            b3 = s6.get("B3_early_cut_no_anchor", {})
            rep_status = rep.get("status", "completed")
            st_badge = f" | status=`{rep_status}`" if rep_status != "completed" else ""
            tp = rep.get("training_params", {})
            samp_badge = (
                f" | sample(T={float(tp['temperature']):g}, top_p={float(tp.get('top_p', 1.0)):g}, "
                f"pp={float(tp.get('presence_penalty', 0.0)):g}, seed={int(tp.get('seed', 42))})"
                if float(tp.get("temperature", 0.0) or 0.0) > 0.0
                else ""
            )
            lines.append(
                f"\n- **`{rep['model']}` (`{rep.get('diag_pool', 'unknown')}`**, n_diag={rep.get('diag_n_items', 0)}) — "
                f"dtype=`{rt.get('dtype', 'n/a')}`{samp_badge}{st_badge} | wall=`{tsec.get('total_wall', 0):.1f}s` "
                f"(calib={tsec.get('calibration', 0):.1f}s, eval={tsec.get('eval_matrix', 0):.1f}s, diag={tsec.get('section6_diagnostics', 0):.1f}s) | "
                f"tau={cp.get('tau_prob', 0):.3f}, esc={100*cp.get('m2_escalation_rate_dev', 0):.1f}%"
            )
            lines.append(
                f"  - **4-Regime**: Helped={_fmt_pct(reg.get('helped_rate'))}, "
                f"Hurt={_fmt_pct(reg.get('hurt_rate'))}, Flip={_fmt_pct(reg.get('flip_rate'))}, "
                f"TiePass={_fmt_pct(reg.get('tie_pass_rate'))}, TieFail={_fmt_pct(reg.get('tie_fail_rate'))}"
            )
            lines.append(
                f"  - **Section 6 Causal (`|E_hurt|={a1.get('n_ehurt', 0)}`)**: "
                f"rho_rec={_fmt_pct(ca.get('rho_recency'))}, rho_int={_fmt_pct(ca.get('rho_interference'))} | "
                f"A1_strip={a1.get('overall_recovery_pct', 0):.1f}%, "
                f"B2a_post={b2a.get('overall_recovery_pct', 0):.1f}%, "
                f"B1_pre={b1.get('overall_recovery_pct', 0):.1f}%, "
                f"B2b_raw={b2b.get('overall_recovery_pct', 0):.1f}%, "
                f"A2_swap={a2.get('overall_recovery_pct', 0):.1f}%, "
                f"B3_cut={b3.get('overall_recovery_pct', 0):.1f}%"
            )
            lines.extend(_format_variant_table_lines(rep))

    if w7_items:
        lines.append("\n#### 6. Optional W7 Extensions")
        for w7 in w7_items:
            parts: list[str] = []
            bl = w7.get("blend")
            if isinstance(bl, dict):
                if bl.get("mode") == "mock":
                    parts.append(f"DeltaNetBlend(mock keys={list(bl.get('sweep', {}).keys())})")
                elif "alphas" in bl:
                    a_str = ", ".join(
                        f"a={k}:{v.get('overall_recovery_pct', 0):.1f}%"
                        for k, v in bl["alphas"].items()
                    )
                    parts.append(f"DeltaNetBlend(n={bl.get('n_ehurt', 0)}: {a_str})")
            ml = w7.get("mlp")
            if isinstance(ml, dict):
                et = ml.get("eval_table", {})
                parts.append(
                    f"PrefillMLP({ml.get('mode')}, tau={ml.get('best_tau', 0):.3f}, "
                    f"Macro={_fmt_pct(et.get('macro_acc'))}, OAA250={_fmt_pct(et.get('oaa_250'))}, "
                    f"U0.05={_fmt_pct(et.get('u_005'))}, Tok={et.get('mean_t_tot', 0):.0f})"
                )
            if parts:
                lines.append(f"- **`{w7['dir']}`**: " + " | ".join(parts))

    if model_rows:
        lines.append("\n#### 7. Phase 7 Statistical Audit & Gate-4 Summary")
        lines.append(
            "| Model | Clean-446 (max delta) | M2 Macro | B1b Macro | M2 OAA_250 | Delta OAA_250 [95% CI] | "
            "Tok Red | p_holm vs B1b | p(M2 vs B0) | p(M2 vs Pre) | Gate 4 | R2 (rec / int) |"
        )
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for mr in model_rows:
            cl_str = (
                f"{mr['n_clean_446']} ({mr['clean_max_delta_pp']:.1f} pp)"
                if mr.get("n_clean_446") is not None and mr.get("clean_max_delta_pp") is not None
                else "n/a"
            )
            ci_str = (
                f"{mr['oaa_diff_pp']:+.1f} [{mr['oaa_ci95_lo_pp']:.1f}, {mr['oaa_ci95_hi_pp']:.1f}] pp"
                if mr.get("oaa_diff_pp") is not None
                else "n/a"
            )
            tred = f"{mr['tok_red_pct']:.1f}%" if mr["tok_red_pct"] is not None else "n/a"
            ph = f"{mr['p_holm']:.4g}" if mr["p_holm"] is not None else "n/a"
            pb0 = f"{mr['p_m2_vs_b0']:.4g}" if mr.get("p_m2_vs_b0") is not None else "n/a"
            ppre = f"{mr['p_m2_vs_m2pre']:.4g}" if mr.get("p_m2_vs_m2pre") is not None else "n/a"
            r2_str = (
                f"{mr['r2_rec_pct']:.1f}% / {mr['r2_int_pct']:.1f}%"
                if mr.get("r2_rec_pct") is not None and mr.get("r2_int_pct") is not None
                else "n/a"
            )
            lines.append(
                f"| `{mr['model']}` | {cl_str} | {_fmt_pct(mr['m2_macro'])} | {_fmt_pct(mr['b1b_macro'])} | "
                f"{_fmt_pct(mr['m2_oaa250'])} | {ci_str} | {tred} | {ph} | {pb0} | {ppre} | "
                f"{'PASS' if mr['gate4_pass'] else 'CHECK'} | {r2_str} |"
            )

    lines.append(f"<!-- CBARR_RUN_END:{run_id} -->")
    return "\n".join(lines)


def _upsert_run_block(md_path: pathlib.Path, run_id: str, block: str, upsert: bool) -> None:
    """Replaces an existing <!-- CBARR_RUN_START/END --> section in place, or appends a new run block."""
    if not md_path.exists():
        md_path.write_text("# C-BARR Experiment & Runtime Log\n\n---\n\n## Run History\n", encoding="utf-8")
    existing = md_path.read_text(encoding="utf-8")
    start_marker = f"<!-- CBARR_RUN_START:{run_id} -->"
    end_marker = f"<!-- CBARR_RUN_END:{run_id} -->"
    if upsert and start_marker in existing and end_marker in existing:
        pattern = re.compile(
            re.escape(start_marker) + r".*?" + re.escape(end_marker),
            flags=re.DOTALL,
        )
        updated = pattern.sub(lambda _m: block, existing, count=1)
        md_path.write_text(updated if updated.endswith("\n") else updated + "\n", encoding="utf-8")
        return

    sep = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
    md_path.write_text(existing + sep + block + "\n", encoding="utf-8")


def _ensure_git_identity(repo_root: pathlib.Path) -> None:
    for key, fallback in (
        ("user.name", "cbarr-runner"),
        ("user.email", "cbarr-runner@users.noreply.github.com"),
    ):
        chk = subprocess.run(
            ["git", "-C", str(repo_root), "config", "--get", key],
            capture_output=True,
            text=True,
        )
        if chk.returncode != 0 or not chk.stdout.strip():
            subprocess.run(["git", "-C", str(repo_root), "config", key, fallback], check=True)


def _run_git_with_askpass(
    args: list[str], repo_root: pathlib.Path, token: str
) -> subprocess.CompletedProcess[str]:
    """Runs git non-interactively using an ephemeral GIT_ASKPASS script deleted immediately in finally."""
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    askpass_path: pathlib.Path | None = None
    try:
        if token:
            with tempfile.NamedTemporaryFile(
                "w", suffix="_cbarr_askpass.sh", delete=False, encoding="utf-8"
            ) as tf:
                tf.write(
                    "#!/bin/sh\n"
                    'case "$1" in\n'
                    '  *Username*|*username*) printf "%s\\n" "x-access-token" ;;\n'
                    '  *) printf "%s\\n" "$CBARR_GIT_ASKPASS_TOKEN" ;;\n'
                    "esac\n"
                )
                askpass_path = pathlib.Path(tf.name)
            os.chmod(askpass_path, stat.S_IRWXU)
            env["GIT_ASKPASS"] = str(askpass_path)
            env["CBARR_GIT_ASKPASS_TOKEN"] = token
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            env=env,
        )
    finally:
        if askpass_path is not None and askpass_path.exists():
            try:
                askpass_path.unlink()
            except OSError:
                pass


def _git_commit_and_push_ledger(
    repo_root: pathlib.Path,
    md_path: pathlib.Path,
    run_id: str,
    loop_name: str,
    remote: str,
    branch: str,
    token_env: str,
) -> tuple[bool, str]:
    """Commits and pushes only the ledger Markdown file to GitHub without persisting tokens."""
    if not (repo_root / ".git").exists():
        return False, "not a git repository"

    try:
        rel_md = str(md_path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return False, f"ledger path {md_path} is outside repo root {repo_root}"

    token = os.environ.get(token_env, "").strip()
    remote_proc = subprocess.run(
        ["git", "-C", str(repo_root), "remote", "get-url", remote],
        capture_output=True,
        text=True,
    )
    if remote_proc.returncode != 0:
        return False, f"git remote '{remote}' not configured"
    remote_url = remote_proc.stdout.strip()
    if remote_url.startswith("https://") and not token:
        return False, f"HTTPS remote '{remote}' requires env var {token_env} to push"

    target_branch = branch.strip()
    if not target_branch:
        br_proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
        )
        target_branch = br_proc.stdout.strip() if br_proc.returncode == 0 else "master"
    if not target_branch or target_branch == "HEAD":
        target_branch = "master"

    _ensure_git_identity(repo_root)
    subprocess.run(["git", "-C", str(repo_root), "add", "--", rel_md], check=True)
    diff_chk = subprocess.run(["git", "-C", str(repo_root), "diff", "--cached", "--quiet"])
    if diff_chk.returncode != 0:
        loop_lbl = f" | {loop_name}" if loop_name else ""
        msg = f"chore(results): update {pathlib.Path(rel_md).name} [{run_id}{loop_lbl}]"
        cm_proc = subprocess.run(
            ["git", "-C", str(repo_root), "commit", "-m", msg, "--", rel_md],
            capture_output=True,
            text=True,
        )
        if cm_proc.returncode != 0:
            err = _sanitize_text(cm_proc.stderr or cm_proc.stdout, repo_root, token)
            return False, f"git commit failed: {err.strip()}"

    last_err = ""
    for attempt in range(2):
        if attempt > 0:
            pull_proc = _run_git_with_askpass(
                ["pull", "--rebase", "--autostash", remote, target_branch],
                repo_root,
                token,
            )
            if pull_proc.returncode != 0:
                subprocess.run(["git", "-C", str(repo_root), "rebase", "--abort"], check=False)
                err = _sanitize_text(pull_proc.stderr or pull_proc.stdout, repo_root, token)
                return False, f"git pull --rebase failed: {err.strip()}"
        push_proc = _run_git_with_askpass(
            ["push", remote, f"HEAD:{target_branch}"], repo_root, token
        )
        if push_proc.returncode == 0:
            return True, f"pushed {rel_md} -> {remote}/{target_branch}"
        last_err = _sanitize_text(push_proc.stderr or push_proc.stdout, repo_root, token)

    return False, f"git push failed: {last_err.strip()}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Record C-BARR training params and per-loop results.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--status", required=True, help="e.g., RUNNING, PASSED, or FAILED")
    parser.add_argument("--command", required=True)
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--timings-file", type=str, default="")
    parser.add_argument("--log-file", type=str, default="")
    parser.add_argument("--md-log", type=str, default="", help="Markdown ledger path to update.")
    parser.add_argument("--total-sec", type=float, default=0.0)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--loop-name", type=str, default="", help="Completed loop/stage label.")
    parser.add_argument("--params-json", type=str, default="", help="JSON string or @path of training params.")
    parser.add_argument("--smoke-report", type=str, default="", help="Optional smoke_test_gate3_report.json path.")
    parser.add_argument("--upsert", dest="upsert", action="store_true", default=True)
    parser.add_argument("--no-upsert", dest="upsert", action="store_false")
    parser.add_argument("--git-push", action="store_true", help="Commit & push --md-log to GitHub.")
    parser.add_argument("--git-remote", type=str, default="origin")
    parser.add_argument("--git-branch", type=str, default="")
    parser.add_argument("--git-token-env", type=str, default="GITHUB_TOKEN")
    parser.add_argument("--strict-push", action="store_true", help="Exit non-zero if --git-push fails.")
    args = parser.parse_args()

    root = ROOT_DIR
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings_path = pathlib.Path(args.timings_file) if args.timings_file else None
    stage_timings = _load_timings(timings_path)
    if args.total_sec <= 0.0 and stage_timings:
        total_sec = sum(float(s.get("duration_sec", 0.0)) for s in stage_timings)
    else:
        total_sec = float(args.total_sec)

    token_val = os.environ.get(args.git_token_env, "").strip()
    cli_params = _parse_params_json(args.params_json) if args.params_json else {}
    hw = _detect_hardware()
    git_rev = _detect_git_rev(root)
    ts_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    smoke_rows = _extract_smoke_report(root, out_dir, args.smoke_report)
    calibrations = _extract_calibrations(out_dir)
    matrix_loops = _extract_matrix_loops(out_dir)
    w7_items = _extract_w7_extensions(out_dir)
    model_rows = _extract_model_summaries(out_dir)
    live_progress = _extract_live_progress(out_dir)
    training_params = _collect_training_params(cli_params, calibrations, matrix_loops)

    rel_log = _sanitize_text(args.log_file, root, token_val) if args.log_file else ""
    cmd_clean = _sanitize_text(args.command, root, token_val)
    notes_clean = _sanitize_text(args.notes, root, token_val)

    record = {
        "run_id": args.run_id,
        "loop_name": args.loop_name,
        "timestamp_utc": ts_utc,
        "status": args.status,
        "git_rev": git_rev,
        "command": cmd_clean,
        "total_sec": round(total_sec, 2),
        "log_file": rel_log,
        "training_params": training_params,
        "hardware": hw,
        "stages": stage_timings,
        "smoke": smoke_rows,
        "calibrations": [
            {
                "model": c["model"],
                "calibrated_params": c["calibrated_params"],
                "classifier_metrics": c.get("classifier_metrics", {}),
                "wall_time_sec": c.get("wall_time_sec"),
            }
            for c in calibrations
        ],
        "matrix_loops": [
            {
                "model": r["model"],
                "diag_pool": r.get("diag_pool"),
                "timing_sec": r.get("timing_sec"),
                "m2_macro": r.get("systems", {}).get("M2", {}).get("macro_acc"),
                "m2_oaa250": r.get("systems", {}).get("M2", {}).get("oaa_250"),
                "b1b_macro": r.get("systems", {}).get("B1b", {}).get("macro_acc"),
            }
            for r in matrix_loops
        ],
        "models": model_rows,
        "notes": notes_clean,
    }
    history_path = out_dir / "experiment_history.jsonl"
    with history_path.open("a", encoding="utf-8") as f_hist:
        f_hist.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not args.md_log:
        print(hl(f"Recorded experiment telemetry to {history_path}", "cyan"), flush=True)
        return

    md_path = pathlib.Path(args.md_log)
    block = _render_run_markdown(
        run_id=args.run_id,
        status=args.status,
        loop_name=args.loop_name,
        ts_utc=ts_utc,
        git_rev=git_rev,
        total_sec=total_sec,
        hw=hw,
        command_clean=cmd_clean,
        rel_log=rel_log,
        notes_clean=notes_clean,
        training_params=training_params,
        stage_timings=stage_timings,
        smoke_rows=smoke_rows,
        calibrations=calibrations,
        matrix_loops=matrix_loops,
        w7_items=w7_items,
        model_rows=model_rows,
        live_progress=live_progress,
    )
    _upsert_run_block(md_path, args.run_id, block, upsert=args.upsert)
    loop_msg = f" [{args.loop_name}]" if args.loop_name else ""
    print(
        hl(f"[log_experiment] Updated `{args.run_id}`{loop_msg} -> {history_path} & {md_path}", "green"),
        flush=True,
    )

    if args.git_push:
        ok, push_msg = _git_commit_and_push_ledger(
            repo_root=root,
            md_path=md_path,
            run_id=args.run_id,
            loop_name=args.loop_name,
            remote=args.git_remote,
            branch=args.git_branch,
            token_env=args.git_token_env,
        )
        if ok:
            print(hl(f"[git-push] {push_msg}", "green"), flush=True)
        else:
            print(hl(f"[WARN] [git-push] {push_msg}", "yellow"), flush=True)
            if args.strict_push:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
