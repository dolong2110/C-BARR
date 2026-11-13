"""Records structured hardware specs, phase wall-clock timings, and C-BARR results into EXPERIMENT_LOG.md."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys
from typing import Any


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


def _detect_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": "unavailable",
        "transformers": "unavailable",
        "cuda_available": False,
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
        if info["cuda_available"]:
            info["gpu_count"] = int(torch.cuda.device_count())
            props = torch.cuda.get_device_properties(0)
            sm = torch.cuda.get_device_capability(0)
            info["gpu_name"] = props.name
            info["vram_gb"] = round(props.total_memory / (1024**3), 1)
            info["sm_cap"] = f"{sm[0]}.{sm[1]}"
            info["bf16_native"] = bool(torch.cuda.is_bf16_supported() and sm[0] >= 8)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["gpu_name"] = "Apple MPS"
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
        p = model_dir / fname
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")).get("systems", {})
    return {}


def _extract_model_summaries(out_dir: pathlib.Path) -> list[dict[str, Any]]:
    stats_path = out_dir / "stats_summary.json"
    if not stats_path.exists():
        return []
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for model_name, m_data in stats.get("models", {}).items():
        g4 = m_data.get("gate_audit", {}).get("gate4_primary", {})
        cont = m_data.get("gate_audit", {}).get("contingencies", {})
        boot = m_data.get("paired_bootstrap_95ci", {}).get("M2_minus_B1b", {})
        sys_tbl = _load_model_systems(out_dir, model_name)
        m2 = sys_tbl.get("M2", {})
        b1b = sys_tbl.get("B1b", {})
        rows.append(
            {
                "model": model_name,
                "m2_macro": m2.get("macro_acc"),
                "b1b_macro": b1b.get("macro_acc"),
                "m2_oaa250": m2.get("oaa_250"),
                "b1b_oaa250": b1b.get("oaa_250"),
                "tok_red_pct": g4.get("token_reduction_pct"),
                "p_holm": g4.get("m2_p_holm_vs_b1b"),
                "gate4_pass": bool(g4.get("m2_macro_ge_b1b") and g4.get("token_reduction_pct", 0) > 0),
                "r2_rec_pct": (100.0 * cont["R2_rho_recency"]) if "R2_rho_recency" in cont else None,
                "oaa_diff_pp": (
                    100.0 * boot["oaa_250"]["diff_b_minus_a"]
                    if "oaa_250" in boot
                    else None
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Append timed experiment run record to repo logs.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--status", required=True, help="e.g., PASSED or FAILED")
    parser.add_argument("--command", required=True)
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--timings-file", type=str, default="")
    parser.add_argument("--log-file", type=str, default="")
    parser.add_argument("--md-log", type=str, default="", help="Markdown ledger path to append to.")
    parser.add_argument("--total-sec", type=float, default=0.0)
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings_path = pathlib.Path(args.timings_file) if args.timings_file else None
    stage_timings = _load_timings(timings_path)
    if args.total_sec <= 0.0 and stage_timings:
        total_sec = sum(float(s.get("duration_sec", 0.0)) for s in stage_timings)
    else:
        total_sec = float(args.total_sec)

    hw = _detect_hardware()
    git_rev = _detect_git_rev(root)
    ts_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    model_rows = _extract_model_summaries(out_dir)

    record = {
        "run_id": args.run_id,
        "timestamp_utc": ts_utc,
        "status": args.status,
        "git_rev": git_rev,
        "command": args.command,
        "total_sec": round(total_sec, 2),
        "log_file": args.log_file,
        "hardware": hw,
        "stages": stage_timings,
        "models": model_rows,
        "notes": args.notes,
    }
    history_path = out_dir / "experiment_history.jsonl"
    with history_path.open("a", encoding="utf-8") as f_hist:
        f_hist.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not args.md_log:
        print(f"Recorded experiment telemetry to {history_path}")
        return

    md_path = pathlib.Path(args.md_log)
    if not md_path.exists():
        md_path.write_text("# C-BARR Experiment & Runtime Log\n\n", encoding="utf-8")

    # Keep paths workspace-relative so double-blind anonymity checks never see local usernames
    rel_log = args.log_file
    try:
        rel_log = str(pathlib.Path(args.log_file).resolve().relative_to(root.resolve()))
    except Exception:
        pass

    hw_desc = (
        f"{hw['gpu_count']}x {hw['gpu_name']} ({hw['vram_gb']} GB, SM {hw['sm_cap']}, "
        f"CUDA {hw['cuda_version']}, BF16={hw['bf16_native']})"
        if hw["cuda_available"]
        else f"{hw['gpu_name']} (CPU/MPS)"
    )
    kernels = f"fla={hw['fla_available']}, causal_conv1d={hw['causal_conv1d_available']}"

    lines = [
        f"\n### Run `{args.run_id}` — **{args.status}** ({ts_utc})",
        f"- **Git Commit**: `{git_rev}` | **Total Wall Time**: `{_fmt_duration(total_sec)}` ({total_sec:.1f}s)",
        f"- **Hardware**: {hw_desc} | PyTorch `{hw['torch']}` | Transformers `{hw['transformers']}` | Kernels: `{kernels}`",
        f"- **Command**: `{args.command}`",
    ]
    if rel_log:
        lines.append(f"- **Raw Log**: `{rel_log}`")
    if args.notes:
        lines.append(f"- **Notes**: {args.notes}")

    if stage_timings:
        lines.append("\n| Stage | Status | Wall Time |")
        lines.append("| :--- | :--- | :--- |")
        for st in stage_timings:
            lines.append(
                f"| {st['stage']} | {st['status']} | {_fmt_duration(st['duration_sec'])} ({st['duration_sec']:.1f}s) |"
            )

    if model_rows:
        lines.append("\n| Model | M2 Macro | B1b Macro | M2 OAA_250 | Tok Red | p_holm | Gate 4 / R2 Recency |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for mr in model_rows:
            m2_m = f"{100.0 * mr['m2_macro']:.1f}%" if mr["m2_macro"] is not None else "n/a"
            b1_m = f"{100.0 * mr['b1b_macro']:.1f}%" if mr["b1b_macro"] is not None else "n/a"
            oaa = f"{100.0 * mr['m2_oaa250']:.1f}%" if mr["m2_oaa250"] is not None else "n/a"
            tred = f"{mr['tok_red_pct']:.1f}%" if mr["tok_red_pct"] is not None else "n/a"
            ph = f"{mr['p_holm']:.4g}" if mr["p_holm"] is not None else "n/a"
            r2 = f"{mr['r2_rec_pct']:.1f}%" if mr.get("r2_rec_pct") is not None else "n/a"
            lines.append(
                f"| `{mr['model']}` | {m2_m} | {b1_m} | {oaa} | {tred} | {ph} | "
                f"{'PASS' if mr['gate4_pass'] else 'CHECK'} / {r2} |"
            )

    with md_path.open("a", encoding="utf-8") as f_md:
        f_md.write("\n".join(lines) + "\n")
    print(f"Recorded experiment entry `{args.run_id}` to {history_path} and {md_path}")


if __name__ == "__main__":
    main()
