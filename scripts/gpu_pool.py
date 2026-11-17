"""Live free-VRAM, SM-utilization, and CPU-aware GPU slot allocator for shared CUDA servers.

Queries physical GPU free memory and compute utilization via `nvidia-smi` (independent of parent
CUDA_VISIBLE_DEVICES) right before each pipeline stage, skips occupied GPUs, sizes concurrent
workers per GPU across 4 bounds (safe VRAM headroom, model HBM-bandwidth saturation, live external
GPU utilization, and host CPU budget), and optionally waits/polls if all GPUs are temporarily busy.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from typing import Any

_PARAM_BILLIONS_RE = re.compile(r"(?:^|[^0-9.])(\d+(?:\.\d+)?)\s*[bB](?:$|[^a-zA-Z0-9])")


def _estimate_model_billions(model_id: str) -> float:
    """Extracts approximate parameter count (in billions) from a Hugging Face model ID."""
    low = str(model_id or "").lower()
    if "0.8b" in low or "08b" in low:
        return 0.8
    matches = _PARAM_BILLIONS_RE.findall(low)
    if matches:
        try:
            return max(0.1, float(matches[-1]))
        except ValueError:
            pass
    return 4.0


def model_vram_per_worker_gb(model_id: str, stage: str = "matrix") -> float:
    """Conservative peak free-VRAM budget (GiB) required per concurrent worker."""
    params_b = _estimate_model_billions(model_id)
    if params_b >= 11.0:
        base = round(2.3 * params_b + 3.0, 1)
    elif params_b >= 7.0:
        base = 23.0
    elif params_b >= 2.8:
        base = 12.5
    elif params_b >= 1.5:
        base = 8.5
    else:
        base = 5.5
    if stage == "blend":
        # W7 DeltaNet blend retains both prompt and thinking DynamicCache states transiently.
        return round(base + max(1.5, min(5.0, 0.5 * params_b)), 1)
    return base


def _model_bandwidth_cap(params_b: float, total_gb: float, stage: str = "matrix") -> int:
    """Caps workers/GPU where batch_size=1 decoding saturates HBM bandwidth & SM warps."""
    if params_b >= 7.0:
        w = 2 if (total_gb >= 64.0 and stage != "blend") else 1
    elif params_b >= 2.8:
        w = 3 if total_gb >= 64.0 else (2 if total_gb >= 32.0 else 1)
    elif params_b >= 1.5:
        w = 4 if total_gb >= 64.0 else (3 if total_gb >= 32.0 else (2 if total_gb >= 16.0 else 1))
    else:
        w = 4 if total_gb >= 32.0 else (2 if total_gb >= 14.0 else 1)
    if stage == "blend":
        return min(w, 1 if params_b >= 7.0 else 2)
    return w


def query_physical_gpus(candidate_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Returns [{'id': str, 'free_gb': float, 'total_gb': float, 'util_pct': float|None}, ...]."""
    cand_set = {str(c).strip() for c in candidate_ids if str(c).strip()} if candidate_ids else None
    gpus: list[dict[str, Any]] = []
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            for raw in proc.stdout.strip().splitlines():
                parts = [p.strip() for p in raw.split(",")]
                if len(parts) < 3:
                    continue
                gid = parts[0]
                if cand_set is not None and gid not in cand_set:
                    continue
                free_gb = float(parts[1]) / 1024.0
                total_gb = float(parts[2]) / 1024.0
                util_pct: float | None = None
                if len(parts) >= 4:
                    try:
                        util_pct = float(parts[3])
                    except ValueError:
                        util_pct = None
                gpus.append(
                    {
                        "id": gid,
                        "free_gb": free_gb,
                        "total_gb": total_gb,
                        "util_pct": util_pct,
                    }
                )
            if gpus:
                return gpus
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            vis_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            vis_ids = [x.strip() for x in vis_env.split(",") if x.strip()] if vis_env else []
            for idx in range(torch.cuda.device_count()):
                gid = vis_ids[idx] if idx < len(vis_ids) else str(idx)
                if cand_set is not None and gid not in cand_set:
                    continue
                free_b, total_b = torch.cuda.mem_get_info(idx)
                gpus.append(
                    {
                        "id": gid,
                        "free_gb": float(free_b) / (1024.0**3),
                        "total_gb": float(total_b) / (1024.0**3),
                        "util_pct": None,
                    }
                )
    except Exception:
        pass
    return gpus


def compute_slots_from_free_mem(
    gpus: list[dict[str, Any]],
    model_id: str,
    workers_per_gpu: str = "auto",
    max_gpus: int = 8,
    *,
    stage: str = "matrix",
    cpu_count: int | None = None,
    cpus_per_worker: int = 2,
    reserve_gb: float = 2.0,
) -> list[str]:
    """Assigns per-worker GPU IDs across VRAM, bandwidth, utilization, and CPU bounds."""
    if not gpus:
        return []
    req_gb = model_vram_per_worker_gb(model_id, stage=stage)
    params_b = _estimate_model_billions(model_id)

    # Prefer GPUs with the most free VRAM and lowest external utilization first.
    eligible = [g for g in gpus if float(g["free_gb"]) >= req_gb]
    eligible.sort(
        key=lambda g: (
            -float(g["free_gb"]),
            float(g["util_pct"]) if g.get("util_pct") is not None else 0.0,
            int(str(g["id"])) if str(g["id"]).isdigit() else 999,
        )
    )
    eligible = eligible[: max(1, int(max_gpus))]
    if not eligible:
        return []

    per_gpu_counts: list[list[Any]] = []
    for g in eligible:
        free_gb = float(g["free_gb"])
        total_gb = float(g["total_gb"])
        eff_reserve = reserve_gb if (total_gb >= 16.0 and free_gb >= req_gb + reserve_gb) else 0.0
        max_fit = max(1, int((free_gb - eff_reserve) // req_gb))
        if workers_per_gpu != "auto":
            target_w = max(1, int(workers_per_gpu))
            w_cnt = min(max_fit, target_w)
        else:
            w_bw = _model_bandwidth_cap(params_b, total_gb, stage=stage)
            util_raw = g.get("util_pct")
            w_util = w_bw
            if util_raw is not None:
                u_val = float(util_raw)
                if u_val >= 60.0:
                    w_util = 1
                elif u_val >= 30.0:
                    w_util = min(w_bw, 2)
            w_cnt = min(max_fit, w_bw, w_util)
        per_gpu_counts.append([str(g["id"]), max(1, w_cnt)])

    # Clamp total workers cleanly to host CPU budget when auto-sizing.
    if workers_per_gpu == "auto" and cpu_count is not None and cpu_count > 0:
        max_total_cpu = max(len(per_gpu_counts), int(cpu_count) // max(1, int(cpus_per_worker)))
        while sum(item[1] for item in per_gpu_counts) > max_total_cpu:
            # Trim from the highest-worker card (and higher GPU ID on ties) down to min 1/GPU.
            idx_trim = max(
                range(len(per_gpu_counts)),
                key=lambda i: (per_gpu_counts[i][1], i),
            )
            if per_gpu_counts[idx_trim][1] <= 1:
                break
            per_gpu_counts[idx_trim][1] -= 1

    # Interleave across distinct GPUs (e.g. ['3', '4', '3', '4']) so shard 0..K spread across cards first.
    slots: list[str] = []
    max_w = max(cnt for _, cnt in per_gpu_counts)
    for round_idx in range(max_w):
        for gid, cnt in per_gpu_counts:
            if round_idx < cnt:
                slots.append(gid)
    return slots


def _format_gpu_live_summary(gpus: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for g in gpus:
        u = g.get("util_pct")
        u_str = f"@{float(u):.0f}%" if u is not None else ""
        parts.append(f"gpu{g['id']}:{g['free_gb']:.1f}G{u_str}")
    return ", ".join(parts)


def select_worker_gpu_slots(
    model_id: str,
    candidate_ids: list[str] | None = None,
    workers_per_gpu: str = "auto",
    max_gpus: int = 8,
    wait_timeout_sec: int = 0,
    poll_sec: int = 30,
    stage: str = "matrix",
) -> list[str]:
    """Queries live free VRAM/util/CPUs and returns worker GPU IDs, optionally waiting if busy."""
    deadline = time.monotonic() + max(0, int(wait_timeout_sec))
    req_gb = model_vram_per_worker_gb(model_id, stage=stage)
    cpus = os.cpu_count()
    while True:
        gpus = query_physical_gpus(candidate_ids=candidate_ids)
        if not gpus:
            return []
        slots = compute_slots_from_free_mem(
            gpus,
            model_id=model_id,
            workers_per_gpu=workers_per_gpu,
            max_gpus=max_gpus,
            stage=stage,
            cpu_count=cpus,
        )
        if slots:
            summary = _format_gpu_live_summary(gpus)
            print(
                f"[gpu_pool] Selected {len(slots)} worker slot(s) {slots} for {model_id} "
                f"(stage={stage}, mode={workers_per_gpu}, need>={req_gb:.1f}G/worker, cpus={cpus or '?'} | live: {summary})",
                file=sys.stderr,
                flush=True,
            )
            return slots
        now = time.monotonic()
        if now >= deadline:
            summary = _format_gpu_live_summary(gpus)
            print(
                f"[gpu_pool] ERROR: 0 GPUs have >= {req_gb:.1f} GB free for {model_id} ({summary})",
                file=sys.stderr,
                flush=True,
            )
            return []
        max_free = max(float(g["free_gb"]) for g in gpus)
        rem_m = int(max(1.0, (deadline - now) / 60.0))
        print(
            f"[gpu_pool] All candidate GPUs busy (max free={max_free:.1f}G < {req_gb:.1f}G for {model_id}); "
            f"waiting {poll_sec}s for free VRAM (up to {rem_m}m remaining)...",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(max(1, int(poll_sec)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Select live free-VRAM/utilization CUDA GPU slots.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--candidates", type=str, default="", help="Optional comma-separated GPU IDs")
    parser.add_argument("--workers-per-gpu", type=str, default="auto")
    parser.add_argument("--max-gpus", type=int, default=8)
    parser.add_argument("--wait-sec", type=int, default=0)
    parser.add_argument("--poll-sec", type=int, default=30)
    parser.add_argument(
        "--stage",
        type=str,
        default="matrix",
        choices=("matrix", "blend", "mlp", "smoke"),
        help="Workload memory profile (matrix, blend, mlp, smoke).",
    )
    parser.add_argument("--single", action="store_true", help="Output only the single freest GPU ID")
    args = parser.parse_args()

    cands = [x.strip() for x in args.candidates.split(",") if x.strip()] if args.candidates else None
    gpus_now = query_physical_gpus(candidate_ids=cands)
    if not gpus_now:
        print("NO_CUDA")
        return

    slots = select_worker_gpu_slots(
        args.model,
        candidate_ids=cands,
        workers_per_gpu="1" if args.single else args.workers_per_gpu,
        max_gpus=1 if args.single else args.max_gpus,
        wait_timeout_sec=args.wait_sec,
        poll_sec=args.poll_sec,
        stage=args.stage,
    )
    if not slots:
        sys.exit(1)
    if args.single:
        print(slots[0])
    else:
        print(",".join(slots))


if __name__ == "__main__":
    main()
