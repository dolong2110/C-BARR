"""Live free-VRAM GPU slot allocator for shared multi-user CUDA servers.

Queries physical GPU free memory via `nvidia-smi` (independent of parent CUDA_VISIBLE_DEVICES) right
before each pipeline stage, skips GPUs occupied by other users, assigns 1-3 worker slots per GPU
strictly within actual free VRAM, and optionally waits/polls if all GPUs are temporarily busy.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Any


def model_vram_per_worker_gb(model_id: str) -> float:
    """Conservative free-VRAM budget (GiB) required per concurrent worker for a given model."""
    low = model_id.lower()
    if "9b" in low:
        return 23.0
    if "4b" in low or "3b" in low:
        return 12.5
    if "2b" in low:
        return 8.5
    if "0.8b" in low or "08b" in low or "1b" in low:
        return 5.5
    return 12.5


def query_physical_gpus(candidate_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Returns [{'id': str, 'free_gb': float, 'total_gb': float}, ...] across physical GPUs."""
    cand_set = {str(c).strip() for c in candidate_ids if str(c).strip()} if candidate_ids else None
    gpus: list[dict[str, Any]] = []
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total",
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
                gpus.append({"id": gid, "free_gb": free_gb, "total_gb": total_gb})
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
) -> list[str]:
    """Assigns per-worker GPU IDs strictly bounded by each GPU's current free VRAM."""
    if not gpus:
        return []
    req_gb = model_vram_per_worker_gb(model_id)
    is_9b = "9b" in model_id.lower()

    # Prefer GPUs with the most free VRAM first so partially loaded GPUs are used last
    eligible = [g for g in gpus if float(g["free_gb"]) >= req_gb]
    eligible.sort(key=lambda g: (-float(g["free_gb"]), int(str(g["id"])) if str(g["id"]).isdigit() else 999))
    eligible = eligible[: max(1, int(max_gpus))]
    if not eligible:
        return []

    per_gpu_counts: list[tuple[str, int]] = []
    for g in eligible:
        free_gb = float(g["free_gb"])
        total_gb = float(g["total_gb"])
        max_fit = max(1, int(free_gb // req_gb))
        if workers_per_gpu != "auto":
            target_w = max(1, int(workers_per_gpu))
        elif is_9b:
            target_w = 1
        elif total_gb >= 64.0 and free_gb >= 3.0 * req_gb + 4.0:
            target_w = 3
        elif total_gb >= 32.0 and free_gb >= 2.0 * req_gb + 1.0:
            target_w = 2
        else:
            target_w = 1
        w_cnt = min(max_fit, target_w)
        per_gpu_counts.append((str(g["id"]), w_cnt))

    # Interleave across distinct GPUs (e.g. ['3', '4', '3', '4']) so shard 0..K spread across cards first
    slots: list[str] = []
    max_w = max(cnt for _, cnt in per_gpu_counts)
    for round_idx in range(max_w):
        for gid, cnt in per_gpu_counts:
            if round_idx < cnt:
                slots.append(gid)
    return slots


def select_worker_gpu_slots(
    model_id: str,
    candidate_ids: list[str] | None = None,
    workers_per_gpu: str = "auto",
    max_gpus: int = 8,
    wait_timeout_sec: int = 0,
    poll_sec: int = 30,
) -> list[str]:
    """Queries live free VRAM and returns worker GPU IDs, optionally waiting if all GPUs are busy."""
    deadline = time.monotonic() + max(0, int(wait_timeout_sec))
    req_gb = model_vram_per_worker_gb(model_id)
    attempt = 0
    while True:
        gpus = query_physical_gpus(candidate_ids=candidate_ids)
        if not gpus:
            return []
        slots = compute_slots_from_free_mem(
            gpus,
            model_id=model_id,
            workers_per_gpu=workers_per_gpu,
            max_gpus=max_gpus,
        )
        if slots:
            summary = ", ".join(f"gpu{g['id']}:{g['free_gb']:.1f}G" for g in gpus)
            print(
                f"[gpu_pool] Selected {len(slots)} worker slot(s) {slots} for {model_id} "
                f"(need>={req_gb:.1f}G/worker | live free: {summary})",
                file=sys.stderr,
                flush=True,
            )
            return slots
        now = time.monotonic()
        if now >= deadline:
            summary = ", ".join(f"gpu{g['id']}:{g['free_gb']:.1f}G" for g in gpus)
            print(
                f"[gpu_pool] ERROR: 0 GPUs have >= {req_gb:.1f} GB free for {model_id} ({summary})",
                file=sys.stderr,
                flush=True,
            )
            return []
        attempt += 1
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
    parser = argparse.ArgumentParser(description="Select live free-VRAM CUDA GPU slots.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--candidates", type=str, default="", help="Optional comma-separated GPU IDs")
    parser.add_argument("--workers-per-gpu", type=str, default="auto")
    parser.add_argument("--max-gpus", type=int, default=8)
    parser.add_argument("--wait-sec", type=int, default=0)
    parser.add_argument("--poll-sec", type=int, default=30)
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
    )
    if not slots:
        sys.exit(1)
    if args.single:
        print(slots[0])
    else:
        print(",".join(slots))


if __name__ == "__main__":
    main()
