"""Live free-VRAM, SM-utilization, and CPU/RAM-aware GPU slot allocator for shared CUDA servers.

Queries physical GPU free memory and compute utilization via `nvidia-smi` (independent of parent
CUDA_VISIBLE_DEVICES) plus available host CPU RAM before each pipeline stage, skips occupied GPUs,
sizes concurrent workers per GPU across stage-specific VRAM/context headroom, model HBM-bandwidth
saturation, live SM utilization, and host CPU/RAM bounds, and polls if GPUs are temporarily busy.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any

_PARAM_BILLIONS_RE = re.compile(r"(?:^|[^0-9.])(\d+(?:\.\d+)?)\s*[bB](?:$|[^a-zA-Z0-9])")
STAGE_CHOICES: tuple[str, ...] = (
    "matrix",
    "calibrate",
    "diag",
    "b512",
    "blend",
    "mlp",
    "smoke",
    "replay",
)


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


def model_vram_per_worker_gb(
    model_id: str, stage: str = "matrix", safety_mult: float = 1.0
) -> float:
    """Conservative peak free-VRAM budget (GiB) required per concurrent worker by stage."""
    params_b = _estimate_model_billions(model_id)
    if params_b >= 11.0:
        base = round(2.3 * params_b + 3.0, 1)
    elif params_b >= 7.0:
        base = 24.0
    elif params_b >= 3.5:
        base = 12.5
    elif params_b >= 2.8:
        base = 11.5
    elif params_b >= 1.5:
        base = 8.5
    else:
        base = 5.5

    st = str(stage or "matrix").lower()
    if st == "blend":
        # W7 DeltaNet blend retains both prompt and thinking DynamicCache states + float32 interp transiently.
        req = round(base + max(2.0, min(5.5, 0.6 * params_b)), 1)
    elif st == "mlp":
        # Single-layer forward hook, use_cache=False, logits_to_keep=1, prompt-only (<512 tokens).
        req = round(max(2.5, 0.78 * base), 1)
    elif st == "replay":
        # Stage-2 solve_first continuation only (<=1024 out tokens, no 2048-tok B1b or probe entropies).
        req = round(max(3.5, 0.88 * base), 1)
    elif st == "b512":
        # Contingency R3 M2_B512 backfill (<=512 think + 1024 answer, no 2048-tok B1b or Section 6).
        req = round(max(3.5, 0.90 * base), 1)
    elif st == "calibrate":
        # Phase 4 seed_60 + dev_60 calibration only (<=2048 tokens, no Section 6 ~2500-tok prefills).
        req = round(max(3.5, 0.95 * base), 1)
    elif st == "smoke":
        # Sequential multi-model Gate-3 smoke test in a single process requires extra allocator headroom.
        req = round(base + 1.0, 1)
    else:
        # matrix & diag execute full <=2048-tok B1b plus ~2500-tok Section 6 counterfactual prefills.
        req = base

    mult = max(1.0, float(safety_mult))
    return round(req * mult, 1)


def host_ram_per_worker_gb(model_id: str, stage: str = "matrix") -> float:
    """Conservative host CPU RAM budget (GiB) per concurrent worker including BF16 safetensors staging."""
    params_b = _estimate_model_billions(model_id)
    st = str(stage or "matrix").lower()
    weight_mult = 1.5 if st == "mlp" else 2.2
    return round(max(3.0, weight_mult * params_b + 2.5), 1)


def _extra_worker_vram_gb(params_b: float, stage: str) -> float:
    """Additional VRAM headroom per extra worker on the same GPU for separate CUDA/Triton contexts."""
    if str(stage or "").lower() == "blend":
        return 1.0
    return 0.5 if params_b < 1.5 else 0.75


def _model_bandwidth_cap(params_b: float, total_gb: float, stage: str = "matrix") -> int:
    """Caps workers/GPU where batch_size=1 decoding saturates HBM bandwidth & SM warps."""
    st = str(stage or "matrix").lower()
    if params_b >= 7.0:
        w = 2 if (total_gb >= 64.0 and st != "blend") else 1
    elif params_b >= 2.8:
        w = 3 if total_gb >= 64.0 else (2 if total_gb >= 32.0 else 1)
    elif params_b >= 1.5:
        w = 4 if total_gb >= 64.0 else (3 if total_gb >= 32.0 else (2 if total_gb >= 16.0 else 1))
    else:
        w = 4 if total_gb >= 32.0 else (2 if total_gb >= 14.0 else 1)
    if st == "blend":
        return min(w, 1 if params_b >= 7.0 else 2)
    return w


def query_host_avail_ram_gb() -> float | None:
    """Returns currently available host CPU RAM in GiB (Linux /proc/meminfo or macOS vm_stat)."""
    meminfo = pathlib.Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            avail_kb = None
            free_kb = 0
            cached_kb = 0
            for raw in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if raw.startswith("MemAvailable:"):
                    avail_kb = int(raw.split()[1])
                    break
                if raw.startswith("MemFree:"):
                    free_kb = int(raw.split()[1])
                elif raw.startswith("Cached:"):
                    cached_kb = int(raw.split()[1])
            kb = avail_kb if avail_kb is not None else (free_kb + cached_kb)
            if kb > 0:
                return round(kb / (1024.0 * 1024.0), 2)
        except Exception:
            pass
    try:
        proc = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=4)
        if proc.returncode == 0 and proc.stdout.strip():
            ps_match = re.search(r"page size of\s+(\d+)\s+bytes", proc.stdout)
            page_sz = int(ps_match.group(1)) if ps_match else 4096
            pages = 0
            for key in ("Pages free", "Pages inactive", "Pages speculative"):
                m = re.search(rf"{key}:\s+(\d+)", proc.stdout)
                if m:
                    pages += int(m.group(1))
            if pages > 0:
                return round((pages * page_sz) / (1024.0**3), 2)
    except Exception:
        pass
    return None


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


def _effective_gpu_reserve_gb(total_gb: float, reserve_gb: float) -> float:
    return float(reserve_gb) if float(total_gb) >= 16.0 else min(float(reserve_gb), 0.5)


def _max_workers_fitting_vram(
    free_gb: float,
    req_gb: float,
    extra_worker_gb: float,
    eff_reserve_gb: float,
) -> int:
    usable = float(free_gb) - float(eff_reserve_gb)
    if usable < req_gb:
        return 0
    k = 1
    while (k + 1) * req_gb + k * extra_worker_gb <= usable + 1e-6:
        k += 1
    return k


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
    host_avail_gb: float | None = None,
    host_reserve_gb: float = 4.0,
    safety_mult: float = 1.0,
) -> list[str]:
    """Assigns per-worker GPU IDs across VRAM+headroom, bandwidth, utilization, CPU, and host-RAM bounds."""
    if not gpus:
        return []
    req_gb = model_vram_per_worker_gb(model_id, stage=stage, safety_mult=safety_mult)
    params_b = _estimate_model_billions(model_id)
    extra_worker_gb = _extra_worker_vram_gb(params_b, stage=stage)

    # Enforce mandatory VRAM reserve directly inside eligibility so nearly-full GPUs never sneak through.
    eligible = [
        g
        for g in gpus
        if float(g["free_gb"]) >= req_gb + _effective_gpu_reserve_gb(float(g["total_gb"]), reserve_gb)
    ]
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
        eff_reserve = _effective_gpu_reserve_gb(total_gb, reserve_gb)
        max_fit = max(1, _max_workers_fitting_vram(free_gb, req_gb, extra_worker_gb, eff_reserve))
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
            idx_trim = max(
                range(len(per_gpu_counts)),
                key=lambda i: (per_gpu_counts[i][1], i),
            )
            if per_gpu_counts[idx_trim][1] <= 1:
                break
            per_gpu_counts[idx_trim][1] -= 1

    # Clamp total workers to available host CPU RAM so parallel weight loads do not trigger SIGKILL (137).
    if workers_per_gpu == "auto" and host_avail_gb is not None and host_avail_gb > 0:
        req_host = host_ram_per_worker_gb(model_id, stage=stage)
        usable_host = max(req_host, float(host_avail_gb) - float(host_reserve_gb))
        max_total_host = max(1, int(usable_host // req_host))
        while sum(item[1] for item in per_gpu_counts) > max_total_host:
            idx_trim = max(
                range(len(per_gpu_counts)),
                key=lambda i: (per_gpu_counts[i][1], i),
            )
            if per_gpu_counts[idx_trim][1] > 1:
                per_gpu_counts[idx_trim][1] -= 1
            elif len(per_gpu_counts) > 1:
                per_gpu_counts.pop()
            else:
                break

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
    safety_mult: float | None = None,
) -> list[str]:
    """Queries live free VRAM/util/CPUs/host RAM and returns worker GPU IDs, optionally waiting if busy."""
    deadline = time.monotonic() + max(0, int(wait_timeout_sec))
    eff_mult = (
        float(safety_mult)
        if safety_mult is not None
        else float(os.environ.get("CBARR_VRAM_SAFETY_MULT", "1.0") or "1.0")
    )
    req_gb = model_vram_per_worker_gb(model_id, stage=stage, safety_mult=eff_mult)
    cpus = os.cpu_count()
    while True:
        gpus = query_physical_gpus(candidate_ids=candidate_ids)
        if not gpus:
            return []
        host_avail = query_host_avail_ram_gb()
        slots = compute_slots_from_free_mem(
            gpus,
            model_id=model_id,
            workers_per_gpu=workers_per_gpu,
            max_gpus=max_gpus,
            stage=stage,
            cpu_count=cpus,
            host_avail_gb=host_avail,
            safety_mult=eff_mult,
        )
        if slots:
            summary = _format_gpu_live_summary(gpus)
            ram_s = f"{host_avail:.1f}G" if host_avail is not None else "?"
            print(
                f"[gpu_pool] Selected {len(slots)} worker slot(s) {slots} for {model_id} "
                f"(stage={stage}, mult={eff_mult:.2f}, mode={workers_per_gpu}, need>={req_gb:.1f}G+res/worker, cpus={cpus or '?'}, "
                f"host_avail={ram_s} | live: {summary})",
                file=sys.stderr,
                flush=True,
            )
            return slots
        now = time.monotonic()
        if now >= deadline:
            summary = _format_gpu_live_summary(gpus)
            print(
                f"[gpu_pool] ERROR: 0 GPUs have >= {req_gb:.1f} GB + headroom free for {model_id} "
                f"(stage={stage} | {summary})",
                file=sys.stderr,
                flush=True,
            )
            return []
        max_free = max(float(g["free_gb"]) for g in gpus)
        rem_m = int(max(1.0, (deadline - now) / 60.0))
        print(
            f"[gpu_pool] All candidate GPUs busy (max free={max_free:.1f}G, stage={stage} needs >= {req_gb:.1f}G+res "
            f"for {model_id}); waiting {poll_sec}s for free VRAM (up to {rem_m}m remaining)...",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(max(1, int(poll_sec)))


def _has_complete_hf_weights(snap_dir: pathlib.Path) -> bool:
    """Verifies a local HF snapshot directory contains config, tokenizer, and ALL weight shards (not tokenizer-only)."""
    if not snap_dir.is_dir() or not (snap_dir / "config.json").is_file():
        return False
    has_tok = any(
        (snap_dir / fn).is_file()
        for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "tokenizer.model")
    )
    if not has_tok:
        return False
    for idx_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        idx_p = snap_dir / idx_name
        if idx_p.is_file():
            try:
                wmap = json.loads(idx_p.read_text(encoding="utf-8")).get("weight_map", {})
                shard_files = {str(v) for v in wmap.values() if v}
                if shard_files and all(
                    (snap_dir / sf).is_file() and (snap_dir / sf).stat().st_size > 0
                    for sf in shard_files
                ):
                    return True
            except Exception:
                pass
            return False
    return any(
        (snap_dir / fn).is_file() and (snap_dir / fn).stat().st_size > 0
        for fn in ("model.safetensors", "pytorch_model.bin")
    )


def prewarm_hf_model_weights(
    model_id: str,
    *,
    offline: bool = False,
    max_retries: int = 3,
    snapshot_fn: Any = None,
) -> bool:
    """Downloads/verifies complete model weights in 1 process; returns True iff safe to set HF_HUB_OFFLINE=1."""
    local_p = pathlib.Path(model_id)
    if local_p.exists():
        return _has_complete_hf_weights(local_p)
    if offline or os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes"):
        return True
    if snapshot_fn is None:
        try:
            from huggingface_hub import snapshot_download

            snapshot_fn = snapshot_download
        except Exception:
            return False

    try:
        cached_dir = pathlib.Path(str(snapshot_fn(repo_id=model_id, local_files_only=True)))
        if _has_complete_hf_weights(cached_dir):
            return True
        print(
            f"[hf_prewarm] Partial local cache for {model_id} (missing weight shards); downloading full checkpoint...",
            flush=True,
        )
    except Exception:
        print(f"[hf_prewarm] Pre-warming HF checkpoint once for {model_id}...", flush=True)

    base_ignore = ["*.gguf", "*.onnx*", "*.msgpack", "*.h5", "*.ot", "original/*", "consolidated*"]
    attempts = max(1, int(max_retries))
    for attempt in range(1, attempts + 1):
        try:
            dl_dir = pathlib.Path(
                str(snapshot_fn(repo_id=model_id, ignore_patterns=[*base_ignore, "*.bin"]))
            )
            if _has_complete_hf_weights(dl_dir):
                return True
            # Fallback for legacy repos that publish only pytorch_model.bin without .safetensors
            dl_dir_bin = pathlib.Path(str(snapshot_fn(repo_id=model_id, ignore_patterns=base_ignore)))
            if _has_complete_hf_weights(dl_dir_bin):
                return True
        except Exception as exc:
            if attempt >= attempts:
                print(
                    f"[hf_prewarm] HF pre-warm failed after {attempts} attempt(s) ({exc}); workers will resolve online.",
                    flush=True,
                )
                return False
            wait_s = 5 * (2 ** (attempt - 1))
            print(
                f"[hf_prewarm] Attempt {attempt}/{attempts} for {model_id} hit transient error ({exc}); retrying in {wait_s}s...",
                flush=True,
            )
            time.sleep(wait_s)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Select live free-VRAM/utilization/RAM CUDA GPU slots.")
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
        choices=STAGE_CHOICES,
        help="Workload memory profile (matrix, calibrate, diag, b512, blend, mlp, smoke, replay).",
    )
    parser.add_argument(
        "--safety-mult",
        type=float,
        default=float(os.environ.get("CBARR_VRAM_SAFETY_MULT", "1.0") or "1.0"),
        help="VRAM requirement multiplier (automatically >1.0 on retry after mid-stage OOM).",
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
        safety_mult=args.safety_mult,
    )
    if not slots:
        sys.exit(1)
    if args.single:
        print(slots[0])
    else:
        print(",".join(slots))


if __name__ == "__main__":
    main()
