#!/usr/bin/env python3
"""Multi-GPU intra-model shard launcher with live per-GPU logging and periodic progress snapshots."""
from __future__ import annotations

import argparse
import codecs
import json
import os
import pathlib
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.logfmt import hl  # noqa: E402

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_TQDM_LINE_RE = re.compile(r"^\s*[^:\n]{1,60}:\s+\d+%\|")
_PROGRESS_METRIC_RE = re.compile(
    r"^\[(?:seed_60|dev_60|eval_450|diag_ehurt|diag_controls)\b|overall=|completed\s+\d+\s+items|COMPLETED\b",
    re.IGNORECASE,
)
_MILESTONE_OR_ERR_RE = re.compile(
    r"completed\s+\d+\s+items|COMPLETED\s+(?:CALIBRATION\s+)?SHARD|\[diag_controls\]|\bslices\s*\("
    r"|\[ERROR\]|Traceback\s*\(|OutOfMemoryError|CUDA out of memory|RuntimeError:|ValueError:|FileNotFoundError:|FAILED",
    re.IGNORECASE,
)


@dataclass
class _ShardState:
    idx: int
    cuda_dev: str
    log_path: pathlib.Path
    shard_dir: pathlib.Path
    proc: subprocess.Popen[bytes]
    start_ts: float
    end_ts: float | None = None
    rc: int | None = None
    last_line: str = ""
    last_progress: str = ""
    last_cr: str = ""
    last_activity_ts: float = 0.0
    last_tqdm_echo_ts: float = 0.0


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_dur(seconds: float) -> str:
    sec = max(0, int(round(seconds)))
    if sec < 60:
        return f"{sec}s"
    mins, rem_s = divmod(sec, 60)
    if mins < 60:
        return f"{mins}m{rem_s:02d}s"
    hrs, rem_m = divmod(mins, 60)
    return f"{hrs}h{rem_m:02d}m{rem_s:02d}s"


def _resolve_model_dir(
    out_dir: pathlib.Path,
    model_id: str,
    cmd_args: list[str],
) -> pathlib.Path:
    tag = model_id.replace("/", "__")

    def _float_arg(flag: str, default: float) -> float:
        for i, tok in enumerate(cmd_args):
            if tok == flag and i + 1 < len(cmd_args):
                try:
                    return float(cmd_args[i + 1])
                except ValueError:
                    return default
            if tok.startswith(f"{flag}="):
                try:
                    return float(tok.split("=", 1)[1])
                except ValueError:
                    return default
        return default

    def _int_arg(flag: str, default: int) -> int:
        for i, tok in enumerate(cmd_args):
            if tok == flag and i + 1 < len(cmd_args):
                try:
                    return int(cmd_args[i + 1])
                except ValueError:
                    return default
            if tok.startswith(f"{flag}="):
                try:
                    return int(tok.split("=", 1)[1])
                except ValueError:
                    return default
        return default

    temp = _float_arg("--temperature", 0.0)
    if temp > 0.0:
        top_p = _float_arg("--top-p", 0.95)
        pp = _float_arg("--presence-penalty", 0.0)
        seed = _int_arg("--seed", 42)
        tag = f"{tag}__T{temp:g}_tp{top_p:g}_pp{pp:g}_s{seed}"
    return out_dir / tag


def _split_specs(calibrate_only: bool, strict_diag: bool) -> tuple[tuple[str, str, str, str], ...]:
    """Returns (tag, short_label, data_jsonl, cache_jsonl) for tracked splits."""
    specs: list[tuple[str, str, str, str]] = [
        ("seed_60", "seed", "seed_60.jsonl", "seed_60_cache.jsonl"),
        ("dev_60", "dev", "dev_60.jsonl", "dev_60_cache.jsonl"),
    ]
    if not calibrate_only:
        diag_data = "ifeval_diag_341_strict.jsonl" if strict_diag else "ifeval_diag_491.jsonl"
        specs.extend(
            [
                ("eval_450", "eval", "eval_450.jsonl", "eval_450_cache.jsonl"),
                ("diag_ehurt", "diag", diag_data, "diag_ehurt_cache.jsonl"),
            ]
        )
    return tuple(specs)


def _read_jsonl_uids(path: pathlib.Path, *, item_nested: bool = False) -> list[str]:
    if not path.is_file():
        return []
    uids: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if item_nested:
                uid = str((obj.get("item") or {}).get("uid") or obj.get("uid") or "")
            else:
                uid = str(obj.get("uid") or (obj.get("item") or {}).get("uid") or "")
            if uid:
                uids.append(uid)
    return uids


def _count_newlines(path: pathlib.Path) -> int:
    if not path.is_file():
        return 0
    total = 0
    try:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(65536)
                if not chunk:
                    break
                total += chunk.count(b"\n")
    except OSError:
        return 0
    return total


def _tail_lines(path: pathlib.Path, max_lines: int = 25) -> list[str]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-max_lines:]


def _load_split_baselines(
    data_dir: pathlib.Path,
    model_dir: pathlib.Path,
    specs: tuple[tuple[str, str, str, str], ...],
    n_shards: int,
) -> tuple[dict[str, int], list[dict[str, int]], list[dict[str, int]], dict[str, int]]:
    total_targets: dict[str, int] = {}
    shard_targets: list[dict[str, int]] = [{} for _ in range(n_shards)]
    shard_canon: list[dict[str, int]] = [{} for _ in range(n_shards)]
    total_canon: dict[str, int] = {}

    for tag, _, data_fname, cache_fname in specs:
        uids = _read_jsonl_uids(data_dir / data_fname, item_nested=False)
        canon_uids = set(_read_jsonl_uids(model_dir / cache_fname, item_nested=True))
        uid_set = set(uids)
        total_targets[tag] = len(uids)
        total_canon[tag] = sum(1 for u in uid_set if u in canon_uids)
        for s_idx in range(n_shards):
            slice_uids = uids[s_idx::n_shards]
            shard_targets[s_idx][tag] = len(slice_uids)
            shard_canon[s_idx][tag] = sum(1 for u in slice_uids if u in canon_uids)

    return total_targets, shard_targets, shard_canon, total_canon


def _should_echo_line(
    shard_idx: int,
    clean_line: str,
    log_mode: str,
    now: float,
    state: _ShardState,
) -> bool:
    if not clean_line:
        return False
    is_err = bool(
        re.search(
            r"\[ERROR\]|Traceback\s*\(|OutOfMemoryError|CUDA out of memory|RuntimeError:|ValueError:|FAILED",
            clean_line,
            re.IGNORECASE,
        )
    )
    if log_mode == "off":
        return is_err
    if log_mode == "milestones":
        return bool(_MILESTONE_OR_ERR_RE.search(clean_line))
    if log_mode == "auto" and shard_idx > 0:
        return bool(_MILESTONE_OR_ERR_RE.search(clean_line))

    # For GPU 0 in 'auto', or all GPUs in 'all': suppress rapid intermediate tqdm bars.
    if _TQDM_LINE_RE.match(clean_line) and "100%|" not in clean_line:
        if (now - state.last_tqdm_echo_ts) < 10.0:
            return False
        state.last_tqdm_echo_ts = now
    return True


def _stream_worker(
    state: _ShardState,
    log_mode: str,
    event_q: queue.Queue[tuple[int, str, str]],
) -> None:
    """Reads shard stdout/stderr in raw chunks, logs complete lines to file, and forwards events."""
    assert state.proc.stdout is not None
    fd = state.proc.stdout.fileno()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    cur_line = ""
    state.log_path.parent.mkdir(parents=True, exist_ok=True)

    with state.log_path.open("a", encoding="utf-8", buffering=1) as log_fh:
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                chunk = b""
            if not chunk:
                tail_text = decoder.decode(b"", final=True)
                cur_line += tail_text
                break
            txt = decoder.decode(chunk, final=False)
            if not txt:
                continue
            cur_line += txt
            while "\n" in cur_line:
                raw_seg, cur_line = cur_line.split("\n", 1)
                collapsed = raw_seg.rstrip("\r").split("\r")[-1]
                clean = _ANSI_RE.sub("", collapsed).rstrip()
                now = time.monotonic()
                log_fh.write(clean + "\n")
                if clean:
                    state.last_line = clean
                    state.last_cr = ""
                    state.last_activity_ts = now
                    if _PROGRESS_METRIC_RE.search(clean):
                        state.last_progress = clean
                    if _should_echo_line(state.idx, clean, log_mode, now, state):
                        event_q.put((state.idx, "line", clean))
            if cur_line and "\r" in cur_line:
                segs = [s for s in cur_line.split("\r") if s]
                cr_seg = _ANSI_RE.sub("", segs[-1]).strip() if segs else ""
                if cr_seg:
                    state.last_cr = cr_seg
                    state.last_activity_ts = time.monotonic()
                cur_line = f"{segs[-1]}\r" if (cur_line.endswith("\r") and segs) else cur_line.split("\r")[-1]

        if cur_line:
            collapsed = cur_line.rstrip("\r").split("\r")[-1]
            clean = _ANSI_RE.sub("", collapsed).rstrip()
            if clean:
                log_fh.write(clean + "\n")
                state.last_line = clean
                state.last_activity_ts = time.monotonic()
                if _PROGRESS_METRIC_RE.search(clean):
                    state.last_progress = clean
                if _should_echo_line(state.idx, clean, log_mode, time.monotonic(), state):
                    event_q.put((state.idx, "line", clean))

    try:
        state.proc.stdout.close()
    except OSError:
        pass
    rc = state.proc.wait()
    state.rc = rc
    state.end_ts = time.monotonic()
    event_q.put((state.idx, "exit", str(rc)))


def _terminate_children(states: list[_ShardState]) -> None:
    """Best-effort graceful SIGTERM -> SIGKILL cleanup of active GPU child shards."""
    alive = [st for st in states if st.proc.poll() is None]
    if not alive:
        return
    for st in alive:
        try:
            st.proc.terminate()
        except OSError:
            pass
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if all(st.proc.poll() is not None for st in alive):
            return
        time.sleep(0.1)
    for st in alive:
        if st.proc.poll() is None:
            try:
                st.proc.kill()
            except OSError:
                pass


def _render_snapshot(
    stage_name: str,
    states: list[_ShardState],
    specs: tuple[tuple[str, str, str, str], ...],
    total_targets: dict[str, int],
    shard_targets: list[dict[str, int]],
    shard_canon: list[dict[str, int]],
    total_canon: dict[str, int],
    start_ts: float,
) -> str:
    now = time.monotonic()
    elapsed = _format_dur(now - start_ts)
    done_ok = sum(1 for st in states if st.rc == 0)
    done_fail = sum(1 for st in states if st.rc is not None and st.rc != 0)
    active = len(states) - done_ok - done_fail

    agg_done: dict[str, int] = {}
    per_shard_done: list[dict[str, int]] = []
    for st in states:
        s_counts: dict[str, int] = {}
        for tag, _, _, cache_fname in specs:
            new_rows = _count_newlines(st.shard_dir / cache_fname)
            target = shard_targets[st.idx].get(tag, 0)
            canon_n = shard_canon[st.idx].get(tag, 0)
            s_counts[tag] = min(target, canon_n + new_rows) if target > 0 else (canon_n + new_rows)
            agg_done[tag] = agg_done.get(tag, 0) + new_rows
        per_shard_done.append(s_counts)

    split_parts: list[str] = []
    for tag, short_lbl, _, _ in specs:
        tgt = total_targets.get(tag, 0)
        done_n = total_canon.get(tag, 0) + agg_done.get(tag, 0)
        if tgt > 0:
            done_n = min(tgt, done_n)
            pct = 100.0 * done_n / tgt
            split_parts.append(f"{short_lbl}={done_n}/{tgt} ({pct:.0f}%)")
        else:
            split_parts.append(f"{short_lbl}={done_n}")

    header = (
        f"[{_utc_iso()}] "
        + hl(f"[Shards Progress | {stage_name}]", "cyan")
        + f" elapsed={elapsed} | active={active}/{len(states)} done={done_ok}/{len(states)} fail={done_fail}\n"
        + f"  Total (canon+shards): {'  '.join(split_parts)}"
    )

    rows = [header]
    for st in states:
        st_dur = _format_dur((st.end_ts or now) - st.start_ts)
        if st.rc is None:
            badge = hl("RUN ", "yellow")
        elif st.rc == 0:
            badge = hl("DONE", "green")
        else:
            badge = hl(f"FAIL({st.rc})", "red")

        s_parts: list[str] = []
        for tag, short_lbl, _, _ in specs:
            d_n = per_shard_done[st.idx].get(tag, 0)
            t_n = shard_targets[st.idx].get(tag, 0)
            s_parts.append(f"{short_lbl}={d_n}/{t_n}" if t_n > 0 else f"{short_lbl}={d_n}")

        latest = st.last_progress or st.last_line or st.last_cr or "starting..."
        if len(latest) > 105:
            latest = latest[:102] + "..."
        idle_s = int(max(0.0, now - st.last_activity_ts)) if st.last_activity_ts > 0 else 0
        idle_note = f" ({idle_s}s ago)" if st.rc is None and idle_s >= 15 else ""
        rows.append(
            f"  GPU{st.idx}(dev={st.cuda_dev}) {badge} {st_dur:>6s} | {' '.join(s_parts)} | {latest}{idle_note}"
        )
    return "\n".join(rows)


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run N parallel run_matrix.py GPU shards with live logging and progress dashboard."
    )
    parser.add_argument("--stage-name", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--log-dir", type=str, default="outputs/logs")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--visible-gpus", type=str, default="")
    parser.add_argument("--progress-sec", type=float, default=20.0)
    parser.add_argument(
        "--shard-log",
        type=str,
        default="auto",
        choices=["auto", "all", "milestones", "off"],
    )
    parser.add_argument("--threads-per-worker", type=int, default=4)
    args, remainder = parser.parse_known_args(argv)
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    if not remainder:
        parser.error("Expected base run_matrix command after '--'")
    if args.shards < 1:
        parser.error("--shards must be >= 1")
    return args, remainder


def _prewarm_hf_snapshot(model_id: str, cmd_base: list[str]) -> bool:
    """Downloads or verifies a single shared HF snapshot before spawning parallel shards."""
    if "--mock" in cmd_base or "--offline" in cmd_base:
        return False
    if not any("run_matrix.py" in str(tok) for tok in cmd_base):
        return False
    if os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes"):
        return True
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return False
    try:
        snapshot_download(repo_id=model_id, local_files_only=True)
        return True
    except Exception:
        pass
    try:
        print(f"[{_utc_iso()}] [sharded] Pre-warming HF cache once for {model_id}...", flush=True)
        snapshot_download(repo_id=model_id)
        return True
    except Exception as exc:
        print(f"[{_utc_iso()}] [sharded] HF pre-warm skipped ({exc}); shards will resolve directly.", flush=True)
        return False


def run_sharded_stage(args: argparse.Namespace, cmd_base: list[str]) -> int:
    data_dir = pathlib.Path(args.data_dir)
    out_dir = pathlib.Path(args.out_dir)
    log_dir = pathlib.Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    n_shards = int(args.shards)
    vis_list = [x.strip() for x in args.visible_gpus.split(",") if x.strip()]
    slug = args.model.replace("/", "__")
    model_dir = _resolve_model_dir(out_dir, args.model, cmd_base)
    strict_diag = "--strict-diag" in cmd_base
    calibrate_only = "--calibrate-only" in cmd_base
    specs = _split_specs(calibrate_only=calibrate_only, strict_diag=strict_diag)
    total_targets, shard_targets, shard_canon, total_canon = _load_split_baselines(
        data_dir, model_dir, specs, n_shards
    )

    prewarmed_offline = _prewarm_hf_snapshot(args.model, cmd_base)
    threads_cap = str(max(1, int(args.threads_per_worker)))

    def _cuda_for(s_idx: int) -> str:
        return vis_list[s_idx % len(vis_list)] if vis_list else str(s_idx)

    gpu_map = ", ".join(f"shard{i}->dev{_cuda_for(i)}" for i in range(n_shards))
    first_log = log_dir / f"cbarr_{args.run_id}_gpu0_{slug}.log"
    print(f"[{_utc_iso()}] [sharded] GPU map ({n_shards}): {gpu_map}", flush=True)
    print(
        f"[{_utc_iso()}] [sharded] Per-GPU logs (threads={threads_cap}/worker): {log_dir}/cbarr_{args.run_id}_gpu{{0..{n_shards - 1}}}_{slug}.log",
        flush=True,
    )
    print(f"[{_utc_iso()}] [sharded] Live follow : tail -f {first_log}", flush=True)

    event_q: queue.Queue[tuple[int, str, str]] = queue.Queue()
    states: list[_ShardState] = []
    threads: list[threading.Thread] = []
    start_ts = time.monotonic()

    interrupted_sig: list[int] = []

    def _on_signal(signum: int, _frame: Any) -> None:
        if not interrupted_sig:
            interrupted_sig.append(signum)
            _terminate_children(states)

    prev_int = signal.signal(signal.SIGINT, _on_signal)
    prev_term = signal.signal(signal.SIGTERM, _on_signal)

    try:
        for s_idx in range(n_shards):
            cuda_dev = _cuda_for(s_idx)
            s_log = log_dir / f"cbarr_{args.run_id}_gpu{s_idx}_{slug}.log"
            shard_dir = model_dir / "shards" / f"shard_{s_idx}_of_{n_shards}"
            cmd = [
                *cmd_base,
                "--shard-idx",
                str(s_idx),
                "--num-shards",
                str(n_shards),
                "--step-checkpoint-every",
                "0",
            ]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = cuda_dev
            env["PYTHONUNBUFFERED"] = "1"
            # Bound per-process OpenMP/BLAS/tokenizer threads so N parallel shards do not oversubscribe host CPUs.
            env.setdefault("OMP_NUM_THREADS", threads_cap)
            env.setdefault("MKL_NUM_THREADS", threads_cap)
            env.setdefault("OPENBLAS_NUM_THREADS", threads_cap)
            env.setdefault("NUMEXPR_NUM_THREADS", threads_cap)
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            if prewarmed_offline:
                env["HF_HUB_OFFLINE"] = "1"
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
            st = _ShardState(
                idx=s_idx,
                cuda_dev=cuda_dev,
                log_path=s_log,
                shard_dir=shard_dir,
                proc=proc,
                start_ts=time.monotonic(),
                last_activity_ts=time.monotonic(),
            )
            states.append(st)
            th = threading.Thread(
                target=_stream_worker,
                args=(st, args.shard_log, event_q),
                daemon=True,
            )
            th.start()
            threads.append(th)

        remaining = n_shards
        rc_all = 0
        last_snap_ts = start_ts
        progress_sec = float(args.progress_sec)

        while remaining > 0:
            if interrupted_sig:
                break
            try:
                s_idx, kind, payload = event_q.get(timeout=0.5)
            except queue.Empty:
                s_idx, kind, payload = -1, "", ""

            if kind == "line":
                tag = hl(f"[GPU{s_idx}]", "cyan" if s_idx == 0 else "dim")
                print(f"{tag} {payload}", flush=True)
            elif kind == "exit":
                remaining -= 1
                rc_val = int(payload)
                st = states[s_idx]
                dur_s = _format_dur((st.end_ts or time.monotonic()) - st.start_ts)
                if rc_val == 0:
                    done_cnt = sum(1 for x in states if x.rc == 0)
                    msg = hl(f"GPU{s_idx} (dev={st.cuda_dev}) DONE in {dur_s}", "green")
                    print(
                        f"[{_utc_iso()}] [sharded] {msg} ({done_cnt}/{n_shards} finished)",
                        flush=True,
                    )
                else:
                    rc_all = 1
                    msg = hl(f"GPU{s_idx} (dev={st.cuda_dev}) FAILED exit={rc_val} after {dur_s}", "red")
                    print(
                        f"[{_utc_iso()}] [sharded] {msg} — log: {st.log_path}",
                        flush=True,
                    )
                    tail = _tail_lines(st.log_path, max_lines=25)
                    if tail:
                        print("--- Last lines of failed shard log ---", flush=True)
                        for ln in tail:
                            print(f"  [GPU{s_idx}!] {ln}", flush=True)
                        print("--- End failed shard log ---", flush=True)
                    if remaining > 0:
                        print(
                            f"[{_utc_iso()}] [sharded] Waiting for {remaining} active sibling shard(s) to finish flushing caches...",
                            flush=True,
                        )

            now = time.monotonic()
            if progress_sec > 0.0 and remaining > 0 and (now - last_snap_ts) >= progress_sec:
                print(
                    _render_snapshot(
                        args.stage_name,
                        states,
                        specs,
                        total_targets,
                        shard_targets,
                        shard_canon,
                        total_canon,
                        start_ts,
                    ),
                    flush=True,
                )
                last_snap_ts = now

        for th in threads:
            th.join(timeout=2.0)

        if interrupted_sig:
            sig_num = interrupted_sig[0]
            print(
                f"[{_utc_iso()}] [sharded] Interrupted by signal {sig_num}; terminated active child shards.",
                flush=True,
            )
            return 128 + sig_num

        return rc_all
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)


def main() -> None:
    args, cmd_base = _parse_args()
    raise SystemExit(run_sharded_stage(args, cmd_base))


if __name__ == "__main__":
    main()
