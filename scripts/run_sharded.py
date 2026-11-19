#!/usr/bin/env python3
"""Multi-GPU intra-model shard launcher with live per-GPU logging and periodic progress snapshots."""
from __future__ import annotations

import argparse
import codecs
from collections import deque
import json
import os
import pathlib
import queue
import re
import shutil
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

from src.logfmt import (  # noqa: E402
    colorize_log_line,
    colors_enabled,
    fmt_progress_bar,
    gpu_color,
    hl,
    status_badge,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_TQDM_LINE_RE = re.compile(r"^\s*[^:\n]{1,60}:\s+\d+%\|")
_PROGRESS_METRIC_RE = re.compile(
    r"^\[(?:seed_60|dev_60|eval_450|diag_ehurt|diag_controls)\b|overall=|completed\s+\d+\s+items|COMPLETED\b",
    re.IGNORECASE,
)
_MILESTONE_OR_ERR_RE = re.compile(
    r"completed\s+\d+\s+items|COMPLETED\s+(?:CALIBRATION\s+)?SHARD|\[diag_controls\]|\bslices\s*\("
    r"|\[ERROR\]|Traceback\s*\(|OutOfMemoryError|CUDA out of memory|RuntimeError:|ValueError:|OSError:|FileNotFoundError:|FAILED",
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
    slot_idx: int = 0
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


def _read_jsonl_uids(
    path: pathlib.Path, *, item_nested: bool = False, require_key: str | None = None
) -> list[str]:
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
            if require_key and require_key not in obj:
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
    *,
    include_b512: bool = False,
) -> tuple[dict[str, int], list[dict[str, int]], list[dict[str, int]], dict[str, int]]:
    total_targets: dict[str, int] = {}
    shard_targets: list[dict[str, int]] = [{} for _ in range(n_shards)]
    shard_canon: list[dict[str, int]] = [{} for _ in range(n_shards)]
    total_canon: dict[str, int] = {}

    for tag, _, data_fname, cache_fname in specs:
        uids = _read_jsonl_uids(data_dir / data_fname, item_nested=False)
        req_k = "m2_b512" if (tag == "eval_450" and include_b512) else None
        canon_uids = set(_read_jsonl_uids(model_dir / cache_fname, item_nested=True, require_key=req_k))
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
            r"\[ERROR\]|Traceback\s*\(|OutOfMemoryError|CUDA out of memory|RuntimeError:|ValueError:|OSError:|FAILED",
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


def _terminate_children(states: list[_ShardState | None]) -> None:
    """Best-effort graceful SIGTERM -> SIGKILL cleanup of active GPU child shards."""
    alive = [st for st in states if st is not None and st.proc.poll() is None]
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
    states: list[_ShardState | None],
    specs: tuple[tuple[str, str, str, str], ...],
    total_targets: dict[str, int],
    shard_targets: list[dict[str, int]],
    shard_canon: list[dict[str, int]],
    total_canon: dict[str, int],
    start_ts: float,
    *,
    n_workers: int = 0,
    pre_done_count: int = 0,
) -> str:
    now = time.monotonic()
    elapsed = _format_dur(now - start_ts)
    total_shards = len(states)
    eff_workers = n_workers if n_workers > 0 else total_shards
    spawned = [st for st in states if st is not None]
    done_ok = pre_done_count + sum(1 for st in spawned if st.rc == 0)
    done_fail = sum(1 for st in spawned if st.rc is not None and st.rc != 0)
    active = sum(1 for st in spawned if st.rc is None)
    queued = max(0, total_shards - pre_done_count - len(spawned))

    agg_done: dict[str, int] = {}
    per_shard_done: dict[int, dict[str, int]] = {}
    for st in spawned:
        s_counts: dict[str, int] = {}
        for tag, _, _, cache_fname in specs:
            new_rows = _count_newlines(st.shard_dir / cache_fname)
            target = shard_targets[st.idx].get(tag, 0)
            canon_n = shard_canon[st.idx].get(tag, 0)
            s_counts[tag] = min(target, canon_n + new_rows) if target > 0 else (canon_n + new_rows)
            agg_done[tag] = agg_done.get(tag, 0) + new_rows
        per_shard_done[st.idx] = s_counts

    split_parts: list[str] = []
    sum_done_all = 0
    sum_tgt_all = 0
    for tag, short_lbl, _, _ in specs:
        tgt = total_targets.get(tag, 0)
        done_n = total_canon.get(tag, 0) + agg_done.get(tag, 0)
        if tgt > 0:
            done_n = min(tgt, done_n)
            sum_done_all += done_n
            sum_tgt_all += tgt
            pct = 100.0 * done_n / tgt
            raw_sp = f"{short_lbl}={done_n}/{tgt} ({pct:.0f}%)"
            sp_style = "green" if done_n >= tgt else ("yellow" if done_n > 0 else "sky")
            split_parts.append(hl(raw_sp, sp_style))
        else:
            split_parts.append(hl(f"{short_lbl}={done_n}", "sky"))

    use_color = colors_enabled()
    title_pill = (
        hl(f" [Shards Progress | {stage_name}] ", "badge_info")
        if use_color
        else f"[Shards Progress | {stage_name}]"
    )
    el_txt = hl(f"elapsed={elapsed}", "white")
    act_txt = hl(f"active={active}/{eff_workers}", "yellow" if active > 0 else "muted")
    done_txt = hl(f"done={done_ok}/{total_shards}", "green" if done_ok > 0 else "muted")
    q_txt = f" {hl(f'queued={queued}', 'cyan')}" if total_shards > eff_workers else ""
    fail_txt = (
        hl(f" fail={done_fail} ", "badge_fail")
        if (use_color and done_fail > 0)
        else hl(f"fail={done_fail}", "red" if done_fail > 0 else "muted")
    )
    overall_bar = fmt_progress_bar(sum_done_all, sum_tgt_all, width=18)
    bar_prefix = f"{overall_bar}  " if overall_bar else ""

    border_style = "red" if done_fail > 0 else ("green" if (active == 0 and queued == 0) else "cyan")
    rows: list[str] = []
    if use_color:
        rows.append(hl("━" * 96, border_style))
    rows.append(
        f"{hl(f'[{_utc_iso()}]', 'muted')} {title_pill} {el_txt} | {act_txt} {done_txt}{q_txt} {fail_txt}\n"
        + f"  {hl('Total (canon+shards):', 'white')} {bar_prefix}{'  '.join(split_parts)}"
    )

    if total_shards > eff_workers:
        display_states = sorted(
            [st for st in spawned if st.rc is None or st.rc != 0],
            key=lambda s: (s.slot_idx, s.idx),
        )
    else:
        display_states = spawned

    for st in display_states:
        st_dur = _format_dur((st.end_ts or now) - st.start_ts)
        if st.rc is None:
            badge = status_badge("run", "RUN ")
        elif st.rc == 0:
            badge = status_badge("done", "DONE")
        else:
            badge = status_badge("fail", f"FAIL({st.rc})")

        s_parts: list[str] = []
        for tag, short_lbl, _, _ in specs:
            d_n = per_shard_done.get(st.idx, {}).get(tag, 0)
            t_n = shard_targets[st.idx].get(tag, 0)
            if t_n > 0:
                cell = f"{short_lbl}={d_n}/{t_n}"
                c_style = "green" if d_n >= t_n else ("yellow" if d_n > 0 else "muted")
                s_parts.append(hl(cell, c_style))
            else:
                s_parts.append(hl(f"{short_lbl}={d_n}", "muted"))

        latest = st.last_progress or st.last_line or st.last_cr or "starting..."
        if len(latest) > 105:
            latest = latest[:102] + "..."
        latest_col = hl(latest, "sky") if latest == "starting..." else colorize_log_line(latest)
        idle_s = int(max(0.0, now - st.last_activity_ts)) if st.last_activity_ts > 0 else 0
        idle_note = hl(f" ({idle_s}s ago)", "orange") if st.rc is None and idle_s >= 15 else ""
        disp_gpu = st.slot_idx if total_shards > eff_workers else st.idx
        micro_lbl = f" [s{st.idx}/{total_shards}]" if total_shards > eff_workers else ""
        gpu_lbl = (
            hl(f"GPU{disp_gpu:<2d}(dev={st.cuda_dev}){micro_lbl}", gpu_color(disp_gpu))
            if use_color
            else f"GPU{disp_gpu}(dev={st.cuda_dev}){micro_lbl}"
        )
        dur_lbl = hl(f"{st_dur:>6s}", "white")
        sep = hl("|", "muted")
        rows.append(
            f"  {gpu_lbl} {badge} {dur_lbl} {sep} {' '.join(s_parts)} {sep} {latest_col}{idle_note}"
        )
    if use_color:
        rows.append(hl("━" * 96, border_style))
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
    parser.add_argument(
        "--queue-mult",
        type=str,
        default=os.environ.get("CBARR_QUEUE_MULT", "auto"),
        help="Micro-shard over-decomposition factor per worker slot for dynamic work-stealing ('auto' -> 2x-3x on live multi-GPU).",
    )
    raw_stagger = os.environ.get("CBARR_SPAWN_STAGGER_SEC", "").strip()
    parser.add_argument(
        "--stagger-sec",
        type=float,
        default=float(raw_stagger) if raw_stagger else -1.0,
        help="Delay (seconds) between spawning concurrent CUDA workers (<0 auto-sizes by model billions).",
    )
    args, remainder = parser.parse_known_args(argv)
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    if not remainder:
        parser.error("Expected base run_matrix command after '--'")
    if args.shards < 1:
        parser.error("--shards must be >= 1")
    return args, remainder


def _prewarm_hf_snapshot(model_id: str, cmd_base: list[str]) -> bool:
    """Downloads or verifies a single complete HF weight snapshot before spawning parallel shards."""
    if "--mock" in cmd_base or "--offline" in cmd_base:
        return False
    if not any("run_matrix.py" in str(tok) for tok in cmd_base):
        return False
    from scripts.gpu_pool import prewarm_hf_model_weights

    return prewarm_hf_model_weights(model_id)


def _sanitize_shard_jsonl_tails(shard_dirs: list[pathlib.Path]) -> None:
    """Drops any truncated trailing JSON line caused by abrupt OOM/signal termination before merging."""
    for sd in shard_dirs:
        for jp in sd.glob("*_cache.jsonl"):
            if not jp.is_file():
                continue
            raw_lines = jp.read_text(encoding="utf-8", errors="replace").splitlines()
            if not raw_lines:
                continue
            valid_lines: list[str] = []
            dropped = 0
            for ln in raw_lines:
                s = ln.strip()
                if not s:
                    continue
                try:
                    json.loads(s)
                    valid_lines.append(s)
                except json.JSONDecodeError:
                    dropped += 1
            if dropped > 0:
                tmp_p = jp.with_suffix(".jsonl.tmp")
                tmp_p.write_text(
                    ("".join(v + "\n" for v in valid_lines)) if valid_lines else "",
                    encoding="utf-8",
                )
                tmp_p.replace(jp)


def _merge_and_prune_matrix_shards(
    data_dir: pathlib.Path,
    model_dir: pathlib.Path,
    strict_diag: bool,
) -> dict[str, int]:
    """Merges partial shard_* caches into canonical JSONL caches and removes merged shard dirs."""
    shards_root = model_dir / "shards"
    if not shards_root.is_dir():
        return {}
    shard_dirs = [p for p in sorted(shards_root.glob("shard_*")) if p.is_dir()]
    if not shard_dirs:
        return {}
    _sanitize_shard_jsonl_tails(shard_dirs)

    from scripts.run_matrix import _load_jsonl, merge_shard_caches

    seed_items = _load_jsonl(data_dir / "seed_60.jsonl")
    dev_items = _load_jsonl(data_dir / "dev_60.jsonl")
    eval_items = _load_jsonl(data_dir / "eval_450.jsonl")
    diag_file = "ifeval_diag_341_strict.jsonl" if strict_diag else "ifeval_diag_491.jsonl"
    diag_items = _load_jsonl(data_dir / diag_file)
    merged = merge_shard_caches(model_dir, seed_items, dev_items, eval_items, diag_items)
    for sd in shard_dirs:
        shutil.rmtree(sd, ignore_errors=True)
    try:
        if shards_root.is_dir() and not any(shards_root.iterdir()):
            shards_root.rmdir()
    except OSError:
        pass
    return merged


def run_sharded_stage(args: argparse.Namespace, cmd_base: list[str]) -> int:
    data_dir = pathlib.Path(args.data_dir)
    out_dir = pathlib.Path(args.out_dir)
    log_dir = pathlib.Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    n_workers = int(args.shards)
    vis_list = [x.strip() for x in args.visible_gpus.split(",") if x.strip()]
    slug = args.model.replace("/", "__")
    model_dir = _resolve_model_dir(out_dir, args.model, cmd_base)
    strict_diag = "--strict-diag" in cmd_base
    calibrate_only = "--calibrate-only" in cmd_base
    include_b512 = "--include-b512" in cmd_base
    is_mock = "--mock" in cmd_base
    is_matrix_cmd = any("run_matrix.py" in str(tok) for tok in cmd_base)
    pre_harvested = _merge_and_prune_matrix_shards(data_dir, model_dir, strict_diag=strict_diag)
    if pre_harvested:
        print(
            f"[{_utc_iso()}] [sharded] Harvested prior partial shards -> {pre_harvested}",
            flush=True,
        )
    specs = _split_specs(calibrate_only=calibrate_only, strict_diag=strict_diag)
    init_targets, _, _, init_canon = _load_split_baselines(
        data_dir, model_dir, specs, n_workers, include_b512=include_b512
    )

    q_raw = str(args.queue_mult or "auto").strip().lower()
    if q_raw == "auto":
        if is_mock or calibrate_only or n_workers <= 1 or not is_matrix_cmd:
            q_mult = 1
        else:
            rem_total = sum(max(0, init_targets.get(t, 0) - init_canon.get(t, 0)) for t, _, _, _ in specs)
            if rem_total >= n_workers * 36:
                q_mult = 3
            elif rem_total >= n_workers * 18:
                q_mult = 2
            else:
                q_mult = 1
    else:
        q_mult = max(1, int(q_raw))

    total_shards = n_workers * q_mult
    total_targets, shard_targets, shard_canon, total_canon = _load_split_baselines(
        data_dir, model_dir, specs, total_shards, include_b512=include_b512
    )

    pre_done_set: set[int] = set()
    if q_mult > 1 and is_matrix_cmd:
        for s_idx in range(total_shards):
            if all(shard_canon[s_idx].get(t, 0) >= shard_targets[s_idx].get(t, 0) for t, _, _, _ in specs):
                pre_done_set.add(s_idx)

    prewarmed_offline = _prewarm_hf_snapshot(args.model, cmd_base)
    threads_cap = str(max(1, int(args.threads_per_worker)))

    def _cuda_for(slot_i: int) -> str:
        return vis_list[slot_i % len(vis_list)] if vis_list else str(slot_i)

    gpu_map = ", ".join(
        hl(f"shard{i}->dev{_cuda_for(i)}", gpu_color(i)) for i in range(n_workers)
    )
    first_log = log_dir / f"cbarr_{args.run_id}_gpu0_{slug}.log"
    sh_tag = hl("[sharded]", "cyan")
    if total_shards > n_workers:
        q_info = hl(
            f"work-stealing queue={total_shards} micro-shards ({len(pre_done_set)} cached, {total_shards - len(pre_done_set)} todo)",
            "yellow",
        )
        print(f"[{_utc_iso()}] {sh_tag} GPU map ({n_workers}): {gpu_map} | {q_info}", flush=True)
    else:
        print(f"[{_utc_iso()}] {sh_tag} GPU map ({n_workers}): {gpu_map}", flush=True)
    print(
        f"[{_utc_iso()}] {sh_tag} Per-GPU logs (threads={threads_cap}/worker): {log_dir}/cbarr_{args.run_id}_gpu{{0..{n_workers - 1}}}_{slug}.log",
        flush=True,
    )
    print(f"[{_utc_iso()}] {sh_tag} Live follow : {hl(f'tail -f {first_log}', 'yellow')}", flush=True)

    event_q: queue.Queue[tuple[int, str, str]] = queue.Queue()
    states: list[_ShardState | None] = [None] * total_shards
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
        if is_mock or n_workers <= 1:
            stagger_sec = 0.0
        elif float(args.stagger_sec) >= 0.0:
            stagger_sec = min(15.0, float(args.stagger_sec))
        else:
            from scripts.gpu_pool import _estimate_model_billions

            pb = _estimate_model_billions(args.model)
            stagger_sec = 8.0 if pb >= 7.0 else (3.0 if pb >= 2.8 else 1.5)

        def _spawn_shard_on_slot(s_idx: int, slot_i: int) -> None:
            cuda_dev = _cuda_for(slot_i)
            s_log = log_dir / f"cbarr_{args.run_id}_gpu{slot_i}_{slug}.log"
            shard_dir = model_dir / "shards" / f"shard_{s_idx}_of_{total_shards}"
            s_log.parent.mkdir(parents=True, exist_ok=True)
            with s_log.open("a", encoding="utf-8") as fh:
                fh.write(
                    f"[{_utc_iso()}] [shard {s_idx}/{total_shards} | slot={slot_i} | dev={cuda_dev} | model={args.model}]\n"
                )
            cmd = [
                *cmd_base,
                "--shard-idx",
                str(s_idx),
                "--num-shards",
                str(total_shards),
                "--step-checkpoint-every",
                "0",
            ]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = cuda_dev
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
                slot_idx=slot_i,
                last_activity_ts=time.monotonic(),
            )
            states[s_idx] = st
            th = threading.Thread(
                target=_stream_worker,
                args=(st, args.shard_log, event_q),
                daemon=True,
            )
            th.start()
            threads.append(th)

        pending_queue: deque[int] = deque(
            s_idx for s_idx in range(total_shards) if s_idx not in pre_done_set
        )
        active_slots = 0
        initial_launches = min(n_workers, len(pending_queue))
        for slot_i in range(initial_launches):
            if slot_i > 0 and stagger_sec > 0.0:
                time.sleep(stagger_sec)
            s_idx = pending_queue.popleft()
            _spawn_shard_on_slot(s_idx, slot_i)
            active_slots += 1

        rc_all = 0
        last_snap_ts = start_ts
        progress_sec = float(args.progress_sec)

        while active_slots > 0:
            if interrupted_sig:
                break
            try:
                s_idx, kind, payload = event_q.get(timeout=0.5)
            except queue.Empty:
                s_idx, kind, payload = -1, "", ""

            if kind == "line":
                st = states[s_idx]
                assert st is not None
                disp_gpu = st.slot_idx if total_shards > n_workers else s_idx
                tag = hl(f"[GPU{disp_gpu}](dev={st.cuda_dev})", gpu_color(disp_gpu))
                print(f"{tag} {colorize_log_line(payload)}", flush=True)
            elif kind == "exit":
                rc_val = int(payload)
                st = states[s_idx]
                assert st is not None
                slot_i = st.slot_idx
                disp_gpu = slot_i if total_shards > n_workers else s_idx
                dur_s = _format_dur((st.end_ts or time.monotonic()) - st.start_ts)
                if rc_val == 0:
                    done_cnt = len(pre_done_set) + sum(1 for x in states if x is not None and x.rc == 0)
                    pill = f"{status_badge('done', 'DONE')} " if colors_enabled() else ""
                    shard_sfx = f" [shard {s_idx}/{total_shards}]" if total_shards > n_workers else ""
                    msg = hl(f"GPU{disp_gpu} (dev={st.cuda_dev}) DONE in {dur_s}{shard_sfx}", "green")
                    if pending_queue and rc_all == 0 and not interrupted_sig:
                        next_s = pending_queue.popleft()
                        steal_lbl = hl(
                            f"-> stealing shard {next_s}/{total_shards} ({done_cnt}/{total_shards} done, {len(pending_queue)} queued)",
                            "cyan",
                        )
                        print(f"[{_utc_iso()}] {sh_tag} {pill}{msg} {steal_lbl}", flush=True)
                        _spawn_shard_on_slot(next_s, slot_i)
                    else:
                        active_slots -= 1
                        cnt_lbl = hl(f"({done_cnt}/{total_shards} finished)", "cyan")
                        print(
                            f"[{_utc_iso()}] {sh_tag} {pill}{msg} {cnt_lbl}",
                            flush=True,
                        )
                else:
                    rc_all = 1
                    active_slots -= 1
                    pending_queue.clear()
                    pill = f"{status_badge('fail', f'EXIT {rc_val}')} " if colors_enabled() else ""
                    msg = hl(f"GPU{disp_gpu} (dev={st.cuda_dev}) FAILED exit={rc_val} after {dur_s}", "red")
                    print(
                        f"[{_utc_iso()}] {sh_tag} {pill}{msg} — log: {hl(str(st.log_path), 'yellow')}",
                        flush=True,
                    )
                    tail = _tail_lines(st.log_path, max_lines=25)
                    if tail:
                        print(hl("--- Last lines of failed shard log ---", "red"), flush=True)
                        for ln in tail:
                            print(f"  {hl(f'[GPU{disp_gpu}!]', 'red')} {colorize_log_line(ln)}", flush=True)
                        print(hl("--- End failed shard log ---", "red"), flush=True)
                    if active_slots > 0:
                        print(
                            hl(
                                f"[{_utc_iso()}] [sharded] Waiting for {active_slots} active sibling shard(s) to finish flushing caches...",
                                "orange",
                            ),
                            flush=True,
                        )

            now = time.monotonic()
            if progress_sec > 0.0 and active_slots > 0 and (now - last_snap_ts) >= progress_sec:
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
                        n_workers=n_workers,
                        pre_done_count=len(pre_done_set),
                    ),
                    flush=True,
                )
                last_snap_ts = now

        for th in threads:
            th.join(timeout=2.0)

        if interrupted_sig or rc_all != 0:
            post_harvested = _merge_and_prune_matrix_shards(
                data_dir, model_dir, strict_diag=strict_diag
            )
            if post_harvested:
                print(
                    f"[{_utc_iso()}] [sharded] Harvested completed shard progress -> {post_harvested}",
                    flush=True,
                )

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
