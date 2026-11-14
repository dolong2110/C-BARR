#!/usr/bin/env python3
"""Compact cross-session result synchronization for split Colab/Kaggle C-BARR runs.

Allows Part 1 (Primary models) to pack per-model summaries + compressed JSONL caches
into the git repository and push to GitHub, and allows Part 2 (Scale Sweep + Phase 7)
to unpack Part 1's caches into OUT_DIR, run combined analysis across all models,
and commit/push the unified final artifacts cleanly without violating GitHub file
size limits or .gitignore exclusions.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.log_experiment import _ensure_git_identity, _run_git_with_askpass, _sanitize_text

COMPACT_CACHE_NAMES: tuple[str, ...] = (
    "seed_60_cache.jsonl",
    "dev_60_cache.jsonl",
    "eval_450_cache.jsonl",
    "diag_ehurt_cache.jsonl",
)

MODEL_SUMMARY_FILES: tuple[str, ...] = (
    "calibration_summary.json",
    "matrix_summary.json",
    "matrix_summary_strict341.json",
    "matrix_summary_diag491.json",
    "tables_rows.tex",
    "tables_rows_strict341.tex",
    "tables_rows_diag491.tex",
    "ext_deltanet_blend.json",
    "prefill_mlp_summary.json",
)

PHASE7_FILES: tuple[str, ...] = (
    "stats_summary.json",
    "paper_tables.tex",
)


def model_slug(model_id: str) -> str:
    """Normalizes a HuggingFace model identifier or directory name to a directory slug."""
    return model_id.strip().replace("/", "__")


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _copy_if_different(src: pathlib.Path, dst: pathlib.Path) -> None:
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        if _sha256_file(src) == _sha256_file(dst):
            return
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _gzip_copy_deterministic(
    src: pathlib.Path, dst_gz: pathlib.Path, max_bytes: int
) -> None:
    """Writes a deterministic gzip archive (mtime=0) only updating dst_gz when content changes."""
    dst_gz.parent.mkdir(parents=True, exist_ok=True)
    tmp_gz = dst_gz.with_name(f".{dst_gz.name}.tmp")
    try:
        with src.open("rb") as f_in, tmp_gz.open("wb") as raw_out:
            with gzip.GzipFile(
                filename="", mode="wb", compresslevel=9, fileobj=raw_out, mtime=0
            ) as gz_out:
                while True:
                    chunk = f_in.read(65536)
                    if not chunk:
                        break
                    gz_out.write(chunk)
        sz = tmp_gz.stat().st_size
        if sz > max_bytes:
            raise ValueError(
                f"Compressed archive {dst_gz.name} ({sz / (1024 * 1024):.2f} MB) exceeds "
                f"max_file_mb ({max_bytes / (1024 * 1024):.2f} MB)."
            )
        if dst_gz.exists() and dst_gz.stat().st_size == sz:
            if _sha256_file(tmp_gz) == _sha256_file(dst_gz):
                tmp_gz.unlink(missing_ok=True)
                return
        os.replace(tmp_gz, dst_gz)
    except Exception:
        tmp_gz.unlink(missing_ok=True)
        raise


def _gunzip_file(src_gz: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    try:
        with gzip.open(src_gz, "rb") as gz_in, tmp.open("wb") as f_out:
            while True:
                chunk = gz_in.read(65536)
                if not chunk:
                    break
                f_out.write(chunk)
        os.replace(tmp, dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _count_plain_lines(path: pathlib.Path) -> int:
    count = 0
    last_nonempty = False
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            count += chunk.count(b"\n")
            last_nonempty = not chunk.endswith(b"\n")
    return count + (1 if last_nonempty and path.stat().st_size > 0 else 0)


def _count_gzip_lines(path_gz: pathlib.Path) -> int:
    count = 0
    nonempty = False
    last_byte = b""
    with gzip.open(path_gz, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            nonempty = True
            count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if nonempty and last_byte != b"\n":
        count += 1
    return count


def _discover_model_slugs(out_dir: pathlib.Path) -> list[str]:
    slugs: list[str] = []
    if not out_dir.is_dir():
        return slugs
    known = set(MODEL_SUMMARY_FILES) | set(COMPACT_CACHE_NAMES) | {
        f"{c}.gz" for c in COMPACT_CACHE_NAMES
    }
    for entry in sorted(out_dir.iterdir()):
        if not entry.is_dir() or entry.name in {"logs", "__pycache__"} or entry.name.startswith("."):
            continue
        if any((entry / fname).is_file() for fname in known):
            slugs.append(entry.name)
    return slugs


def _check_paper_anonymity(repo_root: pathlib.Path) -> None:
    chk = repo_root / "scripts" / "check_anonymity.sh"
    if not chk.is_file():
        return
    proc = subprocess.run(
        ["bash", str(chk)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"Anonymity audit failed; refusing to stage paper/figures: {detail}")


def pack_outputs_for_repo(
    out_dir: pathlib.Path,
    repo_root: pathlib.Path,
    models: list[str] | None = None,
    include_phase7: bool = True,
    max_file_mb: float = 45.0,
    include_ledger: bool = True,
) -> list[pathlib.Path]:
    """Copies compact model summaries and gzips JSONL caches into repo_root/outputs/."""
    out_dir = out_dir.resolve()
    repo_root = repo_root.resolve()
    repo_outputs = repo_root / "outputs"
    repo_outputs.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_file_mb * 1024 * 1024)

    target_slugs = (
        [model_slug(m) for m in models if m.strip()]
        if models
        else _discover_model_slugs(out_dir)
    )

    staged: list[pathlib.Path] = []
    for slug in target_slugs:
        src_dir = out_dir / slug
        if not src_dir.is_dir():
            continue
        dst_dir = repo_outputs / slug
        dst_dir.mkdir(parents=True, exist_ok=True)

        for fname in MODEL_SUMMARY_FILES:
            src_f = src_dir / fname
            if not src_f.is_file():
                continue
            if src_f.stat().st_size > max_bytes:
                raise ValueError(
                    f"{src_f} ({src_f.stat().st_size / (1024 * 1024):.2f} MB) exceeds "
                    f"max_file_mb ({max_file_mb:.2f} MB)."
                )
            dst_f = dst_dir / fname
            _copy_if_different(src_f, dst_f)
            staged.append(dst_f.relative_to(repo_root))

        for cname in COMPACT_CACHE_NAMES:
            src_c = src_dir / cname
            dst_gz = dst_dir / f"{cname}.gz"
            if src_c.is_file() and src_c.stat().st_size > 0:
                _gzip_copy_deterministic(src_c, dst_gz, max_bytes=max_bytes)
                staged.append(dst_gz.relative_to(repo_root))
            elif (src_dir / f"{cname}.gz").is_file():
                src_gz = src_dir / f"{cname}.gz"
                if src_gz.stat().st_size > max_bytes:
                    raise ValueError(
                        f"{src_gz} exceeds max_file_mb ({max_file_mb:.2f} MB)."
                    )
                _copy_if_different(src_gz, dst_gz)
                staged.append(dst_gz.relative_to(repo_root))

    if include_phase7:
        for pf in PHASE7_FILES:
            src_p = out_dir / pf
            if not src_p.is_file():
                continue
            if src_p.stat().st_size > max_bytes:
                raise ValueError(
                    f"{src_p} exceeds max_file_mb ({max_file_mb:.2f} MB)."
                )
            dst_p = repo_outputs / pf
            _copy_if_different(src_p, dst_p)
            staged.append(dst_p.relative_to(repo_root))

        fig_dir = repo_root / "paper" / "figures"
        pdfs = sorted(fig_dir.glob("*.pdf")) if fig_dir.is_dir() else []
        if pdfs:
            _check_paper_anonymity(repo_root)
            for pdf in pdfs:
                if pdf.stat().st_size > max_bytes:
                    raise ValueError(
                        f"{pdf} exceeds max_file_mb ({max_file_mb:.2f} MB)."
                    )
                staged.append(pdf.relative_to(repo_root))

    if include_ledger:
        ledger = repo_root / "EXPERIMENT_LOG.md"
        if ledger.is_file():
            staged.append(ledger.relative_to(repo_root))

    return list(dict.fromkeys(staged))


def unpack_repo_caches(
    repo_root: pathlib.Path,
    out_dir: pathlib.Path,
    overwrite: bool = False,
) -> dict[str, int]:
    """Restores compact summaries and .jsonl.gz caches from repo_root/outputs into out_dir."""
    repo_root = repo_root.resolve()
    out_dir = out_dir.resolve()
    repo_outputs = repo_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"models": 0, "summaries_copied": 0, "caches_unpacked": 0}
    if not repo_outputs.is_dir():
        return counts

    for src_dir in sorted(repo_outputs.iterdir()):
        if not src_dir.is_dir() or src_dir.name in {"logs", "__pycache__"} or src_dir.name.startswith("."):
            continue
        dst_dir = out_dir / src_dir.name
        touched = False

        if src_dir.resolve() != dst_dir.resolve():
            for fname in MODEL_SUMMARY_FILES:
                src_f = src_dir / fname
                dst_f = dst_dir / fname
                if not src_f.is_file():
                    continue
                if (
                    overwrite
                    or not dst_f.exists()
                    or dst_f.stat().st_size == 0
                    or src_f.stat().st_mtime > dst_f.stat().st_mtime
                ):
                    _copy_if_different(src_f, dst_f)
                    counts["summaries_copied"] += 1
                    touched = True

        for cname in COMPACT_CACHE_NAMES:
            src_gz = src_dir / f"{cname}.gz"
            dst_jsonl = dst_dir / cname
            if not src_gz.is_file() or src_gz.stat().st_size == 0:
                continue
            should_unpack = (
                overwrite
                or not dst_jsonl.exists()
                or dst_jsonl.stat().st_size == 0
            )
            if not should_unpack:
                gz_lines = _count_gzip_lines(src_gz)
                cur_lines = _count_plain_lines(dst_jsonl)
                if gz_lines > cur_lines:
                    should_unpack = True
            if should_unpack:
                _gunzip_file(src_gz, dst_jsonl)
                counts["caches_unpacked"] += 1
                touched = True

        if touched:
            counts["models"] += 1

    if repo_outputs.resolve() != out_dir.resolve():
        for pf in PHASE7_FILES:
            src_p = repo_outputs / pf
            dst_p = out_dir / pf
            if src_p.is_file() and (overwrite or not dst_p.exists() or dst_p.stat().st_size == 0):
                _copy_if_different(src_p, dst_p)

    return counts


def sync_and_push_results(
    repo_root: pathlib.Path,
    out_dir: pathlib.Path,
    label: str,
    models: list[str] | None = None,
    include_phase7: bool = True,
    remote: str = "origin",
    branch: str = "",
    token_env: str = "GITHUB_TOKEN",
    git_push: bool = True,
    max_file_mb: float = 45.0,
    include_ledger: bool = True,
) -> tuple[bool, str]:
    """Packs compact results, commits only allowlisted files, and pushes with pull --rebase retry."""
    repo_root = repo_root.resolve()
    rel_paths = pack_outputs_for_repo(
        out_dir=out_dir,
        repo_root=repo_root,
        models=models,
        include_phase7=include_phase7,
        max_file_mb=max_file_mb,
        include_ledger=include_ledger,
    )
    if not rel_paths:
        return False, "No C-BARR result artifacts found to sync."
    if not (repo_root / ".git").exists():
        return False, f"not a git repository: {repo_root}"

    rel_strs = [str(p) for p in rel_paths]
    _ensure_git_identity(repo_root)
    add_proc = subprocess.run(
        ["git", "-C", str(repo_root), "add", "--", *rel_strs],
        capture_output=True,
        text=True,
    )
    if add_proc.returncode != 0:
        err = (add_proc.stderr or add_proc.stdout).strip()
        return False, f"git add failed: {err}"

    diff_chk = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--cached", "--quiet", "--", *rel_strs]
    )
    if diff_chk.returncode != 0:
        msg = f"chore(results): sync {label}"
        cm_proc = subprocess.run(
            ["git", "-C", str(repo_root), "commit", "-m", msg, "--", *rel_strs],
            capture_output=True,
            text=True,
        )
        if cm_proc.returncode != 0:
            err = (cm_proc.stderr or cm_proc.stdout).strip()
            return False, f"git commit failed: {err}"

    if not git_push:
        return True, f"committed {len(rel_strs)} artifact(s) locally (git push skipped)"

    token = os.environ.get(token_env, "").strip()
    remote_proc = subprocess.run(
        ["git", "-C", str(repo_root), "remote", "get-url", remote],
        capture_output=True,
        text=True,
    )
    if remote_proc.returncode != 0:
        return False, f"committed locally, but git remote '{remote}' is not configured"
    remote_url = remote_proc.stdout.strip()
    if remote_url.startswith("https://") and not token:
        return (
            False,
            f"committed locally, but HTTPS remote '{remote}' requires env var {token_env} to push",
        )

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
            return True, f"pushed {len(rel_strs)} artifact(s) -> {remote}/{target_branch}"
        last_err = _sanitize_text(push_proc.stderr or push_proc.stdout, repo_root, token)

    return False, f"git push failed: {last_err.strip()}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack, unpack, and git-sync C-BARR results across split runs."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["pack", "unpack", "push"],
        help="pack into repo/outputs, unpack repo/outputs/*.gz into out-dir, or commit+push.",
    )
    parser.add_argument("--repo-dir", type=str, default=str(REPO_ROOT))
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--label", type=str, default="run")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--no-phase7", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-file-mb", type=float, default=45.0)
    parser.add_argument("--git-push", action="store_true")
    parser.add_argument("--no-git-push", action="store_true")
    parser.add_argument("--git-remote", type=str, default="origin")
    parser.add_argument("--git-branch", type=str, default="")
    parser.add_argument("--git-token-env", type=str, default="GITHUB_TOKEN")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    repo_root = pathlib.Path(args.repo_dir).resolve()
    out_dir = (
        pathlib.Path(args.out_dir).resolve()
        if args.out_dir
        else (repo_root / "outputs").resolve()
    )

    if args.mode == "pack":
        paths = pack_outputs_for_repo(
            out_dir=out_dir,
            repo_root=repo_root,
            models=args.models,
            include_phase7=not args.no_phase7,
            max_file_mb=args.max_file_mb,
            include_ledger=not args.no_ledger,
        )
        print(f"[sync] Packed {len(paths)} artifact(s) into {repo_root / 'outputs'}")
        return

    if args.mode == "unpack":
        counts = unpack_repo_caches(
            repo_root=repo_root,
            out_dir=out_dir,
            overwrite=args.overwrite,
        )
        print(
            f"[sync] Unpacked {counts['caches_unpacked']} cache(s) and copied "
            f"{counts['summaries_copied']} summary file(s) across {counts['models']} model(s) "
            f"-> {out_dir}"
        )
        return

    do_push = False if args.no_git_push else (True if args.git_push or args.mode == "push" else False)
    ok, msg = sync_and_push_results(
        repo_root=repo_root,
        out_dir=out_dir,
        label=args.label,
        models=args.models,
        include_phase7=not args.no_phase7,
        remote=args.git_remote,
        branch=args.git_branch,
        token_env=args.git_token_env,
        git_push=do_push,
        max_file_mb=args.max_file_mb,
        include_ledger=not args.no_ledger,
    )
    status = "OK" if ok else "WARN"
    print(f"[sync] {status}: {msg}")
    if args.strict and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
