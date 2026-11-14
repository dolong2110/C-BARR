#!/usr/bin/env python3
"""Version and provenance tracking utilities for C-BARR code updates and split training runs."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
VERSION_FILENAME = "VERSION"
RESULTS_VERSION_REL = pathlib.Path("outputs") / "RESULTS_VERSION"
RESULTS_MANIFEST_REL = pathlib.Path("outputs") / "results_manifest.json"
SEMVER_RE = re.compile(r"^\s*v?(\d+)\.(\d+)\.(\d+)\s*$")


def parse_semver(version_str: str) -> tuple[int, int, int]:
    """Parses a canonical X.Y.Z SemVer string into a comparable 3-tuple."""
    m = SEMVER_RE.match(str(version_str or ""))
    if not m:
        raise ValueError(f"Invalid semantic version {version_str!r}; expected X.Y.Z")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def format_semver(parts: tuple[int, int, int]) -> str:
    """Formats a 3-tuple as X.Y.Z."""
    return f"{int(parts[0])}.{int(parts[1])}.{int(parts[2])}"


def read_repo_version(repo_root: pathlib.Path = REPO_ROOT, default: str = "0.0.0") -> str:
    """Reads the repository code version from <repo_root>/VERSION cleanly."""
    vpath = pathlib.Path(repo_root) / VERSION_FILENAME
    if not vpath.is_file():
        return default
    raw = vpath.read_text(encoding="utf-8").strip()
    if not raw:
        return default
    return format_semver(parse_semver(raw))


def write_repo_version(version_str: str, repo_root: pathlib.Path = REPO_ROOT) -> pathlib.Path:
    """Atomically writes a validated SemVer string to <repo_root>/VERSION."""
    normalized = format_semver(parse_semver(version_str))
    vpath = pathlib.Path(repo_root) / VERSION_FILENAME
    tmp = vpath.with_name(f".{VERSION_FILENAME}.tmp")
    tmp.write_text(normalized + "\n", encoding="utf-8")
    os.replace(tmp, vpath)
    return vpath


def bump_semver(version_str: str, part: str = "patch") -> str:
    """Increments major, minor, or patch component of a SemVer string."""
    major, minor, patch = parse_semver(version_str)
    kind = part.strip().lower()
    if kind == "patch":
        return format_semver((major, minor, patch + 1))
    if kind == "minor":
        return format_semver((major, minor + 1, 0))
    if kind == "major":
        return format_semver((major + 1, 0, 0))
    raise ValueError(f"Unsupported bump part {part!r}; expected 'patch', 'minor', or 'major'")


def bump_repo_version(repo_root: pathlib.Path = REPO_ROOT, part: str = "patch") -> str:
    """Reads, increments, and writes <repo_root>/VERSION, returning the new version."""
    current = read_repo_version(repo_root)
    nxt = bump_semver(current, part=part)
    write_repo_version(nxt, repo_root=repo_root)
    return nxt


def detect_git_rev(repo_root: pathlib.Path = REPO_ROOT) -> tuple[str, bool]:
    """Returns (short_sha, is_dirty) for <repo_root>, degrading safely outside git."""
    root = pathlib.Path(repo_root)
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return (sha or "unknown"), dirty
    except Exception:
        return "unknown", False


def get_version_provenance(repo_root: pathlib.Path = REPO_ROOT) -> dict[str, Any]:
    """Returns structured code version + git commit provenance for artifact stamping."""
    root = pathlib.Path(repo_root)
    code_ver = read_repo_version(root)
    sha, dirty = detect_git_rev(root)
    git_rev = f"{sha}{'-dirty' if dirty else ''}" if sha != "unknown" else "unknown"
    return {
        "code_version": code_ver,
        "git_rev": git_rev,
        "git_sha": sha,
        "git_dirty": dirty,
        "results_seq": read_results_seq(root),
    }


def read_results_seq(repo_root: pathlib.Path = REPO_ROOT) -> int:
    """Reads the monotonic synced-results release integer from outputs/RESULTS_VERSION."""
    rpath = pathlib.Path(repo_root) / RESULTS_VERSION_REL
    if not rpath.is_file():
        return 0
    raw = rpath.read_text(encoding="utf-8").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        m = re.search(r"(\d+)\s*$", raw)
        return int(m.group(1)) if m else 0


def bump_results_seq(repo_root: pathlib.Path = REPO_ROOT) -> int:
    """Atomically increments outputs/RESULTS_VERSION and returns the new sequence number."""
    rpath = pathlib.Path(repo_root) / RESULTS_VERSION_REL
    rpath.parent.mkdir(parents=True, exist_ok=True)
    nxt = read_results_seq(repo_root) + 1
    tmp = rpath.with_name(".RESULTS_VERSION.tmp")
    tmp.write_text(f"{nxt}\n", encoding="utf-8")
    os.replace(tmp, rpath)
    return nxt


def is_result_only_path(rel_path: str) -> bool:
    """Returns True for output/ledger/figure paths updated by result-sync pushes."""
    norm = rel_path.replace("\\", "/").lstrip("./")
    return (
        norm == "EXPERIMENT_LOG.md"
        or norm == "outputs"
        or norm.startswith("outputs/")
        or norm.startswith("paper/figures/")
    )


def _git_show_version(repo_root: pathlib.Path, rev: str) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{rev}:{VERSION_FILENAME}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    raw = proc.stdout.strip()
    if not raw:
        return None
    return format_semver(parse_semver(raw))


def check_version_increment(
    repo_root: pathlib.Path = REPO_ROOT,
    base_ref: str = "origin/master",
    target_ref: str = "HEAD",
) -> tuple[bool, str]:
    """Verifies VERSION strictly increases whenever non-result files change relative to base_ref."""
    root = pathlib.Path(repo_root)
    base_ok = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "--quiet", base_ref],
        capture_output=True,
        text=True,
    )
    if base_ok.returncode != 0:
        return True, f"base ref {base_ref!r} not found; skipping version diff check."

    diff_proc = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", f"{base_ref}..{target_ref}"],
        capture_output=True,
        text=True,
    )
    if diff_proc.returncode != 0:
        return False, f"git diff failed: {(diff_proc.stderr or diff_proc.stdout).strip()}"

    changed = [line.strip() for line in diff_proc.stdout.splitlines() if line.strip()]
    if not changed:
        return True, "no changes relative to base ref."

    non_result_changed = [
        p for p in changed if not is_result_only_path(p) and p != VERSION_FILENAME
    ]
    if not non_result_changed:
        return True, "result-only changes; VERSION bump not required."

    base_ver = _git_show_version(root, base_ref)
    target_ver = _git_show_version(root, target_ref)
    if base_ver is None:
        if target_ver is not None:
            return True, f"initialized VERSION at v{target_ver}."
        return False, f"non-result files changed without creating {VERSION_FILENAME}."
    if target_ver is None:
        return False, f"{VERSION_FILENAME} was removed in {target_ref}."

    if parse_semver(target_ver) <= parse_semver(base_ver):
        sample = ", ".join(non_result_changed[:4])
        extra = f" (+{len(non_result_changed) - 4} more)" if len(non_result_changed) > 4 else ""
        return (
            False,
            f"non-result files changed ({sample}{extra}) but VERSION stayed v{target_ver} "
            f"(base v{base_ver}). Run: python scripts/versioning.py bump",
        )
    return True, f"VERSION incremented v{base_ver} -> v{target_ver}."


def install_git_hooks(repo_root: pathlib.Path = REPO_ROOT) -> pathlib.Path:
    """Configures git to use the tracked .githooks directory and ensures pre-push is executable."""
    root = pathlib.Path(repo_root)
    hook_path = root / ".githooks" / "pre-push"
    if hook_path.is_file():
        hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    subprocess.run(
        ["git", "-C", str(root), "config", "core.hooksPath", ".githooks"],
        check=True,
    )
    return hook_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage C-BARR code/results versions.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("show", help="Print current repository version and provenance.")
    p_show.add_argument("--repo-dir", type=str, default=str(REPO_ROOT))
    p_show.add_argument("--json", action="store_true")

    p_bump = sub.add_parser("bump", help="Increment VERSION (patch/minor/major) or set explicit X.Y.Z.")
    p_bump.add_argument("--repo-dir", type=str, default=str(REPO_ROOT))
    p_bump.add_argument("--part", choices=["patch", "minor", "major"], default="patch")
    p_bump.add_argument("--set", dest="set_version", type=str, default="")

    p_chk = sub.add_parser("check", help="Check that VERSION was bumped vs base ref for non-result changes.")
    p_chk.add_argument("--repo-dir", type=str, default=str(REPO_ROOT))
    p_chk.add_argument("--base", type=str, default="origin/master")
    p_chk.add_argument("--target", type=str, default="HEAD")

    p_hooks = sub.add_parser("install-hooks", help="Enable tracked .githooks pre-push version check.")
    p_hooks.add_argument("--repo-dir", type=str, default=str(REPO_ROOT))

    args = parser.parse_args()
    root = pathlib.Path(args.repo_dir).resolve()

    if args.cmd == "show":
        prov = get_version_provenance(root)
        if args.json:
            print(json.dumps(prov, indent=2))
        else:
            print(
                f"v{prov['code_version']} (git={prov['git_rev']}, results_seq={prov['results_seq']})"
            )
        return

    if args.cmd == "bump":
        old_v = read_repo_version(root)
        if args.set_version:
            write_repo_version(args.set_version, root)
            new_v = read_repo_version(root)
        else:
            new_v = bump_repo_version(root, part=args.part)
        print(f"Bumped VERSION: v{old_v} -> v{new_v}")
        return

    if args.cmd == "check":
        ok, msg = check_version_increment(root, base_ref=args.base, target_ref=args.target)
        print(f"[version-check] {'OK' if ok else 'FAIL'}: {msg}")
        if not ok:
            raise SystemExit(1)
        return

    if args.cmd == "install-hooks":
        hp = install_git_hooks(root)
        print(f"Configured git core.hooksPath=.githooks (hook: {hp})")


if __name__ == "__main__":
    main()
