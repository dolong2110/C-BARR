"""High-contrast console and Colab/Kaggle log highlighting utilities with clean plain-text fallback."""
from __future__ import annotations

import os
import re
import sys

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# High-contrast bright SGR codes legible on both dark SSH terminals and modern notebook themes.
_ANSI = {
    "bold": "\033[1m",
    "cyan": "\033[1;96m",
    "green": "\033[1;92m",
    "yellow": "\033[1;93m",
    "red": "\033[1;91m",
    "magenta": "\033[1;95m",
    "blue": "\033[1;94m",
    "white": "\033[1;97m",
    "orange": "\033[1;38;5;214m",
    "teal": "\033[1;38;5;49m",
    "violet": "\033[1;38;5;141m",
    "sky": "\033[1;38;5;117m",
    "muted": "\033[38;5;250m",
    # Keep legacy "dim" readable (light silver instead of dark charcoal \033[2m).
    "dim": "\033[38;5;248m",
    # High-visibility pill badges (bold foreground on vivid background)
    "badge_ok": "\033[1;30;102m",
    "badge_run": "\033[1;30;103m",
    "badge_fail": "\033[1;97;41m",
    "badge_info": "\033[1;30;106m",
    "badge_warn": "\033[1;30;48;5;214m",
    "reset": "\033[0m",
}

_GPU_PALETTE = ("cyan", "green", "yellow", "magenta", "sky", "orange", "teal", "violet")

_ERR_LINE_RE = re.compile(
    r"\[ERROR\]|\bERROR:|Traceback\s*\(|OutOfMemoryError|CUDA out of memory|"
    r"RuntimeError:|ValueError:|OSError:|FileNotFoundError:|AssertionError:|\bFAILED\b",
    re.IGNORECASE,
)
_WARN_LINE_RE = re.compile(
    r"\bWARN(?:ING)?\b|retrying|Partial local cache|Waiting for \d+ active|busy \(max free=",
    re.IGNORECASE,
)
_OK_LINE_RE = re.compile(
    r"COMPLETED\b|completed\s+\d+\s+items|\bMerged\s+\d+|\[git-push\]\s+Pushed|"
    r"Environment setup OK|anonymity audit PASSED|\bDONE\b|\bPASSED\b",
)
_SPLIT_PREFIX_RE = re.compile(r"^(\[(?:seed_60|dev_60|eval_450|diag_ehurt|diag_controls|smoke_12)[^\]]*\])(.*)$")
_METRIC_KV_RE = re.compile(r"\b((?:B0|Fast|B1b|M2_250|M2|OAA250|IFEval-Prec(?:\([^)]*\))?)=\d+(?:\.\d+)?%?)")
_ETA_RE = re.compile(r"\b(eta=[0-9hms.]+)")


def colors_enabled() -> bool:
    """Returns True when ANSI highlights are requested (`CBARR_COLOR=1`/`FORCE_COLOR=1`) or stdout is a TTY."""
    if os.environ.get("NO_COLOR", "").strip():
        return False
    override = os.environ.get("CBARR_COLOR", "").strip().lower()
    if override in ("1", "true", "yes", "on"):
        return True
    if override in ("0", "false", "no", "off"):
        return False
    force = os.environ.get("FORCE_COLOR", "").strip().lower()
    if force in ("1", "true", "yes", "on"):
        return True
    return bool(hasattr(sys.stdout, "isatty") and sys.stdout.isatty())


def strip_ansi(text: str) -> str:
    """Removes ANSI SGR escape sequences for clean plain-text log persistence."""
    return _ANSI_ESCAPE_RE.sub("", text)


def hl(text: str, style: str = "bold") -> str:
    """Wraps text in ANSI SGR color codes when enabled, else returns plain text."""
    if not colors_enabled():
        return text
    code = _ANSI.get(style)
    return f"{code}{text}{_ANSI['reset']}" if code else text


def gpu_color(idx: int) -> str:
    """Returns a bright, distinct foreground style name for GPU worker `idx`."""
    return _GPU_PALETTE[max(0, int(idx)) % len(_GPU_PALETTE)]


def status_badge(kind: str, label: str = "") -> str:
    """Formats a high-visibility status pill (`RUN`, `DONE`, `FAIL(...)`) with plain-text fallback."""
    k = kind.lower()
    if k == "run":
        txt = label or "RUN "
        return hl(f" {txt.strip():<4s} ", "badge_run") if colors_enabled() else txt
    if k in ("done", "ok", "pass"):
        txt = label or "DONE"
        return hl(f" {txt.strip():<4s} ", "badge_ok") if colors_enabled() else txt
    txt = label or "FAIL"
    return hl(f" {txt.strip()} ", "badge_fail") if colors_enabled() else txt


def fmt_progress_bar(done: int, total: int, width: int = 16) -> str:
    """Formats a bright Unicode progress bar when colors are enabled, else returns empty string."""
    if total <= 0 or not colors_enabled():
        return ""
    clamped = max(0, min(int(done), int(total)))
    ratio = clamped / float(total)
    filled = int(round(ratio * width))
    fill_style = "green" if clamped >= total else "cyan"
    bar_fill = hl("█" * filled, fill_style) if filled > 0 else ""
    bar_empty = hl("░" * (width - filled), "muted") if filled < width else ""
    pct_style = "green" if clamped >= total else ("yellow" if clamped > 0 else "sky")
    pct_txt = hl(f"{100.0 * ratio:3.0f}%", pct_style)
    return f"[{bar_fill}{bar_empty}] {pct_txt}"


def colorize_log_line(line: str) -> str:
    """Applies semantic high-contrast highlights to plain worker/orchestrator log lines."""
    if not line or not colors_enabled() or "\033[" in line:
        return line
    if _ERR_LINE_RE.search(line):
        return hl(line, "red")
    if _WARN_LINE_RE.search(line):
        return hl(line, "orange")
    stripped = line.lstrip()
    if stripped.startswith("===") or stripped.startswith(">>>"):
        style = "green" if _OK_LINE_RE.search(line) else "cyan"
        return hl(line, style)
    m_sp = _SPLIT_PREFIX_RE.match(line)
    if m_sp:
        head, tail = m_sp.group(1), m_sp.group(2)
        tail = _ETA_RE.sub(lambda m: hl(m.group(1), "yellow"), tail)
        tail = _METRIC_KV_RE.sub(lambda m: hl(m.group(1), "green"), tail)
        return f"{hl(head, 'magenta')}{tail}"
    if _METRIC_KV_RE.search(line):
        return _METRIC_KV_RE.sub(lambda m: hl(m.group(1), "green"), line)
    if _OK_LINE_RE.search(line):
        return hl(line, "green")
    return line


def banner(title: str, style: str = "cyan", width: int = 74) -> str:
    """Formats a high-visibility bordered section banner readable with or without ANSI."""
    bar = "━" * width if colors_enabled() else "=" * width
    return f"{hl(bar, style)}\n{hl(f'>>> {title}', style)}\n{hl(bar, style)}"
