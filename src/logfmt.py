"""Console and Colab/Kaggle log highlighting utilities with clean plain-text fallback."""
from __future__ import annotations

import os
import sys

_ANSI = {
    "bold": "\033[1m",
    "cyan": "\033[1;36m",
    "green": "\033[1;32m",
    "yellow": "\033[1;33m",
    "red": "\033[1;31m",
    "magenta": "\033[1;35m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def colors_enabled() -> bool:
    """Returns True when ANSI highlights are requested (`CBARR_COLOR=1`) or stdout is a TTY."""
    if os.environ.get("NO_COLOR", "").strip():
        return False
    override = os.environ.get("CBARR_COLOR", "").strip().lower()
    if override in ("1", "true", "yes", "on"):
        return True
    if override in ("0", "false", "no", "off"):
        return False
    return bool(hasattr(sys.stdout, "isatty") and sys.stdout.isatty())


def hl(text: str, style: str = "bold") -> str:
    """Wraps text in ANSI SGR color codes when enabled, else returns plain text."""
    if not colors_enabled():
        return text
    code = _ANSI.get(style)
    return f"{code}{text}{_ANSI['reset']}" if code else text


def banner(title: str, style: str = "cyan", width: int = 74) -> str:
    """Formats a high-visibility bordered section banner readable with or without ANSI."""
    bar = "=" * width
    return f"{hl(bar, style)}\n{hl(f'>>> {title}', style)}\n{hl(bar, style)}"
