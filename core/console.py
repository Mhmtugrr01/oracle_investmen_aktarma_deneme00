"""Terminal Matrix çıktı yardımcıları."""

from __future__ import annotations

import sys

GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _enable_ansi() -> None:
    if sys.platform == "win32":
        import os

        os.system("")


def _safe_print(text: str) -> None:
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"), flush=True)


def agent_print(agent: str, message: str, color: str = GREEN) -> None:
    _enable_ansi()
    _safe_print(f"{color}{BOLD}[{agent}]{RESET} {color}{message}{RESET}")


def system_print(message: str, color: str = CYAN) -> None:
    _enable_ansi()
    _safe_print(f"{color}{BOLD}[SYSTEM]{RESET} {color}{message}{RESET}")


def warn_print(message: str) -> None:
    agent_print("WARNING", message, YELLOW)


def error_print(message: str) -> None:
    agent_print("FATAL", message, RED)
