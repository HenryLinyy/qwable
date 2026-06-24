"""Validation helpers for agent patch and test tool calls."""

from __future__ import annotations

from pathlib import PurePosixPath
import re
from typing import Any


ALLOWED_PATCH_TOOL_NAMES = {
    "read_file",
    "search_files",
    "list_files",
    "edit_file",
    "apply_patch",
    "run_tests",
    "shell",
}

DISALLOWED_SHELL_PATTERNS = [
    "rm -rf /",
    "sudo ",
    "curl | sh",
    "wget | sh",
    "chmod 777",
]

# Reject command chaining, backgrounding (&), redirection (>, >>, <, <<), and
# subshells — not just the few enumerated chain operators. Anything containing
# one of these is not a single simple test/shell command.
_SHELL_SEPARATORS = (";", "&&", "||", "`", "$(", "&", ">", "<", "\n", "\r")
_SHELL_PIPE_INSTALL_RE = re.compile(r"\b(?:curl|wget)\b.*\|\s*sh\b", re.IGNORECASE)
_TEST_COMMAND_PREFIXES = (
    "pytest",
    "python -m pytest",
    "npm test",
    "pnpm test",
    "bun test",
    "uv run pytest",
)


def validate_tool_call(tool_name: str, tool_input: dict[str, Any]) -> tuple[bool, str | None]:
    """Return whether an executor tool call is allowed by the patch protocol."""

    if tool_name not in ALLOWED_PATCH_TOOL_NAMES:
        return False, "tool_not_allowed"

    if tool_name in {"edit_file", "apply_patch"}:
        path = tool_input.get("path")
        if not isinstance(path, str) or not path.strip():
            return False, "path_required"
        path_ok, reason = _validate_relative_path(path)
        if not path_ok:
            return False, reason

    if tool_name in {"shell", "run_tests"}:
        command = tool_input.get("command")
        if not isinstance(command, str) or not command.strip():
            return False, "command_required"
        if is_destructive_command(command):
            return False, "destructive_command"
        if not is_test_command(command):
            return False, "shell_command_not_allowed"

    return True, None


def normalize_patch_text(text: str) -> str:
    """Normalize model patch text into plain patch content."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = normalized.split("\n")

    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    cleaned = "\n".join(line.rstrip() for line in lines).strip("\n")
    if not cleaned:
        return ""
    return cleaned + "\n"


def is_test_command(command: str) -> bool:
    """Allow only standard test commands with optional arguments."""

    # Reject control chars / newlines on the RAW command before normalization —
    # _normalize_command collapses whitespace and would merge two lines into one
    # apparently-allowed prefix.
    if any(ch in command for ch in ("\n", "\r", "\x00")):
        return False

    normalized = _normalize_command(command)
    if not normalized or is_destructive_command(normalized):
        return False
    if any(separator in normalized for separator in _SHELL_SEPARATORS):
        return False
    if "|" in normalized:
        return False

    return any(
        normalized == prefix or normalized.startswith(f"{prefix} ")
        for prefix in _TEST_COMMAND_PREFIXES
    )


def is_destructive_command(command: str) -> bool:
    """Detect shell commands that should never be proxied to an executor."""

    normalized = _normalize_command(command).lower()
    if not normalized:
        return False
    if _SHELL_PIPE_INSTALL_RE.search(normalized):
        return True
    return any(pattern.lower() in normalized for pattern in DISALLOWED_SHELL_PATTERNS)


def _validate_relative_path(path: str) -> tuple[bool, str | None]:
    if path.startswith(("/", "~")):
        return False, "absolute_path_not_allowed"

    pure_path = PurePosixPath(path.replace("\\", "/"))
    if pure_path.is_absolute():
        return False, "absolute_path_not_allowed"
    if ".." in pure_path.parts:
        return False, "parent_path_not_allowed"
    return True, None


def _normalize_command(command: str) -> str:
    return " ".join(command.strip().split())
