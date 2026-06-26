"""G15: LM Studio MLX optimizations enablement.

Applies the following optimizations to ~/.lmstudio/settings.json:
- Bump defaultContextLength to 32768 (was 8192) for longer fusion prompts
- Enable speculativeDecoding in configPresetInclusiveness (gated on having a draft model)
- Document other runtime optimizations that are already on by default in 0.4.16+

Idempotent: safe to run multiple times.

Note: speculative decoding requires a separate draft model to be downloaded
and configured. If no draft model is available, the flag has no effect.
We enable the flag defensively so it activates as soon as a draft model is added.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger("qwable.mlx_optimizations")


SETTINGS_PATH = Path.home() / ".lmstudio" / "settings.json"


def _lms_cli() -> str:
    """Resolve the LM Studio CLI path: env override, else the standard location
    under the current user's home (portable across machines)."""
    return os.environ.get("LMSTUDIO_CLI_PATH") or str(
        Path.home() / ".lmstudio" / "bin" / "lms"
    )


# Recommended defaults for M5 Max 128GB running fusion deliberation
RECOMMENDED_CONTEXT_LENGTH = 32768  # 32K — supports long fusion prompts + history
RECOMMENDED_PARALLEL = 2  # 2 concurrent predictions (fusion uses global lock anyway)


def backup_settings() -> Path | None:
    """Create a timestamped backup of settings.json (only if file exists)."""
    if not SETTINGS_PATH.exists():
        return None
    import time

    backup = SETTINGS_PATH.with_suffix(f".json.bak.{int(time.time())}")
    shutil.copy2(SETTINGS_PATH, backup)
    logger.info("backed up settings.json to %s", backup)
    return backup


def apply_recommended_settings(
    *,
    context_length: int = RECOMMENDED_CONTEXT_LENGTH,
    enable_speculative: bool = True,
    dry_run: bool = False,
) -> dict:
    """Apply recommended LM Studio MLX optimization settings.

    Returns the changes that were (or would be) applied.
    """
    changes: dict = {
        "context_length": {"old": None, "new": context_length},
        "speculative_decoding": {"old": None, "new": enable_speculative},
    }

    if not SETTINGS_PATH.exists():
        logger.warning("settings.json not found at %s", SETTINGS_PATH)
        return changes

    with SETTINGS_PATH.open(encoding="utf-8") as f:
        settings = json.load(f)

    # Bump defaultContextLength
    dcl = settings.get("defaultContextLength", {})
    if isinstance(dcl, dict):
        changes["context_length"]["old"] = dcl.get("value")
        dcl["value"] = context_length
        dcl["type"] = "custom"
    else:
        # Old format — replace
        changes["context_length"]["old"] = dcl
        settings["defaultContextLength"] = {
            "type": "custom",
            "value": context_length,
        }

    # Enable speculative decoding flag (no-op without a draft model)
    cpi = settings.get("configPresetInclusiveness")
    if cpi is None:
        cpi = {}
        settings["configPresetInclusiveness"] = cpi
    if isinstance(cpi, dict):
        changes["speculative_decoding"]["old"] = cpi.get("speculativeDecoding")
        cpi["speculativeDecoding"] = enable_speculative

    if not dry_run:
        with SETTINGS_PATH.open("w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        logger.info("applied settings: %s", changes)

    return changes


def get_current_optimizations() -> dict:
    """Read current LM Studio optimization state for /v1/system/optimizations endpoint."""
    info: dict = {
        "lmstudio_version": "unknown",
        "context_length": None,
        "speculative_decoding": False,
        "current_loaded_models": [],
        "active_runtime": None,
        "draft_model_configured": False,
    }

    # Version
    try:
        out = subprocess.run(
            [_lms_cli(), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        info["lmstudio_version"] = out.stdout.strip()
    except Exception as exc:
        info["lmstudio_version_error"] = str(exc)

    # Loaded models
    try:
        out = subprocess.run(
            [_lms_cli(), "ps"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in out.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 4 and "/" in parts[0]:
                info["current_loaded_models"].append(
                    {
                        "identifier": parts[0],
                        "model": parts[1] if len(parts) > 1 else "?",
                        "status": parts[2] if len(parts) > 2 else "?",
                        "size_gb": parts[3] if len(parts) > 3 else "?",
                    }
                )
    except Exception as exc:
        info["loaded_models_error"] = str(exc)

    # Settings
    if SETTINGS_PATH.exists():
        try:
            with SETTINGS_PATH.open(encoding="utf-8") as f:
                settings = json.load(f)
            dcl = settings.get("defaultContextLength", {})
            if isinstance(dcl, dict):
                info["context_length"] = dcl.get("value")
            cpi = settings.get("configPresetInclusiveness", {})
            if isinstance(cpi, dict):
                info["speculative_decoding"] = cpi.get("speculativeDecoding", False)
        except Exception as exc:
            info["settings_error"] = str(exc)

    # Active runtime
    try:
        out = subprocess.run(
            [_lms_cli(), "runtime", "ls"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in out.stdout.splitlines():
            if "✓" in line:
                info["active_runtime"] = (
                    line.split("@")[0].strip() if "@" in line else line.strip()
                )
                break
    except Exception:
        pass

    return info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("LM Studio MLX optimizations — G15")
    print(f"Settings: {SETTINGS_PATH}")
    print()
    if not SETTINGS_PATH.exists():
        print("ERROR: settings.json not found — is LM Studio installed?")
        raise SystemExit(1)
    backup = backup_settings()
    if backup:
        print(f"Backed up to: {backup}")
    changes = apply_recommended_settings()
    print(f"Changes applied: {json.dumps(changes, indent=2)}")
    print()
    print("Restart LM Studio server to apply: ~/.lmstudio/bin/lms server restart")
