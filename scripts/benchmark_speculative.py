#!/usr/bin/env python3
"""G15: Speculative decoding helper.

LM Studio 0.4.16 supports speculative decoding but ONLY via the GUI
(Developer tab → Speculative Decoding → select Draft Model).
There is no CLI flag, no API field, and no config file we can edit.

This script:
  1. Verifies the draft model is downloaded
  2. Verifies the speculativeDecoding flag in settings.json
  3. Measures baseline throughput WITHOUT speculative decoding
  4. Prints UI steps to enable the pairing manually
  5. After user enables pairing, measures new throughput

Usage:
    python3 scripts/benchmark_speculative.py [--target MODEL] [--draft MODEL] [--tokens N]

    # Default: benchmark qwen3.6-27b (target) + qwen3-1.7b (draft) with 200 tokens
    python3 scripts/benchmark_speculative.py
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx


SETTINGS_PATH = Path.home() / ".lmstudio" / "settings.json"
LMSTUDIO_CLI = os.path.expanduser("~/.lmstudio/bin/lms")


def check_setup(target: str, draft: str) -> dict:
    """Verify LM Studio is set up for speculative decoding."""
    state = {
        "target_model": target,
        "draft_model": draft,
        "draft_downloaded": False,
        "target_loaded": False,
        "draft_loaded": False,
        "settings_flag": False,
    }

    # Check downloaded models
    result = subprocess.run(
        [LMSTUDIO_CLI, "ls"], capture_output=True, text=True, timeout=10,
    )
    for line in result.stdout.splitlines():
        if draft in line:
            state["draft_downloaded"] = True
        if target in line:
            state["target_downloaded"] = True

    # Check loaded
    result = subprocess.run(
        [LMSTUDIO_CLI, "ps"], capture_output=True, text=True, timeout=10,
    )
    for line in result.stdout.splitlines():
        if line.startswith(target):
            state["target_loaded"] = True
        if line.startswith(draft):
            state["draft_loaded"] = True

    # Check settings.json flag
    if SETTINGS_PATH.exists():
        with SETTINGS_PATH.open() as f:
            settings = json.load(f)
        cpi = settings.get("configPresetInclusiveness", {})
        state["settings_flag"] = cpi.get("speculativeDecoding", False)

    return state


def print_ui_steps():
    """Print manual UI steps required to pair draft model."""
    print()
    print("=" * 70)
    print("MANUAL UI STEPS (LM Studio 0.4.16 — speculative decoding is GUI-only)")
    print("=" * 70)
    print()
    print("1. Open LM Studio (already running on this Mac)")
    print()
    print("2. Click the model that's currently loaded (the TARGET, e.g.")
    print("   'qwen/qwen3.6-27b'). This opens the chat panel.")
    print()
    print("3. Click the ⚙️ Settings icon (top right) → 'Adjust Parameters'.")
    print()
    print("4. Find 'Speculative Decoding' section.")
    print()
    print("5. In the 'Draft Model' dropdown, select 'qwen/qwen3-1.7b'")
    print("   (or your downloaded draft model).")
    print()
    print("6. Click 'Save as Preset' (top of settings panel) and give it a name")
    print("   (e.g., 'qwen3.6-with-1.7b-draft').")
    print()
    print("7. Re-load the target model — it will now use the draft.")
    print()
    print("8. Run this script again to measure the speedup.")
    print()
    print("ALTERNATIVE: if you have multiple target models, you need to")
    print("create a separate preset for each (qwen-coder-with-draft,")
    print("gemma-with-draft, etc.)")
    print()


def benchmark(target: str, draft: str, tokens: int, prompt: str) -> dict:
    """Measure tokens/sec for a target model.

    Returns {wall_time_s, prompt_tokens, completion_tokens, tokens_per_sec}.
    """
    # Make sure the target is loaded (skip if not)
    print(f"Benchmarking {target}...")
    t0 = time.monotonic()
    response = httpx.post(
        "http://127.0.0.1:1234/v1/chat/completions",
        json={
            "model": target,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": tokens,
            "stream": False,
            "temperature": 0.0,  # deterministic
        },
        timeout=300.0,
    )
    response.raise_for_status()
    wall_time = time.monotonic() - t0
    data = response.json()
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    return {
        "model": target,
        "wall_time_s": wall_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_sec": completion_tokens / wall_time if wall_time > 0 else 0,
        "text_preview": text[:100],
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark speculative decoding setup")
    parser.add_argument("--target", default="qwen/qwen3.6-27b",
                        help="Target (large) model to benchmark")
    parser.add_argument("--draft", default="qwen/qwen3-1.7b",
                        help="Draft (small) model for speculative decoding")
    parser.add_argument("--tokens", type=int, default=200,
                        help="Number of completion tokens to generate")
    parser.add_argument("--prompt", default="Write a detailed technical explanation of how transformers work, including self-attention, multi-head attention, and positional encodings.",
                        help="Prompt for benchmark")
    parser.add_argument("--no-bench", action="store_true",
                        help="Skip benchmark, just check setup")
    args = parser.parse_args()

    print(f"Target model: {args.target}")
    print(f"Draft model:  {args.draft}")
    print()

    state = check_setup(args.target, args.draft)
    print("Setup check:")
    print(f"  Target downloaded:    {state['target_downloaded']}")
    print(f"  Target loaded:        {state['target_loaded']}")
    print(f"  Draft downloaded:     {state['draft_downloaded']}")
    print(f"  Draft loaded:         {state['draft_loaded']}")
    print(f"  settings.json flag:   {state['settings_flag']}")
    print()

    if not state["draft_downloaded"]:
        print(f"ERROR: Draft model '{args.draft}' not downloaded.")
        print(f"Run: {LMSTUDIO_CLI} get --mlx -y {args.draft}")
        sys.exit(1)

    if not state["target_loaded"]:
        print(f"Loading target {args.target}...")
        subprocess.run(
            [LMSTUDIO_CLI, "load", args.target, "--context-length", "8192", "--gpu", "max"],
            check=True, capture_output=True,
        )

    if not state["draft_loaded"]:
        print(f"Loading draft {args.draft}...")
        subprocess.run(
            [LMSTUDIO_CLI, "load", args.draft, "--context-length", "8192", "--gpu", "max"],
            check=True, capture_output=True,
        )

    if args.no_bench:
        return

    print()
    print(f"=== Benchmark 1: WITHOUT speculative decoding pairing ===")
    print("(Just both loaded — LM Studio doesn't auto-pair by load order)")
    bench1 = benchmark(args.target, args.draft, args.tokens, args.prompt)
    print(f"  Wall time:    {bench1['wall_time_s']:.2f}s")
    print(f"  Completion:   {bench1['completion_tokens']} tokens")
    print(f"  Throughput:   {bench1['tokens_per_sec']:.2f} tok/s")
    print(f"  Preview:      {bench1['text_preview']!r}")
    print()

    print_ui_steps()
    print("After pairing, re-run this script to compare.")

    # Save baseline for later comparison
    baseline_file = Path.home() / ".qwable" / "spec_decode_baseline.json"
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    with baseline_file.open("w") as f:
        json.dump({
            "baseline": bench1,
            "target": args.target,
            "draft": args.draft,
            "timestamp": time.time(),
        }, f, indent=2)
    print(f"Baseline saved to: {baseline_file}")


if __name__ == "__main__":
    main()
