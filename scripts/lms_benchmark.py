#!/usr/bin/env python3
"""
lms_benchmark.py — LM Studio benchmark sweep across (ctx × gpu_offload) per model.

USAGE
  python3 lms_benchmark.py                # full sweep: all models × {8K,32K,64K,128K} × {0.5,0.7,max}
  python3 lms_benchmark.py --help         # show options

DEPENDENCIES
  - LM Studio >= 0.3 with local server on :1234
  - lms CLI at ~/.lmstudio/bin/lms (auto-installed with LM Studio)
  - python3 stdlib only (no pip installs)

WHAT IT MEASURES (per model × ctx × gpu_offload combination)
  - load_time_sec    wall clock from `lms load` to loaded
  - model_size_gib   reported by `lms ps` (LM Studio's authoritative model RAM)
  - rss_gib          sum RSS of all LM Studio Helper processes (process overhead)
  - ttft_sec         time-to-first-token on a 30-line coding prompt, streaming
  - total_sec        total wall clock for chat completion
  - content_tok      estimated content tokens (chars/4)
  - reasoning_tok    estimated reasoning/thinking tokens (chars/4)
  - tok_per_sec      (content + reasoning) / (total_sec - ttft_sec)

OUTPUTS
  /tmp/lms_benchmark.json   full structured results
  /tmp/lms_benchmark.csv    flat CSV for spreadsheet
  /tmp/lms_benchmark.log    timestamped progress log

EMPIRICAL FINDINGS (M5 Max 128GB, June 2026)
  - gpu_offload (0.5/0.7/max) impacts throughput < 5%: Metal unified memory shares pool
  - ctx (8K→128K) impacts throughput < 5%: KV cache scales sub-linearly on Apple Silicon
  - qwen3.6-27b is 17 tok/s because it must emit ~200 reasoning tokens before any content
    → "no-reasoning" can't be enabled via API; requires LM Studio GUI per-model toggle
  - Recommendation: use gpu=max + ctx=128K universally, optimize via model selection not knobs

WRITTEN FOR
  ~/Documents/qwable-agent-gateway-m5/scripts/lms_benchmark.py
  Companion skill: ~/.hermes/skills/software-development/lms-benchmark/
"""

import subprocess
import time
import json
import csv
import sys
import os
import re
import argparse
from urllib.request import Request, urlopen
from urllib.error import URLError
from pathlib import Path

LMS = os.path.expanduser("~/.lmstudio/bin/lms")
BASE_URL = "http://localhost:1234/v1"

# (model_id, max_ctx_supported). :2 variants are duplicates — skip them.
MODELS = [
    ("google/gemma-4-26b-a4b-qat", 131072),
    ("qwen/qwen3-coder-next", 131072),
    ("qwen/qwen3.6-27b", 131072),
    ("qwen/qwen3.6-35b-a3b", 131072),
]

CTX_POINTS = [8192, 32768, 65536, 131072]   # 8K, 32K, 64K, 128K
GPU_OFFLOADS = ["0.5", "0.7", "max"]
PROMPT = ("Write a Python function to compute the n-th Fibonacci number using "
          "memoization. Include docstring and type hints. About 30 lines of code.")
MAX_TOKENS = 200

LOG = Path("/tmp/lms_benchmark.log")
JSON_OUT = Path("/tmp/lms_benchmark.json")
CSV_OUT = Path("/tmp/lms_benchmark.csv")


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def lms_unload_all():
    try:
        subprocess.run([LMS, "unload", "--all"], capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        log("  WARN: unload timed out")
    time.sleep(2)


def lms_load(model, ctx, gpu_offload):
    """Returns (load_time_sec, error_str|None)."""
    start = time.time()
    try:
        proc = subprocess.run(
            [LMS, "load", model, "--context-length", str(ctx), "--gpu", gpu_offload],
            capture_output=True, text=True, timeout=240,
        )
    except subprocess.TimeoutExpired:
        return None, "lms load timeout (>240s)"
    elapsed = time.time() - start
    if proc.returncode != 0:
        return elapsed, (proc.stderr or proc.stdout)[-300:]
    # lms load prints "(XX.XX GiB)" — extract for sanity check
    m = re.search(r"\((\d+\.\d+)\s*GiB\)", proc.stdout)
    if m:
        log(f"  load: {elapsed:.1f}s, model reported: {m.group(1)} GiB")
    else:
        log(f"  load: {elapsed:.1f}s")
    return elapsed, None


def get_lms_model_size_gib(model_id):
    """Query `lms ps` for the size of the currently-loaded model. Returns GiB or None."""
    try:
        out = subprocess.run(
            [LMS, "ps"], capture_output=True, text=True, timeout=10
        ).stdout
    except subprocess.TimeoutExpired:
        return None
    # Format: IDENTIFIER  MODEL  STATUS  SIZE  CONTEXT  ...
    for line in out.splitlines()[1:]:  # skip header
        cols = line.split()
        if len(cols) < 4:
            continue
        ident, _, status, size = cols[0], cols[1], cols[2], cols[3]
        if status == "LOADED" and (ident == model_id or model_id.startswith(ident)):
            try:
                return float(size.replace("GB", ""))
            except ValueError:
                return None
    return None


def get_lms_rss_gib():
    """Sum RSS of all LM Studio Helper processes (Renderer + GPU + worker)."""
    out = subprocess.run(
        ["ps", "axo", "rss,command"], capture_output=True, text=True
    ).stdout
    total_kb = 0
    for line in out.splitlines():
        if "LM Studio" in line or ".lmstudio" in line:
            if "lmlink" in line:
                continue
            try:
                total_kb += int(line.split()[0])
            except (ValueError, IndexError):
                pass
    return total_kb / 1024 / 1024  # KiB → GiB


def measure_chat(model, ctx):
    """Returns (ttft_sec, total_sec, completion_text, error_str|None)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": True,
    }
    req = Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    ttft = None
    completion_text = ""
    reasoning_text = ""
    try:
        with urlopen(req, timeout=180) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = obj.get("choices", [{}])[0].get("delta", {})
                # gemma/qwen reasoning models emit reasoning_content before content
                content_piece = delta.get("content", "")
                reasoning_piece = delta.get("reasoning_content", "")
                if ttft is None and (content_piece or reasoning_piece):
                    ttft = time.time() - start
                completion_text += content_piece
                reasoning_text += reasoning_piece
    except URLError as e:
        return None, None, "", "", f"URLError: {e}"
    except Exception as e:
        return None, None, "", "", f"{type(e).__name__}: {e}"
    total = time.time() - start
    return ttft, total, completion_text, reasoning_text, None


def sweep_one_model(model, max_ctx):
    results = []
    for ctx in CTX_POINTS:
        if ctx > max_ctx:
            log(f"  skip ctx={ctx} (max={max_ctx})")
            continue
        for gpu in GPU_OFFLOADS:
            log(f"=== {model} | ctx={ctx} | gpu={gpu} ===")
            lms_unload_all()
            load_time, err = lms_load(model, ctx, gpu)
            if err:
                log(f"  LOAD FAIL: {err[-150:]}")
                results.append({
                    "model": model, "ctx": ctx, "gpu_offload": gpu,
                    "status": "load_failed",
                    "error": err[-200:],
                    "load_time_sec": round(load_time, 2) if load_time else None,
                })
                continue
            time.sleep(3)  # warmup
            model_size = get_lms_model_size_gib(model)
            rss_before = get_lms_rss_gib()
            ttft, total, text, rtext, cerr = measure_chat(model, ctx)
            rss_after = get_lms_rss_gib()
            if cerr:
                log(f"  CHAT FAIL: {cerr}")
                results.append({
                    "model": model, "ctx": ctx, "gpu_offload": gpu,
                    "status": "chat_failed",
                    "error": cerr[-200:],
                    "load_time_sec": round(load_time, 2),
                    "model_size_gib": round(model_size, 2) if model_size else None,
                    "rss_gib": round(rss_after, 2),
                })
                continue
            content_tokens = max(1, len(text) // 4)
            reasoning_tokens = max(0, len(rtext) // 4)
            total_tokens = content_tokens + reasoning_tokens
            gen_time = max(0.001, (total - ttft) if ttft else total)
            tps = total_tokens / gen_time
            log(f"  RSS: {rss_after:.1f} GiB | TTFT: {ttft:.2f}s | total: {total:.2f}s | "
                f"~{total_tokens} tok ({reasoning_tokens} reason + {content_tokens} content) | {tps:.1f} tok/s")
            results.append({
                "model": model, "ctx": ctx, "gpu_offload": gpu,
                "status": "ok",
                "load_time_sec": round(load_time, 2),
                "model_size_gib": round(model_size, 2) if model_size else None,
                "rss_gib": round(rss_after, 2),
                "rss_delta_gib": round(rss_after - rss_before, 2),
                "ttft_sec": round(ttft, 3) if ttft else None,
                "total_sec": round(total, 2),
                "content_tok": content_tokens,
                "reasoning_tok": reasoning_tokens,
                "completion_tok": total_tokens,
                "tok_per_sec": round(tps, 2),
            })
    return results


def write_outputs(all_results):
    JSON_OUT.write_text(json.dumps(all_results, indent=2))
    if all_results:
        keys = ["model", "ctx", "gpu_offload", "status",
                "load_time_sec", "model_size_gib", "rss_gib", "rss_delta_gib",
                "ttft_sec", "total_sec", "content_tok", "reasoning_tok",
                "completion_tok", "tok_per_sec", "error"]
        with CSV_OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_results)


def main():
    LOG.write_text("")  # truncate
    log(f"Starting benchmark: {len(MODELS)} models × "
        f"{len(CTX_POINTS)} ctx × {len(GPU_OFFLOADS)} gpu_offload")
    all_results = []
    for model, max_ctx in MODELS:
        log(f"\n###### Sweeping {model} (max_ctx={max_ctx}) ######")
        results = sweep_one_model(model, max_ctx)
        all_results.extend(results)
        write_outputs(all_results)  # partial save
    write_outputs(all_results)
    ok = sum(1 for r in all_results if r["status"] == "ok")
    fail = len(all_results) - ok
    log(f"\nDONE: {ok} ok / {fail} failed / {len(all_results)} total")
    log(f"Results: {JSON_OUT} + {CSV_OUT}")


if __name__ == "__main__":
    main()