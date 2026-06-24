#!/usr/bin/env python3
"""OpenAI-compatible LM Studio image smoke for local vision models."""

from __future__ import annotations

import json
import os
import sys
import urllib.request


PNG_1X1_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_VISION_PRO", "qwen/qwen3-vl-30b")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:1234/v1")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one short sentence."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{PNG_1X1_BASE64}"},
                    },
                ],
            }
        ],
        "stream": False,
        "temperature": 0,
        "max_tokens": 64,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise SystemExit("FAIL: empty vision response")
    print(f"PASS {model}: {content.strip()[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
