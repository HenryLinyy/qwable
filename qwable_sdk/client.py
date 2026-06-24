"""Qwable SDK client (sync + async).

Thin wrapper around the gateway HTTP API. Handles:
  - Preset resolution (quality / budget / coding / heavy / custom)
  - Non-streaming fusion_chat() → FusionResult
  - Streaming fusion_chat_stream() → iterator of FusionEvent
  - Async variant afusion_chat() / afusion_chat_stream()
  - list_presets() → introspection of gateway state

Usage:
    from qwable_sdk import LocalFusionClient, FusionPreset

    client = LocalFusionClient()
    presets = client.list_presets()
    print(presets["presets"]["quality"]["panel"])

    result = client.fusion_chat(
        messages=[{"role": "user", "content": "..."}],
        preset=FusionPreset.QUALITY,
    )
    print(result.text)
"""

import json
from typing import Any, Iterator, AsyncIterator, Optional

import httpx

from qwable_sdk.events import (
    ErrorEvent,
    FinalEvent,
    FusionEvent,
    JudgeEvent,
    PanelEvent,
)
from qwable_sdk.types import FusionPreset, FusionPresetName, FusionResult


# Default model aliases — can be overridden per call.
MODEL_OPENAI = "qwable-fusion"
MODEL_ANTHROPIC = "claude-qwable-fusion"


class LocalFusionClient:
    """Python client for Qwable Agent Gateway.

    Parameters:
        base_url: Gateway URL (default: http://127.0.0.1:8088)
        timeout: HTTP timeout in seconds (default: 300)
        max_tokens: Default max_tokens for fusion calls (default: 4000)
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8088",
        timeout: float = 300.0,
        max_tokens: int = 4000,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens

    # ─── Introspection ───────────────────────────────────────────────────

    def list_presets(self) -> dict:
        """GET /v1/fusion/presets — preset metadata + runtime state."""
        with httpx.Client(timeout=10.0) as c:
            r = c.get(f"{self.base_url}/v1/fusion/presets")
            r.raise_for_status()
            return r.json()

    def loaded_models(self) -> dict:
        """GET /v1/fusion/presets/loaded — lightweight state probe."""
        with httpx.Client(timeout=5.0) as c:
            r = c.get(f"{self.base_url}/v1/fusion/presets/loaded")
            r.raise_for_status()
            return r.json()

    def health(self) -> dict:
        """GET /health — health + last_used_model."""
        with httpx.Client(timeout=5.0) as c:
            r = c.get(f"{self.base_url}/health")
            r.raise_for_status()
            return r.json()

    # ─── Non-streaming fusion ─────────────────────────────────────────────

    def fusion_chat(
        self,
        *,
        messages: list[dict],
        preset: FusionPreset | str = FusionPreset.QUALITY,
        model: str = MODEL_OPENAI,
        analysis_models: list[str] | None = None,
        judge_model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> FusionResult:
        """Send a non-streaming fusion request and return the synthesized answer.

        Parameters:
            messages: OpenAI-style messages array
            preset: FusionPreset enum or string ("quality", "budget", "coding", "heavy")
            model: External model id (default: "qwable-fusion")
            analysis_models: Optional override for preset panel (required for CUSTOM preset)
            judge_model: Optional override for preset judge
            max_tokens: Default 4000 (reasoning models need headroom)
            temperature: Optional, default 0.3

        Returns:
            FusionResult with text, structured, trace, timing info

        Raises:
            httpx.HTTPStatusError: on 4xx/5xx from gateway
        """
        body = self._build_body(
            messages=messages,
            preset=preset,
            model=model,
            analysis_models=analysis_models,
            judge_model=judge_model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(
                f"{self.base_url}/v1/chat/completions",
                json=body,
            )
            r.raise_for_status()
            data = r.json()
        return self._parse_fusion_response(data)

    # ─── Streaming fusion (sync generator) ─────────────────────────────

    def fusion_chat_stream(
        self,
        *,
        messages: list[dict],
        preset: FusionPreset | str = FusionPreset.QUALITY,
        model: str = MODEL_OPENAI,
        analysis_models: list[str] | None = None,
        judge_model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[FusionEvent]:
        """Streaming variant of fusion_chat() — yields FusionEvent.

        Event types: panel_start, panel_token, panel_done, judge_start,
        judge_token, judge_done, final, error.

        Usage:
            for event in client.fusion_chat_stream(messages=[...], preset="budget"):
                if event.event == "judge_token":
                    print(event.judge.delta, end="", flush=True)
                elif event.event == "final":
                    print()  # newline
        """
        body = self._build_body(
            messages=messages,
            preset=preset,
            model=model,
            analysis_models=analysis_models,
            judge_model=judge_model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        with httpx.Client(timeout=self.timeout) as c:
            with c.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=body,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    ev = self._parse_sse_data_line(data)
                    if ev is not None:
                        yield ev

    # ─── Async variants ──────────────────────────────────────────────────

    async def afusion_chat(
        self,
        *,
        messages: list[dict],
        preset: FusionPreset | str = FusionPreset.QUALITY,
        model: str = MODEL_OPENAI,
        analysis_models: list[str] | None = None,
        judge_model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> FusionResult:
        """Async variant of fusion_chat()."""
        body = self._build_body(
            messages=messages,
            preset=preset,
            model=model,
            analysis_models=analysis_models,
            judge_model=judge_model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(
                f"{self.base_url}/v1/chat/completions",
                json=body,
            )
            r.raise_for_status()
            data = r.json()
        return self._parse_fusion_response(data)

    async def afusion_chat_stream(
        self,
        *,
        messages: list[dict],
        preset: FusionPreset | str = FusionPreset.QUALITY,
        model: str = MODEL_OPENAI,
        analysis_models: list[str] | None = None,
        judge_model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[FusionEvent]:
        """Async streaming variant."""
        body = self._build_body(
            messages=messages,
            preset=preset,
            model=model,
            analysis_models=analysis_models,
            judge_model=judge_model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            async with c.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=body,
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    ev = self._parse_sse_data_line(data)
                    if ev is not None:
                        yield ev

    # ─── Internal helpers ────────────────────────────────────────────────

    def _build_body(
        self,
        *,
        messages: list[dict],
        preset: FusionPreset | str,
        model: str,
        analysis_models: list[str] | None,
        judge_model: str | None,
        max_tokens: int | None,
        temperature: float | None,
        stream: bool,
    ) -> dict:
        """Build OpenAI Chat-compatible request body."""
        preset_str = preset.value if isinstance(preset, FusionPresetName) else str(preset)

        fusion_block: dict = {"preset": preset_str}
        if analysis_models is not None:
            fusion_block["analysis_models"] = analysis_models
        if judge_model is not None:
            fusion_block["judge_model"] = judge_model

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "fusion": fusion_block,
            "stream": stream,
        }
        body["max_tokens"] = max_tokens if max_tokens is not None else self.max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        return body

    def _parse_fusion_response(self, data: dict) -> FusionResult:
        """Parse non-streaming /v1/chat/completions response into FusionResult."""
        # OpenAI chat format: choices[0].message.content
        text = ""
        if data.get("choices"):
            msg = data["choices"][0].get("message", {})
            text = msg.get("content", "") or ""

        # Trace info — if available
        # (current gateway doesn't return full trace in non-streaming response;
        # this is parsed best-effort)
        return FusionResult(
            text=text,
            preset="",  # not echoed back in non-streaming
            total_latency_ms=0,
        )

    def _parse_sse_data_line(self, data: dict) -> FusionEvent | None:
        """Parse one SSE 'data:' line into a FusionEvent.

        OpenAI chat format: {choices: [{delta: {content: "..."}}]}
        Fusion SSE comments: {starts with ': fusion ...'} (we skip those — they're
        for debugging only).
        """
        # Skip SSE comments (lines starting with ':') — not parsed here because
        # the gateway emits them as separate events. The SDK receives them but
        # skips because they don't have a `choices` key.
        choices = data.get("choices")
        if choices:
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            finish = choice.get("finish_reason")
            if content:
                # Treat judge_token as the default token event.
                # (panel_token SSE comments are filtered by curl/httpx because
                # they don't have the data shape; comment lines start with ':'
                # and are ignored by OpenAI clients.)
                return FusionEvent(
                    event="judge_token",
                    data={"delta": content},
                    judge=JudgeEvent(
                        event="judge_token",
                        delta=content,
                    ),
                )
            if finish:
                return FusionEvent(
                    event="judge_done",
                    data={"finish_reason": finish},
                    judge=JudgeEvent(event="judge_done", finish_reason=finish),
                )
        return None
