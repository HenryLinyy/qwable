"""G10: Serial deliberation runner.

Public surface:
  - run_panel_serial(...): run analysis models serially with load → chat → unload
  - run_fusion_agent(...): top-level entry, runs panel + judge + parse + return
  - run_fusion_agent_streaming(...): G11 streaming entry, yields FusionStreamEvent

Design notes:
  - Serial execution (not parallel) to fit M5 Max 128 GB; models are 16-66 GB each.
  - Every panel call is followed by an unload attempt (even on error) so the
    LM Studio `lms unload --all` invariant holds (only one resident at a time).
  - Judge backend is decided by comparing `preset.judge_model` against
    `ds4_model` from config: match → DS4Client, mismatch → OllamaClient.
  - Streaming runner yields FusionStreamEvent at each panel boundary and
    per-token during judge synthesis (G11 MLX optimization).
"""

import asyncio
import logging
import time
from typing import Any, AsyncIterator

from qwable.fusion_presets import FusionPreset
from qwable.fusion_schemas import PanelResponse, SynthesisInput
from qwable.fusion_synthesis import (
    build_synthesis_prompt,
    parse_structured_output,
)
from qwable.prompts import FUSION_AGENT_ANALYSIS_SYSTEM
from qwable.fusion_retry import chat_with_retry
from qwable.streaming_events import (
    FUSION_STREAM_EVENT_FINAL,
    FUSION_STREAM_EVENT_JUDGE_DONE,
    FUSION_STREAM_EVENT_JUDGE_START,
    FUSION_STREAM_EVENT_JUDGE_TOKEN,
    FUSION_STREAM_EVENT_PANEL_DONE,
    FUSION_STREAM_EVENT_PANEL_START,
    FUSION_STREAM_EVENT_PANEL_TOKEN,
    FusionStreamEvent,
)

logger = logging.getLogger("qwable.fusion_deliberation")


def _extract_assistant_text(chat_response: dict) -> tuple[str, str]:
    """Return (text, finish_reason) from an OpenAI-shaped chat response."""
    try:
        choice = chat_response["choices"][0]
    except (KeyError, IndexError, TypeError):
        return "", "error"
    message = choice.get("message") or {}
    return message.get("content", "") or "", choice.get(
        "finish_reason", "stop"
    ) or "stop"


def run_panel_serial(
    *,
    preset: FusionPreset,
    original_prompt: str,
    system_prompt: str,
    panel_client: Any,
    panel_max_tokens: int,
    temperature: float,
    keep_last_resident: bool = False,
) -> list[PanelResponse]:
    """Run each analysis model serially: chat → unload (except last) → next.

    Errors are captured per-model in PanelResponse.error; the runner continues
    to subsequent models. Unload is attempted even on error so we always leave
    the LM Studio backend with at most one resident model.

    G12-3: When `keep_last_resident=True`, the final panel model is NOT
    unloaded — it stays resident for fast follow-up requests (LM Studio
    TTL=1h handles eventual eviction). Saves ~30s on the next fusion call.
    """
    responses: list[PanelResponse] = []
    n = len(preset.analysis_models)
    for idx, model_id in enumerate(preset.analysis_models):
        is_last = idx == n - 1
        t0 = time.monotonic()
        text = ""
        finish_reason = "stop"
        error: str | None = None
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": original_prompt},
            ]
            chat_response = panel_client.chat_completion(
                model=model_id,
                messages=messages,
                max_tokens=panel_max_tokens,
                stream=False,
                temperature=temperature,
            )
            text, finish_reason = _extract_assistant_text(chat_response)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            finish_reason = "error"
            logger.warning("fusion panel model %s failed: %s", model_id, error)
        latency_ms = int((time.monotonic() - t0) * 1000)
        responses.append(
            PanelResponse(
                model_id=model_id,
                text=text,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                error=error,
            )
        )
        # Skip unload for the last model when keep_last_resident is set
        if is_last and keep_last_resident:
            continue
        # Unload — always, even on error
        try:
            panel_client.unload_models([model_id])
        except Exception as exc:
            logger.warning("fusion unload after %s failed: %s", model_id, exc)
    return responses


async def run_fusion_agent(
    *,
    ollama_client: Any,
    ds4_client: Any,
    preset: FusionPreset,
    original_prompt: str,
    panel_max_tokens: int,
    judge_max_tokens: int,
    ds4_model: str,
    temperature: float = 0.3,
    keep_last_resident: bool = True,
    judge_fallback_chain: list[str] | None = None,
) -> dict:
    """Run full fusion deliberation: panel → judge → parse → return.

    Returns dict with keys:
      - text: final answer text (from parsed Final Answer section)
      - structured: FusionStructuredOutput from parse_structured_output
      - trace: dict for FusionAction.trace
      - panel_responses: list[PanelResponse]
      - total_latency_ms: int
    """
    t_total = time.monotonic()

    # Offload the blocking panel (N synchronous model calls) to a thread so the
    # asyncio event loop stays responsive during deliberation.
    panel_responses = await asyncio.to_thread(
        lambda: run_panel_serial(
            preset=preset,
            original_prompt=original_prompt,
            system_prompt=FUSION_AGENT_ANALYSIS_SYSTEM,
            panel_client=ollama_client,
            panel_max_tokens=panel_max_tokens,
            temperature=temperature,
            keep_last_resident=True,  # G12-3
        )
    )

    synthesis_input = SynthesisInput(
        original_prompt=original_prompt,
        panel_responses=panel_responses,
        preset_name=preset.name,
    )
    judge_system, judge_user = build_synthesis_prompt(synthesis_input)

    judge_messages = [
        {"role": "system", "content": judge_system},
        {"role": "user", "content": judge_user},
    ]

    is_ds4_judge = preset.judge_model == ds4_model
    judge_text = ""

    # G12-5: build judge candidate list = [primary] + fallback chain
    # When primary judge fails, try fallbacks in order.
    # DS4 is only tried first if primary judge IS ds4 (no fallback needed
    # across backends — that would lose context).
    if is_ds4_judge:
        judge_candidates: list[tuple[str, str]] = [(preset.judge_model, "ds4")]
    else:
        # Build ollama fallback chain (filter out the primary judge).
        chain = [
            m for m in (judge_fallback_chain or []) if m and m != preset.judge_model
        ]
        judge_candidates = [(preset.judge_model, "ollama")]
        for fb in chain:
            judge_candidates.append((fb, "ollama"))

    # Try each candidate in order until one succeeds
    judge_backend = "ollama"
    for judge_model_id, backend in judge_candidates:
        try:
            if backend == "ds4":
                judge_response = await asyncio.to_thread(
                    lambda: ds4_client.chat_completion(
                        model=ds4_model,
                        messages=judge_messages,
                        max_tokens=judge_max_tokens,
                        stream=False,
                        temperature=temperature,
                    )
                )
            else:
                # G13-3: wrap with retry for transient LM Studio errors.
                # Resolve retry config at call time (don't fail at import if
                # fusion_core not yet wired into server module).
                try:
                    import qwable.server as _server_mod

                    retry_cfg = getattr(
                        getattr(_server_mod, "config", None),
                        "fusion_max_retries",
                        0,
                    )
                    retry_delay = getattr(
                        getattr(_server_mod, "config", None),
                        "fusion_retry_base_delay",
                        1.0,
                    )
                except Exception:
                    retry_cfg, retry_delay = 0, 1.0
                judge_response = await asyncio.to_thread(
                    lambda: chat_with_retry(
                        lambda: ollama_client.chat_completion(
                            model=judge_model_id,
                            messages=judge_messages,
                            max_tokens=judge_max_tokens,
                            stream=False,
                            temperature=temperature,
                        ),
                        max_retries=retry_cfg,
                        base_delay=retry_delay,
                    )
                )
            # Extract for BOTH backends — previously this only ran on the ollama
            # branch, so the ds4 judge path raised UnboundLocalError (swallowed),
            # making the heavy preset always return the judge-error string.
            candidate_text, _ = _extract_assistant_text(judge_response)
            if candidate_text and candidate_text.strip():
                judge_text = candidate_text
                judge_backend = backend
                logger.info("fusion judge %s succeeded", judge_model_id)
                break
            else:
                logger.warning(
                    "fusion judge %s returned empty text, trying fallback",
                    judge_model_id,
                )
        except Exception as exc:
            logger.warning(
                "fusion judge %s (%s) failed: %s — trying fallback",
                judge_model_id,
                backend,
                exc,
            )
            continue
    else:
        # All candidates failed
        logger.error("fusion judge: all candidates failed")
        judge_text = "[fusion judge error: all fallback candidates failed]"
        judge_backend = judge_candidates[-1][1] if judge_candidates else "unknown"

    # G12-3: unload judge for clean state ONLY if not keeping last resident
    # and the winning judge was ollama-based (can't unload ds4)
    if not keep_last_resident and judge_backend == "ollama":
        try:
            ollama_client.unload_models([preset.judge_model])
        except Exception as exc:
            logger.warning("fusion judge unload failed: %s", exc)

    structured = parse_structured_output(judge_text)
    if structured.had_fallback:
        logger.warning(
            "fusion judge %s returned non-structured output (had_fallback=True)",
            preset.judge_model,
        )

    total_latency_ms = int((time.monotonic() - t_total) * 1000)

    trace = {
        "preset": preset.name,
        "panel_responses": [
            {
                "model_id": r.model_id,
                "latency_ms": r.latency_ms,
                "finish_reason": r.finish_reason,
                "error": r.error,
                "text_preview": (r.text or "")[:120],
            }
            for r in panel_responses
        ],
        "judge_model": preset.judge_model,
        "judge_backend": judge_backend,
        "judge_text_preview": (judge_text or "")[:240],
        "structured_had_fallback": structured.had_fallback,
        "total_latency_ms": total_latency_ms,
    }

    final_text = structured.final_answer or judge_text or "[fusion produced no output]"

    return {
        "text": final_text,
        "structured": structured,
        "trace": trace,
        "panel_responses": panel_responses,
        "total_latency_ms": total_latency_ms,
    }


# ─── G11: streaming runner ──────────────────────────────────────────────


async def _run_panel_serial_streaming(
    *,
    preset: FusionPreset,
    original_prompt: str,
    panel_client: Any,
    panel_max_tokens: int,
    temperature: float,
    keep_last_resident: bool = False,
) -> AsyncIterator[FusionStreamEvent | PanelResponse]:
    """Async generator yielding FusionStreamEvent + PanelResponse per model.

    Each model invocation yields:
      1. FusionStreamEvent(panel_start, {model_id, index})
      2. FusionStreamEvent(panel_done, {model_id, latency_ms, ...})
      3. PanelResponse (for the runner's internal use)
    """
    loop = asyncio.get_event_loop()
    n = len(preset.analysis_models)
    for index, model_id in enumerate(preset.analysis_models):
        is_last = index == n - 1
        yield FusionStreamEvent(
            event=FUSION_STREAM_EVENT_PANEL_START,
            data={"model_id": model_id, "index": index},
        )
        t0 = time.monotonic()
        text = ""
        finish_reason = "stop"
        error: str | None = None
        # G12-1: try streaming first; fall back to non-streaming chat_completion
        # if client doesn't support chat_completion_stream (older clients).
        stream_chunks: list[str] = []
        try:
            if hasattr(panel_client, "chat_completion_stream"):

                def _drain_stream(mid=model_id):
                    chunks = []
                    for delta, finish in panel_client.chat_completion_stream(
                        model=mid,
                        messages=[
                            {"role": "system", "content": FUSION_AGENT_ANALYSIS_SYSTEM},
                            {"role": "user", "content": original_prompt},
                        ],
                        max_tokens=panel_max_tokens,
                        temperature=temperature,
                    ):
                        chunks.append((delta, finish))
                    return chunks

                chunks = await loop.run_in_executor(None, _drain_stream)
                for delta, finish in chunks:
                    if delta:
                        stream_chunks.append(delta)
                        yield FusionStreamEvent(
                            event=FUSION_STREAM_EVENT_PANEL_TOKEN,
                            data={"model_id": model_id, "index": index, "delta": delta},
                        )
                    if finish:
                        finish_reason = finish
                text = "".join(stream_chunks)
            else:
                # Fallback to non-streaming chat
                def _chat_call(mid=model_id):
                    return panel_client.chat_completion(
                        model=mid,
                        messages=[
                            {"role": "system", "content": FUSION_AGENT_ANALYSIS_SYSTEM},
                            {"role": "user", "content": original_prompt},
                        ],
                        max_tokens=panel_max_tokens,
                        stream=False,
                        temperature=temperature,
                    )

                # G13-3: wrap with retry for transient LM Studio errors.
                # Resolve retry config defensively — _server_mod was never
                # imported here, so the old code raised NameError (swallowed),
                # skipping this fallback entirely for older non-streaming clients.
                from qwable.fusion_retry import chat_with_retry

                try:
                    import qwable.server as _server_mod

                    _cfg = getattr(_server_mod, "config", None)
                    retry_cfg = getattr(_cfg, "fusion_max_retries", 0)
                    retry_delay = getattr(_cfg, "fusion_retry_base_delay", 1.0)
                except Exception:
                    retry_cfg, retry_delay = 0, 1.0
                response = await loop.run_in_executor(
                    None,
                    lambda: chat_with_retry(
                        _chat_call,
                        max_retries=retry_cfg,
                        base_delay=retry_delay,
                    ),
                )
                text, finish_reason = _extract_assistant_text(response)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            finish_reason = "error"
            logger.warning(
                "fusion streaming panel model %s failed: %s", model_id, error
            )
        latency_ms = int((time.monotonic() - t0) * 1000)

        panel_resp = PanelResponse(
            model_id=model_id,
            text=text,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            error=error,
        )

        yield FusionStreamEvent(
            event=FUSION_STREAM_EVENT_PANEL_DONE,
            data={
                "model_id": model_id,
                "index": index,
                "latency_ms": latency_ms,
                "finish_reason": finish_reason,
                "error": error,
                "text_preview": (text or "")[:120],
            },
        )
        yield panel_resp

        # G12-3: skip unload for last panel when keep_last_resident is set
        if is_last and keep_last_resident:
            continue
        # Unload — always, even on error
        try:
            await loop.run_in_executor(
                None, lambda: panel_client.unload_models([model_id])
            )
        except Exception as exc:
            logger.warning("fusion streaming unload after %s failed: %s", model_id, exc)


async def run_fusion_agent_streaming(
    *,
    ollama_client: Any,
    ds4_client: Any,
    preset: FusionPreset,
    original_prompt: str,
    panel_max_tokens: int,
    judge_max_tokens: int,
    ds4_model: str,
    temperature: float = 0.3,
    keep_last_resident: bool = True,
) -> AsyncIterator[FusionStreamEvent]:
    """Streaming fusion runner — yields FusionStreamEvent per panel + per judge token.

    Final event carries the full structured synthesis (same shape as
    run_fusion_agent's return dict).
    """
    t_total = time.monotonic()

    # ─── Panel phase ────────────────────────────────────────────────────
    panel_responses: list[PanelResponse] = []
    async for item in _run_panel_serial_streaming(
        preset=preset,
        original_prompt=original_prompt,
        panel_client=ollama_client,
        panel_max_tokens=panel_max_tokens,
        temperature=temperature,
        keep_last_resident=keep_last_resident,
    ):
        if isinstance(item, PanelResponse):
            panel_responses.append(item)
        else:
            yield item

    # ─── Build synthesis prompt ─────────────────────────────────────────
    synthesis_input = SynthesisInput(
        original_prompt=original_prompt,
        panel_responses=panel_responses,
        preset_name=preset.name,
    )
    judge_system, judge_user = build_synthesis_prompt(synthesis_input)
    judge_messages = [
        {"role": "system", "content": judge_system},
        {"role": "user", "content": judge_user},
    ]

    is_ds4_judge = preset.judge_model == ds4_model
    judge_backend = "ds4" if is_ds4_judge else "ollama"

    yield FusionStreamEvent(
        event=FUSION_STREAM_EVENT_JUDGE_START,
        data={
            "judge_model": preset.judge_model,
            "judge_backend": judge_backend,
        },
    )

    # ─── Judge streaming ────────────────────────────────────────────────
    loop = asyncio.get_event_loop()
    t_judge = time.monotonic()
    judge_text_chunks: list[str] = []
    judge_finish_reason: str | None = None

    try:
        if is_ds4_judge:
            stream_iter = ds4_client.chat_completion_stream(
                model=ds4_model,
                messages=judge_messages,
                max_tokens=judge_max_tokens,
                temperature=temperature,
            )
        else:
            stream_iter = ollama_client.chat_completion_stream(
                model=preset.judge_model,
                messages=judge_messages,
                max_tokens=judge_max_tokens,
                temperature=temperature,
            )

        # Drain sync iterator in executor, yielding judge_token events
        def _drain_and_collect(it):
            chunks = []
            for delta, finish in it:
                chunks.append((delta, finish))
            return chunks

        chunks = await loop.run_in_executor(
            None, lambda: _drain_and_collect(stream_iter)
        )
        for delta, finish in chunks:
            if delta:
                judge_text_chunks.append(delta)
                yield FusionStreamEvent(
                    event=FUSION_STREAM_EVENT_JUDGE_TOKEN,
                    data={"delta": delta},
                )
            if finish:
                judge_finish_reason = finish
    except Exception as exc:
        logger.warning("fusion streaming judge failed: %s", exc)
        # The streaming judge previously had no retry — a single transient
        # failure became the user-facing answer. Retry ONCE via a non-streaming
        # call (with backoff), but only if no tokens were emitted yet (we can't
        # un-stream a partial answer).
        if not judge_text_chunks:
            try:
                from qwable.fusion_retry import chat_with_retry

                def _judge_retry_call():
                    if is_ds4_judge:
                        return ds4_client.chat_completion(
                            model=ds4_model,
                            messages=judge_messages,
                            max_tokens=judge_max_tokens,
                            stream=False,
                            temperature=temperature,
                        )
                    return ollama_client.chat_completion(
                        model=preset.judge_model,
                        messages=judge_messages,
                        max_tokens=judge_max_tokens,
                        stream=False,
                        temperature=temperature,
                    )

                resp = await loop.run_in_executor(
                    None,
                    lambda: chat_with_retry(
                        _judge_retry_call, max_retries=2, base_delay=1.0
                    ),
                )
                retry_text, _ = _extract_assistant_text(resp)
                if retry_text and retry_text.strip():
                    judge_text_chunks.append(retry_text)
                    yield FusionStreamEvent(
                        event=FUSION_STREAM_EVENT_JUDGE_TOKEN,
                        data={"delta": retry_text},
                    )
                else:
                    judge_text_chunks.append(
                        f"[fusion judge error: {type(exc).__name__}: {exc}]"
                    )
            except Exception as exc2:
                logger.warning("fusion streaming judge retry failed: %s", exc2)
                judge_text_chunks.append(
                    f"[fusion judge error: {type(exc2).__name__}: {exc2}]"
                )
        else:
            judge_text_chunks.append(
                f"[fusion judge error after partial stream: {type(exc).__name__}: {exc}]"
            )

    judge_text = "".join(judge_text_chunks)
    judge_latency_ms = int((time.monotonic() - t_judge) * 1000)

    # G12-3: unload judge for clean state ONLY if not keeping last resident.
    # (only for ollama path; ds4 is external so no unload needed)
    if not is_ds4_judge and not keep_last_resident:
        try:
            await loop.run_in_executor(
                None, lambda: ollama_client.unload_models([preset.judge_model])
            )
        except Exception as exc:
            logger.warning("fusion streaming judge unload failed: %s", exc)

    # ─── Parse + final event ────────────────────────────────────────────
    structured = parse_structured_output(judge_text)
    if structured.had_fallback:
        logger.warning(
            "fusion streaming judge %s returned non-structured output (had_fallback=True)",
            preset.judge_model,
        )

    total_latency_ms = int((time.monotonic() - t_total) * 1000)

    yield FusionStreamEvent(
        event=FUSION_STREAM_EVENT_JUDGE_DONE,
        data={
            "judge_model": preset.judge_model,
            "judge_backend": judge_backend,
            "judge_latency_ms": judge_latency_ms,
            "judge_finish_reason": judge_finish_reason,
            "structured_had_fallback": structured.had_fallback,
        },
    )

    final_text = structured.final_answer or judge_text or "[fusion produced no output]"

    yield FusionStreamEvent(
        event=FUSION_STREAM_EVENT_FINAL,
        data={
            "text": final_text,
            "structured": {
                "final_answer": structured.final_answer,
                "consensus": structured.consensus,
                "contradictions": structured.contradictions,
                "blind_spots": structured.blind_spots,
                "per_model_notes": structured.per_model_notes,
                "had_fallback": structured.had_fallback,
            },
            "trace": {
                "preset": preset.name,
                "panel_responses": [
                    {
                        "model_id": r.model_id,
                        "latency_ms": r.latency_ms,
                        "finish_reason": r.finish_reason,
                        "error": r.error,
                        "text_preview": (r.text or "")[:120],
                    }
                    for r in panel_responses
                ],
                "judge_model": preset.judge_model,
                "judge_backend": judge_backend,
                "judge_text_preview": (judge_text or "")[:240],
                "structured_had_fallback": structured.had_fallback,
                "total_latency_ms": total_latency_ms,
            },
        },
    )
