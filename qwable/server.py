"""FastAPI server for Qwable Agent Gateway."""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import json
import asyncio
import logging
import inspect

from qwable.config import FusionConfig
from qwable.fusion_core import FusionCore
from qwable.message_parsing import (
    parse_openai_chat_input,
    parse_openai_responses_input,
    parse_anthropic_messages_input,
)
from qwable.render_openai_responses import render_final_answer as render_responses_final
from qwable.render_openai_responses import render_tool_call as render_responses_tool
from qwable.render_openai_chat import render_final_answer as render_chat_final
from qwable.render_openai_chat import render_tool_call as render_chat_tool
from qwable.render_anthropic_messages import (
    render_final_answer as render_anthropic_final,
)
from qwable.render_anthropic_messages import render_tool_use
from qwable.token_estimator import estimate_tokens
from qwable.streaming import sse_event

logger = logging.getLogger("qwable.server")

# Global state
config: FusionConfig | None = None
fusion_core: FusionCore | None = None
global_lock: asyncio.Lock | None = None


async def _try_acquire_global_lock() -> bool:
    """Acquire the global model lock without queueing behind an active request."""
    if global_lock is None:
        raise RuntimeError("global lock is not initialized")
    if global_lock.locked():
        return False
    await global_lock.acquire()
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, fusion_core, global_lock
    config = FusionConfig()
    fusion_core = FusionCore(config)
    global_lock = asyncio.Lock()
    logger.info("Qwable Gateway started")
    yield
    fusion_core.close()
    logger.info("Qwable Gateway stopped")


app = FastAPI(title="Qwable Agent Gateway", version="1.5.0", lifespan=lifespan)


# ─── Health ──────────────────────────────────────────────


@app.get("/health")
async def health():
    return _health_payload()


@app.get("/v1/health")
async def v1_health():
    return _health_payload()


def _health_payload() -> dict:
    cfg = config or FusionConfig()
    return {
        "status": "ok",
        "version": "1.5.0",
        "agent_runtime_enabled": True,
        "agent_store_path": cfg.agent_store_path,
        "agent_profiles": [
            "qwable-agent",
            "qwable-code-agent",
            "qwable-review-agent",
        ],
        "model_roles": {
            "planner": cfg.model_role_planner,
            "executor": cfg.model_role_executor,
            "repair": cfg.model_role_repair,
            "critic": cfg.model_role_critic,
            "judge": cfg.model_role_judge,
        },
    }


# ─── Models ──────────────────────────────────────────────

MODELS_LIST = {
    "object": "list",
    "data": [
        {"id": "qwable", "object": "model", "owned_by": "local"},
        {"id": "qwable-fast", "object": "model", "owned_by": "local"},
        {"id": "qwable-full", "object": "model", "owned_by": "local"},
        {"id": "qwable-heavy", "object": "model", "owned_by": "local"},
        {"id": "qwable-chat", "object": "model", "owned_by": "local"},
        {"id": "qwable-vision-fast", "object": "model", "owned_by": "local"},
        {"id": "qwable-vision-pro", "object": "model", "owned_by": "local"},
        {"id": "qwable-vision-heavy", "object": "model", "owned_by": "local"},
        {"id": "qwable-agentic-pro", "object": "model", "owned_by": "local"},
        {"id": "qwable-hermes-pro", "object": "model", "owned_by": "local"},
        {"id": "qwable-agentic-mlx", "object": "model", "owned_by": "local"},
        {"id": "qwable-formatter-mlx", "object": "model", "owned_by": "local"},
        {"id": "qwable-fusion", "object": "model", "owned_by": "local"},
        {"id": "qwable-fusion-budget", "object": "model", "owned_by": "local"},
        {"id": "qwable-fusion-quality", "object": "model", "owned_by": "local"},
        {"id": "qwable-fusion-coding", "object": "model", "owned_by": "local"},
        {"id": "qwable-fusion-heavy", "object": "model", "owned_by": "local"},
        {"id": "qwable-agent", "object": "model", "owned_by": "local"},
        {"id": "qwable-code-agent", "object": "model", "owned_by": "local"},
        {"id": "qwable-review-agent", "object": "model", "owned_by": "local"},
        {
            "id": "claude-qwable",
            "object": "model",
            "owned_by": "local",
            "type": "model",
            "display_name": "Claude Qwable",
        },
        {
            "id": "claude-qwable-fast",
            "object": "model",
            "owned_by": "local",
            "type": "model",
            "display_name": "Claude Qwable Fast",
        },
        {
            "id": "claude-qwable-full",
            "object": "model",
            "owned_by": "local",
            "type": "model",
            "display_name": "Claude Qwable Full",
        },
        {
            "id": "claude-qwable-heavy",
            "object": "model",
            "owned_by": "local",
            "type": "model",
            "display_name": "Claude Qwable Heavy",
        },
        {
            "id": "claude-qwable-vision-fast",
            "object": "model",
            "owned_by": "local",
            "type": "model",
            "display_name": "Claude Qwable Vision Fast",
        },
        {
            "id": "claude-qwable-vision-pro",
            "object": "model",
            "owned_by": "local",
            "type": "model",
            "display_name": "Claude Qwable Vision Pro",
        },
        {
            "id": "claude-qwable-vision-heavy",
            "object": "model",
            "owned_by": "local",
            "type": "model",
            "display_name": "Claude Qwable Vision Heavy",
        },
        {
            "id": "claude-qwable-agentic-pro",
            "object": "model",
            "owned_by": "local",
            "type": "model",
            "display_name": "Claude Qwable Agentic Pro",
        },
        {
            "id": "claude-qwable-hermes-pro",
            "object": "model",
            "owned_by": "local",
            "type": "model",
            "display_name": "Claude Qwable Hermes Pro",
        },
        {
            "id": "claude-qwable-agentic-mlx",
            "object": "model",
            "owned_by": "local",
            "type": "model",
            "display_name": "Claude Qwable Agentic MLX",
        },
        {
            "id": "claude-qwable-formatter-mlx",
            "object": "model",
            "owned_by": "local",
            "type": "model",
            "display_name": "Claude Qwable Formatter MLX",
        },
        {
            "id": "claude-qwable-fusion",
            "object": "model",
            "owned_by": "local",
            "type": "model",
            "display_name": "Claude Qwable Fusion (Deliberation Router)",
        },
        {
            "id": "claude-qwable-fusion-budget",
            "object": "model",
            "owned_by": "local",
            "type": "model",
            "display_name": "Claude Qwable Fusion (budget)",
        },
        {
            "id": "claude-qwable-fusion-quality",
            "object": "model",
            "owned_by": "local",
            "type": "model",
            "display_name": "Claude Qwable Fusion (quality)",
        },
        {
            "id": "claude-qwable-fusion-coding",
            "object": "model",
            "owned_by": "local",
            "type": "model",
            "display_name": "Claude Qwable Fusion (coding)",
        },
        {
            "id": "claude-qwable-fusion-heavy",
            "object": "model",
            "owned_by": "local",
            "type": "model",
            "display_name": "Claude Qwable Fusion (heavy)",
        },
        {
            "id": "claude-qwable-agent",
            "object": "model",
            "owned_by": "local",
            "type": "model",
            "display_name": "Claude Qwable Agent",
        },
        {
            "id": "claude-qwable-code-agent",
            "object": "model",
            "owned_by": "local",
            "type": "model",
            "display_name": "Claude Qwable Code Agent",
        },
        {
            "id": "claude-qwable-review-agent",
            "object": "model",
            "owned_by": "local",
            "type": "model",
            "display_name": "Claude Qwable Review Agent",
        },
    ],
}


@app.get("/v1/models")
async def v1_models():
    return MODELS_LIST


@app.get("/v1/fusion/presets")
async def v1_fusion_presets():
    """G12-4: list all fusion presets with their panel/judge/peak RAM."""
    from qwable.fusion_presets import PRESETS

    cfg = globals().get("config")
    fc = globals().get("fusion_core")

    # Probe ds4 reachability (best-effort, 1s timeout)
    ds4_reachable = False
    try:
        import httpx as _httpx

        base = getattr(cfg, "ds4_base_url", "http://127.0.0.1:8000/v1")
        # Strip the exact '/v1' suffix, not the char-set {'/', 'v', '1'} that
        # rstrip('/v1') would chew off hosts/ports ending in those chars.
        base_root = base[:-3] if base.endswith("/v1") else base
        with _httpx.Client(timeout=1.0) as c:
            r = c.get(base_root.rstrip("/") + "/v1/models")
            ds4_reachable = r.status_code == 200
    except Exception:
        pass

    loaded = []
    try:
        import subprocess as _sp

        out = _sp.run(
            [(config.lmstudio_cli_path if config else "lms"), "ps"],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
        for line in out.splitlines():
            parts = line.split()
            if (
                len(parts) >= 2
                and "/" in parts[0]
                and parts[0]
                not in (
                    "IDENTIFIER",
                    "No",
                    "To",
                )
            ):
                loaded.append(parts[0])
    except Exception:
        pass

    last_used = getattr(fc, "last_used_model", None) if fc else None

    presets_dict = {}
    for name, p in PRESETS.items():
        ds4_model = (
            getattr(cfg, "ds4_model", "deepseek-v4-flash")
            if cfg
            else "deepseek-v4-flash"
        )
        presets_dict[name] = {
            "panel": list(p.analysis_models),
            "judge": p.judge_model,
            "description": p.description,
            "judge_backend": "ds4" if p.judge_model == ds4_model else "ollama",
        }

    return {
        "presets": presets_dict,
        "loaded_now": loaded,
        "last_used_model": last_used,
        "ds4_reachable": ds4_reachable,
        "default_preset": getattr(cfg, "fusion_default_preset", "quality")
        if cfg
        else "quality",
    }


@app.get("/v1/fusion/presets/loaded")
async def v1_fusion_presets_loaded():
    """G12-4 helper: lightweight endpoint returning only currently loaded models."""
    loaded = []
    try:
        import subprocess as _sp

        out = _sp.run(
            [(config.lmstudio_cli_path if config else "lms"), "ps"],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
        for line in out.splitlines():
            parts = line.split()
            if (
                len(parts) >= 2
                and "/" in parts[0]
                and parts[0]
                not in (
                    "IDENTIFIER",
                    "No",
                    "To",
                )
            ):
                loaded.append(parts[0])
    except Exception:
        pass
    fc = globals().get("fusion_core")
    return {
        "loaded_models": loaded,
        "last_used_model": getattr(fc, "last_used_model", None) if fc else None,
    }


@app.get("/v1/system/optimizations")
async def v1_system_optimizations():
    """G15: LM Studio MLX optimization state for current session.

    Returns runtime info, currently loaded models, settings context
    length, and recommended M5 Max tuning values for comparison.
    """
    from qwable.mlx_optimizations import get_current_optimizations

    info = get_current_optimizations()
    info["recommended_settings"] = {
        "context_length": 32768,
        "parallel": 2,
        "gpu_mode": "max",
        "note": "M5 Max 128GB tuned",
    }
    return info


def _build_models_health() -> dict:
    """Pure builder for /health/models — kept out of the route for testability.

    Per v1.8 plan §14.2: report role → primary / fallback / stages,
    plus v1.8 enable flags (Qwable, Qwythos, health-check).
    """
    from qwable.model_roles import WorkflowStage
    from qwable.model_selector import ModelSelector

    cfg = config or FusionConfig()
    if fusion_core is not None and hasattr(fusion_core, "model_selector"):
        selector = fusion_core.model_selector
    else:
        selector = ModelSelector(cfg)

    out: dict = {
        "status": "ok",
        "version": "1.8.0",
        "roles": {},
        "flags": {
            "enable_qwable_executor": cfg.enable_qwable_executor,
            "enable_qwythos_long_context": cfg.enable_qwythos_long_context,
            "model_health_check_on_startup": cfg.model_health_check_on_startup,
        },
    }

    for stage in (
        WorkflowStage.EXECUTE_PATCH,
        WorkflowStage.REPAIR_PATCH,
        WorkflowStage.CONTEXT_COMPACTION,
        WorkflowStage.PLAN_REVISION,
        WorkflowStage.PLAN_REVIEW,
        WorkflowStage.FINAL_REPORT,
    ):
        try:
            sel = selector.select_for_stage(stage)
        except (RuntimeError, KeyError):
            continue
        role = sel.role.value
        out["roles"].setdefault(
            role,
            {
                "primary": sel.model_name,
                "fallbacks": list(sel.fallback_chain),
                "stages": [],
            },
        )
        out["roles"][role]["stages"].append(stage.value)

    return out


@app.get("/health/models")
async def health_models():
    """v1.8: report role → primary / fallback / availability.

    Per plan §14.2, this is the manual-verification entry point used in
    plan §17.2. Pure data — no live LM Studio probe unless
    MODEL_HEALTH_CHECK_ON_STARTUP=true (which is opt-in).
    """
    return _build_models_health()


@app.get("/v1/system/models")
async def v1_system_models():
    """v1.8: OpenAI-flavoured alias of /health/models (some clients prefer /v1/...)."""
    return _build_models_health()


@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/dashboard/", response_class=HTMLResponse)
async def dashboard():
    """G14-1: Web UI dashboard for fusion deliberation.

    Serves a single-page HTML+JS app that:
    - Lists available presets with panel/judge info
    - Shows currently loaded models (polls every 5s)
    - Lets user send fusion requests and watch streaming events live
    - Works with both OpenAI Chat and Anthropic Messages protocols
    """
    from pathlib import Path

    html_path = Path(__file__).parent / "web" / "dashboard.html"
    if not html_path.exists():
        return HTMLResponse(
            content="<h1>Dashboard not found</h1><p>web/dashboard.html missing</p>",
            status_code=404,
        )
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ─── Conversations (G14-2) ───────────────────────────────────


@app.post("/v1/conversations")
async def post_conversations(body: dict = None):
    """Create a new conversation. Returns {id, created_at, ...}."""
    store = _get_conv_store()
    metadata = (body or {}).get("metadata", {})
    conv = store.create(metadata=metadata)
    return conv.to_dict()


@app.get("/v1/conversations")
async def list_conversations():
    """List all non-expired conversations (most recent first)."""
    store = _get_conv_store()
    return [c.to_dict() for c in store.list_all()]


@app.get("/v1/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """Retrieve a specific conversation."""
    store = _get_conv_store()
    conv = store.get(conv_id)
    if conv is None:
        return JSONResponse({"error": "not_found"}, status_code=404)
    return conv.to_dict()


@app.delete("/v1/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation. Returns {deleted: true|false}."""
    store = _get_conv_store()
    deleted = store.delete(conv_id)
    return {"deleted": deleted}


# Lazy conversation store init (avoid file IO at module import time)
_CONV_STORE: dict = {}


def _get_conv_store():
    if "store" not in _CONV_STORE:
        from qwable.conversation_store import ConversationStore

        _CONV_STORE["store"] = ConversationStore()
    return _CONV_STORE["store"]


# ─── OpenAI Chat Completions ─────────────────────────────


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    task = parse_openai_chat_input(body)

    # Acquire global lock
    if not await _try_acquire_global_lock():
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": "qwable is busy; another request is running",
                    "type": "rate_limit_exceeded",
                }
            },
        )

    if task.stream:
        return StreamingResponse(
            _stream_chat_response(body, task),
            media_type="text/event-stream",
        )

    try:
        action = await fusion_core.execute(task)
    finally:
        global_lock.release()

    # Build response
    if action.type == "tool_call":
        message = render_chat_tool(action)
        finish_reason = "tool_calls"
    else:
        message = render_chat_final(action)
        finish_reason = "stop"

    return JSONResponse(
        {
            "id": "chatcmpl-qwable-1",
            "object": "chat.completion",
            "created": int(__import__("time").time()),
            "model": body.get("model", "qwable-chat"),
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish_reason}
            ],
            "usage": _compute_chat_usage(task),
        }
    )


# ─── OpenAI Responses ───────────────────────────────────


@app.post("/v1/responses")
async def openai_responses(request: Request):
    body = await request.json()
    task = parse_openai_responses_input(body)

    # Acquire global lock
    if not await _try_acquire_global_lock():
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": "qwable is busy; another request is running",
                    "type": "rate_limit_exceeded",
                }
            },
        )

    if task.stream:
        return StreamingResponse(
            _stream_openai_response(body, task),
            media_type="text/event-stream",
        )

    try:
        action = await fusion_core.execute(task)
    finally:
        global_lock.release()

    # Build response
    output_items = []
    if action.type == "tool_call":
        output_items.append(render_responses_tool(action))
    else:
        output_items.append(render_responses_final(action))

    usage = _compute_usage(task)

    response_body = {
        "id": "resp_qwable_1",
        "object": "response",
        "created": int(__import__("time").time()),
        "model": body.get("model", "qwable-fast"),
        "output": output_items,
        "status": "completed",
        "usage": usage,
    }
    if _debug_requested(body):
        response_body["debug"] = _build_debug_payload(action)

    return JSONResponse(response_body)


# ─── Anthropic Messages ─────────────────────────────────


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    body = await request.json()
    task = parse_anthropic_messages_input(body)

    # Acquire global lock
    if not await _try_acquire_global_lock():
        return JSONResponse(
            status_code=429,
            content={
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "qwable is busy; another request is running",
                },
            },
        )

    if task.stream:
        return StreamingResponse(
            _stream_anthropic_response(body, task),
            media_type="text/event-stream",
        )

    try:
        action = await fusion_core.execute(task)
    finally:
        global_lock.release()

    # Build response
    content = []
    if action.type == "tool_call":
        content.append(render_tool_use(action))
        stop_reason = "tool_use"
    else:
        content.append(render_anthropic_final(action))
        stop_reason = "end_turn"

    usage = _compute_usage(task)

    return JSONResponse(
        {
            "id": "msg_qwable_1",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": body.get("model", "claude-qwable-fast"),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage,
        }
    )


# ─── Count Tokens ───────────────────────────────────────


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    body = await request.json()
    text = ""
    messages = body.get("messages", [])
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            text += content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text += block.get("text", "")

    tokens = estimate_tokens(text)
    return JSONResponse(
        {
            "input_tokens": tokens,
            "output_tokens": 0,
        }
    )


# ─── Helpers ─────────────────────────────────────────────


def _compute_usage(task, output_text: str = "") -> dict:
    """Compute estimated token usage for a task.

    output_text drives the output token estimate; pass the produced answer
    (or accumulated stream deltas) so usage isn't a hardcoded placeholder.
    """
    input_tokens = estimate_tokens(task.text)
    output_tokens = estimate_tokens(output_text) if output_text else 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _compute_chat_usage(task, output_text: str = "") -> dict:
    """Compute OpenAI Chat-compatible token usage keys."""
    input_tokens = estimate_tokens(task.text)
    output_tokens = estimate_tokens(output_text) if output_text else 0
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _debug_requested(body: dict) -> bool:
    """Return whether a request explicitly opts into fusion debug metadata."""
    metadata = body.get("metadata")
    return isinstance(metadata, dict) and metadata.get("debug") is True


def _build_debug_payload(action) -> dict:
    """Build opt-in debug metadata without changing normal response shape."""
    trace = getattr(action, "trace", None)
    debug = dict(trace) if isinstance(trace, dict) else {}
    rationale = getattr(action, "rationale_summary", None)
    if rationale:
        debug["rationale_summary"] = rationale
    return debug


async def _execute_fusion_streaming(task):
    """Run fusion execution off the event loop so streaming keepalive can fire."""
    core = fusion_core
    if core is None:
        raise RuntimeError("fusion core is not initialized")
    return await asyncio.to_thread(_run_fusion_execute_sync, core, task)


def _run_fusion_execute_sync(core, task):
    result = core.execute(task)
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result


async def _stream_chat_response(body: dict, task):
    response_id = "chatcmpl-qwable-1"
    model = body.get("model", "qwable-chat")
    created = int(__import__("time").time())
    yield sse_event(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
    )

    # G11: fusion deliberation streaming branch — only for fusion-agent profile.
    # Emit judge tokens as OpenAI chat.completion.chunk events, panel events
    # as SSE comments.
    if task.profile == "fusion-agent":
        from qwable.streaming_events import (
            FUSION_STREAM_EVENT_FINAL,
            FUSION_STREAM_EVENT_JUDGE_TOKEN,
        )
        from qwable.fusion_deliberation import run_fusion_agent_streaming
        from qwable.fusion_request import extract_fusion_request
        from qwable.fusion_presets import FusionPresetError, resolve_preset

        raw = task.raw_request or {}
        fusion_req = extract_fusion_request(raw)
        try:
            preset = resolve_preset(
                fusion_req, default=fusion_core.config.fusion_default_preset
            )
        except FusionPresetError:
            # Bad preset → fall through to non-streaming path
            action_task = asyncio.create_task(_execute_fusion_streaming(task))
            try:
                async for frame in _keepalive_until(action_task):
                    yield frame
                action = action_task.result()
                delta = {"content": action.text or ""}
                yield sse_event(
                    {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": None}
                        ],
                    }
                )
                yield sse_event(
                    {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                )
                yield sse_event("[DONE]")
            finally:
                if not action_task.done():
                    action_task.cancel()
                global_lock.release()
            return

        try:
            yield sse_event(
                ": fusion preset resolved: " + preset.name + "\n\n", event=None
            )
            streamed_text = ""
            async for ev in _events_with_keepalive(
                run_fusion_agent_streaming(
                    ollama_client=fusion_core.ollama,
                    ds4_client=fusion_core.ds4,
                    preset=preset,
                    original_prompt=task.text or "",
                    panel_max_tokens=fusion_core._request_max_tokens(
                        task, fusion_core.config.fusion_max_tokens_panel
                    ),
                    judge_max_tokens=fusion_core._request_max_tokens(
                        task, fusion_core.config.fusion_max_tokens_judge
                    ),
                    ds4_model=fusion_core.config.ds4_model,
                    temperature=fusion_core._request_temperature(task, 0.3),
                )
            ):
                if ev is _KEEPALIVE_MARKER:
                    yield sse_event(comment="keepalive")
                    continue
                if ev.event == FUSION_STREAM_EVENT_JUDGE_TOKEN:
                    delta = ev.data.get("delta", "")
                    streamed_text += delta
                    yield sse_event(
                        {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": delta},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                elif ev.event == FUSION_STREAM_EVENT_FINAL:
                    # If the judge produced no token deltas, the final answer
                    # would otherwise be lost — emit it as a content delta now.
                    final_text = ev.data.get("text", "") or ""
                    if not streamed_text.strip() and final_text.strip():
                        streamed_text = final_text
                        yield sse_event(
                            {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": final_text},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    yield sse_event(
                        {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": "stop"}
                            ],
                        }
                    )
                    # G12 micro C: emit usage chunk so callers can track tokens
                    yield sse_event(
                        {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": None}
                            ],
                            "usage": _compute_usage(task, streamed_text),
                        }
                    )
                    yield sse_event("[DONE]")
                    break
                else:
                    yield sse_event(f": fusion {ev.event}: " + str(ev.data)[:200])
            else:
                yield sse_event("[DONE]")
        finally:
            global_lock.release()
        return

    # Non-fusion streaming: existing keepalive + final chunk path.
    action_task = asyncio.create_task(_execute_fusion_streaming(task))
    try:
        async for frame in _keepalive_until(action_task):
            yield frame
        action = action_task.result()
        if action.type == "tool_call":
            message = render_chat_tool(action)
            delta = {"tool_calls": message["tool_calls"]}
            finish_reason = "tool_calls"
        else:
            delta = {"content": action.text or ""}
            finish_reason = "stop"
        yield sse_event(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
        )
        yield sse_event(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
        )
        yield sse_event("[DONE]")
    finally:
        if not action_task.done():
            action_task.cancel()
        global_lock.release()


async def _stream_openai_response(body: dict, task):
    response_id = "resp_qwable_1"
    model = body.get("model", "qwable-fast")
    created = int(__import__("time").time())

    # G13-2: OpenAI Responses streaming fusion branch
    if task.profile == "fusion-agent":
        from qwable.streaming_events import (
            FUSION_STREAM_EVENT_FINAL,
            FUSION_STREAM_EVENT_JUDGE_TOKEN,
        )
        from qwable.fusion_deliberation import run_fusion_agent_streaming
        from qwable.fusion_request import extract_fusion_request
        from qwable.fusion_presets import FusionPresetError, resolve_preset

        raw = task.raw_request or {}
        fusion_req = extract_fusion_request(raw)
        try:
            preset = resolve_preset(
                fusion_req, default=fusion_core.config.fusion_default_preset
            )
        except FusionPresetError:
            # Fall through to non-streaming path
            action_task = asyncio.create_task(_execute_fusion_streaming(task))
            try:
                async for frame in _keepalive_until(action_task):
                    yield frame
                action = action_task.result()
                yield sse_event(
                    {
                        "type": "response.created",
                        "response": {"id": response_id, "model": model},
                    },
                    event="response.created",
                )
                yield sse_event(
                    {
                        "type": "response.in_progress",
                        "response": {"id": response_id, "model": model},
                    },
                    event="response.in_progress",
                )
                output_item = render_responses_final(action)
                yield sse_event(
                    {"type": "response.output_text.delta", "delta": action.text or ""},
                    event="response.output_text.delta",
                )
                yield sse_event(
                    {
                        "type": "response.completed",
                        "response": {
                            "id": response_id,
                            "object": "response",
                            "created": created,
                            "model": model,
                            "output": [output_item],
                            "usage": _compute_usage(task),
                        },
                    },
                    event="response.completed",
                )
            finally:
                if not action_task.done():
                    action_task.cancel()
                global_lock.release()
            return

        try:
            yield sse_event(
                {
                    "type": "response.created",
                    "response": {"id": response_id, "model": model},
                },
                event="response.created",
            )
            yield sse_event(
                {
                    "type": "response.in_progress",
                    "response": {"id": response_id, "model": model},
                },
                event="response.in_progress",
            )
            streamed_text = ""
            final_text = ""
            async for ev in _events_with_keepalive(
                run_fusion_agent_streaming(
                    ollama_client=fusion_core.ollama,
                    ds4_client=fusion_core.ds4,
                    preset=preset,
                    original_prompt=task.text or "",
                    panel_max_tokens=fusion_core._request_max_tokens(
                        task, fusion_core.config.fusion_max_tokens_panel
                    ),
                    judge_max_tokens=fusion_core._request_max_tokens(
                        task, fusion_core.config.fusion_max_tokens_judge
                    ),
                    ds4_model=fusion_core.config.ds4_model,
                    temperature=fusion_core._request_temperature(task, 0.3),
                )
            ):
                if ev is _KEEPALIVE_MARKER:
                    yield sse_event(comment="keepalive")
                    continue
                if ev.event == FUSION_STREAM_EVENT_JUDGE_TOKEN:
                    delta = ev.data.get("delta", "")
                    streamed_text += delta
                    yield sse_event(
                        {"type": "response.output_text.delta", "delta": delta},
                        event="response.output_text.delta",
                    )
                elif ev.event == FUSION_STREAM_EVENT_FINAL:
                    final_text = ev.data.get("text", "") or ""
                    break
            # If nothing was streamed, surface the final text as a delta so the
            # response isn't an empty success.
            if not streamed_text.strip() and final_text.strip():
                streamed_text = final_text
                yield sse_event(
                    {"type": "response.output_text.delta", "delta": final_text},
                    event="response.output_text.delta",
                )
            output_block = (
                [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": streamed_text}],
                    }
                ]
                if streamed_text.strip()
                else []
            )
            yield sse_event(
                {
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created": created,
                        "model": model,
                        "output": output_block,
                        "usage": _compute_usage(task, streamed_text),
                    },
                },
                event="response.completed",
            )
        finally:
            global_lock.release()
        return

    # Non-fusion streaming: existing OpenAI Responses path
    yield sse_event(
        {"type": "response.created", "response": {"id": response_id, "model": model}},
        event="response.created",
    )
    yield sse_event(
        {
            "type": "response.in_progress",
            "response": {"id": response_id, "model": model},
        },
        event="response.in_progress",
    )

    action_task = asyncio.create_task(_execute_fusion_streaming(task))
    try:
        async for frame in _keepalive_until(action_task):
            yield frame
        action = action_task.result()
        if action.type == "tool_call":
            output_item = render_responses_tool(action)
            yield sse_event(
                {"type": "response.output_item.added", "item": output_item},
                event="response.output_item.added",
            )
        else:
            output_item = render_responses_final(action)
            yield sse_event(
                {"type": "response.output_text.delta", "delta": action.text or ""},
                event="response.output_text.delta",
            )
        yield sse_event(
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created": created,
                    "model": model,
                    "output": [output_item],
                    "usage": _compute_usage(task),
                },
            },
            event="response.completed",
        )
    finally:
        if not action_task.done():
            action_task.cancel()
        global_lock.release()


async def _stream_anthropic_response(body: dict, task):
    message_id = "msg_qwable_1"
    model = body.get("model", "claude-qwable-fast")

    # G12-2: Fusion deliberation streaming branch (Anthropic Messages SSE)
    if task.profile == "fusion-agent":
        from qwable.streaming_events import (
            FUSION_STREAM_EVENT_FINAL,
            FUSION_STREAM_EVENT_JUDGE_TOKEN,
        )
        from qwable.fusion_deliberation import run_fusion_agent_streaming
        from qwable.fusion_request import extract_fusion_request
        from qwable.fusion_presets import FusionPresetError, resolve_preset

        raw = task.raw_request or {}
        fusion_req = extract_fusion_request(raw)
        try:
            preset = resolve_preset(
                fusion_req, default=fusion_core.config.fusion_default_preset
            )
        except FusionPresetError:
            # Bad preset → fall through to non-streaming path
            action_task = asyncio.create_task(_execute_fusion_streaming(task))
            try:
                async for frame in _keepalive_until(action_task, ping_event="ping"):
                    yield frame
                action = action_task.result()
                yield sse_event(
                    {
                        "type": "message_start",
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": _compute_usage(task),
                        },
                    },
                    event="message_start",
                )
                yield sse_event(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    event="content_block_start",
                )
                yield sse_event(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": action.text or ""},
                    },
                    event="content_block_delta",
                )
                yield sse_event(
                    {"type": "content_block_stop", "index": 0},
                    event="content_block_stop",
                )
                yield sse_event(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    },
                    event="message_delta",
                )
                yield sse_event({"type": "message_stop"}, event="message_stop")
            finally:
                if not action_task.done():
                    action_task.cancel()
                global_lock.release()
            return

        # Streaming path: emit Anthropic-format events
        try:
            yield sse_event(
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": _compute_usage(task),
                    },
                },
                event="message_start",
            )
            # Open the text content block
            yield sse_event(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                event="content_block_start",
            )
            streamed_text = ""
            final_text = ""
            async for ev in _events_with_keepalive(
                run_fusion_agent_streaming(
                    ollama_client=fusion_core.ollama,
                    ds4_client=fusion_core.ds4,
                    preset=preset,
                    original_prompt=task.text or "",
                    panel_max_tokens=fusion_core._request_max_tokens(
                        task, fusion_core.config.fusion_max_tokens_panel
                    ),
                    judge_max_tokens=fusion_core._request_max_tokens(
                        task, fusion_core.config.fusion_max_tokens_judge
                    ),
                    ds4_model=fusion_core.config.ds4_model,
                    temperature=fusion_core._request_temperature(task, 0.3),
                )
            ):
                if ev is _KEEPALIVE_MARKER:
                    yield sse_event(comment="keepalive")
                    continue
                if ev.event == FUSION_STREAM_EVENT_JUDGE_TOKEN:
                    delta = ev.data.get("delta", "")
                    streamed_text += delta
                    yield sse_event(
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": delta},
                        },
                        event="content_block_delta",
                    )
                elif ev.event == FUSION_STREAM_EVENT_FINAL:
                    final_text = ev.data.get("text", "") or ""
                    break
            # If the judge streamed no deltas, emit the final answer text so it
            # isn't silently dropped from the content block.
            if not streamed_text.strip() and final_text.strip():
                streamed_text = final_text
                yield sse_event(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": final_text},
                    },
                    event="content_block_delta",
                )
            yield sse_event(
                {"type": "content_block_stop", "index": 0}, event="content_block_stop"
            )
            yield sse_event(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {
                        "output_tokens": estimate_tokens(streamed_text)
                        if streamed_text
                        else 0
                    },
                },
                event="message_delta",
            )
            yield sse_event({"type": "message_stop"}, event="message_stop")
        finally:
            global_lock.release()
        return

    # Non-fusion streaming: existing Anthropic path
    yield sse_event(
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": _compute_usage(task),
            },
        },
        event="message_start",
    )

    action_task = asyncio.create_task(_execute_fusion_streaming(task))
    try:
        async for frame in _keepalive_until(action_task, ping_event="ping"):
            yield frame
        action = action_task.result()
        if action.type == "tool_call":
            block = render_tool_use(action)
            stop_reason = "tool_use"
            yield sse_event(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {**block, "input": {}},
                },
                event="content_block_start",
            )
            yield sse_event(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block["input"], ensure_ascii=False),
                    },
                },
                event="content_block_delta",
            )
        else:
            stop_reason = "end_turn"
            yield sse_event(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                event="content_block_start",
            )
            yield sse_event(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": action.text or ""},
                },
                event="content_block_delta",
            )

        yield sse_event(
            {"type": "content_block_stop", "index": 0}, event="content_block_stop"
        )
        yield sse_event(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            },
            event="message_delta",
        )
        yield sse_event({"type": "message_stop"}, event="message_stop")
    finally:
        if not action_task.done():
            action_task.cancel()
        global_lock.release()


async def _keepalive_until(task_handle: asyncio.Task, ping_event: str | None = None):
    keepalive_seconds = config.stream_keepalive_seconds if config else 10
    while True:
        done, _ = await asyncio.wait({task_handle}, timeout=keepalive_seconds)
        if done:
            return
        if ping_event:
            yield sse_event({"type": "ping"}, event=ping_event)
        else:
            yield sse_event(comment="keepalive")


_KEEPALIVE_MARKER = object()


async def _events_with_keepalive(agen):
    """Wrap a fusion streaming async-generator, yielding _KEEPALIVE_MARKER when no
    real event arrives within the keepalive window — so the long panel/judge gaps
    in run_fusion_agent_streaming don't trip client/proxy idle timeouts.
    """
    keepalive_seconds = config.stream_keepalive_seconds if config else 10
    it = agen.__aiter__()
    while True:
        nxt = asyncio.ensure_future(it.__anext__())
        while True:
            done, _ = await asyncio.wait({nxt}, timeout=keepalive_seconds)
            if done:
                break
            yield _KEEPALIVE_MARKER
        try:
            ev = nxt.result()
        except StopAsyncIteration:
            return
        yield ev
