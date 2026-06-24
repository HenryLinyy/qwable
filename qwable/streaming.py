"""Streaming helpers for keepalive and event delivery."""

from typing import AsyncGenerator
import contextlib
import json
import asyncio
import logging

logger = logging.getLogger("qwable.streaming")


def sse_event(
    data: dict | list | str | None = None,
    event: str | None = None,
    comment: str | None = None,
) -> str:
    """Format one server-sent event frame."""
    if comment is not None:
        return f": {comment}\n\n"

    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")
    if isinstance(data, str):
        encoded = data
    else:
        encoded = json.dumps(data if data is not None else {}, ensure_ascii=False)
    # SSE frames one "data:" line per physical line; a multi-line payload with
    # raw newlines would break framing (everything after the first \n would be
    # parsed as separate, unlabelled lines). Emit one data: line per segment.
    for segment in encoded.split("\n"):
        lines.append(f"data: {segment}")
    return "\n".join(lines) + "\n\n"


async def keepalive_stream(
    event_generator: AsyncGenerator[dict, None],
    keepalive_seconds: int = 10,
) -> AsyncGenerator[str, None]:
    """Wrap an event generator with keepalive pings.

    Sends a keepalive event every `keepalive_seconds` if no real event
    has been sent in that window.
    """
    async for event in event_generator:
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        # Reset keepalive timer after each real event
        await asyncio.sleep(0)

    # After final event, send done signal
    yield "data: [DONE]\n\n"


async def keepalive_task(keepalive_seconds: int, event_queue: asyncio.Queue):
    """Background task that sends keepalive pings at regular intervals."""
    while True:
        await asyncio.sleep(keepalive_seconds)
        await event_queue.put({"type": "keepalive", "data": {}})


async def stream_with_keepalive(
    event_generator: AsyncGenerator[dict, None],
    keepalive_seconds: int = 10,
) -> AsyncGenerator[str, None]:
    """Stream events with keepalive pings from a background task."""
    event_queue: asyncio.Queue = asyncio.Queue()

    async def producer():
        async for event in event_generator:
            await event_queue.put(event)

    keepalive_task_handle = asyncio.create_task(
        keepalive_task(keepalive_seconds, event_queue)
    )

    # Run producer in background
    producer_task = asyncio.create_task(producer())

    try:
        while True:
            if producer_task.done():
                # Surface a producer error, then drain the remaining queue
                # directly instead of busy-re-waiting on the finished producer.
                if producer_task.exception():
                    raise producer_task.exception()
                while not event_queue.empty():
                    event = event_queue.get_nowait()
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                break

            # Producer still running: the keepalive task feeds pings into the
            # same queue, so get() returns either a real event or a keepalive.
            try:
                event = await asyncio.wait_for(
                    event_queue.get(), timeout=keepalive_seconds + 2
                )
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                continue
    finally:
        # Always cancel BOTH background tasks (previously a producer exception
        # raised before the cancel, leaking the infinite keepalive task) and
        # await them so cancellation actually takes effect.
        keepalive_task_handle.cancel()
        producer_task.cancel()
        for _t in (keepalive_task_handle, producer_task):
            with contextlib.suppress(asyncio.CancelledError):
                await _t

    yield "data: [DONE]\n\n"
