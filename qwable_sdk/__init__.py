"""Qwable SDK — Python client for the Qwable Agent Gateway v1.5.

Public surface:
    from qwable_sdk import LocalFusionClient, FusionPreset, FusionEvent

    client = LocalFusionClient("http://127.0.0.1:8088")

    # Non-streaming
    result = client.fusion_chat(
        messages=[{"role": "user", "content": "Compare sort algorithms"}],
        preset=FusionPreset.QUALITY,
    )
    print(result.text)
    print(result.panel_models)
    print(result.judge_model)

    # Streaming (sync generator)
    for event in client.fusion_chat_stream(
        messages=[{"role": "user", "content": "..."}],
        preset=FusionPreset.BUDGET,
    ):
        if event.event == "judge_token":
            print(event.delta, end="", flush=True)
        elif event.event == "final":
            print()

    # Async variant
    result = await client.afusion_chat(
        messages=[{"role": "user", "content": "..."}],
        preset=FusionPreset.CODING,
    )
"""

from qwable_sdk.client import LocalFusionClient
from qwable_sdk.events import (
    FusionEvent,
    PanelEvent,
    JudgeEvent,
    FinalEvent,
    ErrorEvent,
)
from qwable_sdk.types import (
    FusionPreset,
    FusionPresetName,
    FusionResult,
)


__all__ = [
    "LocalFusionClient",
    "FusionEvent",
    "PanelEvent",
    "JudgeEvent",
    "FinalEvent",
    "ErrorEvent",
    "FusionPreset",
    "FusionPresetName",
    "FusionResult",
]
