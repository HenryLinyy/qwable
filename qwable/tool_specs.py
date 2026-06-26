"""ToolSpec normalization from various protocols."""

from qwable.schemas import ToolSpec
from typing import Literal


def normalize_openai_tools(
    tools: list[dict], source_protocol: Literal["openai_responses", "openai_chat"]
) -> list[ToolSpec]:
    """Normalize OpenAI-style tool definitions to ToolSpec list.

    Supports both flat format (type, function, name, parameters)
    and nested format (type, function { name, parameters }).
    """
    result: list[ToolSpec] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        t_type = t.get("type")
        if t_type != "function":
            continue

        func = t.get("function", t)
        name = func.get("name", t.get("name", ""))
        description = func.get("description", t.get("description"))
        params = func.get("parameters", t.get("parameters", {}))
        if not name:
            continue

        result.append(
            ToolSpec(
                name=name,
                description=description,
                input_schema=params,
                source_protocol=source_protocol,
                raw=t,
            )
        )
    return result


def normalize_anthropic_tools(tools: list[dict]) -> list[ToolSpec]:
    """Normalize Anthropic-style tool definitions to ToolSpec list."""
    result: list[ToolSpec] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name", "")
        description = t.get("description", "")
        input_schema = t.get("input_schema", {})
        if not name:
            continue
        result.append(
            ToolSpec(
                name=name,
                description=description,
                input_schema=input_schema,
                source_protocol="anthropic_messages",
                raw=t,
            )
        )
    return result
