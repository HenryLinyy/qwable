"""Parse incoming requests into ParsedAgentTask."""

from qwable.schemas import ToolSpec, ToolResult, ParsedAgentTask
from qwable.tool_specs import normalize_openai_tools, normalize_anthropic_tools
from qwable.profiles import resolve_profile
from qwable.vision import (
    ImageInput,
    image_from_base64_source,
    image_from_url_value,
)
from typing import Literal


def _openai_image_url_value(block: dict) -> tuple[str | None, str | None]:
    image_url = block.get("image_url")
    detail = block.get("detail")
    if isinstance(image_url, dict):
        return image_url.get("url"), image_url.get("detail", detail)
    if isinstance(image_url, str):
        return image_url, detail
    url = block.get("url")
    return url if isinstance(url, str) else None, detail


def parse_openai_responses_input(
    body: dict,
) -> ParsedAgentTask:
    """Parse an OpenAI Responses API request body into ParsedAgentTask."""
    model = body.get("model", "qwable-fast")
    profile = resolve_profile(model, "openai_responses")
    stream = body.get("stream", False)

    raw_input = body.get("input", "")
    text_parts: list[str] = []
    tool_results: list[ToolResult] = []
    images: list[ImageInput] = []

    if isinstance(raw_input, str):
        text_parts.append(raw_input)
    elif isinstance(raw_input, list):
        for item in raw_input:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                if role == "user" or role == "system":
                    if isinstance(content, str):
                        text_parts.append(content)
                    elif isinstance(content, list):
                        for block in content:
                            if not isinstance(block, dict):
                                continue
                            block_type = block.get("type", "")
                            if block_type in ("input_text", "text", "output_text"):
                                text_parts.append(block.get("text", ""))
                            elif block_type in ("input_image", "image_url"):
                                url_value, detail = _openai_image_url_value(block)
                                image = image_from_url_value(
                                    "openai_responses",
                                    url_value,
                                    detail=detail,
                                    raw=block,
                                )
                                if image:
                                    images.append(image)
                elif role == "assistant":
                    if isinstance(content, str):
                        text_parts.append(content)
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") in ("output_text", "text"):
                                text_parts.append(block.get("text", ""))
                elif item.get("type") == "function_call_output":
                    # Convert function_call_output to ToolResult
                    call_id = item.get("call_id", item.get("id", ""))
                    raw_output = item.get("output", "")
                    if isinstance(raw_output, list):
                        output = "".join(
                            b.get("text", "")
                            for b in raw_output
                            if isinstance(b, dict) and b.get("type") in ("output_text", "text")
                        )
                    elif isinstance(raw_output, str):
                        output = raw_output
                    else:
                        output = str(raw_output)
                    tool_results.append(ToolResult(
                        tool_call_id=call_id,
                        name=item.get("name", ""),
                        content=output,
                        # Honor a client-supplied error flag (parity with the
                        # anthropic / openai-chat parsers) instead of hardcoding.
                        is_error=bool(item.get("is_error", False)),
                        source_protocol="openai_responses",
                        raw=item,
                    ))
                elif item.get("type") in ("input_text", "text"):
                    text_parts.append(item.get("text", ""))
                elif item.get("type") in ("input_image", "image_url"):
                    url_value, detail = _openai_image_url_value(item)
                    image = image_from_url_value(
                        "openai_responses",
                        url_value,
                        detail=detail,
                        raw=item,
                    )
                    if image:
                        images.append(image)
    text = "\n".join(text_parts)

    # Parse tools
    tools: list[ToolSpec] = []
    raw_tools = body.get("tools", [])
    if raw_tools:
        tools = normalize_openai_tools(raw_tools, "openai_responses")

    return ParsedAgentTask(
        text=text,
        tools=tools,
        tool_results=tool_results,
        profile=profile,
        source_protocol="openai_responses",
        stream=stream,
        raw_request=body,
        images=images,
    )


def parse_anthropic_messages_input(
    body: dict,
) -> ParsedAgentTask:
    """Parse an Anthropic Messages API request body into ParsedAgentTask."""
    model = body.get("model", "claude-qwable-fast")
    profile = resolve_profile(model, "anthropic_messages")
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 1024)

    # Build text from messages
    text_parts: list[str] = []
    tool_results: list[ToolResult] = []
    images: list[ImageInput] = []

    # System prompt
    system = body.get("system", "")
    if system:
        text_parts.append(f"[system]: {system}")

    messages = body.get("messages", [])
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, str):
            text_parts.append(f"[{role}]: {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type == "text":
                        text_parts.append(f"[{role}]: {block.get('text', '')}")
                    elif block_type == "image":
                        source = block.get("source", {})
                        if isinstance(source, dict) and source.get("type") == "base64":
                            image = image_from_base64_source(
                                "anthropic_messages",
                                data_base64=source.get("data"),
                                mime_type=source.get("media_type"),
                                raw=block,
                            )
                        elif isinstance(source, dict):
                            image = image_from_url_value(
                                "anthropic_messages",
                                source.get("url"),
                                raw=block,
                            )
                        else:
                            image = None
                        if image:
                            images.append(image)
                    elif block_type == "tool_use":
                        # Tool use from assistant — will be passed back to model
                        text_parts.append(
                            f"[tool_use id={block.get('id','')} name={block.get('name','')}]"
                        )
                    elif block_type == "tool_result":
                        # Convert tool_result to internal format
                        tr_id = block.get("tool_use_id", "")
                        tr_name = block.get("name", "")
                        tr_content = ""
                        tr_is_error = block.get("is_error", False)
                        tr_content_blocks = block.get("content", "")
                        had_non_text = False
                        if isinstance(tr_content_blocks, list):
                            for tb in tr_content_blocks:
                                if isinstance(tb, dict) and tb.get("type") == "text":
                                    tr_content += tb.get("text", "") + "\n"
                                else:
                                    had_non_text = True
                        elif isinstance(tr_content_blocks, str):
                            tr_content = tr_content_blocks
                        # Only substitute the placeholder when non-text blocks were
                        # actually dropped — a legitimately empty result stays empty.
                        if not tr_content.strip() and had_non_text:
                            tr_content = "[unsupported non-text block omitted]"
                        tool_results.append(ToolResult(
                            tool_call_id=tr_id,
                            name=tr_name,
                            content=tr_content,
                            is_error=tr_is_error,
                            source_protocol="anthropic_messages",
                            raw=block,
                        ))
                        text_parts.append(f"[tool_result {tr_id}]: {tr_content}")

    text = "\n".join(text_parts)

    # Parse tools
    tools: list[ToolSpec] = []
    raw_tools = body.get("tools", [])
    if raw_tools:
        tools = normalize_anthropic_tools(raw_tools)

    return ParsedAgentTask(
        text=text,
        tools=tools,
        tool_results=tool_results,
        profile=profile,
        source_protocol="anthropic_messages",
        stream=stream,
        raw_request=body,
        images=images,
    )


def parse_openai_chat_input(
    body: dict,
) -> ParsedAgentTask:
    """Parse an OpenAI Chat Completions API request body into ParsedAgentTask."""
    model = body.get("model", "qwable-chat")
    profile = resolve_profile(model, "openai_chat")
    stream = body.get("stream", False)

    # Build text from messages
    text_parts: list[str] = []
    tool_results: list[ToolResult] = []
    images: list[ImageInput] = []

    messages = body.get("messages", [])
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, str):
            text_parts.append(f"[{role}]: {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    bt = block.get("type", "")
                    if bt == "text":
                        text_parts.append(f"[{role}]: {block.get('text', '')}")
                    elif bt == "image_url":
                        url_value, detail = _openai_image_url_value(block)
                        image = image_from_url_value(
                            "openai_chat",
                            url_value,
                            detail=detail,
                            raw=block,
                        )
                        if image:
                            images.append(image)
                    elif bt == "tool_result":
                        tr_id = block.get("tool_use_id", "")
                        tr_content = ""
                        tr_blocks = block.get("content", "")
                        if isinstance(tr_blocks, list):
                            for tb in tr_blocks:
                                if isinstance(tb, dict) and tb.get("type") == "text":
                                    tr_content += tb.get("text", "") + "\n"
                        elif isinstance(tr_blocks, str):
                            tr_content = tr_blocks
                        tool_results.append(ToolResult(
                            tool_call_id=tr_id,
                            name=block.get("name", ""),
                            content=tr_content,
                            is_error=block.get("is_error", False),
                            source_protocol="openai_chat",
                            raw=block,
                        ))

    text = "\n".join(text_parts)

    # Parse tools
    tools: list[ToolSpec] = []
    raw_tools = body.get("tools", [])
    if raw_tools:
        tools = normalize_openai_tools(raw_tools, "openai_chat")

    return ParsedAgentTask(
        text=text,
        tools=tools,
        tool_results=tool_results,
        profile=profile,
        source_protocol="openai_chat",
        stream=stream,
        raw_request=body,
        images=images,
    )
