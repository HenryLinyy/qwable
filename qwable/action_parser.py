"""Parse model output into FusionAction."""

import json
from qwable.schemas import FusionAction
from qwable.text_filters import clean_model_output
from qwable.tool_validation import parse_tool_arguments, validate_tool_call


def parse_action_from_text(
    raw: str | dict,
    tools: list[dict] | None = None,
) -> FusionAction:
    """Parse model output into a FusionAction.

    Strategy:
    1. Try JSON action schema from model (judge-style output)
    2. Try native tool_calls from model (OpenAI-style)
    3. Fall back to final_answer with full text
    """
    # Strategy 1: Check for JSON action schema
    if isinstance(raw, dict):
        text = ""
    else:
        cleaned = clean_model_output(raw)
        if not cleaned.ok:
            return FusionAction(
                type="final_answer",
                text="",
                tool_name=None,
                tool_input=None,
                confidence=0.0,
                rationale_summary=cleaned.error,
            )
        text = cleaned.clean_text.strip()

    # Try to find a JSON block
    json_start = text.find("{")
    json_end = text.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        try:
            parsed = json.loads(text[json_start : json_end + 1])
            action_type = parsed.get("type", "")
            if action_type == "tool_call":
                tool_input, input_error = parse_tool_arguments(
                    parsed.get("tool_input", {})
                )
                if input_error:
                    return _validation_failure(input_error)
                ok, error = validate_tool_call(
                    parsed.get("tool_name"), tool_input, tools
                )
                if not ok:
                    return _validation_failure(error or "tool validation failed")
                return FusionAction(
                    type="tool_call",
                    text=None,
                    tool_name=parsed.get("tool_name"),
                    tool_input=tool_input,
                    confidence=parsed.get("confidence"),
                    rationale_summary=parsed.get("rationale_summary"),
                )
            elif action_type == "final_answer":
                return FusionAction(
                    type="final_answer",
                    # #9: default to empty, not the raw JSON substring.
                    text=parsed.get("text") or "",
                    tool_name=None,
                    tool_input=None,
                    confidence=parsed.get("confidence"),
                    rationale_summary=parsed.get("rationale_summary"),
                )
        except json.JSONDecodeError:
            pass

    # Strategy 2: Check for native tool_calls from Ollama
    if tools and isinstance(raw, dict):
        # Handle two formats:
        # 1. Direct: {"tool_calls": [...]}
        # 2. Nested in choices[0].message: {"choices": [{"message": {"tool_calls": [...]}}]}
        tool_calls = None
        if "tool_calls" in raw:
            tool_calls = raw["tool_calls"]
        elif "choices" in raw:
            choices = raw["choices"]
            # Guard: choices[0] may not be a dict on malformed model output.
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                msg = choices[0].get("message", {})
                if isinstance(msg, dict) and "tool_calls" in msg:
                    tool_calls = msg["tool_calls"]

        # Guard every shape: malformed tool_calls (non-list, non-dict element,
        # or non-dict function) must fall through to final_answer, not crash.
        if (
            isinstance(tool_calls, list)
            and tool_calls
            and isinstance(tool_calls[0], dict)
        ):
            tc0 = tool_calls[0]
            func = tc0.get("function", {})
            if isinstance(func, dict):
                tool_input, input_error = parse_tool_arguments(
                    func.get("arguments", {})
                )
                if input_error:
                    return _validation_failure(input_error)
                ok, error = validate_tool_call(func.get("name"), tool_input, tools)
                if not ok:
                    return _validation_failure(error or "tool validation failed")
                return FusionAction(
                    type="tool_call",
                    text=None,
                    tool_name=func.get("name"),
                    tool_input=tool_input,
                    confidence=1.0,
                    rationale_summary="Native tool call from model",
                )

    # Strategy 3: Fall back to final_answer
    return FusionAction(
        type="final_answer",
        text=text,
        tool_name=None,
        tool_input=None,
        confidence=1.0,
        rationale_summary="Direct answer from model",
    )


def _validation_failure(error: str) -> FusionAction:
    return FusionAction(
        type="final_answer",
        text="",
        tool_name=None,
        tool_input=None,
        confidence=0.0,
        rationale_summary=f"Tool validation failed: {error}",
    )
