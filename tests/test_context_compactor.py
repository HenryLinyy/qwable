"""Tests for deterministic context pack construction."""


def _task(text, *, tool_results=None, vision_evidence=None):
    from qwable.schemas import ParsedAgentTask

    return ParsedAgentTask(
        text=text,
        tools=[],
        tool_results=tool_results or [],
        profile="coding-workflow",
        source_protocol="openai_responses",
        stream=False,
        raw_request={"input": text},
        vision_evidence=vision_evidence or [],
    )


def test_build_pack_uses_task_text_and_tool_results_as_evidence():
    from qwable.config import FusionConfig
    from qwable.context_compactor import ContextCompactor
    from qwable.schemas import ToolResult

    task = _task(
        "Fix qwable/server.py using tests/test_server.py",
        tool_results=[
            ToolResult(
                tool_call_id="call_1",
                name="list_files",
                content="qwable/config.py\nREADME.md",
                is_error=False,
                source_protocol="openai_responses",
                raw={},
            )
        ],
    )

    pack = ContextCompactor(FusionConfig()).build_pack(task, "coding-workflow")

    assert pack.goal == task.text
    assert pack.workflow == "coding-workflow"
    assert "Fix qwable/server.py" in pack.raw_evidence[0]
    assert "tool_result:list_files" in pack.raw_evidence[1]
    assert [file.path for file in pack.files] == [
        "qwable/server.py",
        "tests/test_server.py",
        "qwable/config.py",
        "README.md",
    ]
    assert pack.constraints == [
        "RepoIndex uses only request text, tool results, and vision evidence; it does not read arbitrary filesystem paths."
    ]
    assert pack.metadata["tool_result_count"] == 1


def test_build_pack_adds_error_risk_and_vision_evidence():
    from qwable.config import FusionConfig
    from qwable.context_compactor import ContextCompactor
    from qwable.schemas import ToolResult
    from qwable.vision import VisionEvidence

    task = _task(
        "Review screenshot",
        tool_results=[
            ToolResult(
                tool_call_id="call_2",
                name="read_file",
                content="permission denied for qwable/server.py",
                is_error=True,
                source_protocol="openai_responses",
                raw={},
            )
        ],
        vision_evidence=[
            VisionEvidence(
                model="vision",
                profile="vision-pro",
                summary="UI shows error in README.md",
                visible_text="Traceback in tests/test_ui.py",
                warnings=["low confidence"],
                confidence=0.4,
            )
        ],
    )

    pack = ContextCompactor(FusionConfig()).build_pack(task, "review-workflow")

    assert any("tool_result_error:read_file" in risk for risk in pack.risks)
    assert any("vision:vision-pro" in evidence for evidence in pack.raw_evidence)
    assert any(file.path == "README.md" for file in pack.files)
    assert any(file.path == "tests/test_ui.py" for file in pack.files)
    assert pack.metadata["vision_evidence_count"] == 1
