"""Tests for context pack rendering."""


def test_context_pack_prompt_text_includes_sections_and_truncates():
    from qwable.context_pack import ContextFileSummary, ContextPack

    pack = ContextPack(
        goal="Implement v1.7",
        workflow="coding-workflow",
        files=[
            ContextFileSummary(
                path="qwable/server.py",
                reason="mentioned in tool result",
                summary="FastAPI routes live here",
                symbols=["app", "openai_responses"],
                score=9.0,
            )
        ],
        raw_evidence=["pytest tests/test_server.py"],
        constraints=["do not read arbitrary filesystem paths"],
        risks=["missing model selector"],
        metadata={"source": "test"},
    )

    text = pack.to_prompt_text(max_chars=10_000)

    assert "CONTEXT_PACK" in text
    assert "workflow=coding-workflow" in text
    assert "goal=Implement v1.7" in text
    assert "CONSTRAINTS:" in text
    assert "- do not read arbitrary filesystem paths" in text
    assert "FILES:" in text
    assert "- qwable/server.py: mentioned in tool result" in text
    assert "summary: FastAPI routes live here" in text
    assert "symbols: app, openai_responses" in text
    assert "RAW_EVIDENCE:" in text
    assert "RISKS:" in text

    assert pack.to_prompt_text(max_chars=32) == text[:32]


def test_context_pack_mutable_defaults_are_not_shared():
    from qwable.context_pack import ContextFileSummary, ContextPack

    one = ContextPack(goal="one", workflow="agentic-workflow")
    two = ContextPack(goal="two", workflow="review-workflow")
    one.files.append(
        ContextFileSummary(path="README.md", reason="doc", summary="readme")
    )
    one.raw_evidence.append("evidence")
    one.metadata["key"] = "value"

    assert two.files == []
    assert two.raw_evidence == []
    assert two.metadata == {}
