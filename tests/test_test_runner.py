"""Tests for standard agent test tool-call planning."""


def test_infer_test_commands_uses_explicit_prompt_command_first():
    from qwable.context_pack import ContextPack
    from qwable.test_runner import TestCommandPlanner

    pack = ContextPack(
        goal="Fix tests",
        workflow="coding-workflow",
        raw_evidence=["Run tests: python -m pytest tests/test_agent_state.py"],
    )

    assert TestCommandPlanner().infer_test_commands(pack, "coding-workflow") == [
        "python -m pytest tests/test_agent_state.py"
    ]


def test_infer_test_commands_detects_python_project():
    from qwable.context_pack import ContextFileSummary, ContextPack
    from qwable.test_runner import TestCommandPlanner

    pack = ContextPack(
        goal="Implement Python feature",
        workflow="coding-workflow",
        files=[
            ContextFileSummary(path="pyproject.toml", reason="project metadata", summary="Python project"),
            ContextFileSummary(path="tests/test_feature.py", reason="test", summary="pytest file"),
        ],
    )

    assert TestCommandPlanner().infer_test_commands(pack, "coding-workflow") == ["python -m pytest"]


def test_infer_test_commands_detects_node_project():
    from qwable.context_pack import ContextFileSummary, ContextPack
    from qwable.test_runner import TestCommandPlanner

    pack = ContextPack(
        goal="Implement UI feature",
        workflow="coding-workflow",
        files=[
            ContextFileSummary(path="package.json", reason="project metadata", summary="Node project"),
            ContextFileSummary(path="src/App.test.tsx", reason="test", summary="frontend test"),
        ],
    )

    assert TestCommandPlanner().infer_test_commands(pack, "coding-workflow") == ["npm test"]


def test_infer_test_commands_detects_node_package_managers():
    from qwable.context_pack import ContextFileSummary, ContextPack
    from qwable.test_runner import TestCommandPlanner

    pnpm_pack = ContextPack(
        goal="Implement UI feature",
        workflow="coding-workflow",
        files=[
            ContextFileSummary(path="package.json", reason="project metadata", summary="Node project"),
            ContextFileSummary(path="pnpm-lock.yaml", reason="lockfile", summary="pnpm lock"),
        ],
    )
    bun_pack = ContextPack(
        goal="Implement UI feature",
        workflow="coding-workflow",
        files=[
            ContextFileSummary(path="package.json", reason="project metadata", summary="Node project"),
            ContextFileSummary(path="bun.lockb", reason="lockfile", summary="bun lock"),
        ],
    )

    assert TestCommandPlanner().infer_test_commands(pnpm_pack, "coding-workflow") == ["pnpm test"]
    assert TestCommandPlanner().infer_test_commands(bun_pack, "coding-workflow") == ["bun test"]


def test_infer_test_commands_returns_empty_when_no_signal_exists():
    from qwable.context_pack import ContextPack
    from qwable.test_runner import TestCommandPlanner

    pack = ContextPack(goal="Unknown project", workflow="agentic-workflow")

    assert TestCommandPlanner().infer_test_commands(pack, "agentic-workflow") == []


def test_build_test_tool_call_uses_standard_shape():
    from qwable.test_runner import TestCommandPlanner

    assert TestCommandPlanner().build_test_tool_call("python -m pytest") == {
        "tool_name": "run_tests",
        "tool_input": {"command": "python -m pytest"},
    }
