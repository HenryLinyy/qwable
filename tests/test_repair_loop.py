"""Tests for repair-loop decisions."""


def _run_with_step(attempt_count: int = 0):
    from qwable.agent_state import AgentRun, AgentStep

    run = AgentRun.create(goal="Fix failing tests", workflow="coding-workflow")
    run.plan = [
        AgentStep(
            step_id="step_1",
            title="Patch",
            intent="Repair test failure",
            attempt_count=attempt_count,
        )
    ]
    return run


def test_decide_repairs_actionable_failure_under_limits():
    from qwable.config import FusionConfig
    from qwable.repair_loop import RepairLoop

    run = _run_with_step()
    config = FusionConfig(agent_max_repair_attempts=2)

    decision = RepairLoop(config).decide(
        run, "AssertionError in tests/test_agent_store.py"
    )

    assert decision.should_repair is True
    assert decision.reason == "repairable_failure"
    assert decision.failure_summary == "AssertionError in tests/test_agent_store.py"


def test_decide_stops_after_max_repair_attempts():
    from qwable.config import FusionConfig
    from qwable.repair_loop import RepairLoop

    run = _run_with_step()
    run.repair_count = 2
    config = FusionConfig(agent_max_repair_attempts=2)

    decision = RepairLoop(config).decide(
        run, "AssertionError in tests/test_agent_store.py"
    )

    assert decision.should_repair is False
    assert decision.reason == "max_repair_attempts_exceeded"


def test_decide_stops_after_step_attempt_limit():
    from qwable.config import FusionConfig
    from qwable.repair_loop import RepairLoop

    run = _run_with_step(attempt_count=3)
    config = FusionConfig(agent_max_repair_attempts=5)

    decision = RepairLoop(config).decide(
        run, "AssertionError in tests/test_agent_store.py"
    )

    assert decision.should_repair is False
    assert decision.reason == "step_attempt_limit_exceeded"


def test_decide_stops_when_failure_is_not_actionable():
    from qwable.config import FusionConfig
    from qwable.repair_loop import RepairLoop

    decision = RepairLoop(FusionConfig()).decide(_run_with_step(), "nope")

    assert decision.should_repair is False
    assert decision.reason == "not_actionable"


def test_decide_blocks_security_permission_and_external_service_failures():
    from qwable.config import FusionConfig
    from qwable.repair_loop import RepairLoop

    loop = RepairLoop(FusionConfig())
    failures = [
        "Permission denied while opening /private/config",
        "Security policy blocked shell command",
        "Connection refused: external service unavailable",
        "401 unauthorized from remote API",
    ]

    for failure in failures:
        decision = loop.decide(_run_with_step(), failure)
        assert decision.should_repair is False
        assert decision.reason == "blocked_external_or_security"


def test_failure_summary_is_trimmed_to_first_meaningful_text():
    from qwable.config import FusionConfig
    from qwable.repair_loop import RepairLoop

    decision = RepairLoop(FusionConfig()).decide(
        _run_with_step(),
        "\n\nTraceback (most recent call last):\n" + ("x" * 800),
    )

    assert decision.should_repair is True
    assert decision.failure_summary.startswith("Traceback")
    assert len(decision.failure_summary) <= 500
