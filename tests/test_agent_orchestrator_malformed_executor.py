"""Regression tests: malformed executor/repair JSON must not 500.

Real-world bug: the executor model emitted output that contained braces but was
not valid JSON. ``_parse_json_object`` called ``json.loads`` without a guard, so
a ``json.JSONDecodeError`` escaped ``_execute_current_step`` and the whole
request returned HTTP 500 with an empty body.

The fix normalizes the parse failure to ``ValueError`` and wraps the executor
parse so malformed output is retained as a ``final_answer`` instead of crashing.
"""

import json

import pytest

from qwable.agent_orchestrator import _parse_json_object
from tests import test_agent_orchestrator as t


_MALFORMED_WITH_BRACES = '{"a": "b" "c": "d"}'


def test_parse_json_object_raises_plain_valueerror_on_malformed_braces():
    """Must raise ValueError, NOT a leaking json.JSONDecodeError."""
    with pytest.raises(ValueError) as exc_info:
        _parse_json_object(_MALFORMED_WITH_BRACES)
    assert not isinstance(exc_info.value, json.JSONDecodeError)


async def test_coding_workflow_malformed_executor_json_does_not_500(tmp_path):
    """Executor emits brace-y-but-invalid JSON → graceful final_answer, no raise.

    v1.8's ``_orchestrator`` takes plain response strings (returned as
    ``{"content": ...}``), unlike main's reasoning-response helper.
    """
    # Brace-y but invalid JSON (prose prefix + missing comma). Before the fix
    # this raised JSONDecodeError out of orchestrator.run() (-> HTTP 500).
    malformed_executor = 'Here is the code: {"step_result": {"status": "done" "content": "x"}}'
    orchestrator, _client, _selector, _store, _compactor = t._orchestrator(
        tmp_path,
        [
            t._planner_json(),
            t._plan_critic_json(fatal=False),
            malformed_executor,
        ],
    )

    action = await orchestrator.run(t._task("do a coding task"), "coding-workflow")

    assert action.type == "final_answer"
    assert "step_result" in (action.text or "") or "x" in (action.text or "")
