"""Tests for the v1.7 agent orchestrator core."""

import json


def _task(text="Implement qwable/agent_orchestrator.py", *, raw_request=None, tool_results=None):
    from qwable.schemas import ParsedAgentTask

    return ParsedAgentTask(
        text=text,
        tools=[],
        tool_results=tool_results or [],
        profile="coding-workflow",
        source_protocol="openai_responses",
        stream=False,
        raw_request=raw_request or {"input": text},
    )


class FakeModelClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, *, model, messages, max_tokens, temperature):
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return {"content": response}


class FakeSelector:
    def __init__(self):
        self.calls = []

    def select(self, workflow, stage, *, temperature=0.2):
        from qwable.model_roles import ModelRole, RoleSelection

        self.calls.append((workflow, stage, temperature))
        role_by_stage = {
            "planner": ModelRole.PLANNER,
            "plan_critic": ModelRole.CRITIC,
            "executor": ModelRole.EXECUTOR,
            "test": ModelRole.EXECUTOR,
            "repair": ModelRole.REPAIR,
            "reviewer": ModelRole.CRITIC,
            "judge": ModelRole.JUDGE,
            "finalizer": ModelRole.JUDGE,
        }
        return RoleSelection(
            workflow=workflow,
            stage=stage,
            role=role_by_stage[stage],
            model=f"model/{stage}",
            fallback_chain=[f"model/{stage}", f"fallback/{stage}"],
            max_tokens=111 if stage == "planner" else 222,
            temperature=temperature,
            reason=f"fake {stage}",
        )


class FakeCompactor:
    def __init__(self):
        self.calls = []

    def build_pack(self, task, workflow):
        from qwable.context_pack import ContextPack

        self.calls.append((task.text, workflow))
        return ContextPack(goal=task.text, workflow=workflow, raw_evidence=[f"request:{task.text}"])


def _store(tmp_path):
    from qwable.agent_store import AgentStore

    store = AgentStore(str(tmp_path / "agent_runs.sqlite3"))
    store.init_schema()
    return store


def _orchestrator(tmp_path, responses, **config_overrides):
    from qwable.agent_orchestrator import AgentOrchestrator
    from qwable.config import FusionConfig

    selector = FakeSelector()
    client = FakeModelClient(responses)
    compactor = FakeCompactor()
    store = _store(tmp_path)
    orchestrator = AgentOrchestrator(
        config=FusionConfig(agent_store_path=str(tmp_path / "unused.sqlite3"), **config_overrides),
        model_client=client,
        model_selector=selector,
        store=store,
        compactor=compactor,
    )
    return orchestrator, client, selector, store, compactor


def _planner_json():
    return json.dumps(
        {
            "steps": [
                {
                    "title": "Inspect target",
                    "intent": "Read requested file",
                    "required_tools": ["read_file"],
                    "success_criteria": ["target inspected"],
                    "failure_criteria": ["file missing"],
                }
            ],
            "risks": ["needs tool result"],
            "test_strategy": ["run focused pytest"],
        }
    )


def _two_step_planner_json():
    return json.dumps(
        {
            "steps": [
                {
                    "title": "Inspect target",
                    "intent": "Read requested file",
                    "required_tools": ["read_file"],
                    "success_criteria": ["target inspected"],
                },
                {
                    "title": "Patch target",
                    "intent": "Apply patch to target file",
                    "required_tools": ["apply_patch"],
                    "success_criteria": ["patch applied"],
                },
            ]
        }
    )


def _plan_critic_json(*, fatal=False):
    return json.dumps(
        {
            "fatal_blocker": fatal,
            "blockers": ["missing source evidence"] if fatal else [],
            "risks": ["needs cited evidence"],
        }
    )


def _tool_result(name, content, *, is_error=False):
    from qwable.schemas import ToolResult

    return ToolResult(
        tool_call_id="call_1",
        name=name,
        content=content,
        is_error=is_error,
        source_protocol="openai_responses",
        raw={"content": content},
    )


def _saved_testing_run(store):
    from qwable.agent_state import AgentRun, AgentStep

    run = AgentRun.create(goal="Fix tests", workflow="coding-workflow")
    run.status = "testing"
    run.plan = [
        AgentStep(
            step_id="step_1",
            title="Patch",
            intent="Apply change",
            status="done",
            attempt_count=1,
        )
    ]
    store.save_run(run)
    return run


def _assert_standard_agent_trace(trace, *, stage, status, role):
    assert trace["agent_runtime"] is True
    assert trace["agent_run_id"].startswith("run_")
    assert trace["workflow"] in {"agentic-workflow", "coding-workflow", "review-workflow"}
    assert trace["agent_status"] == status
    assert trace["stage"] == stage
    assert trace["model_role"] == role
    assert isinstance(trace["selected_model"], str)
    assert isinstance(trace["fallback_chain"], list)
    assert isinstance(trace["current_step_index"], int)
    assert isinstance(trace["repair_count"], int)
    assert isinstance(trace["tool_call_count"], int)


def test_extract_agent_run_id_accepts_valid_metadata_only():
    from qwable.agent_orchestrator import extract_agent_run_id

    assert extract_agent_run_id(_task(raw_request={"metadata": {"agent_run_id": "run_abc"}})) == "run_abc"
    assert extract_agent_run_id(_task(raw_request={"metadata": {"agent_run_id": "bad_abc"}})) is None
    assert extract_agent_run_id(_task(raw_request={"metadata": []})) is None
    assert extract_agent_run_id(_task(raw_request={})) is None


async def test_run_uses_model_selector_for_planner_and_executor(tmp_path):
    orchestrator, client, selector, store, compactor = _orchestrator(
        tmp_path,
        [
            _planner_json(),
            _plan_critic_json(),
            json.dumps({"tool_call": {"name": "read_file", "input": {"path": "qwable/server.py"}}}),
        ],
    )

    action = await orchestrator.run(_task(), "coding-workflow")

    assert [call[:2] for call in selector.calls] == [
        ("coding-workflow", "planner"),
        ("coding-workflow", "plan_critic"),
        ("coding-workflow", "executor"),
    ]
    assert [call["model"] for call in client.calls] == [
        "model/planner",
        "model/plan_critic",
        "model/executor",
    ]
    assert action.type == "tool_call"
    assert action.tool_name == "read_file"
    assert action.tool_input == {"path": "qwable/server.py"}
    assert action.confidence == 0.8
    assert action.rationale_summary == "executor_requested_tool"
    _assert_standard_agent_trace(
        action.trace,
        stage="executor",
        status="waiting_for_tool",
        role="executor",
    )
    assert action.trace["selected_model"] == "model/executor"
    assert action.trace["fallback_chain"] == ["model/executor", "fallback/executor"]
    assert compactor.calls == [("Implement qwable/agent_orchestrator.py", "coding-workflow")]

    loaded = store.load_run(action.trace["agent_run_id"])
    assert loaded is not None
    assert loaded.status == "waiting_for_tool"
    assert loaded.plan[0].title == "Inspect target"
    assert loaded.plan[0].required_tools == ["read_file"]
    assert loaded.plan[0].status == "waiting_for_tool"
    assert loaded.tool_call_count == 1


async def test_invalid_planner_json_returns_failed_final_answer(tmp_path):
    orchestrator, client, selector, store, _compactor = _orchestrator(tmp_path, ["not json"])

    action = await orchestrator.run(_task("Plan impossible"), "agentic-workflow")

    assert action.type == "final_answer"
    assert action.tool_name is None
    assert action.text is not None
    assert "planner_json_parse_failed" in action.text
    assert action.trace["agent_status"] == "failed"
    assert action.trace["stage"] == "planner"
    assert action.trace["model_role"] == "planner"
    assert action.trace["selected_model"] == "model/planner"
    assert action.trace["raw_planner_output"] == "not json"

    loaded = store.load_run(action.trace["agent_run_id"])
    assert loaded is not None
    assert loaded.status == "failed"
    assert loaded.failures[0].stage == "planner"
    assert loaded.failures[0].metadata["reason"] == "planner_json_parse_failed"


async def test_agentic_workflow_runs_plan_critic_before_executor(tmp_path):
    orchestrator, client, selector, store, _compactor = _orchestrator(
        tmp_path,
        [
            _planner_json(),
            _plan_critic_json(),
            json.dumps({"step_result": {"summary": "research organized"}}),
        ],
    )

    action = await orchestrator.run(_task("Organize notes into a plan"), "agentic-workflow")

    assert action.type == "final_answer"
    assert action.text == json.dumps({"step_result": {"summary": "research organized"}})
    assert [call[:2] for call in selector.calls] == [
        ("agentic-workflow", "planner"),
        ("agentic-workflow", "plan_critic"),
        ("agentic-workflow", "executor"),
    ]
    assert [call["model"] for call in client.calls] == [
        "model/planner",
        "model/plan_critic",
        "model/executor",
    ]
    assert action.trace["workflow"] == "agentic-workflow"
    assert action.trace["stage"] == "executor"

    loaded = store.load_run(action.trace["agent_run_id"])
    assert loaded is not None
    assert loaded.status == "completed"
    assert any(artifact.kind == "plan_review" for artifact in loaded.artifacts)


async def test_agentic_workflow_blocks_on_fatal_plan_critic(tmp_path):
    orchestrator, client, selector, store, _compactor = _orchestrator(
        tmp_path,
        [
            _planner_json(),
            _plan_critic_json(fatal=True),
        ],
    )

    action = await orchestrator.run(_task("Organize unsupported claims"), "agentic-workflow")

    assert action.type == "final_answer"
    assert "plan_critic_blocked:fatal_blocker" in action.text
    assert action.trace["agent_status"] == "blocked"
    assert action.trace["stage"] == "plan_critic"
    assert action.trace["fatal_blocker"] is True
    assert [call[:2] for call in selector.calls] == [
        ("agentic-workflow", "planner"),
        ("agentic-workflow", "plan_critic"),
    ]
    assert [call["model"] for call in client.calls] == ["model/planner", "model/plan_critic"]

    loaded = store.load_run(action.trace["agent_run_id"])
    assert loaded is not None
    assert loaded.status == "blocked"
    assert loaded.current_step().status == "blocked"
    assert loaded.failures[0].stage == "plan_critic"
    assert loaded.failures[0].metadata["reason"] == "fatal_blocker"


async def test_invalid_executor_tool_call_returns_blocked_final_answer(tmp_path):
    orchestrator, _client, _selector, store, _compactor = _orchestrator(
        tmp_path,
        [
            _planner_json(),
            _plan_critic_json(),
            json.dumps({"tool_call": {"name": "shell", "input": {"command": "rm -rf /"}}}),
        ],
    )

    action = await orchestrator.run(_task("Do dangerous thing"), "coding-workflow")

    assert action.type == "final_answer"
    assert "tool_call_rejected" in action.text
    assert action.trace["agent_status"] == "blocked"
    assert action.trace["stage"] == "executor"
    assert action.trace["tool_validation_error"] == "destructive_command"

    loaded = store.load_run(action.trace["agent_run_id"])
    assert loaded is not None
    assert loaded.status == "blocked"
    assert loaded.current_step().status == "blocked"


async def test_call_with_fallback_tries_next_model_after_failure(tmp_path):
    orchestrator, client, _selector, _store, _compactor = _orchestrator(
        tmp_path,
        [RuntimeError("primary down"), json.dumps({"ok": True})],
    )
    selection = orchestrator.model_selector.select("coding-workflow", "executor")

    response = await orchestrator._call_with_fallback(selection, [{"role": "user", "content": "hi"}])

    assert response == {"content": json.dumps({"ok": True})}
    assert [call["model"] for call in client.calls] == ["model/executor", "fallback/executor"]
    assert orchestrator.core_last_used_model == "fallback/executor"


async def test_tool_result_continuation_appends_evidence_and_requests_coding_tests(tmp_path):
    orchestrator, client, selector, store, _compactor = _orchestrator(
        tmp_path,
        [
            _planner_json(),
            _plan_critic_json(),
            json.dumps({"tool_call": {"name": "read_file", "input": {"path": "qwable/server.py"}}}),
        ],
    )
    first = await orchestrator.run(
        _task("Inspect qwable/server.py. Run tests: python -m pytest tests/test_agent_orchestrator.py"),
        "coding-workflow",
    )

    continuation = await orchestrator.run(
        _task(
            "Inspect qwable/server.py. Run tests: python -m pytest tests/test_agent_orchestrator.py",
            raw_request={"metadata": {"agent_run_id": first.trace["agent_run_id"]}},
            tool_results=[_tool_result("read_file", "def app(): pass")],
        ),
        "coding-workflow",
    )

    assert continuation.type == "tool_call"
    assert continuation.tool_name == "run_tests"
    assert continuation.tool_input == {"command": "python -m pytest tests/test_agent_orchestrator.py"}
    assert continuation.trace["stage"] == "test"
    assert continuation.trace["agent_status"] == "testing"
    assert continuation.trace["model_role"] == "executor"
    assert continuation.confidence == 0.8
    assert continuation.rationale_summary == "test_requested_tool"
    _assert_standard_agent_trace(
        continuation.trace,
        stage="test",
        status="testing",
        role="executor",
    )
    assert [call["model"] for call in client.calls] == [
        "model/planner",
        "model/plan_critic",
        "model/executor",
    ]
    assert ("coding-workflow", "test", 0.2) in selector.calls

    loaded = store.load_run(first.trace["agent_run_id"])
    assert loaded is not None
    assert loaded.status == "testing"
    assert loaded.current_step().status == "done"
    assert "read_file:def app(): pass" in loaded.current_step().evidence
    assert any(artifact.kind == "tool_result" for artifact in loaded.artifacts)
    assert any(
        artifact.kind == "tool_call" and '"tool_name": "run_tests"' in artifact.content
        for artifact in loaded.artifacts
    )


async def test_coding_workflow_advances_to_next_step_after_tool_result(tmp_path):
    orchestrator, client, selector, store, _compactor = _orchestrator(
        tmp_path,
        [
            _two_step_planner_json(),
            _plan_critic_json(),
            json.dumps({"tool_call": {"name": "read_file", "input": {"path": "qwable/server.py"}}}),
            json.dumps(
                {
                    "tool_call": {
                        "name": "apply_patch",
                        "input": {
                            "path": "qwable/server.py",
                            "patch": "*** Begin Patch\n*** End Patch\n",
                        },
                    }
                }
            ),
        ],
    )
    first = await orchestrator.run(_task("Inspect and patch qwable/server.py"), "coding-workflow")

    second = await orchestrator.run(
        _task(
            "Inspect and patch qwable/server.py",
            raw_request={"metadata": {"agent_run_id": first.trace["agent_run_id"]}},
            tool_results=[_tool_result("read_file", "server source")],
        ),
        "coding-workflow",
    )

    assert second.type == "tool_call"
    assert second.tool_name == "apply_patch"
    assert second.trace["stage"] == "executor"
    assert second.trace["current_step_index"] == 1
    assert [call[:2] for call in selector.calls] == [
        ("coding-workflow", "planner"),
        ("coding-workflow", "plan_critic"),
        ("coding-workflow", "executor"),
        ("coding-workflow", "executor"),
    ]
    assert [call["model"] for call in client.calls] == [
        "model/planner",
        "model/plan_critic",
        "model/executor",
        "model/executor",
    ]

    loaded = store.load_run(first.trace["agent_run_id"])
    assert loaded is not None
    assert loaded.current_step_index == 1
    assert loaded.plan[0].status == "done"
    assert loaded.plan[1].status == "waiting_for_tool"
    assert loaded.tool_call_count == 2


async def test_agentic_workflow_advances_tool_steps_without_requesting_tests(tmp_path):
    orchestrator, client, selector, store, _compactor = _orchestrator(
        tmp_path,
        [
            _two_step_planner_json(),
            _plan_critic_json(),
            json.dumps({"tool_call": {"name": "read_file", "input": {"path": "qwable/server.py"}}}),
            json.dumps({"tool_call": {"name": "search_files", "input": {"query": "FusionCore"}}}),
        ],
    )
    first = await orchestrator.run(_task("Read and organize repository notes"), "agentic-workflow")

    second = await orchestrator.run(
        _task(
            "Read and organize repository notes",
            raw_request={"metadata": {"agent_run_id": first.trace["agent_run_id"]}},
            tool_results=[_tool_result("read_file", "server source")],
        ),
        "agentic-workflow",
    )

    assert second.type == "tool_call"
    assert second.tool_name == "search_files"
    assert second.trace["stage"] == "executor"
    assert second.trace["current_step_index"] == 1
    assert ("agentic-workflow", "test", 0.2) not in selector.calls
    assert [call[:2] for call in selector.calls] == [
        ("agentic-workflow", "planner"),
        ("agentic-workflow", "plan_critic"),
        ("agentic-workflow", "executor"),
        ("agentic-workflow", "executor"),
    ]
    assert [call["model"] for call in client.calls] == [
        "model/planner",
        "model/plan_critic",
        "model/executor",
        "model/executor",
    ]

    loaded = store.load_run(first.trace["agent_run_id"])
    assert loaded is not None
    assert loaded.status == "waiting_for_tool"
    assert loaded.current_step_index == 1
    assert loaded.plan[0].status == "done"
    assert loaded.plan[1].status == "waiting_for_tool"
    assert loaded.tool_call_count == 2


async def test_coding_workflow_blocks_when_tool_call_limit_would_be_exceeded(tmp_path):
    orchestrator, _client, _selector, store, _compactor = _orchestrator(
        tmp_path,
        [
            _planner_json(),
            _plan_critic_json(),
            json.dumps({"tool_call": {"name": "read_file", "input": {"path": "qwable/server.py"}}}),
        ],
        agent_max_tool_calls=0,
    )

    action = await orchestrator.run(_task("Inspect qwable/server.py"), "coding-workflow")

    assert action.type == "final_answer"
    assert "tool_call_limit_exceeded" in action.text
    assert action.trace["agent_status"] == "failed"
    assert action.trace["stage"] == "executor"
    assert action.trace["tool_call_limit"] == 0

    loaded = store.load_run(action.trace["agent_run_id"])
    assert loaded is not None
    assert loaded.status == "failed"
    assert loaded.tool_call_count == 0
    assert loaded.failures[0].metadata["reason"] == "tool_call_limit_exceeded"


async def test_coding_workflow_requires_tests_after_mutating_step(tmp_path):
    orchestrator, client, selector, store, _compactor = _orchestrator(
        tmp_path,
        [
            _planner_json(),
            _plan_critic_json(),
            json.dumps(
                {
                    "tool_call": {
                        "name": "apply_patch",
                        "input": {
                            "path": "qwable/server.py",
                            "patch": "*** Begin Patch\n*** End Patch\n",
                        },
                    }
                }
            ),
        ],
    )
    first = await orchestrator.run(
        _task("Patch qwable/server.py and tests/test_agent_orchestrator.py"),
        "coding-workflow",
    )

    continuation = await orchestrator.run(
        _task(
            "Patch qwable/server.py and tests/test_agent_orchestrator.py",
            raw_request={"metadata": {"agent_run_id": first.trace["agent_run_id"]}},
            tool_results=[_tool_result("apply_patch", "patch applied")],
        ),
        "coding-workflow",
    )

    assert continuation.type == "tool_call"
    assert continuation.tool_name == "run_tests"
    assert continuation.tool_input == {"command": "python -m pytest"}
    assert continuation.trace["stage"] == "test"
    assert ("coding-workflow", "test", 0.2) in selector.calls
    assert [call["model"] for call in client.calls] == [
        "model/planner",
        "model/plan_critic",
        "model/executor",
    ]

    loaded = store.load_run(first.trace["agent_run_id"])
    assert loaded is not None
    assert loaded.plan[0].status == "done"
    assert loaded.status == "testing"


async def test_successful_test_result_runs_finalizer(tmp_path):
    orchestrator, client, selector, store, _compactor = _orchestrator(tmp_path, ["Final report: tests passed"])
    run = _saved_testing_run(store)

    action = await orchestrator.run(
        _task(
            "Finish after tests",
            raw_request={"metadata": {"agent_run_id": run.run_id}},
            tool_results=[_tool_result("run_tests", "5 passed")],
        ),
        "coding-workflow",
    )

    assert action.type == "final_answer"
    assert action.text == "Final report: tests passed"
    assert action.trace["stage"] == "finalizer"
    assert action.trace["agent_status"] == "completed"
    assert action.trace["model_role"] == "judge"
    assert action.confidence == 0.8
    assert action.rationale_summary == "agent_completed"
    _assert_standard_agent_trace(
        action.trace,
        stage="finalizer",
        status="completed",
        role="judge",
    )
    assert ("coding-workflow", "finalizer", 0.2) in selector.calls

    loaded = store.load_run(run.run_id)
    assert loaded is not None
    assert loaded.status == "completed"
    assert loaded.artifacts[-1].kind == "final_report"
    assert loaded.artifacts[-1].content == "Final report: tests passed"


async def test_failed_test_result_uses_repair_loop_and_requests_repair_tool(tmp_path):
    orchestrator, client, selector, store, _compactor = _orchestrator(
        tmp_path,
        [
            json.dumps(
                {
                    "tool_call": {
                        "name": "apply_patch",
                        "input": {
                            "path": "qwable/server.py",
                            "patch": "*** Begin Patch\n*** End Patch\n",
                        },
                    }
                }
            )
        ],
    )
    run = _saved_testing_run(store)

    action = await orchestrator.run(
        _task(
            "Repair failed tests",
            raw_request={"metadata": {"agent_run_id": run.run_id}},
            tool_results=[_tool_result("run_tests", "AssertionError in tests/test_agent_orchestrator.py", is_error=True)],
        ),
        "coding-workflow",
    )

    assert action.type == "tool_call"
    assert action.tool_name == "apply_patch"
    assert action.tool_input["path"] == "qwable/server.py"
    assert action.trace["stage"] == "repair"
    assert action.trace["agent_status"] == "waiting_for_tool"
    assert action.trace["model_role"] == "repair"
    assert action.confidence == 0.8
    assert action.rationale_summary == "repair_requested_tool"
    _assert_standard_agent_trace(
        action.trace,
        stage="repair",
        status="waiting_for_tool",
        role="repair",
    )
    assert ("coding-workflow", "repair", 0.2) in selector.calls

    loaded = store.load_run(run.run_id)
    assert loaded is not None
    assert loaded.status == "waiting_for_tool"
    assert loaded.repair_count == 1
    assert loaded.current_step().status == "waiting_for_tool"


async def test_unrepairable_test_failure_returns_blocked_final_answer(tmp_path):
    orchestrator, _client, _selector, store, _compactor = _orchestrator(tmp_path, [])
    run = _saved_testing_run(store)
    run.repair_count = orchestrator.config.agent_max_repair_attempts
    store.save_run(run)

    action = await orchestrator.run(
        _task(
            "Repair failed tests",
            raw_request={"metadata": {"agent_run_id": run.run_id}},
            tool_results=[_tool_result("run_tests", "AssertionError in tests/test_agent_orchestrator.py", is_error=True)],
        ),
        "coding-workflow",
    )

    assert action.type == "final_answer"
    assert "repair_blocked:max_repair_attempts_exceeded" in action.text
    assert action.trace["stage"] == "repair"
    assert action.trace["agent_status"] == "blocked"
    assert action.trace["repair_decision"] == "max_repair_attempts_exceeded"

    loaded = store.load_run(run.run_id)
    assert loaded is not None
    assert loaded.status == "blocked"
    assert loaded.current_step().status == "blocked"


async def test_review_workflow_uses_reviewer_judge_finalizer_without_patch_loop(tmp_path):
    orchestrator, client, selector, store, _compactor = _orchestrator(
        tmp_path,
        [
            "1. Blockers\n- none\n2. High-risk issues\n- risky call",
            "Judge: review is evidence-backed",
            "Final review report",
        ],
    )

    action = await orchestrator.run(_task("Review qwable/server.py"), "review-workflow")

    assert action.type == "final_answer"
    assert action.text == "Final review report"
    assert action.tool_name is None
    assert [call[:2] for call in selector.calls] == [
        ("review-workflow", "reviewer"),
        ("review-workflow", "judge"),
        ("review-workflow", "finalizer"),
    ]
    assert [call["model"] for call in client.calls] == [
        "model/reviewer",
        "model/judge",
        "model/finalizer",
    ]
    assert action.trace["workflow"] == "review-workflow"
    assert action.trace["stage"] == "finalizer"
    assert action.trace["agent_status"] == "completed"
    assert action.trace["review_only"] is True
    assert action.rationale_summary == "review_completed"

    loaded = store.load_run(action.trace["agent_run_id"])
    assert loaded is not None
    assert loaded.status == "completed"
    assert loaded.tool_call_count == 0
    assert loaded.plan == []
    assert [artifact.kind for artifact in loaded.artifacts if artifact.kind in {"review", "final_report"}] == [
        "review",
        "review",
        "final_report",
    ]


async def test_review_workflow_suppresses_apply_patch_tool_call(tmp_path):
    orchestrator, _client, selector, store, _compactor = _orchestrator(
        tmp_path,
        [
            json.dumps(
                {
                    "tool_call": {
                        "name": "apply_patch",
                        "input": {
                            "path": "qwable/server.py",
                            "patch": "*** Begin Patch\n*** End Patch\n",
                        },
                    }
                }
            ),
            "Judge: reviewer suggested a patch but review mode is report-only",
            "Final review report without patch",
        ],
    )

    action = await orchestrator.run(_task("Review only; do not fix"), "review-workflow")

    assert action.type == "final_answer"
    assert action.tool_name is None
    assert action.text == "Final review report without patch"
    assert action.trace["review_only"] is True
    assert action.trace["suppressed_tool_call"] == "apply_patch"
    assert [call[:2] for call in selector.calls] == [
        ("review-workflow", "reviewer"),
        ("review-workflow", "judge"),
        ("review-workflow", "finalizer"),
    ]

    loaded = store.load_run(action.trace["agent_run_id"])
    assert loaded is not None
    assert loaded.tool_call_count == 0
    assert all(artifact.kind != "tool_call" for artifact in loaded.artifacts)
    assert any("suppressed_tool_call" in artifact.content for artifact in loaded.artifacts)
