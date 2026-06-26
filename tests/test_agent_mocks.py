"""Contract tests for reusable agent workflow model fakes."""

import json


def test_fake_chat_response_matches_ollama_shape():
    from tests.agent_mocks import fake_chat_response

    assert fake_chat_response("hello") == {
        "choices": [{"message": {"content": "hello"}}],
    }


def test_fake_agent_model_client_returns_scripted_outputs_and_records_calls():
    from tests.agent_mocks import FakeAgentModelClient, fake_chat_response

    client = FakeAgentModelClient(
        [
            fake_chat_response("first"),
            fake_chat_response("second"),
        ]
    )
    messages = [{"role": "user", "content": "hi"}]

    first = client.chat_completion(
        model="model/planner",
        messages=messages,
        max_tokens=123,
        stream=True,
        tools=[{"type": "function"}],
        temperature=0.4,
    )
    second = client.chat_completion(
        model="model/executor",
        messages=[],
        max_tokens=456,
        temperature=0.2,
    )

    assert first["choices"][0]["message"]["content"] == "first"
    assert second["choices"][0]["message"]["content"] == "second"
    assert client.calls == [
        {
            "model": "model/planner",
            "messages": messages,
            "max_tokens": 123,
            "temperature": 0.4,
        },
        {
            "model": "model/executor",
            "messages": [],
            "max_tokens": 456,
            "temperature": 0.2,
        },
    ]
    assert client.scripted_outputs == []


def test_fake_agent_model_client_raises_when_script_is_exhausted():
    import pytest

    from tests.agent_mocks import FakeAgentModelClient

    client = FakeAgentModelClient([])

    with pytest.raises(RuntimeError, match="no scripted output"):
        client.chat_completion(model="model/planner", messages=[])

    assert client.calls == [
        {
            "model": "model/planner",
            "messages": [],
            "max_tokens": 1200,
            "temperature": 0.7,
        }
    ]


def test_phase18_planner_and_executor_fake_outputs_match_plan_contract():
    from tests.agent_mocks import (
        fake_executor_tool_call_response,
        fake_planner_response,
    )

    planner = json.loads(fake_planner_response()["choices"][0]["message"]["content"])
    executor = json.loads(
        fake_executor_tool_call_response()["choices"][0]["message"]["content"]
    )

    assert planner["steps"] == [
        {
            "title": "Inspect target files",
            "intent": "Find relevant files before patching",
            "required_tools": ["search_files"],
            "success_criteria": ["Relevant files identified"],
            "failure_criteria": ["No files found"],
        }
    ]
    assert planner["risks"] == []
    assert planner["test_strategy"] == ["run pytest"]
    assert executor == {
        "tool_call": {
            "name": "search_files",
            "input": {"query": "agent_orchestrator"},
        }
    }
