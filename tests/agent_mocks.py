"""Reusable fake model client helpers for agent workflow tests."""

from __future__ import annotations

import json
from typing import Any


class FakeAgentModelClient:
    def __init__(self, scripted_outputs: list[dict]):
        self.calls = []
        self.scripted_outputs = list(scripted_outputs)

    def chat_completion(
        self,
        model,
        messages,
        max_tokens=1200,
        stream=False,
        tools=None,
        temperature=0.7,
    ):
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if not self.scripted_outputs:
            raise RuntimeError("no scripted output")
        return self.scripted_outputs.pop(0)


def fake_chat_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def fake_planner_response() -> dict:
    return fake_chat_response(
        json.dumps(
            {
                "steps": [
                    {
                        "title": "Inspect target files",
                        "intent": "Find relevant files before patching",
                        "required_tools": ["search_files"],
                        "success_criteria": ["Relevant files identified"],
                        "failure_criteria": ["No files found"],
                    }
                ],
                "risks": [],
                "test_strategy": ["run pytest"],
            }
        )
    )


def fake_executor_tool_call_response(
    *,
    name: str = "search_files",
    tool_input: dict[str, Any] | None = None,
) -> dict:
    return fake_chat_response(
        json.dumps(
            {
                "tool_call": {
                    "name": name,
                    "input": tool_input or {"query": "agent_orchestrator"},
                }
            }
        )
    )
