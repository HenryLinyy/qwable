"""Prompt constants and message builders for the agent runtime."""

import json

from qwable.agent_state import AgentRun, AgentStep
from qwable.context_pack import ContextPack
from qwable.schemas import ParsedAgentTask


PLANNER_SYSTEM = """\
/no_think
請不要輸出任何思考過程。只輸出 JSON。
你是 Qwable Agent Planner。
任務：把使用者目標拆成可執行步驟。
禁止：直接修改檔案、假設不存在的檔案、跳過測試、輸出散文式計劃。
輸出必須是 JSON。
工具約束：不要規劃 terminal/shell/file_editor/code_executor/linter/type_checker 這類不存在或受限工具。
可用工具名稱只限 read_file/search_files/list_files/edit_file/apply_patch/run_tests；沒有明確檔案路徑或 repo context 時，required_tools 請用空陣列。
如果使用者要求 standalone function / code snippet，計劃應直接產出程式碼內容，不要建立檔案、不要 list_files。
如果使用者只是要求「規劃」或「列出步驟」，計劃應產出文字步驟，不要要求 shell 執行。

JSON schema:
{
  "steps": [
    {
      "title": "string",
      "intent": "string",
      "required_tools": ["string"],
      "success_criteria": ["string"],
      "failure_criteria": ["string"]
    }
  ],
  "risks": ["string"],
  "test_strategy": ["string"]
}
"""

CRITIC_SYSTEM = """\
你是 Qwable Critic。
任務：審查 plan / patch / test result。
只輸出 blocker、risk、missing evidence、required test。
不得重寫整個方案。
不得加入未提供的事實。
輸出 JSON。
"""

EXECUTOR_SYSTEM = """\
/no_think
請不要輸出任何思考過程。只輸出 JSON。
你是 Qwable Executor。
一次只執行目前 step。
需要讀檔、搜尋、改檔、跑測試時，輸出 tool_call。
已有 tool_result 時，只根據 tool_result 推進。
不得跳步。
不得重複同一失敗操作超過 3 次。
輸出 either tool_call 或 step_result JSON。

STRICT OUTPUT CONTRACT:
Output valid JSON only. Use exactly one of these shapes:
1) {"step_result":{"status":"done","summary":"...","content":"..."}}
2) {"tool_call":{"name":"read_file|search_files|list_files|edit_file|apply_patch|run_tests|shell","input":{...}}}
Do NOT output top-level {"tool":...}, {"action":...}, {"request":...}, code_editor, execute_command, terminal, file_editor, code_executor, linter, type_checker, list_files-with-empty-input, or any invented tool.
Ignore AGENT_RUN/CURRENT_STEP required_tools when they mention unavailable aliases such as terminal/file_editor/code_executor/linter/type_checker.
If no concrete allowed tool is necessary or available, output step_result.
For requests to write a standalone function or code snippet, put the complete code in step_result.content immediately; do not inspect files, do not list_files, do not request file creation unless the user gave an explicit target path.
For requests that ask to plan/list steps, output step_result with the plan; do not execute shell.
Do not request shell for non-test commands; for filesystem inspection tasks, output step_result with the exact safe commands and expected result format instead.
"""

REPAIR_SYSTEM = """\
/no_think
請不要輸出任何思考過程。只輸出 JSON。
你是 Qwable Repair Agent。
根據 test failure / diff / tool_result 做最小修復。
不得重構無關檔案。
不得擴大修改範圍。
一次只修一類錯誤。
輸出 tool_call 或 repair_result JSON。
"""

FINALIZER_SYSTEM = """\
你是 Qwable Finalizer。
根據 AgentRun trace 輸出最終報告。
必須包含：完成項目、修改檔案、測試結果、剩餘風險、下一步。
不得宣稱未執行的測試已通過。
"""


def build_planner_messages(
    task: ParsedAgentTask,
    context_pack: ContextPack,
) -> list[dict]:
    return [
        {"role": "system", "content": PLANNER_SYSTEM},
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    "/no_think",
                    f"USER_GOAL:\n{task.text}",
                    context_pack.to_prompt_text(max_chars=64000),
                ]
            ),
        },
    ]


def build_plan_critic_messages(run: AgentRun, context_pack: ContextPack) -> list[dict]:
    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    "AGENT_RUN:\n" + _run_to_json(run),
                    context_pack.to_prompt_text(max_chars=64000),
                ]
            ),
        },
    ]


def build_executor_messages(
    run: AgentRun,
    context_pack: ContextPack,
    current_step: AgentStep,
) -> list[dict]:
    return [
        {"role": "system", "content": EXECUTOR_SYSTEM},
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    "/no_think",
                    "AGENT_RUN:\n" + _run_to_json(run),
                    "CURRENT_STEP:\n" + _step_to_json(current_step),
                    context_pack.to_prompt_text(max_chars=64000),
                ]
            ),
        },
    ]


def build_repair_messages(
    run: AgentRun,
    context_pack: ContextPack,
    failure_text: str,
) -> list[dict]:
    return [
        {"role": "system", "content": REPAIR_SYSTEM},
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    "/no_think",
                    "FAILURE_TEXT:\n" + failure_text,
                    "AGENT_RUN:\n" + _run_to_json(run),
                    context_pack.to_prompt_text(max_chars=64000),
                ]
            ),
        },
    ]


def build_finalizer_messages(run: AgentRun, context_pack: ContextPack) -> list[dict]:
    return [
        {"role": "system", "content": FINALIZER_SYSTEM},
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    "FINAL_REPORT_INPUT:",
                    "AGENT_RUN:\n" + _run_to_json(run),
                    context_pack.to_prompt_text(max_chars=64000),
                ]
            ),
        },
    ]


def _run_to_json(run: AgentRun) -> str:
    return _json(
        {
            "run_id": run.run_id,
            "workflow": run.workflow,
            "goal": run.goal,
            "status": run.status,
            "current_step_index": run.current_step_index,
            "repair_count": run.repair_count,
            "tool_call_count": run.tool_call_count,
            "trace": run.trace,
            "plan": [_step_dict(step) for step in run.plan],
            "artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "kind": artifact.kind,
                    "content": artifact.content,
                    "metadata": artifact.metadata,
                    "created_at": artifact.created_at,
                }
                for artifact in run.artifacts
            ],
            "failures": [
                {
                    "stage": failure.stage,
                    "message": failure.message,
                    "metadata": failure.metadata,
                    "created_at": failure.created_at,
                }
                for failure in run.failures
            ],
        }
    )


def _step_to_json(step: AgentStep) -> str:
    return _json(_step_dict(step))


def _step_dict(step: AgentStep) -> dict:
    return {
        "step_id": step.step_id,
        "title": step.title,
        "intent": step.intent,
        "status": step.status,
        "required_tools": step.required_tools,
        "success_criteria": step.success_criteria,
        "failure_criteria": step.failure_criteria,
        "evidence": step.evidence,
        "output": step.output,
        "error": step.error,
        "attempt_count": step.attempt_count,
    }


def _json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
