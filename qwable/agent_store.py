"""SQLite persistence for agent runs."""

import json
import sqlite3
from pathlib import Path
from typing import Any

from qwable.agent_state import (
    AgentArtifact,
    AgentFailure,
    AgentRun,
    AgentStep,
    new_id,
)


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_json(value: str) -> Any:
    return json.loads(value or "null")


class AgentStore:
    def __init__(self, path: str):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                  run_id TEXT PRIMARY KEY,
                  workflow TEXT NOT NULL,
                  status TEXT NOT NULL,
                  goal TEXT NOT NULL,
                  current_step_index INTEGER NOT NULL,
                  repair_count INTEGER NOT NULL,
                  tool_call_count INTEGER NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  trace_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_steps (
                  run_id TEXT NOT NULL,
                  step_id TEXT NOT NULL,
                  idx INTEGER NOT NULL,
                  title TEXT NOT NULL,
                  intent TEXT NOT NULL,
                  status TEXT NOT NULL,
                  required_tools_json TEXT NOT NULL,
                  success_criteria_json TEXT NOT NULL,
                  failure_criteria_json TEXT NOT NULL,
                  evidence_json TEXT NOT NULL,
                  output TEXT,
                  error TEXT,
                  attempt_count INTEGER NOT NULL,
                  PRIMARY KEY (run_id, step_id)
                );

                CREATE TABLE IF NOT EXISTS agent_artifacts (
                  artifact_id TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  content TEXT NOT NULL,
                  metadata_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_failures (
                  failure_id TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  stage TEXT NOT NULL,
                  message TEXT NOT NULL,
                  metadata_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                """
            )

    def save_run(self, run: AgentRun) -> None:
        # Refresh updated_at on every persist so list_runs (ORDER BY updated_at
        # DESC) reflects last-write order, not creation order. touch() was never
        # called anywhere, so updated_at stayed == created_at for a run's life.
        run.touch()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_runs (
                  run_id, workflow, status, goal, current_step_index,
                  repair_count, tool_call_count, created_at, updated_at, trace_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.workflow,
                    run.status,
                    run.goal,
                    run.current_step_index,
                    run.repair_count,
                    run.tool_call_count,
                    run.created_at,
                    run.updated_at,
                    _dump_json(run.trace),
                ),
            )
            conn.execute("DELETE FROM agent_steps WHERE run_id = ?", (run.run_id,))
            conn.execute("DELETE FROM agent_artifacts WHERE run_id = ?", (run.run_id,))
            conn.execute("DELETE FROM agent_failures WHERE run_id = ?", (run.run_id,))
            self._insert_steps(conn, run.run_id, run.plan)
            for artifact in run.artifacts:
                self._insert_artifact(conn, run.run_id, artifact)
            for failure in run.failures:
                self._insert_failure(conn, run.run_id, failure)

    def load_run(self, run_id: str) -> AgentRun | None:
        with self._connect() as conn:
            run_row = conn.execute(
                "SELECT * FROM agent_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run_row is None:
                return None
            run = AgentRun(
                run_id=run_row["run_id"],
                workflow=run_row["workflow"],
                goal=run_row["goal"],
                status=run_row["status"],
                current_step_index=run_row["current_step_index"],
                repair_count=run_row["repair_count"],
                tool_call_count=run_row["tool_call_count"],
                created_at=run_row["created_at"],
                updated_at=run_row["updated_at"],
                trace=_load_json(run_row["trace_json"]) or {},
            )
            run.plan = [
                self._row_to_step(row)
                for row in conn.execute(
                    "SELECT * FROM agent_steps WHERE run_id = ? ORDER BY idx ASC",
                    (run_id,),
                )
            ]
            run.artifacts = [
                self._row_to_artifact(row)
                for row in conn.execute(
                    "SELECT * FROM agent_artifacts WHERE run_id = ? ORDER BY created_at ASC",
                    (run_id,),
                )
            ]
            run.failures = [
                self._row_to_failure(row)
                for row in conn.execute(
                    "SELECT * FROM agent_failures WHERE run_id = ? ORDER BY created_at ASC, failure_id ASC",
                    (run_id,),
                )
            ]
            return run

    def list_runs(self, limit: int = 20) -> list[AgentRun]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id FROM agent_runs ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            run for row in rows if (run := self.load_run(row["run_id"])) is not None
        ]

    def append_artifact(self, run_id: str, artifact: AgentArtifact) -> None:
        with self._connect() as conn:
            self._insert_artifact(conn, run_id, artifact)

    def append_failure(self, run_id: str, failure: AgentFailure) -> None:
        with self._connect() as conn:
            self._insert_failure(conn, run_id, failure)

    def _insert_steps(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        steps: list[AgentStep],
    ) -> None:
        for idx, step in enumerate(steps):
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_steps (
                  run_id, step_id, idx, title, intent, status,
                  required_tools_json, success_criteria_json,
                  failure_criteria_json, evidence_json, output, error, attempt_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    step.step_id,
                    idx,
                    step.title,
                    step.intent,
                    step.status,
                    _dump_json(step.required_tools),
                    _dump_json(step.success_criteria),
                    _dump_json(step.failure_criteria),
                    _dump_json(step.evidence),
                    step.output,
                    step.error,
                    step.attempt_count,
                ),
            )

    def _insert_artifact(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        artifact: AgentArtifact,
    ) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO agent_artifacts (
              artifact_id, run_id, kind, content, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.artifact_id,
                run_id,
                artifact.kind,
                artifact.content,
                _dump_json(artifact.metadata),
                artifact.created_at,
            ),
        )

    def _insert_failure(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        failure: AgentFailure,
    ) -> None:
        conn.execute(
            """
            INSERT INTO agent_failures (
              failure_id, run_id, stage, message, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                new_id("failure"),
                run_id,
                failure.stage,
                failure.message,
                _dump_json(failure.metadata),
                failure.created_at,
            ),
        )

    def _row_to_step(self, row: sqlite3.Row) -> AgentStep:
        return AgentStep(
            step_id=row["step_id"],
            title=row["title"],
            intent=row["intent"],
            status=row["status"],
            required_tools=_load_json(row["required_tools_json"]) or [],
            success_criteria=_load_json(row["success_criteria_json"]) or [],
            failure_criteria=_load_json(row["failure_criteria_json"]) or [],
            evidence=_load_json(row["evidence_json"]) or [],
            output=row["output"],
            error=row["error"],
            attempt_count=row["attempt_count"],
        )

    def _row_to_artifact(self, row: sqlite3.Row) -> AgentArtifact:
        return AgentArtifact(
            artifact_id=row["artifact_id"],
            run_id=row["run_id"],
            kind=row["kind"],
            content=row["content"],
            metadata=_load_json(row["metadata_json"]) or {},
            created_at=row["created_at"],
        )

    def _row_to_failure(self, row: sqlite3.Row) -> AgentFailure:
        return AgentFailure(
            stage=row["stage"],
            message=row["message"],
            metadata=_load_json(row["metadata_json"]) or {},
            created_at=row["created_at"],
        )
