"""Plan standard test tool calls without executing them."""

from __future__ import annotations

import re

from qwable.context_pack import ContextPack
from qwable.patch_protocol import is_test_command


_EXPLICIT_TEST_RE = re.compile(
    r"(?:run\s+tests?|test\s+command|tests?)\s*[:=]\s*`?(?P<command>[^\n`]+)`?",
    re.IGNORECASE,
)


class TestCommandPlanner:
    """Infer test commands from context and produce standard tool-call objects."""

    def infer_test_commands(
        self, context_pack: ContextPack, workflow: str
    ) -> list[str]:
        explicit_command = self._find_explicit_test_command(context_pack)
        if explicit_command:
            return [explicit_command]

        paths = {file.path for file in context_pack.files}
        lower_paths = {path.lower() for path in paths}
        if self._looks_like_python_project(lower_paths):
            return ["python -m pytest"]
        if "package.json" in lower_paths:
            if "bun.lockb" in lower_paths:
                return ["bun test"]
            if "pnpm-lock.yaml" in lower_paths:
                return ["pnpm test"]
            return ["npm test"]
        return []

    def build_test_tool_call(self, command: str) -> dict:
        return {"tool_name": "run_tests", "tool_input": {"command": command}}

    def _find_explicit_test_command(self, context_pack: ContextPack) -> str | None:
        for evidence in context_pack.raw_evidence:
            match = _EXPLICIT_TEST_RE.search(evidence)
            if not match:
                continue
            command = match.group("command").strip().rstrip(".")
            if is_test_command(command):
                return command
        return None

    def _looks_like_python_project(self, lower_paths: set[str]) -> bool:
        if lower_paths.intersection(
            {"pyproject.toml", "pytest.ini", "setup.cfg", "setup.py"}
        ):
            return True
        return any(
            path.startswith("tests/test_") or path.endswith("_test.py")
            for path in lower_paths
        )
