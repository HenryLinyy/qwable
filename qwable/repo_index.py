"""Deterministic path extraction from user-provided repository evidence."""

import re


SUPPORTED_EXTENSIONS = (
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".md",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".sh",
)
IGNORED_PARTS = {"node_modules", ".git", "dist", "build", "__pycache__"}
PATH_RE = re.compile(
    r"(?<![\w./-])([A-Za-z0-9_./-]+(?:\.json|\.toml|\.yaml|\.tsx|\.jsx|\.py|\.ts|\.js|\.md|\.yml|\.sh))"
)


class RepoIndex:
    def __init__(self, max_files: int = 300):
        self.max_files = max_files

    def from_text(self, text: str, goal: str | None = None) -> dict:
        paths = self.extract_candidate_paths(text)
        return {
            "paths": paths,
            "scored_paths": self.score_paths(goal or text, paths),
        }

    def extract_candidate_paths(self, text: str) -> list[str]:
        seen: set[str] = set()
        paths: list[str] = []
        for match in PATH_RE.finditer(text or ""):
            path = match.group(1).strip("`'\"),.;:")
            path = path.removeprefix("./")
            if not self._is_supported(path):
                continue
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
            if len(paths) >= self.max_files:
                break
        return paths

    def score_paths(self, goal: str, paths: list[str]) -> list[tuple[str, float]]:
        goal_tokens = self._tokens(goal)
        scored = [(path, self._score_path(path, goal_tokens)) for path in paths]
        return sorted(scored, key=lambda item: (-item[1], paths.index(item[0])))

    def _is_supported(self, path: str) -> bool:
        if not path or path.startswith("/") or ".." in path.split("/"):
            return False
        parts = set(path.split("/"))
        if parts & IGNORED_PARTS:
            return False
        return path.endswith(SUPPORTED_EXTENSIONS)

    def _score_path(self, path: str, goal_tokens: set[str]) -> float:
        lower = path.lower()
        score = 0.0
        if lower.startswith("qwable/") and lower.endswith(".py"):
            score += 4.0
        if lower.startswith("tests/"):
            score += 3.0
        if lower == "pyproject.toml":
            score += 2.5
        if lower in {"package.json", "README.md".lower()}:
            score += 2.0
        path_tokens = self._tokens(path.replace("/", " ").replace(".", " "))
        score += 1.5 * len(goal_tokens & path_tokens)
        return score

    def _tokens(self, text: str) -> set[str]:
        return {
            token
            for token in re.split(r"[^a-zA-Z0-9_]+", (text or "").lower())
            if token
        }
