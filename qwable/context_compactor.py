"""Build deterministic context packs without model calls."""

from qwable.config import FusionConfig
from qwable.context_pack import ContextFileSummary, ContextPack
from qwable.repo_index import RepoIndex
from qwable.schemas import ParsedAgentTask


class ContextCompactor:
    def __init__(self, config: FusionConfig):
        self.config = config
        self.repo_index = RepoIndex(max_files=config.agent_repo_index_max_files)

    def build_pack(self, task: ParsedAgentTask, workflow: str) -> ContextPack:
        raw_evidence = self._raw_evidence(task)
        combined_text = "\n".join(raw_evidence)
        paths = self.repo_index.extract_candidate_paths(combined_text)
        score_map = dict(self.repo_index.score_paths(task.text, paths))
        files = [
            ContextFileSummary(
                path=path,
                reason="mentioned in request, tool result, or vision evidence",
                summary="Candidate path extracted from provided evidence.",
                score=score_map.get(path, 0.0),
            )
            for path in paths
        ]
        risks = [
            f"tool_result_error:{result.name or 'unknown'}"
            for result in task.tool_results
            if result.is_error
        ]
        return ContextPack(
            goal=task.text,
            workflow=workflow,
            files=files,
            raw_evidence=raw_evidence,
            constraints=[
                "RepoIndex uses only request text, tool results, and vision evidence; it does not read arbitrary filesystem paths."
            ],
            risks=risks,
            metadata={
                "tool_result_count": len(task.tool_results),
                "vision_evidence_count": len(task.vision_evidence),
                "max_prompt_chars": self.config.agent_context_pack_max_chars,
            },
        )

    def _raw_evidence(self, task: ParsedAgentTask) -> list[str]:
        evidence = [f"request:{task.text}"]
        for result in task.tool_results:
            name = result.name or "unknown"
            evidence.append(f"tool_result:{name}\n{result.content}")
        for item in task.vision_evidence:
            parts = [f"vision:{item.profile}\nsummary:{item.summary}"]
            if item.visible_text:
                parts.append(f"visible_text:{item.visible_text}")
            if item.raw_text:
                parts.append(f"raw_text:{item.raw_text}")
            if item.warnings:
                parts.append("warnings:" + ", ".join(item.warnings))
            if item.confidence is not None:
                parts.append(f"confidence:{item.confidence}")
            evidence.append("\n".join(parts))
        return evidence
