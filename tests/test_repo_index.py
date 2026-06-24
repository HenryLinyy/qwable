"""Tests for deterministic repo path indexing from text evidence."""


def test_extract_candidate_paths_filters_ignored_and_unsupported_paths():
    from qwable.repo_index import RepoIndex

    text = """
    qwable/server.py
    tests/test_server.py
    README.md
    pyproject.toml
    package.json
    scripts/run.sh
    config/settings.yaml
    node_modules/pkg/index.js
    .git/config
    dist/app.js
    build/out.js
    __pycache__/server.py
    notes.txt
    """

    paths = RepoIndex(max_files=20).extract_candidate_paths(text)

    assert paths == [
        "qwable/server.py",
        "tests/test_server.py",
        "README.md",
        "pyproject.toml",
        "package.json",
        "scripts/run.sh",
        "config/settings.yaml",
    ]


def test_extract_candidate_paths_deduplicates_and_respects_limit():
    from qwable.repo_index import RepoIndex

    text = "a.py b.py a.py c.py"

    assert RepoIndex(max_files=2).extract_candidate_paths(text) == ["a.py", "b.py"]


def test_score_paths_prioritizes_goal_matches_and_core_project_files():
    from qwable.repo_index import RepoIndex

    index = RepoIndex()
    scored = index.score_paths(
        goal="fix server route tests",
        paths=[
            "docs/notes.md",
            "qwable/server.py",
            "tests/test_server.py",
            "README.md",
            "pyproject.toml",
        ],
    )

    assert [path for path, _score in scored[:3]] == [
        "qwable/server.py",
        "tests/test_server.py",
        "pyproject.toml",
    ]
    assert scored[0][1] > scored[-1][1]


def test_from_text_returns_paths_and_scores_without_filesystem_reads():
    from qwable.repo_index import RepoIndex

    result = RepoIndex(max_files=10).from_text(
        "Check qwable/config.py and tests/test_config.py",
        goal="config test",
    )

    assert result["paths"] == ["qwable/config.py", "tests/test_config.py"]
    assert result["scored_paths"][0][0] == "qwable/config.py"
