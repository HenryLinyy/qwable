"""Tests for token counting."""

from qwable.token_estimator import count_cjk_chars, estimate_tokens


def test_count_cjk_chars():
    """Count CJK characters correctly."""
    assert count_cjk_chars("Hello 世界") == 2
    assert count_cjk_chars("你好世界") == 4
    assert count_cjk_chars("Hello") == 0
    assert count_cjk_chars("") == 0


def test_estimate_tokens_pure_cjk():
    """CJK-heavy text should use CJK rate."""
    # 100 CJK chars: ~67 tokens
    text = "你好" * 50  # 100 chars
    tokens = estimate_tokens(text)
    assert tokens >= 66
    assert tokens <= 68


def test_estimate_tokens_pure_ascii():
    """ASCII text should use ASCII rate."""
    text = "Hello world this is a test " * 10
    tokens = estimate_tokens(text)
    assert tokens > 0


def test_estimate_tokens_empty():
    """Empty text should return 1."""
    assert estimate_tokens("") == 1


def test_estimate_tokens_mixed():
    """Mixed CJK + ASCII text."""
    text = "Hello 你好 world 世界"
    tokens = estimate_tokens(text)
    assert tokens > 0
