"""Tests for text filters (think blocks)."""

from qwable.text_filters import strip_think_blocks, clean_model_output


def test_strip_single_block():
    """Strip a single complete think block."""
    text = "Hello <thinking>internal reasoning</thinking> world"
    result = strip_think_blocks(text)
    assert result == "Hello  world"


def test_strip_multiple_blocks():
    """Strip multiple complete think blocks."""
    text = "A <thinking>1</thinking> B <thinking>2</thinking> C"
    result = strip_think_blocks(text)
    assert result == "A  B  C"


def test_no_think_blocks():
    """Text without think blocks should be unchanged."""
    text = "Hello world"
    result = strip_think_blocks(text)
    assert result == "Hello world"


def test_unclosed_think_block():
    """Unclosed think block should be detected."""
    text = "Hello <thinking>no close"
    result = clean_model_output(text)
    assert not result.ok
    assert "Unclosed think block" in result.error


def test_clean_think_block():
    """Clean model output with closed think block."""
    text = "Hello <thinking>reasoning</thinking> world"
    result = clean_model_output(text)
    assert result.ok
    assert result.clean_text == "Hello  world"


def test_cjk_think_blocks():
    """Support Chinese think block markers."""
    text = "你好<思考>内部推理</思考>世界"
    result = strip_think_blocks(text)
    assert result == "你好世界"


def test_strip_think_tag():
    """Support DeepSeek-style <think> blocks."""
    text = "A<think>hidden</think>B"
    result = strip_think_blocks(text)
    assert result == "AB"


def test_unclosed_later_think_block():
    """Detect an unclosed block even after a closed block."""
    text = "<think>closed</think>answer<think>open"
    result = clean_model_output(text)
    assert not result.ok
    assert "Unclosed think block" in result.error
