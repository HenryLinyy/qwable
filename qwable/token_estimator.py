"""CJK-aware token estimator for usage reporting."""

import re

CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")


def count_cjk_chars(text: str) -> int:
    """Count CJK (Chinese/Japanese/Korean) characters in text."""
    return len(CJK_PATTERN.findall(text))


def estimate_tokens(text: str) -> int:
    """Estimate token count for CJK-mixed text.

    CJK chars: ~1.5 tokens each
    Non-CJK: ~4 chars per token
    """
    cjk = count_cjk_chars(text)
    non_cjk = len(text) - cjk
    return max(1, int(cjk / 1.5 + non_cjk / 4))
