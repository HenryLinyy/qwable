"""Text filters for model think blocks."""

import re
from dataclasses import dataclass


THINK_OPEN = re.compile(r"<think>|<思维>|<思考>|<thinking>", re.IGNORECASE)
THINK_CLOSE = re.compile(r"</think>|</思维>|</思考>|</thinking>", re.IGNORECASE)


@dataclass
class ThinkFilterResult:
    ok: bool
    clean_text: str
    error: str | None


def strip_think_blocks(text: str) -> str:
    """Remove all complete  think...  blocks.

    Supports multiple blocks. Case-insensitive.
    Returns the text with all blocks removed.
    """
    # Find all opening and closing positions
    opens = [(m.start(), m.end()) for m in THINK_OPEN.finditer(text)]
    closes = [(m.start(), m.end()) for m in THINK_CLOSE.finditer(text)]

    if not opens and not closes:
        return text  # no think blocks at all

    # Build a list of segments to keep
    segments = []
    pos = 0
    open_idx = 0
    close_idx = 0

    while open_idx < len(opens):
        open_start, open_end = opens[open_idx]
        # Find matching close (first close after this open)
        while close_idx < len(closes) and closes[close_idx][0] < open_start:
            close_idx += 1
        if close_idx >= len(closes):
            # Unclosed think block — stop processing
            break
        close_start, close_end = closes[close_idx]
        if close_start <= open_start:
            close_idx += 1
            continue

        # Add text before this think block
        segments.append(text[pos:open_start])
        # Skip the content between open and close
        pos = close_end
        open_idx += 1
        close_idx += 1

    # Add remaining text after last closed block
    segments.append(text[pos:])
    return "".join(segments)


def clean_model_output(raw: str) -> ThinkFilterResult:
    """Clean model output by removing think blocks.

    Returns ThinkFilterResult with ok=True if all think blocks were closed,
    ok=False if an unclosed think block was found.
    """
    if _has_unclosed_think_block(raw):
        return ThinkFilterResult(
            ok=False,
            clean_text=raw,
            error="Unclosed think block: model output is incomplete",
        )

    clean = strip_think_blocks(raw)
    return ThinkFilterResult(
        ok=True,
        clean_text=clean,
        error=None,
    )


def _has_unclosed_think_block(text: str) -> bool:
    pos = 0
    while True:
        opening = THINK_OPEN.search(text, pos)
        if not opening:
            return False
        closing = THINK_CLOSE.search(text, opening.end())
        if not closing:
            return True
        pos = closing.end()
