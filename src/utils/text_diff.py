"""Token-aware text differences for the interactive demo."""

from __future__ import annotations

import re
from difflib import SequenceMatcher


TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|\s+|[^\w\s]", flags=re.UNICODE)


def build_text_diff(original: str, corrected: str) -> list[dict[str, str]]:
    """Return safe display segments whose visible edits preserve both texts."""
    original_tokens = TOKEN_PATTERN.findall(original)
    corrected_tokens = TOKEN_PATTERN.findall(corrected)
    matcher = SequenceMatcher(
        None,
        original_tokens,
        corrected_tokens,
        autojunk=False,
    )

    segments: list[dict[str, str]] = []
    for operation, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if operation == "equal":
            _append_segment(segments, "equal", corrected_tokens[new_start:new_end])
        elif operation == "delete":
            _append_segment(segments, "deleted", original_tokens[old_start:old_end])
        elif operation == "insert":
            _append_segment(segments, "inserted", corrected_tokens[new_start:new_end])
        else:
            _append_segment(segments, "deleted", original_tokens[old_start:old_end])
            _append_segment(segments, "inserted", corrected_tokens[new_start:new_end])
    return segments


def _append_segment(
    segments: list[dict[str, str]],
    segment_type: str,
    tokens: list[str],
) -> None:
    text = "".join(tokens)
    if not text:
        return
    if segments and segments[-1]["type"] == segment_type:
        segments[-1]["text"] += text
    else:
        segments.append({"type": segment_type, "text": text})
