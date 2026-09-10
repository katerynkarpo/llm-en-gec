from __future__ import annotations

import re


def parse_numbered_batch_output(
    generated_text: str,
    expected_count: int,
) -> tuple[list[str] | None, str | None]:
    """Parse and validate ``Sentence N: ...`` model output."""
    predictions: list[str | None] = [None] * expected_count
    seen_indices: set[int] = set()
    duplicate_indices: set[int] = set()
    out_of_range_indices: list[int] = []
    unnumbered_lines: list[str] = []

    for raw_line in generated_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = re.match(r"^(?:sentence\s*)?(\d+)\s*[:.)-]\s*(.*)$", line, re.I)
        if not match:
            unnumbered_lines.append(line)
            continue

        index = int(match.group(1)) - 1
        if not 0 <= index < expected_count:
            out_of_range_indices.append(index + 1)
            continue
        if index in seen_indices:
            duplicate_indices.add(index)
        seen_indices.add(index)
        predictions[index] = match.group(2).strip()

    missing_indices = [i for i, prediction in enumerate(predictions) if prediction is None]
    empty_indices = [i for i, prediction in enumerate(predictions) if prediction == ""]
    errors: list[str] = []
    if missing_indices:
        errors.append("missing " + ", ".join(str(i + 1) for i in missing_indices))
    if empty_indices:
        errors.append("empty " + ", ".join(str(i + 1) for i in empty_indices))
    if duplicate_indices:
        errors.append("duplicate " + ", ".join(str(i + 1) for i in sorted(duplicate_indices)))
    if out_of_range_indices:
        errors.append("out-of-range " + ", ".join(str(i) for i in out_of_range_indices))
    if unnumbered_lines:
        errors.append(f"{len(unnumbered_lines)} unnumbered line(s)")

    if errors:
        return None, "; ".join(errors)
    return [str(prediction) for prediction in predictions], None
