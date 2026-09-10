"""Deterministic few-shot sampling shared by experiments and the demo."""

from __future__ import annotations

import random
import re
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def load_parallel_examples(
    src_path: Path,
    ref_path: Path,
) -> tuple[list[str], list[str]]:
    """Load a line-aligned source/reference corpus."""
    if not src_path.exists():
        raise FileNotFoundError(f"Few-shot source file not found: {src_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Few-shot reference file not found: {ref_path}")

    with src_path.open("r", encoding="utf-8") as stream:
        sources = [line.rstrip("\n") for line in stream]
    with ref_path.open("r", encoding="utf-8") as stream:
        references = [line.rstrip("\n") for line in stream]

    if len(sources) != len(references):
        raise ValueError(
            f"Few-shot length mismatch: {len(sources)} sources vs "
            f"{len(references)} references"
        )
    return sources, references


def detokenize_prompt_example(text: str) -> str:
    """Apply the lightweight detokenization used in the paper experiments."""
    text = str(text).strip()
    replacements = [
        (r"\s+([,.;:!?%])", r"\1"),
        (r"\(\s+", "("),
        (r"\s+\)", ")"),
        (r"\[\s+", "["),
        (r"\s+\]", "]"),
        (r"\{\s+", "{"),
        (r"\s+\}", "}"),
        (r"\s+/'", "'"),
        (r"\s+'(s|m|re|ve|d|ll)\b", r"'\1"),
        (r"\s+n\s*'\s*t\b", "n't"),
        (r"\s+n't\b", "n't"),
        (r"\s+-\s+", "-"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def select_few_shot_examples(
    *,
    src_path: Path,
    ref_path: Path,
    n: int,
    seed: int,
    detokenize_examples: bool = True,
) -> list[dict]:
    """Select the same deterministic examples for a given corpus, n, and seed."""
    if n <= 0:
        return []

    sources, references = load_parallel_examples(src_path, ref_path)
    if n > len(sources):
        raise ValueError(
            f"few-shot n={n} exceeds available examples ({len(sources)}) "
            f"in {src_path}"
        )

    selected_indices = random.Random(seed).sample(range(len(sources)), n)
    return [
        {
            "train_index": index + 1,
            "source": (
                detokenize_prompt_example(sources[index])
                if detokenize_examples
                else sources[index]
            ),
            "reference": (
                detokenize_prompt_example(references[index])
                if detokenize_examples
                else references[index]
            ),
        }
        for index in selected_indices
    ]


def format_few_shot_prompt(examples: list[dict]) -> str:
    """Format selected examples for system-prompt delivery."""
    if not examples:
        return ""
    lines = [
        "Few-shot examples from BEA train:",
        "Follow the same input-to-correction style. "
        "Do not copy these examples; use them only as guidance.",
    ]
    for example_number, example in enumerate(examples, start=1):
        lines.extend(
            [
                "",
                f"Example {example_number}:",
                f"Input: {example['source']}",
                f"Correction: {example['reference']}",
            ]
        )
    return "\n".join(lines)


def build_few_shot_prompt(
    *,
    src_path: Path,
    ref_path: Path,
    n: int,
    seed: int,
    detokenize_examples: bool = True,
) -> tuple[str, list[dict]]:
    """Select and format few-shot examples for an experiment run."""
    examples = select_few_shot_examples(
        src_path=src_path,
        ref_path=ref_path,
        n=n,
        seed=seed,
        detokenize_examples=detokenize_examples,
    )
    return format_few_shot_prompt(examples), examples
