"""
M2 Scorer metric for GEC evaluation.

Wrapper around the official CoNLL-2014 M2 scorer from NUS.
Repository: https://github.com/nusnlp/m2scorer

Note: This uses subprocess to call the original m2scorer script since it's
written for Python 2.
"""

import subprocess
import tempfile
import re
from pathlib import Path
from typing import List, Tuple
import os
import shutil


def _normalize_sentence(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())


def extract_m2_subset(m2_file_path: str, num_sentences: int, offset: int = 0) -> str:
    """
    Extract a subset of sentences from an M2 file.

    Args:
        m2_file_path: Path to the full M2 file
        num_sentences: Number of sentences to extract
        offset: Starting sentence index (default: 0)

    Returns:
        Path to temporary M2 file containing the subset
    """
    with open(m2_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Parse M2 file to find sentence boundaries
    sentence_blocks = []
    current_block = []

    for line in lines:
        if line.startswith('S '):
            # Start of a new sentence block
            if current_block:
                sentence_blocks.append(current_block)
            current_block = [line]
        else:
            # Annotation line (A) or other
            current_block.append(line)

    # Don't forget the last block
    if current_block:
        sentence_blocks.append(current_block)

    # Extract the subset based on offset and limit
    subset_blocks = sentence_blocks[offset:offset + num_sentences]

    # Write subset to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.m2', delete=False, encoding='utf-8') as f:
        for block in subset_blocks:
            f.writelines(block)
            f.write('\n')  # Add blank line between blocks
        temp_m2_file = f.name

    return temp_m2_file


def calculate_m2score(
    hypotheses: List[str],
    m2_file_path: str,
    m2scorer_path: str | None = None,
    max_unchanged_words: int = 2,
    beta: float = 0.5,
    ignore_whitespace_casing: bool = False,
    verbose: bool = False,
    offset: int = 0
) -> Tuple[float, float, float]:
    """
    Calculate M2 score (Precision, Recall, F-score) using the official CoNLL-2014 scorer.

    Args:
        hypotheses: List of corrected sentences (system output)
        m2_file_path: Path to the M2 annotation file (gold standard)
        m2scorer_path: Optional explicit path to m2scorer script
        max_unchanged_words: Maximum unchanged words when extracting edits (default: 2)
        beta: Beta value for F-measure (default: 0.5 for F_0.5)
        ignore_whitespace_casing: Ignore edits affecting only whitespace/casing (default: False)
        verbose: Print verbose output (default: False)
        offset: Starting sentence index in the M2 file (default: 0)

    Returns:
        Tuple of (precision, recall, f_score)
    """
    # Resolve m2scorer script path
    repo_root = Path(__file__).resolve().parents[1]
    env_path = os.getenv("M2SCORER_PATH")
    candidates = [
        Path(m2scorer_path) if m2scorer_path else None,
        Path(env_path) if env_path else None,
        repo_root / "m2scorer" / "m2scorer",
        repo_root.parent / "m2scorer" / "m2scorer",
    ]
    m2scorer_script = next((p for p in candidates if p is not None and p.exists()), None)
    if m2scorer_script is None:
        # Fallback: use ERRANT scoring if m2scorer script is unavailable.
        return calculate_m2score_with_errant(hypotheses=hypotheses, m2_file_path=m2_file_path, offset=offset)

    # Extract subset of M2 file matching the hypotheses
    temp_m2_file = extract_m2_subset(m2_file_path, len(hypotheses), offset)

    # Write hypotheses to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        for hyp in hypotheses:
            f.write(hyp + '\n')
        hyp_file = f.name

    try:
        # Build command - m2scorer has been converted to Python 3
        # Note: Options must come before positional arguments for getopt
        cmd = [
            'python3',
            str(m2scorer_script),
            '--max_unchanged_words', str(max_unchanged_words),
            '--beta', str(beta)
        ]

        if ignore_whitespace_casing:
            cmd.append('--ignore_whitespace_casing')

        if verbose:
            cmd.append('--verbose')

        # Add positional arguments at the end - use temp M2 file subset
        cmd.extend([hyp_file, temp_m2_file])

        # Run m2scorer
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            raise RuntimeError(f"M2 scorer failed: {result.stderr}")

        # Parse output to extract scores
        # Expected format:
        # Precision   : 0.1234
        # Recall      : 0.5678
        # F_0.5       : 0.9012
        output = result.stdout

        precision_match = re.search(r'Precision\s*:\s*([\d.]+)', output)
        recall_match = re.search(r'Recall\s*:\s*([\d.]+)', output)
        f_score_match = re.search(rf'F_{beta}\s*:\s*([\d.]+)', output)

        if not (precision_match and recall_match and f_score_match):
            raise RuntimeError(f"Failed to parse m2scorer output:\n{output}")

        precision = float(precision_match.group(1))
        recall = float(recall_match.group(1))
        f_score = float(f_score_match.group(1))

        return precision, recall, f_score

    finally:
        # Clean up temporary files
        Path(hyp_file).unlink(missing_ok=True)
        Path(temp_m2_file).unlink(missing_ok=True)


def _parse_errant_compare_output(output: str) -> Tuple[float, float, float]:
    """Parse Precision/Recall/F0.5 from errant_compare output."""
    # Table format:
    # TP FP FN Prec Rec F0.5
    # 580 599 528 0.4919 0.5235 0.4979
    table_match = re.search(
        r"TP\s+FP\s+FN\s+Prec\s+Rec\s+F0\.5\s*\n\s*\d+\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        output,
        flags=re.IGNORECASE,
    )
    if table_match:
        return float(table_match.group(1)), float(table_match.group(2)), float(table_match.group(3))

    # Alternate format:
    # Precision : 0.1234
    # Recall    : 0.5678
    # F0.5      : 0.9012
    prec_match = re.search(r"Precision\s*:?\s*([\d.]+)", output, flags=re.IGNORECASE)
    rec_match = re.search(r"Recall\s*:?\s*([\d.]+)", output, flags=re.IGNORECASE)
    f_match = re.search(r"F0?\.?5\s*:?\s*([\d.]+)", output, flags=re.IGNORECASE)
    if prec_match and rec_match and f_match:
        return float(prec_match.group(1)), float(rec_match.group(1)), float(f_match.group(1))

    raise RuntimeError(f"Failed to parse errant_compare output:\n{output}")


def calculate_m2score_with_errant(
    hypotheses: List[str],
    m2_file_path: str,
    offset: int = 0,
) -> Tuple[float, float, float]:
    """Fallback scorer using errant_parallel + errant_compare."""
    if shutil.which("errant_parallel") is None or shutil.which("errant_compare") is None:
        raise FileNotFoundError(
            "Neither m2scorer nor ERRANT CLI is available. "
            "Install errant (commands: errant_parallel, errant_compare) "
            "or provide m2scorer via en_m2scorer_path/M2SCORER_PATH."
        )

    temp_m2_file = extract_m2_subset(m2_file_path, len(hypotheses), offset)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f_hyp:
        for hyp in hypotheses:
            f_hyp.write(hyp + "\n")
        hyp_file = f_hyp.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".src", delete=False, encoding="utf-8") as f_src:
        with open(temp_m2_file, "r", encoding="utf-8") as m2f:
            for line in m2f:
                if line.startswith("S "):
                    f_src.write(line[2:])
        src_file = f_src.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".m2", delete=False, encoding="utf-8") as f_hyp_m2:
        hyp_m2_file = f_hyp_m2.name

    try:
        align_cmd = ["errant_parallel", "-orig", src_file, "-cor", hyp_file, "-out", hyp_m2_file]
        align_res = subprocess.run(align_cmd, capture_output=True, text=True, check=False)
        if align_res.returncode != 0:
            raise RuntimeError(
                "ERRANT alignment failed.\n"
                f"CMD: {' '.join(align_cmd)}\nSTDERR:\n{align_res.stderr}\nSTDOUT:\n{align_res.stdout}"
            )

        compare_cmd = ["errant_compare", "-hyp", hyp_m2_file, "-ref", temp_m2_file]
        compare_res = subprocess.run(compare_cmd, capture_output=True, text=True, check=False)
        if compare_res.returncode != 0:
            raise RuntimeError(
                "ERRANT scoring failed.\n"
                f"CMD: {' '.join(compare_cmd)}\nSTDERR:\n{compare_res.stderr}\nSTDOUT:\n{compare_res.stdout}"
            )

        return _parse_errant_compare_output(compare_res.stdout)
    finally:
        Path(hyp_file).unlink(missing_ok=True)
        Path(src_file).unlink(missing_ok=True)
        Path(hyp_m2_file).unlink(missing_ok=True)
        Path(temp_m2_file).unlink(missing_ok=True)


def print_m2score_results(precision: float, recall: float, f_score: float, beta: float = 0.5) -> None:
    """
    Print M2 score results in a formatted way.

    Args:
        precision: Precision score
        recall: Recall score
        f_score: F-score
        beta: Beta value used for F-measure (default: 0.5)
    """
    print(f"M2 Scorer Results:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F_{beta}:      {f_score:.4f}")
