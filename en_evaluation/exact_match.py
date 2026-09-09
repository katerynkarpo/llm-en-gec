from typing import List, Dict


def calculate_exact_matches(
    hypotheses: List[str],
    references: List[str]
) -> tuple[int, int, float]:
    """
    Calculate exact match accuracy between hypotheses and references.

    Args:
        hypotheses: List of corrected hypotheses (model reference_outputs)
        references: List of reference corrections

    Returns:
        Tuple of (exact_matches_count, total_count, accuracy_percentage)
    """
    if len(hypotheses) != len(references):
        raise ValueError(f"Length mismatch: {len(hypotheses)} hypotheses vs {len(references)} references")

    exact_matches = sum(
        1 for hyp, ref in zip(hypotheses, references)
        if hyp.strip() == ref.strip()
    )

    total = len(hypotheses)
    accuracy = (exact_matches / total * 100) if total > 0 else 0.0

    return exact_matches, total, accuracy


def print_exact_match_results(exact_matches: int, total: int, accuracy: float) -> None:
    """
    Print exact match results in a formatted way.

    Args:
        exact_matches: Number of exact matches
        total: Total number of samples
        accuracy: Accuracy percentage
    """
    print(f"Exact matches: {exact_matches}/{total} ({accuracy:.1f}%)")
