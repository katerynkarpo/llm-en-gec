#!/usr/bin/env python3
"""
Convert M2 format files to parallel format (source + reference files).

This script parses CoNLL-2014 M2 annotation files and generates:
- .src file: Original sentences with errors (tokenized)
- .ref0, .ref1, ... files: Reference corrections from each annotator (tokenized)

The M2 format contains:
- S lines: Source sentences (tokenized)
- A lines: Annotations with format: start end|||error_type|||correction|||fields|||annotator_id

Usage:
    python convert_m2_to_parallel.py [M2_FILE] [OUTPUT_PREFIX]

Arguments:
    M2_FILE: Path to M2 annotation file (default: official-2014.combined-withalt.m2)
    OUTPUT_PREFIX: Output file prefix (default: conll14.test)

Examples:
    python convert_m2_to_parallel.py
    python convert_m2_to_parallel.py official-2014.combined-withalt.m2 conll14.test
    python convert_m2_to_parallel.py custom.m2 custom
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def parse_m2_file(m2_path: Path) -> List[Tuple[str, Dict[int, List[Tuple[int, int, str]]]]]:
    """
    Parse M2 file and extract source sentences with edits grouped by annotator.

    Args:
        m2_path: Path to M2 annotation file

    Returns:
        List of (source_sentence, annotator_edits) tuples where:
        - source_sentence: space-separated tokenized sentence
        - annotator_edits: Dict mapping annotator_id to list of (start, end, correction) tuples
    """
    sentences = []
    current_source = None
    current_edits = defaultdict(list)

    with open(m2_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')

            if not line:
                # Empty line marks end of sentence block
                if current_source is not None:
                    sentences.append((current_source, dict(current_edits)))
                    current_source = None
                    current_edits = defaultdict(list)
                continue

            if line.startswith('S '):
                # Source sentence (remove 'S ' prefix)
                current_source = line[2:].strip()

            elif line.startswith('A '):
                # Annotation line: A start end|||error_type|||correction|||field4|||field5|||annotator_id
                annotation = line[2:].strip()
                parts = annotation.split('|||')

                if len(parts) < 6:
                    print(f"Warning: Skipping malformed annotation: {line}", file=sys.stderr)
                    continue

                # Parse offsets
                offsets = parts[0].split()
                if len(offsets) != 2:
                    print(f"Warning: Invalid offsets in annotation: {line}", file=sys.stderr)
                    continue

                start = int(offsets[0])
                end = int(offsets[1])
                error_type = parts[1]
                correction = parts[2]
                annotator_id = int(parts[5])

                # Skip noop edits (no correction needed)
                if error_type == 'noop':
                    continue

                # Handle -NONE- as empty string (deletion)
                if correction == '-NONE-':
                    correction = ''

                current_edits[annotator_id].append((start, end, correction))

    # Don't forget the last sentence
    if current_source is not None:
        sentences.append((current_source, dict(current_edits)))

    return sentences


def apply_edits(source: str, edits: List[Tuple[int, int, str]]) -> str:
    """
    Apply edits to source sentence to generate corrected version.

    Args:
        source: Space-separated tokenized source sentence
        edits: List of (start, end, correction) tuples

    Returns:
        Corrected sentence (space-separated tokens)
    """
    tokens = source.split()

    # Sort edits by start position (descending) to apply from right to left
    # This prevents offset shifts when applying multiple edits
    sorted_edits = sorted(edits, key=lambda x: x[0], reverse=True)

    for start, end, correction in sorted_edits:
        # Validate offsets
        if start < 0 or end < 0 or start > len(tokens) or end > len(tokens):
            print(f"Warning: Invalid offsets ({start}, {end}) for sentence: {source[:50]}...", file=sys.stderr)
            continue

        # Apply edit: replace tokens[start:end] with correction
        correction_tokens = correction.split() if correction else []
        tokens[start:end] = correction_tokens

    return ' '.join(tokens)


def convert_m2_to_parallel(m2_path: Path, output_prefix: str):
    """
    Convert M2 file to parallel format (source + reference files).

    Args:
        m2_path: Path to M2 annotation file
        output_prefix: Prefix for output files (e.g., 'conll14.test')
    """
    print(f"Reading M2 file: {m2_path}")
    sentences = parse_m2_file(m2_path)
    print(f"Parsed {len(sentences)} sentences")

    if not sentences:
        print("Error: No sentences found in M2 file", file=sys.stderr)
        sys.exit(1)

    # Extract source sentences
    sources = [source for source, _ in sentences]

    # Collect all unique annotator IDs
    all_annotator_ids = set()
    for _, edits_dict in sentences:
        all_annotator_ids.update(edits_dict.keys())

    annotator_ids = sorted(all_annotator_ids)
    print(f"Found {len(annotator_ids)} annotators: {annotator_ids}")

    # Generate references for each annotator
    references_by_annotator = {}
    for annotator_id in annotator_ids:
        references = []
        for source, edits_dict in sentences:
            if annotator_id in edits_dict:
                # Apply this annotator's edits
                corrected = apply_edits(source, edits_dict[annotator_id])
            else:
                # No edits from this annotator, use source as-is
                corrected = source
            references.append(corrected)
        references_by_annotator[annotator_id] = references

    # Write output files
    output_dir = m2_path.parent

    # Write source file
    src_file = output_dir / f"{output_prefix}.src"
    with open(src_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sources) + '\n')
    print(f"Written source file: {src_file} ({len(sources)} sentences)")

    # Write reference files
    for idx, annotator_id in enumerate(annotator_ids):
        ref_file = output_dir / f"{output_prefix}.ref{idx}"
        references = references_by_annotator[annotator_id]
        with open(ref_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(references) + '\n')
        print(f"Written reference file: {ref_file} (annotator {annotator_id}, {len(references)} sentences)")

    # Copy M2 file to match naming convention
    m2_output = output_dir / f"{output_prefix}.m2"
    if m2_output != m2_path:
        import shutil
        shutil.copy2(m2_path, m2_output)
        print(f"Copied M2 file: {m2_output}")

    print(f"\nConversion complete!")
    print(f"Generated files:")
    print(f"  - {src_file.name}")
    for idx in range(len(annotator_ids)):
        print(f"  - {output_prefix}.ref{idx}")
    print(f"  - {m2_output.name}")


def main():
    """Main entry point for the conversion utility."""
    # Default values
    default_m2_file = "official-2014.combined-withalt.m2"
    default_output_prefix = "conll14.test"

    # Parse command line arguments
    if len(sys.argv) > 3:
        print(__doc__)
        sys.exit(1)

    # Get M2 file path
    if len(sys.argv) >= 2:
        m2_file = sys.argv[1]
    else:
        m2_file = default_m2_file

    # Get output prefix
    if len(sys.argv) >= 3:
        output_prefix = sys.argv[2]
    else:
        output_prefix = default_output_prefix

    # Convert to Path and validate
    m2_path = Path(m2_file)
    if not m2_path.is_absolute():
        # Assume relative to script directory
        m2_path = Path(__file__).parent / m2_file

    if not m2_path.exists():
        print(f"Error: M2 file not found: {m2_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("M2 to Parallel Format Converter")
    print("=" * 80)
    print()

    # Run conversion
    convert_m2_to_parallel(m2_path, output_prefix)


if __name__ == "__main__":
    main()
