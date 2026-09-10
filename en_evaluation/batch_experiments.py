"""Position-probe and batch-shuffle experiments reported in the paper."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.utils import detokenize, process_in_parallel, tokenize


EvaluatePredictions = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class BatchExperimentContext:
    """Runtime state shared by the two batch-position experiments."""

    run_name: str
    mode: str
    split: str
    model: str
    prompt_name: str
    prompt_run_name: str
    few_shot: dict[str, Any]
    batch_size: int
    sources: list[str]
    references: list[str]
    agent: Any
    log_input_output: bool
    num_threads: int
    show_progress: bool
    run_dir: Path
    m2_path: Path
    m2scorer_path: str | None
    start_index: int
    evaluate_predictions: EvaluatePredictions


def validate_batch_experiment(
    *,
    position_config: dict[str, Any],
    shuffle_config: dict[str, Any],
    batch_size: int,
    total_sentences: int,
    agent: Any,
) -> int | str | None:
    """Validate the selected experiment and return its run-name descriptor."""
    position_enabled = bool(position_config.get("enabled", False))
    shuffle_enabled = bool(shuffle_config.get("enabled", False))

    if position_enabled and shuffle_enabled:
        raise ValueError(
            "Set only one of position_experiment.enabled or "
            "batch_shuffle_experiment.enabled."
        )

    if shuffle_enabled:
        num_permutations = int(shuffle_config.get("num_permutations", 100))
        batch_start_index = int(shuffle_config.get("batch_start_index", 0))
        random_sample = bool(shuffle_config.get("random_sample", False))
        no_repeat_positions = bool(shuffle_config.get("no_repeat_positions", False))
        num_batches = int(shuffle_config.get("num_batches", 1))

        if batch_size <= 1:
            raise ValueError("batch_shuffle_experiment requires batch_size > 1.")
        if num_permutations <= 0:
            raise ValueError(
                "batch_shuffle_experiment.num_permutations must be positive."
            )
        if num_batches <= 0:
            raise ValueError("batch_shuffle_experiment.num_batches must be positive.")
        if no_repeat_positions and num_permutations > batch_size:
            raise ValueError(
                "batch_shuffle_experiment.no_repeat_positions=true allows at most "
                f"batch_size={batch_size} permutations. Got {num_permutations}."
            )
        if not random_sample and batch_start_index < 0:
            raise ValueError(
                "batch_shuffle_experiment.batch_start_index must be non-negative."
            )
        if not random_sample and batch_start_index + batch_size > total_sentences:
            raise ValueError(
                "batch_shuffle_experiment fixed batch exceeds loaded samples: "
                f"start={batch_start_index}, batch_size={batch_size}, "
                f"loaded={total_sentences}."
            )
        if random_sample and batch_size * num_batches > total_sentences:
            raise ValueError(
                "batch_shuffle_experiment cannot sample disjoint random batches: "
                f"batch_size={batch_size}, num_batches={num_batches}, "
                f"loaded={total_sentences}."
            )
        if not hasattr(agent, "execute_sentence_batch"):
            raise ValueError(
                "batch_shuffle_experiment requires an agent with "
                "execute_sentence_batch support."
            )

        return f"shuffle_{batch_size}_b{num_batches}_n{num_permutations}"

    if position_enabled:
        positions = [int(value) for value in position_config.get("positions", [])]
        if batch_size <= 1:
            raise ValueError("position_experiment requires batch_size > 1.")
        if not positions:
            raise ValueError(
                "position_experiment.enabled=true requires at least one probe position."
            )
        if min(positions) < 1 or max(positions) > batch_size:
            raise ValueError(
                f"position_experiment positions {positions} must be within "
                f"[1, {batch_size}]."
            )
        if not hasattr(agent, "execute_sentence_batch"):
            raise ValueError(
                "position_experiment requires an agent with "
                "execute_sentence_batch support."
            )
        return "position_probe"

    return None


def run_batch_experiment(
    *,
    position_config: dict[str, Any],
    shuffle_config: dict[str, Any],
    context: BatchExperimentContext,
) -> bool:
    """Run the enabled batch experiment and return whether one was executed."""
    if bool(shuffle_config.get("enabled", False)):
        _run_batch_shuffle(shuffle_config, context)
        return True
    if bool(position_config.get("enabled", False)):
        _run_position_probe(position_config, context)
        return True
    return False


def build_position_probe_batch(
    target_index: int,
    total_sentences: int,
    batch_size: int,
    position: int,
) -> list[int]:
    """Build a synthetic batch with the target sentence at a fixed position."""
    if total_sentences < batch_size:
        raise ValueError(
            f"Position probe requires at least batch_size={batch_size} sentences, "
            f"got {total_sentences}."
        )
    if position < 1 or position > batch_size:
        raise ValueError(
            f"Probe position must be within [1, {batch_size}], got {position}."
        )

    filler_indices: list[int] = []
    candidate = (target_index + 1) % total_sentences
    while len(filler_indices) < batch_size - 1:
        if candidate != target_index:
            filler_indices.append(candidate)
        candidate = (candidate + 1) % total_sentences

    batch_indices = list(filler_indices)
    batch_indices.insert(position - 1, target_index)
    return batch_indices


def process_position_probe(
    probe_index: int,
    target_index: int,
    position: int,
    batch_indices: list[int],
    sources: list[str],
    references: list[str],
    agent: Any,
    log_input_output: bool,
    include_full_batch_outputs: bool = False,
) -> dict[str, Any]:
    """Run one synthetic batch and return the target sentence output."""
    batch_sources = [sources[index] for index in batch_indices]
    target_batch_offset = position - 1

    try:
        corrected_batch = agent.execute_sentence_batch(batch_sources)
        corrected = corrected_batch[target_batch_offset]
    except Exception as exc:
        print(
            f"[probe target={target_index + 1} position={position}] "
            f"LLM request failed: {exc}"
        )
        corrected = sources[target_index]
        result = {
            "probe_index": probe_index,
            "target_index": target_index + 1,
            "position": position,
            "source": sources[target_index],
            "reference": references[target_index],
            "corrected": corrected,
            "batch_indices": [index + 1 for index in batch_indices],
            "error": str(exc),
        }
        if include_full_batch_outputs:
            result["batch_rows"] = [
                {
                    "batch_offset": batch_offset + 1,
                    "sentence_index": sentence_index + 1,
                    "source": sources[sentence_index],
                    "reference": references[sentence_index],
                    "corrected": sources[sentence_index],
                    "is_target": batch_offset == target_batch_offset,
                }
                for batch_offset, sentence_index in enumerate(batch_indices)
            ]
        return result

    if log_input_output:
        print(f"[probe {probe_index}] TARGET POSITION {position}")
        print(f"[probe {probe_index}] INPUT : {sources[target_index]}")
        print(f"[probe {probe_index}] OUTPUT: {corrected}")

    result = {
        "probe_index": probe_index,
        "target_index": target_index + 1,
        "position": position,
        "source": sources[target_index],
        "reference": references[target_index],
        "corrected": corrected,
        "batch_indices": [index + 1 for index in batch_indices],
    }
    if include_full_batch_outputs:
        result["batch_rows"] = [
            {
                "batch_offset": batch_offset + 1,
                "sentence_index": sentence_index + 1,
                "source": sources[sentence_index],
                "reference": references[sentence_index],
                "corrected": corrected_row,
                "is_target": batch_offset == target_batch_offset,
            }
            for batch_offset, (sentence_index, corrected_row) in enumerate(
                zip(batch_indices, corrected_batch)
            )
        ]
    return result


def process_batch_shuffle_permutation(
    permutation_index: int,
    batch_indices: list[int],
    sources: list[str],
    references: list[str],
    agent: Any,
    log_input_output: bool,
    include_full_outputs: bool = True,
    batch_sample_index: int = 1,
    rotation_offset: int | None = None,
) -> dict[str, Any]:
    """Run one sampled permutation of a fixed batch."""
    batch_sources = [sources[index] for index in batch_indices]

    try:
        corrected_batch = agent.execute_sentence_batch(batch_sources)
    except Exception as exc:
        print(f"[shuffle permutation={permutation_index}] LLM request failed: {exc}")
        corrected_batch = batch_sources
        error = str(exc)
    else:
        error = None

    rows = []
    for position, (sentence_index, corrected) in enumerate(
        zip(batch_indices, corrected_batch), start=1
    ):
        if log_input_output:
            print(
                f"[perm {permutation_index} pos {position}] "
                f"INPUT : {sources[sentence_index]}"
            )
            print(
                f"[perm {permutation_index} pos {position}] OUTPUT: {corrected}"
            )
        rows.append(
            {
                "batch_sample_index": batch_sample_index,
                "permutation_index": permutation_index,
                "rotation_offset": rotation_offset,
                "position": position,
                "sentence_index": sentence_index + 1,
                "source": sources[sentence_index],
                "reference": references[sentence_index],
                "corrected": corrected,
            }
        )

    result = {
        "batch_sample_index": batch_sample_index,
        "permutation_index": permutation_index,
        "rotation_offset": rotation_offset,
        "batch_indices": [index + 1 for index in batch_indices],
    }
    if error:
        result["error"] = error
    if include_full_outputs:
        result["rows"] = rows
    return result


def build_no_repeat_position_batches(
    *,
    total_sentences: int,
    batch_size: int,
    num_permutations: int,
    seed: int,
    num_batches: int = 1,
) -> tuple[list[list[int]], list[tuple[int, int, list[int], int]]]:
    """Sample disjoint batches and rotate them without repeated positions."""
    total_required = batch_size * num_batches
    if total_required > total_sentences:
        raise ValueError(
            f"Cannot sample {num_batches} disjoint batches of size {batch_size} "
            f"from {total_sentences} loaded samples."
        )
    if num_permutations > batch_size:
        raise ValueError(
            "batch_shuffle_experiment.no_repeat_positions=true allows at most "
            f"batch_size permutations. Got num_permutations={num_permutations}, "
            f"batch_size={batch_size}."
        )
    if num_batches <= 0:
        raise ValueError("batch_shuffle_experiment.num_batches must be positive.")

    rng = random.Random(seed)
    sampled_indices = rng.sample(range(total_sentences), total_required)
    selected_batches = [
        sampled_indices[index * batch_size : (index + 1) * batch_size]
        for index in range(num_batches)
    ]

    permutation_specs = []
    permutation_index = 1
    for batch_sample_index, base_indices in enumerate(selected_batches, start=1):
        offsets = list(range(batch_size))
        rng.shuffle(offsets)
        for offset in offsets[:num_permutations]:
            batch_indices = base_indices[offset:] + base_indices[:offset]
            permutation_specs.append(
                (permutation_index, batch_sample_index, batch_indices, offset)
            )
            permutation_index += 1
    return selected_batches, permutation_specs


def _strip_full_batch_outputs(probe_results: list[dict]) -> list[dict]:
    compact_results = []
    for row in probe_results:
        compact_row = dict(row)
        compact_row.pop("batch_rows", None)
        compact_results.append(compact_row)
    return compact_results


def _save_position_probe_full_batches(
    *,
    probe_results: list[dict],
    probe_positions: list[int],
    run_dir: Path,
) -> None:
    batch_outputs_dir = run_dir / "batch_outputs"
    batch_outputs_dir.mkdir(exist_ok=True)

    for position in probe_positions:
        position_rows = sorted(
            [
                row
                for row in probe_results
                if row["position"] == position and "batch_rows" in row
            ],
            key=lambda row: row["target_index"],
        )
        output_path = batch_outputs_dir / f"pos{position}.jsonl"
        with open(output_path, "w", encoding="utf-8") as output_file:
            for row in position_rows:
                batch_payload = {
                    "probe_index": row["probe_index"],
                    "target_index": row["target_index"],
                    "position": row["position"],
                    "source": row["source"],
                    "reference": row["reference"],
                    "corrected": row["corrected"],
                    "batch_indices": row["batch_indices"],
                    "batch_rows": row["batch_rows"],
                }
                if row.get("error"):
                    batch_payload["error"] = row["error"]
                output_file.write(
                    json.dumps(batch_payload, ensure_ascii=False) + "\n"
                )


def _save_batch_shuffle_permutations(
    *, permutation_results: list[dict], run_dir: Path
) -> None:
    output_path = run_dir / "batch_shuffle_permutations.jsonl"
    with open(output_path, "w", encoding="utf-8") as output_file:
        for result in sorted(
            permutation_results, key=lambda row: row["permutation_index"]
        ):
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")


def _normalize_for_metric(text: str) -> str:
    try:
        text = tokenize(detokenize(str(text)))
    except Exception:
        text = str(text)
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def _token_edit_distance(source: str, corrected: str) -> int:
    source_tokens = _normalize_for_metric(source).split()
    corrected_tokens = _normalize_for_metric(corrected).split()
    if not source_tokens:
        return len(corrected_tokens)

    previous = list(range(len(corrected_tokens) + 1))
    for source_index, source_token in enumerate(source_tokens, start=1):
        current = [source_index]
        for corrected_index, corrected_token in enumerate(
            corrected_tokens, start=1
        ):
            cost = 0 if source_token == corrected_token else 1
            current.append(
                min(
                    previous[corrected_index] + 1,
                    current[corrected_index - 1] + 1,
                    previous[corrected_index - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def _add_edit_rate_fields(results: list[dict]) -> None:
    for row in results:
        source = row.get("source", "")
        corrected = row.get("corrected", "")
        source_length = max(1, len(_normalize_for_metric(source).split()))
        token_edits = _token_edit_distance(source, corrected)
        row["changed"] = _normalize_for_metric(source) != _normalize_for_metric(
            corrected
        )
        row["token_edit_distance"] = token_edits
        row["token_edit_rate"] = token_edits / source_length


def _build_m2_subset_file(
    m2_path: Path, sentence_indices: list[int], output_path: Path
) -> None:
    with open(m2_path, "r", encoding="utf-8") as source_file:
        lines = source_file.readlines()

    blocks: list[list[str]] = []
    current_block: list[str] = []
    for line in lines:
        if line.startswith("S "):
            if current_block:
                blocks.append(current_block)
            current_block = [line]
        else:
            current_block.append(line)
    if current_block:
        blocks.append(current_block)

    missing = [
        index for index in sentence_indices if index < 1 or index > len(blocks)
    ]
    if missing:
        raise ValueError(f"M2 sentence indices out of range: {missing[:10]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        for sentence_index in sentence_indices:
            output_file.writelines(blocks[sentence_index - 1])
            output_file.write("\n")


def _evaluate_arbitrary_predictions(
    *,
    results: list[dict],
    references_multi: list[list[str]],
    context: BatchExperimentContext,
    prefix: str,
) -> dict[str, Any]:
    sentence_indices = [int(row["sentence_index"]) for row in results]
    ordered_m2_path = context.run_dir / f"{prefix}references.m2"
    _build_m2_subset_file(context.m2_path, sentence_indices, ordered_m2_path)
    return context.evaluate_predictions(
        results=results,
        references_multi=references_multi,
        m2_path=ordered_m2_path,
        m2scorer_path=context.m2scorer_path,
        start_index=0,
        run_dir=context.run_dir,
        prefix=prefix,
    )


def _summarize_position_rows(
    summary: dict[str, Any], position_results: list[dict]
) -> None:
    changed_count = sum(1 for row in position_results if row.get("changed"))
    total = len(position_results)
    mean_token_edit_rate = (
        sum(float(row.get("token_edit_rate", 0.0)) for row in position_results)
        / total
        if total
        else 0.0
    )
    mean_token_edit_distance = (
        sum(float(row.get("token_edit_distance", 0.0)) for row in position_results)
        / total
        if total
        else 0.0
    )
    summary["changed_count"] = changed_count
    summary["edit_rate"] = changed_count / total if total else 0.0
    summary["mean_token_edit_rate"] = mean_token_edit_rate
    summary["mean_token_edit_distance"] = mean_token_edit_distance


def _write_batch_shuffle_svg_plot(
    summary_rows: list[dict], output_path: Path
) -> None:
    if not summary_rows:
        return

    width = 980
    height = 520
    margin_left = 70
    margin_right = 40
    margin_top = 35
    margin_bottom = 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    positions = [int(row["position"]) for row in summary_rows]
    max_position = max(positions)
    min_position = min(positions)

    def x_for(position: int) -> float:
        if max_position == min_position:
            return margin_left + plot_width / 2
        return (
            margin_left
            + (position - min_position)
            / (max_position - min_position)
            * plot_width
        )

    def y_for(value: float) -> float:
        clipped = max(0.0, min(1.0, value))
        return margin_top + (1.0 - clipped) * plot_height

    def polyline(metric: str) -> str:
        return " ".join(
            f"{x_for(int(row['position'])):.1f},"
            f"{y_for(float(row.get(metric) or 0.0)):.1f}"
            for row in summary_rows
        )

    x_ticks = []
    for position in positions:
        if position == min_position or position == max_position or position % 5 == 0:
            x = x_for(position)
            x_ticks.append(
                f'<line x1="{x:.1f}" y1="{height - margin_bottom}" '
                f'x2="{x:.1f}" y2="{height - margin_bottom + 6}" stroke="#444" />'
                f'<text x="{x:.1f}" y="{height - margin_bottom + 24}" '
                f'text-anchor="middle">{position}</text>'
            )

    y_ticks = []
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_for(tick)
        y_ticks.append(
            f'<line x1="{margin_left - 6}" y1="{y:.1f}" '
            f'x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e7eb" />'
            f'<text x="{margin_left - 12}" y="{y + 4:.1f}" '
            f'text-anchor="end">{tick:.2f}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 13px; fill: #111827; }}
    .title {{ font-size: 20px; font-weight: 700; }}
    .label {{ font-size: 14px; font-weight: 600; }}
  </style>
  <text class="title" x="{margin_left}" y="24">Batch Position Sensitivity</text>
  {''.join(y_ticks)}
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#111827" />
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#111827" />
  {''.join(x_ticks)}
  <polyline points="{polyline('f0_5')}" fill="none" stroke="#2563eb" stroke-width="3" />
  <polyline points="{polyline('edit_rate')}" fill="none" stroke="#dc2626" stroke-width="3" />
  <text class="label" x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle">Position in batch</text>
  <text class="label" x="20" y="{height / 2:.1f}" text-anchor="middle" transform="rotate(-90 20 {height / 2:.1f})">Metric value</text>
  <rect x="{width - 230}" y="26" width="170" height="58" fill="#ffffff" stroke="#d1d5db" />
  <line x1="{width - 212}" y1="48" x2="{width - 172}" y2="48" stroke="#2563eb" stroke-width="3" />
  <text x="{width - 162}" y="52">F0.5</text>
  <line x1="{width - 212}" y1="70" x2="{width - 172}" y2="70" stroke="#dc2626" stroke-width="3" />
  <text x="{width - 162}" y="74">Edit rate</text>
</svg>
'''
    output_path.write_text(svg, encoding="utf-8")


def _run_batch_shuffle(
    settings: dict[str, Any], context: BatchExperimentContext
) -> None:
    num_permutations = int(settings.get("num_permutations", 100))
    seed = int(settings.get("seed", 13))
    batch_start_index = int(settings.get("batch_start_index", 0))
    save_permutations = bool(settings.get("save_permutations", True))
    random_sample = bool(settings.get("random_sample", False))
    no_repeat_positions = bool(settings.get("no_repeat_positions", False))
    num_batches = int(settings.get("num_batches", 1))

    print("Running sampled batch-shuffle position experiment...")
    print(f"Shuffle permutations: {num_permutations}")
    print(f"Shuffle seed: {seed}")
    print(f"Shuffle random batches: {num_batches}")
    print(f"Shuffle random sample: {random_sample}")
    print(f"Shuffle no-repeat positions: {no_repeat_positions}")
    if not random_sample:
        print(f"Shuffle batch start index: {batch_start_index}")

    if random_sample and no_repeat_positions:
        selected_batch_groups, permutation_specs = build_no_repeat_position_batches(
            total_sentences=len(context.sources),
            batch_size=context.batch_size,
            num_permutations=num_permutations,
            seed=seed,
            num_batches=num_batches,
        )
        fixed_batch_indices = [
            index for batch_group in selected_batch_groups for index in batch_group
        ]
    else:
        rng = random.Random(seed)
        selected_batch_groups = [
            (
                rng.sample(range(len(context.sources)), context.batch_size)
                if random_sample
                else list(
                    range(batch_start_index, batch_start_index + context.batch_size)
                )
            )
        ]
        fixed_batch_indices = selected_batch_groups[0]
        permutation_specs = []
        for permutation_index in range(1, num_permutations + 1):
            sampled_indices = list(fixed_batch_indices)
            rng.shuffle(sampled_indices)
            permutation_specs.append(
                (permutation_index, 1, sampled_indices, None)
            )

    print(
        "Selected batch groups: "
        + "; ".join(
            f"{group_index}: {', '.join(str(index + 1) for index in batch_group)}"
            for group_index, batch_group in enumerate(
                selected_batch_groups, start=1
            )
        )
    )
    permutation_items = [
        (
            permutation_index,
            sampled_indices,
            context.sources,
            context.references,
            context.agent,
            context.log_input_output,
            True,
            batch_sample_index,
            rotation_offset,
        )
        for (
            permutation_index,
            batch_sample_index,
            sampled_indices,
            rotation_offset,
        ) in permutation_specs
    ]

    permutation_results = process_in_parallel(
        permutation_items,
        process_batch_shuffle_permutation,
        max_workers=context.num_threads,
        show_progress=context.show_progress,
    )
    shuffle_rows = [
        row for result in permutation_results for row in result.get("rows", [])
    ]
    _add_edit_rate_fields(shuffle_rows)

    if save_permutations:
        _save_batch_shuffle_permutations(
            permutation_results=permutation_results,
            run_dir=context.run_dir,
        )

    selected_groups_payload = [
        {
            "batch_sample_index": group_index,
            "sentence_indices": [index + 1 for index in batch_group],
        }
        for group_index, batch_group in enumerate(selected_batch_groups, start=1)
    ]
    offsets_payload = [
        {
            "permutation_index": permutation_index,
            "batch_sample_index": batch_sample_index,
            "rotation_offset": offset,
        }
        for permutation_index, batch_sample_index, _, offset in permutation_specs
        if offset is not None
    ]

    _write_json(
        context.run_dir / "batch_shuffle_results.json",
        {
            "run_name": context.run_name,
            "mode": context.mode,
            "split": context.split,
            "model": context.model,
            "prompt": context.prompt_name,
            "prompt_run": context.prompt_run_name,
            "random_sample": random_sample,
            "no_repeat_positions": no_repeat_positions,
            "few_shot": context.few_shot,
            "batch_size": context.batch_size,
            "num_batches": num_batches,
            "num_permutations": num_permutations,
            "seed": seed,
            "batch_start_index": batch_start_index,
            "fixed_batch_indices": [index + 1 for index in fixed_batch_indices],
            "selected_batch_groups": selected_groups_payload,
            "permutation_offsets": offsets_payload,
            "rows": shuffle_rows,
        },
    )

    summary_rows = []
    for position in range(1, context.batch_size + 1):
        position_results = [
            row for row in shuffle_rows if row["position"] == position
        ]
        position_references = [[row["reference"]] for row in position_results]

        with open(
            context.run_dir / f"predictions.pos{position}.txt",
            "w",
            encoding="utf-8",
        ) as output_file:
            for row in position_results:
                output_file.write(f"{row['corrected']}\n")

        summary = _evaluate_arbitrary_predictions(
            results=position_results,
            references_multi=position_references,
            context=context,
            prefix=f"pos{position}.",
        )
        summary["position"] = position
        _summarize_position_rows(summary, position_results)
        summary_rows.append(summary)

        if summary["f0_5"] is not None:
            print(
                f"Position {position}: n={summary['total_samples']} "
                f"edit_rate={summary['edit_rate']:.4f} "
                f"F0.5={summary['f0_5']:.4f}"
            )
        else:
            print(
                f"Position {position}: n={summary['total_samples']} "
                f"edit_rate={summary['edit_rate']:.4f} F0.5=NA"
            )

    _write_json(
        context.run_dir / "batch_shuffle_summary.json",
        {
            "run_name": context.run_name,
            "batch_size": context.batch_size,
            "num_batches": num_batches,
            "num_permutations": num_permutations,
            "seed": seed,
            "random_sample": random_sample,
            "no_repeat_positions": no_repeat_positions,
            "few_shot": context.few_shot,
            "batch_start_index": batch_start_index,
            "fixed_batch_indices": [index + 1 for index in fixed_batch_indices],
            "selected_batch_groups": selected_groups_payload,
            "permutation_offsets": offsets_payload,
            "summaries": summary_rows,
        },
    )

    with open(
        context.run_dir / "batch_shuffle_summary.txt", "w", encoding="utf-8"
    ) as output_file:
        output_file.write(
            "position\ttotal_samples\texact_match_percent\tprecision\trecall\t"
            "f0_5\tedit_rate\tmean_token_edit_rate\tmean_token_edit_distance\n"
        )
        for row in summary_rows:
            precision = (
                f"{row['precision']:.4f}" if row["precision"] is not None else "NA"
            )
            recall = f"{row['recall']:.4f}" if row["recall"] is not None else "NA"
            f05 = f"{row['f0_5']:.4f}" if row["f0_5"] is not None else "NA"
            output_file.write(
                f"{row['position']}\t{row['total_samples']}\t"
                f"{row['exact_match_percent']:.1f}\t{precision}\t{recall}\t{f05}\t"
                f"{row['edit_rate']:.6f}\t{row['mean_token_edit_rate']:.6f}\t"
                f"{row['mean_token_edit_distance']:.6f}\n"
            )

    with open(
        context.run_dir / "batch_shuffle_summary.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        fieldnames = [
            "position",
            "total_samples",
            "exact_match_percent",
            "precision",
            "recall",
            "f0_5",
            "edit_rate",
            "mean_token_edit_rate",
            "mean_token_edit_distance",
            "metric_source",
            "m2_status",
            "m2_error",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field) for field in fieldnames})

    _write_batch_shuffle_svg_plot(
        summary_rows, context.run_dir / "batch_shuffle_summary.svg"
    )


def _run_position_probe(
    settings: dict[str, Any], context: BatchExperimentContext
) -> None:
    positions = [int(value) for value in settings.get("positions", [])]
    save_full_batches = bool(settings.get("save_full_batches", False))

    print("Running position-in-batch probe...")
    print(f"Probe positions: {positions}")
    print(f"Save full batches: {save_full_batches}")

    probe_items = []
    probe_counter = 1
    total_sentences = len(context.sources)
    for target_index in range(total_sentences):
        for position in positions:
            batch_indices = build_position_probe_batch(
                target_index=target_index,
                total_sentences=total_sentences,
                batch_size=context.batch_size,
                position=position,
            )
            probe_items.append(
                (
                    probe_counter,
                    target_index,
                    position,
                    batch_indices,
                    context.sources,
                    context.references,
                    context.agent,
                    context.log_input_output,
                    save_full_batches,
                )
            )
            probe_counter += 1

    probe_results = process_in_parallel(
        probe_items,
        process_position_probe,
        max_workers=context.num_threads,
        show_progress=context.show_progress,
    )
    compact_results = _strip_full_batch_outputs(probe_results)

    _write_json(
        context.run_dir / "position_probe_results.json",
        {
            "run_name": context.run_name,
            "mode": context.mode,
            "split": context.split,
            "model": context.model,
            "prompt": context.prompt_name,
            "prompt_run": context.prompt_run_name,
            "few_shot": context.few_shot,
            "batch_size": context.batch_size,
            "positions": positions,
            "results": compact_results,
        },
    )

    if save_full_batches:
        _save_position_probe_full_batches(
            probe_results=probe_results,
            probe_positions=positions,
            run_dir=context.run_dir,
        )

    summary_rows = []
    for position in positions:
        position_results = sorted(
            [row for row in compact_results if row["position"] == position],
            key=lambda row: row["target_index"],
        )
        position_references = [
            [context.references[row["target_index"] - 1]]
            for row in position_results
        ]

        with open(
            context.run_dir / f"predictions.pos{position}.txt",
            "w",
            encoding="utf-8",
        ) as output_file:
            for row in position_results:
                output_file.write(f"{row['corrected']}\n")

        summary = context.evaluate_predictions(
            results=position_results,
            references_multi=position_references,
            m2_path=context.m2_path,
            m2scorer_path=context.m2scorer_path,
            start_index=context.start_index,
            run_dir=context.run_dir,
            prefix=f"pos{position}.",
        )
        summary["position"] = position
        summary_rows.append(summary)

        if summary["f0_5"] is not None:
            print(
                f"Position {position}: exact={summary['exact_match_percent']:.1f}% "
                f"F0.5={summary['f0_5']:.4f}"
            )
        else:
            print(
                f"Position {position}: exact={summary['exact_match_percent']:.1f}% "
                "F0.5=NA"
            )

    _write_json(
        context.run_dir / "position_probe_summary.json",
        {
            "run_name": context.run_name,
            "batch_size": context.batch_size,
            "positions": positions,
            "few_shot": context.few_shot,
            "summaries": summary_rows,
        },
    )

    with open(
        context.run_dir / "position_probe_summary.txt", "w", encoding="utf-8"
    ) as output_file:
        output_file.write(
            "position\texact_match_percent\tprecision\trecall\tf0_5\n"
        )
        for row in summary_rows:
            precision = (
                f"{row['precision']:.4f}" if row["precision"] is not None else "NA"
            )
            recall = f"{row['recall']:.4f}" if row["recall"] is not None else "NA"
            f05 = f"{row['f0_5']:.4f}" if row["f0_5"] is not None else "NA"
            output_file.write(
                f"{row['position']}\t{row['exact_match_percent']:.1f}\t"
                f"{precision}\t{recall}\t{f05}\n"
            )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
