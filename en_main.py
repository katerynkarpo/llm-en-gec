import argparse
import asyncio
import json
import random
import re
import shutil
from pathlib import Path

import yaml
from dotenv import load_dotenv

from en_evaluation.batch_experiments import (
    BatchExperimentContext,
    run_batch_experiment,
    validate_batch_experiment,
)
from en_evaluation.exact_match import calculate_exact_matches, print_exact_match_results
from en_evaluation.m2scorer import calculate_m2score, print_m2score_results
from src.agent_registry import get_agent, register_agent
from src.agents import SinglePromptGECAgent
from src.llm import create_router
from src.utils import detokenize, tokenize, process_in_parallel

try:
    from gec_metrics import get_metric
except Exception:
    get_metric = None


load_dotenv()


def cleanup_litellm_async_clients() -> None:
    """Close LiteLLM async HTTP clients to avoid pending-task warnings on exit."""
    try:
        import litellm
    except Exception:
        return

    close_fn = getattr(litellm, "close_litellm_async_clients", None)
    if not callable(close_fn):
        return

    try:
        asyncio.run(close_fn())
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run English GEC experiments (BEA-2019 style).")
    parser.add_argument(
        "--config",
        default="configs/bea_dev.yaml",
        help="Path to YAML config file (default: configs/bea_dev.yaml).",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    config_file = Path(__file__).parent / config_path
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_agents() -> None:
    register_agent("gec_single_prompt", SinglePromptGECAgent)


def normalise_run_name_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    normalized = normalized.strip("-")
    return normalized or "unknown"


def build_run_name(
    mode: str,
    split: str,
    model: str,
    prompt_name: str,
    temperature: float | None,
    reasoning_effort: str | None,
    batch_descriptor: int | str | None,
) -> str:
    if isinstance(temperature, (int, float)):
        temp_token = f"temp{temperature:g}"
    else:
        temp_token = "tempNA"
    effort_token = f"effort{reasoning_effort}" if reasoning_effort else "effortNA"
    if isinstance(batch_descriptor, (int, str)):
        batch_token = f"batch_{batch_descriptor}"
    else:
        batch_token = "batch_NA"
    return "_".join(
        [
            normalise_run_name_component(mode),
            normalise_run_name_component(split),
            normalise_run_name_component(model),
            normalise_run_name_component(prompt_name),
            normalise_run_name_component(batch_token),
            normalise_run_name_component(temp_token),
            normalise_run_name_component(effort_token),
        ]
    )


def load_parallel_texts(src_path: Path, ref_path: Path) -> tuple[list[str], list[str]]:
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")

    with open(src_path, "r", encoding="utf-8") as f:
        sources = [line.rstrip("\n") for line in f]
    with open(ref_path, "r", encoding="utf-8") as f:
        refs = [line.rstrip("\n") for line in f]

    if len(sources) != len(refs):
        raise ValueError(f"Length mismatch: {len(sources)} sources vs {len(refs)} refs")
    return sources, refs


def detokenize_prompt_example(text: str) -> str:
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


def build_few_shot_prompt(
    *,
    src_path: Path,
    ref_path: Path,
    n: int,
    seed: int,
    detokenize_examples: bool = True,
) -> tuple[str, list[dict]]:
    if n <= 0:
        return "", []

    sources, refs = load_parallel_texts(src_path, ref_path)
    if n > len(sources):
        raise ValueError(
            f"few_shot.n={n} exceeds available train examples ({len(sources)}) in {src_path}"
        )

    rng = random.Random(seed)
    selected_indices = rng.sample(range(len(sources)), n)
    examples = [
        {
            "train_index": idx + 1,
            "source": detokenize_prompt_example(sources[idx]) if detokenize_examples else sources[idx],
            "reference": detokenize_prompt_example(refs[idx]) if detokenize_examples else refs[idx],
        }
        for idx in selected_indices
    ]

    lines = [
        "Few-shot examples from BEA train:",
        "Follow the same input-to-correction style. Do not copy these examples; use them only as guidance.",
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
    return "\n".join(lines), examples


def build_few_shot_metadata(
    *,
    enabled: bool,
    n: int,
    seed: int,
    detokenize_examples: bool,
    delivery: str,
    demo_chunk: int,
    src_path: Path,
    ref_path: Path,
    examples: list[dict],
) -> dict:
    """Build the canonical few-shot payload saved with every run artifact."""
    return {
        "enabled": enabled,
        "n": n if enabled else 0,
        "seed": seed if enabled else None,
        "detokenize": detokenize_examples if enabled else None,
        "delivery": delivery if enabled else None,
        "demo_chunk": demo_chunk if enabled else None,
        "src_path": str(src_path) if enabled else None,
        "ref_path": str(ref_path) if enabled else None,
        "examples": examples if enabled else [],
    }


def load_parallel_texts_multi(src_path: Path, ref_paths: list[Path]) -> tuple[list[str], list[list[str]]]:
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    if not ref_paths:
        raise ValueError("At least one reference path is required")

    with open(src_path, "r", encoding="utf-8") as f:
        sources = [line.rstrip("\n") for line in f]

    all_refs: list[list[str]] = []
    for ref_path in ref_paths:
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference file not found: {ref_path}")
        with open(ref_path, "r", encoding="utf-8") as f:
            refs = [line.rstrip("\n") for line in f]
        if len(sources) != len(refs):
            raise ValueError(f"Length mismatch: {len(sources)} sources vs {len(refs)} refs in {ref_path}")
        all_refs.append(refs)

    refs_nested = [[ref_list[i] for ref_list in all_refs] for i in range(len(sources))]
    return sources, refs_nested


def _iter_chunks(seq: list, chunk_size: int):
    for i in range(0, len(seq), chunk_size):
        yield seq[i:i + chunk_size]


def process_batch_chunk(
    chunk_index: int,
    chunk: list[tuple[int, str, str]],
    gec_agent,
    log_input_output: bool,
) -> dict:
    """Process one batch chunk via execute_sentence_batch."""
    sources = [src for _, src, _ in chunk]
    rows = []
    try:
        corrected_batch = gec_agent.execute_sentence_batch(sources)
    except Exception as exc:
        print(f"[batch {chunk[0][0]}-{chunk[-1][0]}] LLM request failed: {exc}")
        corrected_batch = sources
        for index, source, reference in chunk:
            rows.append({
                "index": index,
                "source": source,
                "reference": reference,
                "corrected": source,
                "error": str(exc),
            })
        return {"batch_index": chunk_index, "rows": rows}

    for (index, source, reference), corrected in zip(chunk, corrected_batch):
        if log_input_output:
            print(f"[{index}] INPUT : {source}")
            print(f"[{index}] OUTPUT: {corrected}")
        rows.append({
            "index": index,
            "source": source,
            "reference": reference,
            "corrected": corrected,
        })
    return {"batch_index": chunk_index, "rows": rows}


def process_sample(index: int, source: str, reference: str, gec_agent, log_input_output: bool) -> dict:
    try:
        response = gec_agent.execute(source)
        corrected = response.corrected_sentence
    except Exception as exc:
        print(f"[{index}] LLM request failed: {exc}")
        corrected = source
        return {
            "index": index,
            "source": source,
            "reference": reference,
            "corrected": corrected,
            "error": str(exc),
        }

    if log_input_output:
        print(f"[{index}] INPUT : {source}")
        print(f"[{index}] OUTPUT: {corrected}")

    return {
        "index": index,
        "source": source,
        "reference": reference,
        "corrected": corrected,
    }


def write_eval_ready_output(results: list[dict], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            corrected = str(r.get("corrected", ""))
            normalized = " ".join(corrected.replace("\r", " ").replace("\n", " ").split())
            f.write(f"{normalized}\n")


def evaluate_en_predictions(
    *,
    results: list[dict],
    references_multi: list[list[str]],
    m2_path: Path,
    m2scorer_path: str | None,
    start_index: int,
    run_dir: Path,
    prefix: str = "",
) -> dict:
    """Evaluate English sentence-level predictions and persist compact artifacts."""
    hypotheses = [_normalize_for_en_metric(r["corrected"]) for r in results]
    references = [_normalize_for_en_metric(r["reference"]) for r in results]
    normalized_refs_multi = [
        [_normalize_for_en_metric(ref) for ref in sent_refs]
        for sent_refs in references_multi
    ]

    exact_matches, total, exact_acc = calculate_exact_matches(hypotheses, references)

    eval_summary = {
        "status": "ok",
        "metric_source": None,
        "errant_status": "skipped",
        "errant_error": None,
        "m2_status": "ok",
        "m2_error": None,
        "precision": None,
        "recall": None,
        "f0_5": None,
        "exact_matches": exact_matches,
        "total_samples": total,
        "exact_match_percent": exact_acc,
    }

    if get_metric is not None:
        try:
            errant_metric_cls = get_metric("errant")
            errant_metric = errant_metric_cls(errant_metric_cls.Config(beta=0.5, language="en"))
            valid_inputs = [_normalize_for_en_metric(r["source"]) for r in results]
            valid_outputs = hypotheses
            max_refs = max(len(sent_refs) for sent_refs in normalized_refs_multi) if normalized_refs_multi else 1
            valid_refs = [
                [sent_refs[i] if i < len(sent_refs) else sent_refs[0] for sent_refs in normalized_refs_multi]
                for i in range(max_refs)
            ]

            errant_score = errant_metric.score_corpus_verbose(
                sources=valid_inputs,
                hypotheses=valid_outputs,
                references=valid_refs,
            )
            eval_summary["metric_source"] = "gec_metrics_errant"
            eval_summary["errant_status"] = "ok"
            eval_summary["precision"] = float(errant_score.precision)
            eval_summary["recall"] = float(errant_score.recall)
            eval_summary["f0_5"] = float(errant_score.f)
        except Exception as exc:
            eval_summary["errant_status"] = "failed"
            eval_summary["errant_error"] = str(exc)

    if eval_summary["precision"] is None:
        try:
            precision, recall, f05 = calculate_m2score(
                hypotheses=hypotheses,
                m2_file_path=str(m2_path),
                m2scorer_path=m2scorer_path,
                beta=0.5,
                offset=start_index,
            )
            eval_summary["metric_source"] = "m2scorer_or_errant_fallback"
            eval_summary["m2_status"] = "ok"
            eval_summary["precision"] = precision
            eval_summary["recall"] = recall
            eval_summary["f0_5"] = f05
        except Exception as exc:
            eval_summary["m2_status"] = "failed"
            eval_summary["m2_error"] = str(exc)

    predictions_path = run_dir / f"{prefix}predictions.eval.txt"
    write_eval_ready_output(results, predictions_path)

    evaluation_path = run_dir / f"{prefix}evaluation.txt"
    with open(evaluation_path, "w", encoding="utf-8") as f:
        f.write("=== EN Evaluation ===\n")
        f.write(f"Exact matches: {exact_matches}/{total} ({exact_acc:.1f}%)\n")
        f.write(f"Metric source: {eval_summary['metric_source']}\n")
        f.write(f"ERRANT status: {eval_summary['errant_status']}\n")
        if eval_summary["errant_error"]:
            f.write(f"ERRANT error: {eval_summary['errant_error']}\n")
        f.write(f"M2 status: {eval_summary['m2_status']}\n")
        if eval_summary["m2_error"]:
            f.write(f"M2 error: {eval_summary['m2_error']}\n")
        if eval_summary["precision"] is not None:
            f.write(f"Precision: {eval_summary['precision']:.4f}\n")
            f.write(f"Recall: {eval_summary['recall']:.4f}\n")
            f.write(f"F0.5: {eval_summary['f0_5']:.4f}\n")

    metrics_path = run_dir / f"{prefix}metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"exact_matches: {exact_matches}\n")
        f.write(f"total_samples: {total}\n")
        f.write(f"exact_match_percent: {exact_acc:.1f}\n")
        if eval_summary["precision"] is not None:
            f.write(f"precision: {eval_summary['precision']:.4f}\n")
            f.write(f"recall: {eval_summary['recall']:.4f}\n")
            f.write(f"f0_5: {eval_summary['f0_5']:.4f}\n")

    return eval_summary


def _normalize_for_en_metric(text: str) -> str:
    # English normalization for metric pipeline.
    try:
        text = tokenize(detokenize(str(text)))
    except Exception:
        text = str(text)
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def main(config_path: str = "configs/bea_dev.yaml") -> None:
    config = load_config(config_path)
    config_file_path = (Path(__file__).parent / config_path).resolve()

    setup_agents()

    mode = config.get("mode", "full")
    split = config.get("en_split", "en_bea_dev")
    src_path = Path(config.get("en_src_path", "data/en_bea/dev/bea-dev.src"))
    ref_paths_config = config.get("en_ref_paths")
    ref_path = Path(config.get("en_ref_path", "data/en_bea/dev/bea-dev.ref0"))
    m2_path = Path(config.get("en_m2_path", "data/en_bea/dev/bea-dev.m2"))
    m2scorer_path = config.get("en_m2scorer_path")
    model = config.get("model", "gpt-4.1-mini")
    temperature = config.get("temperature")
    top_p = config.get("top_p")
    reasoning_effort = config.get("reasoning_effort")
    thinking_config = config.get("thinking_config")
    llm_request_kwargs = config.get("llm_request_kwargs", {}) or {}
    output_format = config.get("output_format", "structured_json")
    retry_malformed_batches = bool(config.get("retry_malformed_batches", True))
    agent_name = config.get("agent", "gec_single_prompt")
    prompt_name = config.get("prompt_name", "base_zero_shot")
    llm_timeout_seconds = float(config.get("llm_timeout_seconds", 120.0))
    num_threads = int(config.get("num_threads", 10))
    verbosity = bool(config.get("verbosity", False))
    show_progress = bool(config.get("show_progress", True))
    log_input_output = bool(config.get("log_input_output", False))
    start_index = int(config.get("start_index", 0))
    limit = config.get("limit")
    batch_size = int(config.get("batch_size", 1))
    llm_router_config = config.get("llm_router")
    position_experiment = config.get("position_experiment", {}) or {}
    position_experiment_enabled = bool(position_experiment.get("enabled", False))
    batch_shuffle_experiment = config.get("batch_shuffle_experiment", {}) or {}
    batch_shuffle_enabled = bool(batch_shuffle_experiment.get("enabled", False))
    few_shot_config = config.get("few_shot", {}) or {}
    few_shot_enabled = bool(few_shot_config.get("enabled", False))
    few_shot_n = int(few_shot_config.get("n", 0))
    few_shot_seed = int(few_shot_config.get("seed", 13))
    few_shot_src_path = Path(few_shot_config.get("src_path", "data/en_bea/train/bea-train.src"))
    few_shot_ref_path = Path(few_shot_config.get("ref_path", "data/en_bea/train/bea-train.ref0"))
    few_shot_detokenize = bool(few_shot_config.get("detokenize", True))
    few_shot_delivery = str(few_shot_config.get("delivery", "system"))
    few_shot_demo_chunk = int(few_shot_config.get("demo_chunk", 0))
    few_shot_prompt = ""
    few_shot_examples: list[dict] = []
    prompt_name_for_run = prompt_name

    mode_limits = {"fast": 10, "mini-test": 99, "full": None}
    if mode not in mode_limits:
        raise ValueError(f"Unknown mode: {mode}")

    if few_shot_enabled:
        if few_shot_n <= 0:
            raise ValueError("few_shot.enabled=true requires few_shot.n > 0.")
        few_shot_prompt, few_shot_examples = build_few_shot_prompt(
            src_path=few_shot_src_path,
            ref_path=few_shot_ref_path,
            n=few_shot_n,
            seed=few_shot_seed,
            detokenize_examples=few_shot_detokenize,
        )
        prompt_name_for_run = f"{prompt_name}_{few_shot_n}_shots_seed{few_shot_seed}"

    few_shot_metadata = build_few_shot_metadata(
        enabled=few_shot_enabled,
        n=few_shot_n,
        seed=few_shot_seed,
        detokenize_examples=few_shot_detokenize,
        delivery=few_shot_delivery,
        demo_chunk=few_shot_demo_chunk,
        src_path=few_shot_src_path,
        ref_path=few_shot_ref_path,
        examples=few_shot_examples,
    )

    print(f"Running EN GEC on split: {split}")
    print(f"Mode: {mode}")
    print(f"Model: {model}")
    print(f"Prompt: {prompt_name}")
    print(f"Output format: {output_format}")
    if few_shot_enabled:
        print(f"Few-shot examples: n={few_shot_n}, seed={few_shot_seed}")
        print(f"Few-shot detokenize: {few_shot_detokenize}")
        print(f"Few-shot source: {few_shot_src_path}")
        print(f"Few-shot reference: {few_shot_ref_path}")
        print(f"Few-shot delivery: {few_shot_delivery}")
    print(f"Position experiment: {position_experiment_enabled}")
    print(f"Batch shuffle experiment: {batch_shuffle_enabled}")
    print(f"Source: {src_path}")
    if ref_paths_config:
        ref_paths = [Path(p) for p in ref_paths_config]
        print(f"References: {', '.join(str(p) for p in ref_paths)}")
    else:
        ref_paths = [ref_path]
        print(f"Reference: {ref_path}")
    print(f"M2: {m2_path}")
    if m2scorer_path:
        print(f"M2 scorer: {m2scorer_path}")

    if len(ref_paths) == 1:
        sources, refs_single = load_parallel_texts(src_path, ref_paths[0])
        refs_nested = [[ref] for ref in refs_single]
    else:
        sources, refs_nested = load_parallel_texts_multi(src_path, ref_paths)
        refs_single = [refs[0] for refs in refs_nested]

    effective_limit = mode_limits[mode]
    if limit is not None:
        effective_limit = int(limit)

    end_index = start_index + effective_limit if effective_limit is not None else None
    sources = sources[start_index:end_index]
    refs_nested = refs_nested[start_index:end_index]
    refs_single = refs_single[start_index:end_index]

    print(f"Loaded {len(sources)} samples")

    if llm_router_config:
        llm_router = create_router(llm_router_config)
    else:
        from src.llm.litellm import router as llm_router

    model_groups = sorted({d["model_name"] for d in llm_router.model_list if "model_name" in d})
    if model not in model_groups:
        raise ValueError(
            f"Config 'model' must match one of the router models: "
            f"{', '.join(model_groups)}. Got: '{model}'."
        )

    agent_kwargs = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "request_timeout": llm_timeout_seconds,
        "prompt_name": prompt_name,
        "llm_router": llm_router,
        "reasoning_effort": reasoning_effort,
        "thinking_config": thinking_config,
        "few_shot_prompt": few_shot_prompt,
        "few_shot_examples": few_shot_examples,
        "few_shot_delivery": few_shot_delivery,
        "few_shot_demo_chunk": few_shot_demo_chunk,
        "output_format": output_format,
        "retry_malformed_batches": retry_malformed_batches,
        "request_kwargs": llm_request_kwargs,
    }
    gec_agent = get_agent(agent_name, **agent_kwargs)
    # Validate prompt selection before submitting many parallel requests.
    gec_agent.get_system_prompt()

    experiment_descriptor = validate_batch_experiment(
        position_config=position_experiment,
        shuffle_config=batch_shuffle_experiment,
        batch_size=batch_size,
        total_sentences=len(sources),
        agent=gec_agent,
    )

    if experiment_descriptor is not None:
        results = []
    elif batch_size > 1 and hasattr(gec_agent, "execute_sentence_batch"):
        indexed = [(i, src, ref) for i, (src, ref) in enumerate(zip(sources, refs_single), 1)]
        chunks = list(_iter_chunks(indexed, batch_size))
        items = [
            (i, chunk, gec_agent, log_input_output)
            for i, chunk in enumerate(chunks, 1)
        ]
        batch_results = process_in_parallel(
            items,
            process_batch_chunk,
            max_workers=num_threads,
            show_progress=show_progress,
        )
        results = []
        for br in batch_results:
            results.extend(br["rows"])
    else:
        items = [
            (i, source, reference, gec_agent, log_input_output)
            for i, (source, reference) in enumerate(zip(sources, refs_single), 1)
        ]
        results = process_in_parallel(
            items,
            process_sample,
            max_workers=num_threads,
            show_progress=show_progress,
        )

    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    run_name = build_run_name(
        mode=mode,
        split=split,
        model=model,
        prompt_name=prompt_name_for_run,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        batch_descriptor=experiment_descriptor or batch_size,
    )
    run_dir = outputs_dir / run_name
    run_dir.mkdir(exist_ok=True)
    print(f"Run directory: {run_dir}")

    print(f"Evaluation M2: {m2_path}")

    run_config_path = run_dir / Path(config_path).name
    shutil.copy2(config_file_path, run_config_path)
    if few_shot_enabled:
        with open(run_dir / "few_shot_examples.json", "w", encoding="utf-8") as f:
            json.dump(
                few_shot_metadata,
                f,
                ensure_ascii=False,
                indent=2,
            )
    experiment_context = BatchExperimentContext(
        run_name=run_name,
        mode=mode,
        split=split,
        model=model,
        prompt_name=prompt_name,
        prompt_run_name=prompt_name_for_run,
        few_shot=few_shot_metadata,
        batch_size=batch_size,
        sources=sources,
        references=refs_single,
        agent=gec_agent,
        log_input_output=log_input_output,
        num_threads=num_threads,
        show_progress=show_progress,
        run_dir=run_dir,
        m2_path=m2_path,
        m2scorer_path=m2scorer_path,
        start_index=start_index,
        evaluate_predictions=evaluate_en_predictions,
    )
    if run_batch_experiment(
        position_config=position_experiment,
        shuffle_config=batch_shuffle_experiment,
        context=experiment_context,
    ):
        return

    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "mode": mode,
                "split": split,
                "model": model,
                "prompt": prompt_name,
                "prompt_run": prompt_name_for_run,
                "few_shot": few_shot_metadata,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(run_dir / "predictions.txt", "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"{r['corrected']}\n")

    eval_summary = evaluate_en_predictions(
        results=results,
        references_multi=refs_nested,
        m2_path=m2_path,
        m2scorer_path=m2scorer_path,
        start_index=start_index,
        run_dir=run_dir,
    )

    print_exact_match_results(
        eval_summary["exact_matches"],
        eval_summary["total_samples"],
        eval_summary["exact_match_percent"],
    )
    if eval_summary["precision"] is not None:
        print_m2score_results(eval_summary["precision"], eval_summary["recall"], eval_summary["f0_5"], beta=0.5)

    if verbosity:
        print("Saved:")
        print(f"  {run_dir / 'results.json'}")
        print(f"  {run_dir / 'predictions.txt'}")
        print(f"  {run_dir / 'predictions.eval.txt'}")
        print(f"  {run_dir / 'evaluation.txt'}")
        print(f"  {run_dir / 'metrics.txt'}")


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args.config)
    finally:
        cleanup_litellm_async_clients()
