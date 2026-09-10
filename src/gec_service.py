"""Reusable single-text GEC service for interactive applications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import SystemRandom
from typing import Any

import yaml
from dotenv import load_dotenv

from src.agents import SinglePromptGECAgent
from src.agents.prompts.base import GEC_PROMPTS
from src.few_shot import (
    detokenize_prompt_example,
    format_few_shot_prompt,
    load_parallel_examples,
    select_few_shot_examples,
)
from src.llm import create_router
from src.utils import sentence_spans


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROMPT_LABELS = {
    "base_zero_shot": "A1 · Vanilla zero-shot",
    "minimal_edits_zero_shot": "A2/A3 · Minimal edits",
    "taxonomy_based_zero_shot": "A4/A5 · Taxonomy-based",
    "qwen3_8b_optimized": "A6.1 · Qwen3-8B optimized",
    "taxonomy_optimised_gpt_family": "A6.2 · GPT optimized",
    "taxonomy_optimised_opus_family": "A6.3 · Claude optimized",
    "taxonomy_optimised_gemini_family": "A6.4 · Gemini optimized",
}


@dataclass(frozen=True)
class PromptRecipe:
    """Paper prompt plus its reproducible default few-shot selection."""

    name: str
    label: str
    prompt_name: str
    paper_n: int = 0
    paper_seed: int | None = None
    optimized_for: str | None = None


PROMPT_RECIPES = {
    recipe.name: recipe
    for recipe in [
        PromptRecipe("a1", "A1 · Vanilla zero-shot", "base_zero_shot"),
        PromptRecipe("a2", "A2 · Minimal edits · zero-shot", "minimal_edits_zero_shot"),
        PromptRecipe(
            "a3",
            "A3 · Minimal edits · 8-shot",
            "minimal_edits_zero_shot",
            paper_n=8,
            paper_seed=123,
        ),
        PromptRecipe("a4", "A4 · Taxonomy · zero-shot", "taxonomy_based_zero_shot"),
        PromptRecipe(
            "a5",
            "A5 · Taxonomy · 8-shot",
            "taxonomy_based_zero_shot",
            paper_n=8,
            paper_seed=123,
        ),
        PromptRecipe(
            "a6_1",
            "A6.1 · Qwen-optimized · 8-shot",
            "qwen3_8b_optimized",
            paper_n=8,
            paper_seed=123,
            optimized_for="Qwen",
        ),
        PromptRecipe(
            "a6_2",
            "A6.2 · GPT-optimized · 8-shot",
            "taxonomy_optimised_gpt_family",
            paper_n=8,
            paper_seed=712,
            optimized_for="GPT",
        ),
        PromptRecipe(
            "a6_3",
            "A6.3 · Claude-optimized · 8-shot",
            "taxonomy_optimised_opus_family",
            paper_n=8,
            paper_seed=123,
            optimized_for="Claude",
        ),
        PromptRecipe(
            "a6_4",
            "A6.4 · Gemini-optimized · zero-shot",
            "taxonomy_optimised_gemini_family",
            optimized_for="Gemini",
        ),
    ]
}


@dataclass(frozen=True)
class ModelProfile:
    """Safe model settings exposed to the browser."""

    name: str
    label: str
    family: str
    default_recipe: str
    output_format: str = "structured_json"
    request_timeout: float = 120.0
    request_kwargs: dict[str, Any] | None = None
    supports_reasoning_effort: bool = False
    temperature_min: float = 0.0
    temperature_max: float = 1.0
    temperature_default: float = 0.0
    temperature_requires_no_reasoning: bool = False

    @classmethod
    def from_dict(cls, name: str, values: dict[str, Any]) -> "ModelProfile":
        return cls(
            name=name,
            label=str(values.get("label", name)),
            family=str(values.get("family", "Other")),
            default_recipe=str(values.get("default_recipe", "a2")),
            output_format=str(values.get("output_format", "structured_json")),
            request_timeout=float(values.get("request_timeout", 120.0)),
            request_kwargs=dict(values.get("request_kwargs", {}) or {}),
            supports_reasoning_effort=bool(values.get("supports_reasoning_effort", False)),
            temperature_min=float(values.get("temperature_min", 0.0)),
            temperature_max=float(values.get("temperature_max", 1.0)),
            temperature_default=float(values.get("temperature_default", 0.0)),
            temperature_requires_no_reasoning=bool(
                values.get("temperature_requires_no_reasoning", False)
            ),
        )


class GECService:
    """Correct text with user-selectable, server-approved inference settings."""

    def __init__(
        self,
        *,
        router: Any,
        profiles: dict[str, ModelProfile],
        default_model: str,
        max_input_chars: int = 10_000,
        max_batch_size: int = 120,
        few_shot_src_path: Path | None = None,
        few_shot_ref_path: Path | None = None,
        max_few_shot_examples: int = 32,
    ) -> None:
        if not profiles:
            raise ValueError("The demo must define at least one model profile.")
        if default_model not in profiles:
            raise ValueError(f"Unknown default model profile: {default_model}")

        self.router = router
        self.profiles = profiles
        self.default_model = default_model
        self.max_input_chars = max_input_chars
        self.max_batch_size = max_batch_size
        self.few_shot_src_path = few_shot_src_path
        self.few_shot_ref_path = few_shot_ref_path
        self.max_few_shot_examples = max_few_shot_examples
        self._validate_profiles()

    @classmethod
    def from_config(cls, config_path: str | Path) -> "GECService":
        """Create a service from a demo YAML configuration."""
        load_dotenv(PROJECT_ROOT / ".env")
        path = Path(config_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Demo configuration not found: {path}")

        with path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}

        router_config = config.get("llm_router")
        if not isinstance(router_config, dict):
            raise ValueError("Demo config must contain an 'llm_router' mapping.")

        raw_profiles = config.get("model_profiles")
        if not isinstance(raw_profiles, dict) or not raw_profiles:
            raise ValueError("Demo config must contain non-empty 'model_profiles'.")
        profiles = {
            str(name): ModelProfile.from_dict(str(name), values or {})
            for name, values in raw_profiles.items()
        }

        demo_config = config.get("demo", {}) or {}
        few_shot_config = config.get("few_shot", {}) or {}
        default_model = str(demo_config.get("default_model", next(iter(profiles))))
        max_input_chars = int(demo_config.get("max_input_chars", 10_000))
        return cls(
            router=create_router(router_config),
            profiles=profiles,
            default_model=default_model,
            max_input_chars=max_input_chars,
            max_batch_size=int(demo_config.get("max_batch_size", 120)),
            few_shot_src_path=_project_path(few_shot_config.get("src_path")),
            few_shot_ref_path=_project_path(few_shot_config.get("ref_path")),
            max_few_shot_examples=int(few_shot_config.get("max_n", 32)),
        )

    def options(self) -> dict[str, Any]:
        """Return non-secret choices and defaults for the browser UI."""
        models = [
            {
                "name": profile.name,
                "label": profile.label,
                "family": profile.family,
                "default_recipe": profile.default_recipe,
                "supports_reasoning_effort": profile.supports_reasoning_effort,
                "temperature_min": profile.temperature_min,
                "temperature_max": profile.temperature_max,
                "temperature_default": profile.temperature_default,
                "temperature_requires_no_reasoning": (
                    profile.temperature_requires_no_reasoning
                ),
            }
            for profile in self.profiles.values()
        ]
        prompts = [
            {"name": name, "label": PROMPT_LABELS.get(name, name)}
            for name in GEC_PROMPTS
        ]
        recipes = [
            {
                "name": recipe.name,
                "label": recipe.label,
                "prompt_name": recipe.prompt_name,
                "paper_n": recipe.paper_n,
                "paper_seed": recipe.paper_seed,
                "optimized_for": recipe.optimized_for,
            }
            for recipe in PROMPT_RECIPES.values()
        ]
        return {
            "default_model": self.default_model,
            "max_input_chars": self.max_input_chars,
            "max_batch_size": self.max_batch_size,
            "models": models,
            "prompts": prompts,
            "recipes": recipes,
            "few_shot": {
                "sampling_available": self._few_shot_corpus_available(),
                "max_n": self.max_few_shot_examples,
                "default_n": 8,
                "default_seed": 123,
            },
        }

    def correct(
        self,
        text: str,
        *,
        model: str | None = None,
        prompt: str | None = None,
        recipe: str | None = None,
        batch_size: int = 1,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        few_shot_mode: str | None = None,
        few_shot_n: int = 8,
        few_shot_seed: int = 123,
        few_shot_examples: list[dict[str, str]] | None = None,
    ) -> str:
        """Correct one text using settings selected in the demo UI."""
        if not text.strip():
            raise ValueError("Text must not be empty.")
        if len(text) > self.max_input_chars:
            raise ValueError(
                f"Text is too long ({len(text)} characters); "
                f"the limit is {self.max_input_chars}."
            )
        if not 1 <= batch_size <= self.max_batch_size:
            raise ValueError(
                f"Batch size must be between 1 and {self.max_batch_size}."
            )

        spans = sentence_spans(text)
        sentence_count = len(spans)
        if sentence_count < batch_size:
            raise ValueError(
                f"The text contains {sentence_count} sentence(s), but batch size is "
                f"{batch_size}. Add more sentences or reduce batch size."
            )

        profile = self.profiles.get(model or self.default_model)
        if profile is None:
            raise ValueError(f"Unknown model profile: {model}")

        if reasoning_effort and profile.temperature_requires_no_reasoning:
            if temperature is not None:
                raise ValueError(
                    f"Temperature cannot be set for {profile.label} when reasoning "
                    "effort is enabled."
                )
            resolved_temperature = None
        else:
            resolved_temperature = (
                profile.temperature_default if temperature is None else temperature
            )
            if not profile.temperature_min <= resolved_temperature <= profile.temperature_max:
                raise ValueError(
                    f"Temperature for {profile.label} must be between "
                    f"{profile.temperature_min:g} and {profile.temperature_max:g}."
                )

        recipe_name = recipe or profile.default_recipe
        selected_recipe = PROMPT_RECIPES.get(recipe_name)
        if selected_recipe is None:
            raise ValueError(f"Unknown prompt recipe: {recipe_name}")

        prompt_name = prompt or selected_recipe.prompt_name
        if prompt_name not in GEC_PROMPTS:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        if reasoning_effort and not profile.supports_reasoning_effort:
            raise ValueError(f"Reasoning effort is not supported by {profile.label}.")

        mode = few_shot_mode or ("custom" if few_shot_examples else "paper")
        if mode == "paper":
            normalized_examples = self._sample_examples(
                n=selected_recipe.paper_n,
                seed=selected_recipe.paper_seed or 123,
            )
        elif mode == "seeded":
            normalized_examples = self._sample_examples(n=few_shot_n, seed=few_shot_seed)
        elif mode == "custom":
            examples = few_shot_examples or []
            if len(examples) > 8:
                raise ValueError("At most 8 custom few-shot examples are supported.")
            normalized_examples = self._validate_examples(examples)
        elif mode == "none":
            normalized_examples = []
        else:
            raise ValueError(
                "Few-shot mode must be paper, seeded, custom, or none."
            )
        uses_chat_examples = profile.output_format == "numbered_text"

        agent = SinglePromptGECAgent(
            model=profile.name,
            temperature=resolved_temperature,
            request_timeout=profile.request_timeout,
            prompt_name=prompt_name,
            llm_router=self.router,
            reasoning_effort=reasoning_effort,
            few_shot_prompt=(
                "" if uses_chat_examples else format_few_shot_prompt(normalized_examples)
            ),
            few_shot_examples=normalized_examples,
            few_shot_delivery="chat" if uses_chat_examples else "system",
            output_format=profile.output_format,
            request_kwargs=dict(profile.request_kwargs or {}),
        )
        agent.get_system_prompt()
        return self._correct_sentence_batches(
            text=text,
            spans=spans,
            batch_size=batch_size,
            agent=agent,
        )

    @staticmethod
    def count_sentences(text: str) -> int:
        """Count sentences exactly as the batching pipeline will segment them."""
        return len(sentence_spans(text))

    def random_example(self, *, sentence_count: int = 1) -> dict[str, Any]:
        """Sample erroneous single-sentence examples from the configured train corpus."""
        if not 1 <= sentence_count <= self.max_batch_size:
            raise ValueError(
                f"Example sentence count must be between 1 and {self.max_batch_size}."
            )
        if not self._few_shot_corpus_available():
            raise ValueError(
                "Example corpus is unavailable. Check the paths in configs/demo.yaml."
            )

        sources, references = load_parallel_examples(
            self.few_shot_src_path,
            self.few_shot_ref_path,
        )
        randomizer = SystemRandom()
        selected: list[tuple[int, str]] = []
        seen: set[int] = set()
        max_attempts = min(len(sources), max(1_000, sentence_count * 100))

        while len(selected) < sentence_count and len(seen) < max_attempts:
            index = randomizer.randrange(len(sources))
            if index in seen:
                continue
            seen.add(index)
            source = detokenize_prompt_example(sources[index])
            reference = detokenize_prompt_example(references[index])
            if source == reference or not source.endswith((".", "!", "?")):
                continue
            if self.count_sentences(source) != 1:
                continue
            projected_length = sum(len(item) for _, item in selected)
            projected_length += len(selected) + len(source)
            if projected_length > self.max_input_chars:
                continue
            selected.append((index + 1, source))

        if len(selected) < sentence_count:
            raise RuntimeError(
                f"Could not find {sentence_count} suitable train examples."
            )
        text = " ".join(source for _, source in selected)
        return {
            "text": text,
            "sentence_count": self.count_sentences(text),
            "train_indices": [index for index, _ in selected],
        }

    @staticmethod
    def _correct_sentence_batches(
        *,
        text: str,
        spans: list[tuple[int, int]],
        batch_size: int,
        agent: SinglePromptGECAgent,
    ) -> str:
        sentences = [text[start:end] for start, end in spans]
        corrected: list[str] = []
        if batch_size == 1:
            corrected = [
                agent.execute(sentence).corrected_sentence.strip()
                for sentence in sentences
            ]
        else:
            for start in range(0, len(sentences), batch_size):
                corrected.extend(
                    item.strip()
                    for item in agent.execute_sentence_batch(
                        sentences[start:start + batch_size]
                    )
                )

        parts: list[str] = []
        cursor = 0
        for (start, end), replacement in zip(spans, corrected):
            parts.append(text[cursor:start])
            parts.append(replacement)
            cursor = end
        parts.append(text[cursor:])
        return "".join(parts)

    def _sample_examples(self, *, n: int, seed: int) -> list[dict]:
        if n == 0:
            return []
        if n < 0 or n > self.max_few_shot_examples:
            raise ValueError(
                f"Few-shot n must be between 0 and {self.max_few_shot_examples}."
            )
        if not self._few_shot_corpus_available():
            raise ValueError(
                "Few-shot corpus is unavailable. Check the paths in configs/demo.yaml."
            )
        return select_few_shot_examples(
            src_path=self.few_shot_src_path,
            ref_path=self.few_shot_ref_path,
            n=n,
            seed=seed,
            detokenize_examples=True,
        )

    def _few_shot_corpus_available(self) -> bool:
        return bool(
            self.few_shot_src_path
            and self.few_shot_ref_path
            and self.few_shot_src_path.exists()
            and self.few_shot_ref_path.exists()
        )

    def _validate_profiles(self) -> None:
        router_models = {
            item.get("model_name")
            for item in getattr(self.router, "model_list", [])
            if isinstance(item, dict)
        }
        missing = set(self.profiles) - router_models
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Model profiles missing from llm_router.model_list: {names}")
        for profile in self.profiles.values():
            if profile.default_recipe not in PROMPT_RECIPES:
                raise ValueError(
                    f"Unknown default recipe '{profile.default_recipe}' "
                    f"for model profile '{profile.name}'."
                )
            if profile.output_format not in {"structured_json", "numbered_text"}:
                raise ValueError(
                    f"Unsupported output format for '{profile.name}': "
                    f"{profile.output_format}"
                )
            if profile.temperature_min > profile.temperature_max:
                raise ValueError(
                    f"Invalid temperature range for '{profile.name}': minimum "
                    "exceeds maximum."
                )
            if not (
                profile.temperature_min
                <= profile.temperature_default
                <= profile.temperature_max
            ):
                raise ValueError(
                    f"Temperature default for '{profile.name}' must be inside its "
                    "configured range."
                )
            if (
                profile.temperature_requires_no_reasoning
                and not profile.supports_reasoning_effort
            ):
                raise ValueError(
                    f"Profile '{profile.name}' requires reasoning support before "
                    "temperature can conflict with it."
                )

    @staticmethod
    def _validate_examples(examples: list[dict[str, str]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for index, example in enumerate(examples, start=1):
            source = str(example.get("source", "")).strip()
            reference = str(example.get("reference", "")).strip()
            if not source or not reference:
                raise ValueError(
                    f"Few-shot example {index} needs both source and correction."
                )
            if len(source) > 2_000 or len(reference) > 2_000:
                raise ValueError(f"Few-shot example {index} is too long.")
            normalized.append({"source": source, "reference": reference})
        return normalized


def _project_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path
