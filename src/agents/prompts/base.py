from .prompt_01_vanilla_zero_shot import (EN_BASE_GEC_PROMPT)
from .prompt_02_minimal_edits_zero_shot import (MINIMAL_EDITS_PROMPT)
from .prompt_03_minimal_edits_taxonomy_instructions_zero_shot import (TAXONOMY_BASED_PROMPT)
from .prompt_04_qwen3_8b_optimized import QWEN3_8B_OPTIMIZED_PROMPT
from .prompt_05_gpt_family_optimized import LLM_OPTIMISED_GPT_FAMILY_PROMPT
from .prompt_06_claude_family_optimized import LLM_OPTIMISED_OPUS_FAMILY_PROMPT
from .prompt_07_gemini_family_optimized import LLM_OPTIMISED_GEMINI_FAMILY_PROMPT


GEC_PROMPTS = {
    "base_zero_shot": EN_BASE_GEC_PROMPT,
    "minimal_edits_zero_shot": MINIMAL_EDITS_PROMPT,
    "taxonomy_based_zero_shot": TAXONOMY_BASED_PROMPT,
    "taxonomy_optimised_gemini_family": LLM_OPTIMISED_GEMINI_FAMILY_PROMPT,
    "taxonomy_optimised_opus_family": LLM_OPTIMISED_OPUS_FAMILY_PROMPT,
    "taxonomy_optimised_gpt_family": LLM_OPTIMISED_GPT_FAMILY_PROMPT,
    "qwen3_8b_optimized": QWEN3_8B_OPTIMIZED_PROMPT,
}


def get_gec_prompt(name: str) -> str:
    """Return a named GEC system prompt."""
    if name not in GEC_PROMPTS:
        available = ", ".join(sorted(GEC_PROMPTS.keys()))
        raise ValueError(f"Unknown prompt '{name}'. Available prompts: {available}")
    return GEC_PROMPTS[name]
