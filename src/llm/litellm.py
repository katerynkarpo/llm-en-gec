"""LiteLLM model registry for the six API models evaluated in the paper."""

from __future__ import annotations

import os

import litellm
from litellm import Router


litellm.drop_params = True


def _key(name: str) -> str:
    return os.environ.get(name, "")


model_list = [
    {
        "model_name": "gpt-4.1-mini",
        "litellm_params": {
            "model": "gpt-4.1-mini-2025-04-14",
            "api_key": _key("OPENAI_API_KEY"),
        },
    },
    {
        "model_name": "gpt-5.4",
        "litellm_params": {
            "model": "gpt-5.4-2026-03-05",
            "api_key": _key("OPENAI_API_KEY"),
        },
    },
    {
        "model_name": "claude-sonnet-4.6",
        "litellm_params": {
            "model": "claude-sonnet-4-6",
            "api_key": _key("ANTHROPIC_API_KEY"),
        },
    },
    {
        "model_name": "claude-opus-4.6",
        "litellm_params": {
            "model": "claude-opus-4-6",
            "api_key": _key("ANTHROPIC_API_KEY"),
        },
    },
    {
        "model_name": "gemini-3-flash",
        "litellm_params": {
            "model": "gemini/gemini-3-flash-preview",
            "api_key": _key("GEMINI_API_KEY"),
        },
    },
    {
        "model_name": "gemini-3.1-pro",
        "litellm_params": {
            "model": "gemini/gemini-3.1-pro-preview",
            "api_key": _key("GEMINI_API_KEY"),
        },
    },
]


# Reproducibility requires a failed deployment to fail visibly. In particular,
# do not fall back from one model family or snapshot to another.
router = Router(
    model_list=model_list,
    fallbacks=[],
    enable_pre_call_checks=True,
    cache_responses=True,
)

