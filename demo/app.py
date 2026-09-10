"""FastAPI application for the interactive GEC demo."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from src.gec_service import GECService, PROMPT_RECIPES
from src.utils import build_text_diff


DEFAULT_CONFIG = "configs/demo.yaml"
INDEX_PATH = Path(__file__).parent / "static" / "index.html"
DOCS_PATH = Path(__file__).parent / "static" / "docs.html"


class FewShotExample(BaseModel):
    source: str = Field(max_length=2_000)
    reference: str = Field(max_length=2_000)


class CorrectionRequest(BaseModel):
    text: str
    model: str | None = None
    recipe: str | None = None
    prompt: str | None = None
    batch_size: int = Field(default=1, ge=1, le=120)
    temperature: float | None = Field(default=None, ge=0.0)
    reasoning_effort: str | None = None
    few_shot_mode: Literal["paper", "seeded", "custom", "none"] = "paper"
    few_shot_n: int = Field(default=8, ge=1, le=32)
    few_shot_seed: int = 123
    few_shot_examples: list[FewShotExample] = Field(default_factory=list, max_length=8)

    @field_validator("text")
    @classmethod
    def text_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not be empty.")
        return value

    @field_validator("reasoning_effort")
    @classmethod
    def validate_reasoning_effort(cls, value: str | None) -> str | None:
        if value not in {None, "low", "medium", "high"}:
            raise ValueError("Reasoning effort must be low, medium, high, or unset.")
        return value


class CorrectionResponse(BaseModel):
    original_text: str
    corrected_text: str
    changed: bool
    model: str
    recipe: str
    prompt: str
    sentence_count: int
    batch_size: int
    batch_count: int
    few_shot_mode: str
    few_shot_n: int
    few_shot_seed: int | None
    latency_ms: int
    diff: list[dict[str, str]]


class ExampleResponse(BaseModel):
    text: str
    sentence_count: int
    train_indices: list[int]


def create_app(service: GECService | None = None) -> FastAPI:
    """Build the app; an injected service keeps endpoint tests API-free."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            config_path = os.getenv("GEC_DEMO_CONFIG", DEFAULT_CONFIG)
            app.state.gec_service = GECService.from_config(config_path)
        else:
            app.state.gec_service = service
        yield

    app = FastAPI(
        title="LLM English GEC Demo",
        description="Minimal-edit English grammatical error correction.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))

    @app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
    def docs() -> HTMLResponse:
        """Serve dependency-free API documentation that also works offline."""
        return HTMLResponse(DOCS_PATH.read_text(encoding="utf-8"))

    @app.get("/api/health", tags=["service"])
    def health(request: Request) -> dict[str, str]:
        current_service = _service(request)
        return {"status": "ok", "default_model": current_service.default_model}

    @app.get("/api/options", tags=["service"])
    def options(request: Request) -> dict[str, Any]:
        return _service(request).options()

    @app.get("/api/example", response_model=ExampleResponse, tags=["service"])
    def example(
        request: Request,
        count: int = Query(default=1, ge=1, le=120),
    ) -> ExampleResponse:
        try:
            return ExampleResponse(**_service(request).random_example(sentence_count=count))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not sample a train example: {exc}",
            ) from exc

    @app.post("/api/correct", response_model=CorrectionResponse, tags=["correction"])
    def correct(payload: CorrectionRequest, request: Request) -> CorrectionResponse:
        current_service = _service(request)
        model = payload.model or current_service.default_model
        profile = current_service.profiles.get(model)
        recipe = payload.recipe or (profile.default_recipe if profile else "")
        recipe_config = PROMPT_RECIPES.get(recipe)
        prompt = payload.prompt or (recipe_config.prompt_name if recipe_config else "")
        sentence_count = current_service.count_sentences(payload.text)
        started = perf_counter()
        try:
            corrected = current_service.correct(
                payload.text,
                model=model,
                prompt=prompt,
                recipe=recipe,
                batch_size=payload.batch_size,
                temperature=payload.temperature,
                reasoning_effort=payload.reasoning_effort,
                few_shot_mode=payload.few_shot_mode,
                few_shot_n=payload.few_shot_n,
                few_shot_seed=payload.few_shot_seed,
                few_shot_examples=[
                    example.model_dump() for example in payload.few_shot_examples
                ],
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"The model request failed: {exc}",
            ) from exc

        return CorrectionResponse(
            original_text=payload.text,
            corrected_text=corrected,
            changed=payload.text != corrected,
            model=model,
            recipe=recipe,
            prompt=prompt,
            sentence_count=sentence_count,
            batch_size=payload.batch_size,
            batch_count=(sentence_count + payload.batch_size - 1) // payload.batch_size,
            few_shot_mode=payload.few_shot_mode,
            few_shot_n=_used_few_shot_n(payload, recipe_config),
            few_shot_seed=_used_few_shot_seed(payload, recipe_config),
            latency_ms=round((perf_counter() - started) * 1_000),
            diff=build_text_diff(payload.text, corrected),
        )

    return app


def _service(request: Request) -> GECService:
    return request.app.state.gec_service


def _used_few_shot_n(
    payload: CorrectionRequest,
    recipe: Any,
) -> int:
    if payload.few_shot_mode == "paper":
        return recipe.paper_n if recipe else 0
    if payload.few_shot_mode == "seeded":
        return payload.few_shot_n
    if payload.few_shot_mode == "custom":
        return len(payload.few_shot_examples)
    return 0


def _used_few_shot_seed(
    payload: CorrectionRequest,
    recipe: Any,
) -> int | None:
    if payload.few_shot_mode == "paper":
        return recipe.paper_seed if recipe and recipe.paper_n else None
    if payload.few_shot_mode == "seeded":
        return payload.few_shot_seed
    return None


app = create_app()
