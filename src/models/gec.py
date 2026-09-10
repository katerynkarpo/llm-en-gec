from pydantic import BaseModel, Field


class GECResponse(BaseModel):
    """Response model for Grammatical Error Correction."""
    corrected_sentence: str = Field(..., description="Corrected version of the input sentence")


class BatchSentenceGECItem(BaseModel):
    """Structured correction for one sentence inside a batch."""

    SENTENCE_ID: int = Field(..., description="1-based index within the batch input.")
    CORRECTED_SENTENCE: str = Field(..., description="Corrected version of the sentence.")


class BatchGECResponse(BaseModel):
    """Structured response for multi-sentence batch correction."""

    SENTENCES: list[BatchSentenceGECItem] = Field(
        ...,
        description="Corrections for each input sentence.",
    )
