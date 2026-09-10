from typing import Type
from pydantic import BaseModel
from .base import BaseAgent
from ..models.gec import (
    BatchGECResponse,
    GECResponse,
)
from .prompts.base import get_gec_prompt
from .response_parsers import parse_numbered_batch_output


CORRECTION_PROMPT = """{}

### Text to correct:
{}

### Corrected text:
{}"""
SINGLE_CONTINUATION = """### Text to correct:
{}

### Corrected text:
"""
BATCH_CORRECTION_PROMPT = """{}

Correct each numbered text independently. Return only the corrected texts in the same numbered format, one per line.

### Texts to correct:
{}

### Corrected texts:
"""
BATCH_CONTINUATION = """### Texts to correct:
{}

### Corrected texts:
"""


class SinglePromptGECAgent(BaseAgent):
    """Agent for Grammatical Error Correction using zero-shot instruction."""

    def get_system_prompt(self) -> str:
        """Return the configured instruction prompt for GEC."""
        prompt = get_gec_prompt(self.prompt_name or "ua_single")
        if self.few_shot_prompt and self.few_shot_delivery == "system":
            return f"{prompt.rstrip()}\n\n{self.few_shot_prompt.strip()}"
        return prompt

    def get_response_model(self) -> Type[BaseModel]:
        """Return the GEC response model."""
        return GECResponse

    def execute(self, input_text: str, **kwargs) -> BaseModel:
        """Correct one sentence using structured JSON or plain-text output."""
        if self.output_format != "numbered_text":
            return super().execute(input_text, **kwargs)

        corrected = self._execute_plain_text_single(input_text, **kwargs)
        return self.get_response_model().model_validate({"corrected_sentence": corrected})

    def execute_sentence_batch(self, input_texts: list[str], **kwargs) -> list[str]:
        """Correct multiple sentences in one structured LLM call."""
        if not input_texts:
            return []

        if self.output_format == "numbered_text":
            return self._execute_numbered_text_batch(input_texts, **kwargs)

        user_payload_lines = [
            "Correct each sentence separately. Return a result for each SENTENCE_ID.",
        ]
        for i, text in enumerate(input_texts, start=1):
            user_payload_lines.append(f"Sentence {i}: {text}")
        user_payload = "\n".join(user_payload_lines)

        response = self.execute_with_model(
            input_text=user_payload,
            response_model=BatchGECResponse,
            **kwargs,
        )

        by_id = {item.SENTENCE_ID: item.CORRECTED_SENTENCE for item in response.SENTENCES}
        corrected: list[str] = []
        for i, src in enumerate(input_texts, start=1):
            value = by_id.get(i)
            corrected.append(value if value is not None else src)
        return corrected

    def _execute_plain_text_single(self, input_text: str, **kwargs) -> str:
        messages = self._build_plain_text_single_messages(input_text)
        kwargs.setdefault("stop", ["\n"])
        return self.complete_text(messages, **kwargs).strip()

    def _execute_numbered_text_batch(self, input_texts: list[str], **kwargs) -> list[str]:
        messages = self._build_numbered_batch_messages(input_texts)
        content = self.complete_text(messages, **kwargs)
        predictions, error = parse_numbered_batch_output(content, len(input_texts))
        if predictions is not None:
            return predictions
        if not self.retry_malformed_batches:
            raise ValueError(f"Malformed numbered batch response: {error}")
        return [self._execute_plain_text_single(text, **kwargs) for text in input_texts]

    def _build_plain_text_single_messages(self, input_text: str) -> list[dict[str, str]]:
        prompt = self.get_system_prompt()
        if self.few_shot_delivery != "chat" or not self.few_shot_examples:
            return [
                {
                    "role": "user",
                    "content": CORRECTION_PROMPT.format(prompt, input_text, ""),
                }
            ]

        messages: list[dict[str, str]] = []
        for index, example in enumerate(self.few_shot_examples):
            source = str(example["source"])
            reference = str(example["reference"])
            user_content = (
                CORRECTION_PROMPT.format(prompt, source, "")
                if index == 0
                else SINGLE_CONTINUATION.format(source)
            )
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": reference})
        messages.append(
            {"role": "user", "content": SINGLE_CONTINUATION.format(input_text)}
        )
        return messages

    def _build_numbered_batch_messages(
        self,
        input_texts: list[str],
    ) -> list[dict[str, str]]:
        prompt = self.get_system_prompt()
        numbered_inputs = self._number_lines(input_texts)
        if self.few_shot_delivery != "chat" or not self.few_shot_examples:
            return [
                {
                    "role": "user",
                    "content": BATCH_CORRECTION_PROMPT.format(prompt, numbered_inputs),
                }
            ]

        chunk_size = self.few_shot_demo_chunk or len(self.few_shot_examples)
        demo_chunks = [
            self.few_shot_examples[index:index + chunk_size]
            for index in range(0, len(self.few_shot_examples), chunk_size)
        ]
        messages: list[dict[str, str]] = []
        for chunk_index, chunk in enumerate(demo_chunks):
            demo_sources = self._number_lines([str(example["source"]) for example in chunk])
            demo_refs = self._number_lines([str(example["reference"]) for example in chunk])
            user_content = (
                BATCH_CORRECTION_PROMPT.format(prompt, demo_sources)
                if chunk_index == 0
                else BATCH_CONTINUATION.format(demo_sources)
            )
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": demo_refs})
        messages.append(
            {"role": "user", "content": BATCH_CONTINUATION.format(numbered_inputs)}
        )
        return messages

    @staticmethod
    def _number_lines(texts: list[str]) -> str:
        return "\n".join(
            f"Sentence {index}: {text}" for index, text in enumerate(texts, start=1)
        )
