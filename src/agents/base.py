from abc import ABC, abstractmethod
import json
import re
from typing import Any, Optional, Type
from pydantic import BaseModel


class _SafeFormatDict(dict):
    """Keep unknown format placeholders unchanged."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float | None = 0.0,
        top_p: Optional[float] = None,
        request_timeout: float = 120.0,
        prompt_name: Optional[str] = None,
        llm_router: Any = None,
        reasoning_effort: Optional[str] = None,
        thinking_config: Optional[dict] = None,
        few_shot_prompt: Optional[str] = None,
        few_shot_examples: Optional[list[dict[str, Any]]] = None,
        few_shot_delivery: str = "system",
        few_shot_demo_chunk: int = 0,
        output_format: str = "structured_json",
        retry_malformed_batches: bool = True,
        request_kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the base agent.

        Args:
            model: Model group/name resolved by LiteLLM Router
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            request_timeout: Per-request timeout (seconds) passed to LiteLLM
            prompt_name: Optional prompt selector for agent subclasses
            llm_router: Optional LiteLLM Router instance
            reasoning_effort: Optional reasoning effort level (low/medium/high) for supported models
        """
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.request_timeout = request_timeout
        self.prompt_name = prompt_name
        self.llm_router = llm_router
        self.reasoning_effort = reasoning_effort
        self.thinking_config = thinking_config
        self.few_shot_prompt = few_shot_prompt
        self.few_shot_examples = few_shot_examples or []
        self.few_shot_delivery = few_shot_delivery
        self.few_shot_demo_chunk = few_shot_demo_chunk
        self.output_format = output_format
        self.retry_malformed_batches = retry_malformed_batches
        self.request_kwargs = request_kwargs or {}

        if self.llm_router is None:
            raise ValueError("llm_router must be provided.")
        if self.reasoning_effort is not None and self.thinking_config is not None:
            raise ValueError(
                "Set either reasoning_effort or thinking_config, not both — "
                "LiteLLM rejects the combination."
            )
        if self.few_shot_delivery not in {"system", "chat"}:
            raise ValueError("few_shot_delivery must be either 'system' or 'chat'.")
        if self.output_format not in {"structured_json", "numbered_text"}:
            raise ValueError("output_format must be 'structured_json' or 'numbered_text'.")
        if self.few_shot_demo_chunk < 0:
            raise ValueError("few_shot_demo_chunk must be non-negative.")

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    @abstractmethod
    def get_response_model(self) -> Type[BaseModel]:
        """Return the Pydantic model for structured output."""
        pass

    def execute(self, input_text: str, **kwargs) -> BaseModel:
        """
        Execute the agent on input text.

        Args:
            input_text: Input text to process
            **kwargs: Additional parameters for the API call

        Returns:
            Structured response according to the agent's response model
        """
        return self.execute_with_model(input_text=input_text, response_model=self.get_response_model(), **kwargs)

    def execute_with_model(
        self,
        input_text: str,
        response_model: Type[BaseModel],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> BaseModel:
        """Execute using a custom structured response model."""
        raw_system_prompt = system_prompt if system_prompt is not None else self.get_system_prompt()
        system_prompt = self._render_system_prompt(raw_system_prompt, input_text)

        router_messages = [
            {"role": "system", "content": system_prompt},
        ]
        # Anthropic via LiteLLM requires at least one non-system message.
        router_user_content = input_text if str(input_text).strip() else "Please continue."
        router_messages.append({"role": "user", "content": router_user_content})

        router_kwargs = self._build_router_kwargs(
            router_messages,
            response_model=response_model,
            **kwargs,
        )

        completion = self._router_completion_with_fallback(router_kwargs)
        content = self._completion_content(completion)
        if isinstance(content, dict):
            return response_model.model_validate(
                self._coerce_payload_for_response_model(content, response_model)
            )
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        try:
            parsed_payload = self._extract_json_payload(content)
            return response_model.model_validate(
                self._coerce_payload_for_response_model(parsed_payload, response_model)
            )
        except Exception:
            # Fallback for single-field sentence responses when model returns malformed JSON.
            if set(response_model.model_fields.keys()) == {"corrected_sentence"}:
                repaired = self._recover_corrected_sentence(content)
                return response_model.model_validate({"corrected_sentence": repaired})
            raise

    def complete_text(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Run an unstructured chat completion and return its text content."""
        router_kwargs = self._build_router_kwargs(messages, **kwargs)
        completion = self.llm_router.completion(**router_kwargs)
        content = self._completion_content(completion)
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)
        return str(content or "").strip()

    def _build_router_kwargs(
        self,
        messages: list[dict[str, str]],
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Combine common decoding options with per-run provider parameters."""
        router_kwargs = dict(self.request_kwargs)
        router_kwargs.update({"model": self.model, "messages": messages})
        if self.temperature is not None:
            router_kwargs["temperature"] = self.temperature
        if response_model is not None:
            router_kwargs["response_format"] = self._build_router_response_format(response_model)
        if self.top_p is not None:
            router_kwargs["top_p"] = self.top_p
        if self.reasoning_effort is not None:
            router_kwargs["reasoning_effort"] = self.reasoning_effort
        if self.thinking_config is not None and self._supports_thinking_config():
            router_kwargs["thinkingConfig"] = self.thinking_config
        router_kwargs.update(kwargs)
        router_kwargs.setdefault("timeout", self.request_timeout)
        return router_kwargs

    @staticmethod
    def _completion_content(completion: Any) -> Any:
        """Normalize text-style content returned by OpenAI-compatible providers."""
        content = completion.choices[0].message.content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return content

    def _router_completion_with_fallback(self, router_kwargs: dict[str, Any]) -> Any:
        """Retry router completion with response_format downgrade fallback.

        Most unsupported params (top_p, temperature, reasoning_effort, etc.) are
        handled globally by ``litellm.drop_params = True``. This method only
        retries when the provider rejects the ``json_schema`` response format,
        downgrading it to ``json_object``.
        """
        try:
            return self.llm_router.completion(**router_kwargs)
        except Exception as exc:
            message = str(exc)
            if ("response_format" in message or "json_schema" in message) and "response_format" in router_kwargs:
                fallback_kwargs = dict(router_kwargs)
                fallback_kwargs["response_format"] = {"type": "json_object"}
                fallback_kwargs["messages"] = self._ensure_json_object_instruction(
                    fallback_kwargs.get("messages", [])
                )
                return self.llm_router.completion(**fallback_kwargs)
            raise

    def _supports_thinking_config(self) -> bool:
        return "gemini" in self.model.lower()

    @staticmethod
    def _build_router_response_format(response_model: Type[BaseModel]) -> dict[str, Any]:
        """Build provider-agnostic schema config for LiteLLM structured output."""
        schema = BaseAgent._enforce_closed_object_schema(response_model.model_json_schema())
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": schema,
                "strict": True,
            },
        }

    @staticmethod
    def _enforce_closed_object_schema(node: Any) -> Any:
        """Ensure all JSON Schema object nodes declare additionalProperties=false."""
        if isinstance(node, dict):
            is_object = node.get("type") == "object" or "properties" in node
            if is_object and "additionalProperties" not in node:
                node["additionalProperties"] = False
            for key, value in list(node.items()):
                node[key] = BaseAgent._enforce_closed_object_schema(value)
            return node
        if isinstance(node, list):
            return [BaseAgent._enforce_closed_object_schema(item) for item in node]
        return node

    @staticmethod
    def _render_system_prompt(system_prompt: str, input_text: str) -> str:
        """Inject input text into prompt templates that use {input_text}."""
        if "{input_text}" in system_prompt:
            return system_prompt.format_map(_SafeFormatDict(input_text=input_text))
        return system_prompt

    @staticmethod
    def _extract_json_payload(content: str) -> dict[str, Any]:
        """Extract a JSON object from model output."""
        if not content:
            raise ValueError("Model returned empty response while JSON was expected.")

        text = content.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))

        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            return json.loads(text[first_brace:last_brace + 1])

        raise ValueError(f"Could not parse JSON object from model response: {text}")

    @staticmethod
    def _recover_corrected_sentence(content: str) -> str:
        """Best-effort recovery for malformed single-field JSON outputs."""
        text = (content or "").strip()
        if not text:
            return ""

        # Common case: malformed JSON containing quoted chunks.
        quoted = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', text, flags=re.DOTALL)
        if quoted:
            if quoted[0] == "corrected_sentence":
                parts = [p for p in quoted[1:] if p.strip()]
                if parts:
                    return "".join(parts).strip()
            return quoted[-1].strip()

        # Fallback to plain text cleanup.
        cleaned = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        cleaned = cleaned.replace("{", "").replace("}", "")
        cleaned = cleaned.replace("corrected_sentence", "")
        cleaned = cleaned.replace(":", " ").strip(" \n\t\"',")
        return cleaned.strip()

    @staticmethod
    def _coerce_payload_for_response_model(
        payload: dict[str, Any],
        response_model: Type[BaseModel],
    ) -> dict[str, Any]:
        """Normalize common provider/model JSON variants into the expected schema."""
        if set(response_model.model_fields.keys()) == {"SENTENCES"} and "SENTENCES" not in payload:
            sentence_entries: list[dict[str, Any]] = []
            for key, value in payload.items():
                match = re.fullmatch(r"SENTENCE[_\s-]?(\d+)", str(key), flags=re.IGNORECASE)
                if match:
                    sentence_entries.append(
                        {
                            "SENTENCE_ID": int(match.group(1)),
                            "CORRECTED_SENTENCE": str(value),
                        }
                    )
            if sentence_entries:
                sentence_entries.sort(key=lambda item: item["SENTENCE_ID"])
                return {"SENTENCES": sentence_entries}
        return payload

    @staticmethod
    def _ensure_json_object_instruction(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ensure at least one message explicitly asks for a JSON object response."""
        patched_messages: list[dict[str, Any]] = []
        instruction = "Return a valid JSON object only."
        found_json_word = False

        for message in messages:
            cloned = dict(message)
            content = cloned.get("content")
            if isinstance(content, str) and "json" in content.lower():
                found_json_word = True
            patched_messages.append(cloned)

        if found_json_word:
            return patched_messages

        for idx, message in enumerate(patched_messages):
            if message.get("role") == "system" and isinstance(message.get("content"), str):
                patched_messages[idx] = dict(message)
                patched_messages[idx]["content"] = f"{message['content'].rstrip()}\n\n{instruction}"
                return patched_messages

        patched_messages.append({"role": "system", "content": instruction})
        return patched_messages

    def batch_execute(self, input_texts: list[str], **kwargs) -> list[BaseModel]:
        """
        Execute the agent on multiple input texts.

        Args:
            input_texts: List of input texts to process
            **kwargs: Additional parameters for the API call

        Returns:
            List of structured responses
        """
        return [self.execute(text, **kwargs) for text in input_texts]
