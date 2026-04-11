from __future__ import annotations

import json

import structlog
from ollama import AsyncClient

from src.config import OllamaConfig
from src.llm.prompts import CLASSIFY_PROMPT, EXTRACT_KNOWLEDGE_PROMPT

log = structlog.get_logger()


class OllamaClient:
    """Ollama client with structured JSON output and retry logic."""

    def __init__(self, config: OllamaConfig):
        self.client = AsyncClient(host=config.base_url)
        self.model = config.model
        self.timeout = config.timeout
        self.default_temperature = config.temperature

    async def generate_json(self, prompt: str, temperature: float | None = None) -> dict:
        """
        Generate structured JSON output from the SLM.
        Retries once on JSON parse failure with a stricter prompt suffix.
        """
        temp = temperature if temperature is not None else self.default_temperature

        # First attempt
        try:
            response = await self.client.generate(
                model=self.model,
                prompt=prompt,
                format="json",
                options={"temperature": temp},
            )
            text = response.get("response", "")
            return json.loads(text)
        except (json.JSONDecodeError, KeyError):
            log.warning("ollama_json_parse_failed_first_attempt")

        # Second attempt with stricter prompt
        strict_prompt = prompt + "\n\nIMPORTANT: Return ONLY valid JSON, no other text."
        try:
            response = await self.client.generate(
                model=self.model,
                prompt=strict_prompt,
                format="json",
                options={"temperature": temp},
            )
            text = response.get("response", "")
            return json.loads(text)
        except (json.JSONDecodeError, KeyError):
            log.error("ollama_json_parse_failed_second_attempt")
            raise

    async def classify_message(self, content: str, categories_block: str) -> dict:
        """Classify a message against categories. Returns parsed result."""
        prompt = CLASSIFY_PROMPT.format(
            categories_block=categories_block,
            message_content=content,
        )
        return await self.generate_json(prompt, temperature=0.1)

    async def extract_knowledge(self, content: str) -> dict:
        """Extract entity relationship triples from message content."""
        prompt = EXTRACT_KNOWLEDGE_PROMPT.format(message_content=content)
        return await self.generate_json(prompt, temperature=0.1)

    async def check_health(self) -> bool:
        """Check if Ollama is reachable and the model is available."""
        try:
            models = await self.client.list()
            model_names = [m.get("name", "") for m in models.get("models", [])]
            return any(self.model in name for name in model_names)
        except Exception:
            return False
