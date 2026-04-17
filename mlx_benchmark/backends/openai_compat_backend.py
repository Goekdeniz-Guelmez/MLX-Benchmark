"""OpenAI-compatible backend — works with OpenAI, Groq, OpenRouter, and any
OpenAI-compatible API."""

from .base import Backend


class OpenAICompatBackend(Backend):
    """Backend for any OpenAI-compatible chat API (OpenAI, Groq, OpenRouter, etc.)."""

    PROVIDER_BASE_URLS = {
        "openai": "https://api.openai.com/v1",
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }

    def __init__(
        self,
        model: str,
        provider: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        from openai import OpenAI

        self._model = model
        self._provider = provider

        resolved_base_url = base_url or self.PROVIDER_BASE_URLS.get(provider)
        self._client = OpenAI(api_key=api_key, base_url=resolved_base_url)

    @property
    def name(self) -> str:
        return f"{self._provider}/{self._model}"

    def generate(self, prompt: str, system: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content