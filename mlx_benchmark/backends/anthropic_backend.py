"""Anthropic backend — Claude models via the Anthropic API."""

from .base import Backend


class AnthropicBackend(Backend):
    """Backend that calls Anthropic's Claude models."""

    def __init__(self, model: str, api_key: str | None = None):
        import anthropic

        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return f"anthropic/{self._model}"

    def generate(self, prompt: str, system: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text