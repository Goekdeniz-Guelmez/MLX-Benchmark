"""Ollama backend — local LLM inference via Ollama."""

from .base import Backend


class OllamaBackend(Backend):
    """Backend that calls a locally-running Ollama model."""

    def __init__(self, model: str, host: str | None = None):
        import ollama

        self._model = model
        self._client = ollama.Client(host=host) if host else ollama.Client()

        # Ensure the model is available
        try:
            self._client.show(model)
        except Exception:
            print(f"Model '{model}' not found locally. Pulling...")
            self._client.pull(model)
            print(f"Model '{model}' pulled successfully.")

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    def generate(self, prompt: str, system: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        )
        return response["message"]["content"]