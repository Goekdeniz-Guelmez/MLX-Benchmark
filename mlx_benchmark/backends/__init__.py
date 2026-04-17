"""Backend factory and registry."""

from .base import Backend
from .ollama_backend import OllamaBackend
from .anthropic_backend import AnthropicBackend
from .openai_compat_backend import OpenAICompatBackend


def create_backend(
    model: str,
    provider: str = "ollama",
    api_key: str | None = None,
    base_url: str | None = None,
    host: str | None = None,
) -> Backend:
    """Create a backend instance.

    Args:
        model: Model identifier (e.g. 'llama3.2', 'claude-sonnet-4-20250514').
        provider: One of 'ollama', 'anthropic', 'openai', 'groq', 'openrouter'.
        api_key: API key (for cloud providers). Falls back to env vars.
        base_url: Custom base URL (overrides provider default).
        host: Ollama host URL (e.g. 'http://localhost:11434').

    Returns:
        A Backend instance.
    """
    if provider == "ollama":
        return OllamaBackend(model=model, host=host)
    elif provider == "anthropic":
        return AnthropicBackend(model=model, api_key=api_key)
    elif provider in ("openai", "groq", "openrouter"):
        return OpenAICompatBackend(model=model, provider=provider, api_key=api_key, base_url=base_url)
    else:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            f"Choose from: ollama, anthropic, openai, groq, openrouter"
        )


__all__ = ["Backend", "OllamaBackend", "AnthropicBackend", "OpenAICompatBackend", "create_backend"]