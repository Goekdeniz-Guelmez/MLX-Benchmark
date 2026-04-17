"""Base class for LLM backends."""

from abc import ABC, abstractmethod


class Backend(ABC):
    """Abstract interface for an LLM inference backend."""

    @abstractmethod
    def generate(self, prompt: str, system: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: User message content.
            system: System prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The generated text.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name, e.g. 'ollama/llama3.2'."""
        ...

    def __repr__(self) -> str:
        return f"<Backend {self.name}>"