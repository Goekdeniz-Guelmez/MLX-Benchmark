"""MLX Benchmark — evaluate LLMs on Apple MLX knowledge and coding tasks."""

from .backends import Backend, OllamaBackend, AnthropicBackend, OpenAICompatBackend, create_backend
from .dataset import load_dataset, filter_dataset
from .export import generate_latex_table, generate_plot, load_result_files, find_result_files
from .judge import evaluate
from .runner import run_benchmark, BenchResult, BenchStats
from .version import __version__

__all__ = [
    "Backend",
    "OllamaBackend",
    "AnthropicBackend",
    "OpenAICompatBackend",
    "create_backend",
    "load_dataset",
    "filter_dataset",
    "evaluate",
    "run_benchmark",
    "BenchResult",
    "BenchStats",
    "generate_latex_table",
    "generate_plot",
    "load_result_files",
    "find_result_files",
]