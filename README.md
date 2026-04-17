# MLX Benchmark

Benchmark LLMs on Apple MLX framework knowledge and coding tasks.

## Install

```bash
pip install mlx-benchmark
```

For cloud provider support, install extras:

```bash
pip install "mlx-benchmark[anthropic]"   # Claude models
pip install "mlx-benchmark[openai]"       # OpenAI, Groq, OpenRouter
pip install "mlx-benchmark[all]"          # All providers
pip install "mlx-benchmark[plot]"         # PNG chart export (matplotlib)
```

## Quick Start

Benchmark a local Ollama model:

```bash
mlx-bench --model llama3.2
```

Benchmark multiple models sequentially:

```bash
mlx-bench --model llama3.2,mistral,qwen2.5-coder
```

Use a stronger model as judge:

```bash
mlx-bench --model llama3.2 --judge-model gemma4
```

## Cloud Providers

```bash
# Anthropic Claude
mlx-bench --provider anthropic --model claude-sonnet-4-20250514

# OpenAI
mlx-bench --provider openai --model gpt-4o

# OpenRouter (access to many models)
mlx-bench --provider openrouter --model anthropic/claude-sonnet-4-20250514

# Groq (fast inference)
mlx-bench --provider groq --model llama-3.2-70b-versatile
```

API keys are read from environment variables:

| Provider    | Environment Variable |
|-------------|---------------------|
| Anthropic   | `ANTHROPIC_API_KEY` |
| OpenAI      | `OPENAI_API_KEY`    |
| Groq        | `GROQ_API_KEY`      |
| OpenRouter  | `OPEN_ROUTER_API_KEY` |

## Filtering

```bash
# Only coding and debug questions
mlx-bench --model llama3.2 --types coding debug

# Only hard questions
mlx-bench --model llama3.2 --difficulties hard

# Specific categories
mlx-bench --model llama3.2 --categories mlx_core mlx_nn

# Quick test run (10 samples)
mlx-bench --model llama3.2 --limit 10
```

## Export

Generate LaTeX tables and PNG charts from benchmark results:

```bash
# Export LaTeX + PNG from all results in ./results/
mlx-bench --latex --plot

# Export only LaTeX from specific result files
mlx-bench --latex --results results/bench_ollama_llama3-2_*.json

# Run benchmark and also generate exports
mlx-bench --config models.yml --latex --plot
```

Outputs:
- `bench_results.tex` — two `booktabs` tables (accuracy by difficulty, accuracy by type)
- `bench_results.png` — grouped bar chart comparing models

> The `--plot` flag requires matplotlib. Install with `pip install mlx-benchmark[plot]`.

## All Options

```
--model MODEL          Model name (comma-separated for multiple)
--provider PROVIDER    ollama | anthropic | openai | groq | openrouter
--judge-model MODEL   Judge model (default: same as --model)
--judge-provider PROVIDER  Judge provider (default: same as --provider)
--dataset PATH         Custom dataset JSONL (default: bundled v2)
--output-dir DIR       Where to save results (default: ./results)
--max-tokens N         Max response tokens (default: 1024)
--temperature T        Sampling temperature (default: 0.0)
--limit N              Limit number of samples
--categories [...]     Filter by category
--difficulties [...]   Filter by difficulty (easy, medium, hard, very-hard)
--types [...]          Filter by type (qa, fill_blank, mcq, true_false, coding, debug)
--rate-limit SECS      Delay between API calls (default: 0.5)
--host URL             Ollama host (default: http://localhost:11434)
--api-key KEY          API key for cloud providers
--base-url URL         Custom base URL for OpenAI-compatible APIs
--latex                Generate LaTeX table from results
--plot                 Generate PNG bar chart from results
--results [FILES ...]  Result JSON files to export (default: all in --output-dir)
```

## Python API

```python
from mlx_benchmark import run_benchmark

results, stats = run_benchmark(
    model="llama3.2",
    provider="ollama",
    limit=20,
    types=["coding", "debug"],
)

print(f"Accuracy: {stats.accuracy:.1f}%")
```

Export results programmatically:

```python
from mlx_benchmark import load_result_files, generate_latex_table, generate_plot

data = load_result_files(["results/bench_ollama_llama3-2_20260412.json"])
latex = generate_latex_table(data, output_dir="results")
generate_plot(data, output_dir="results")
```

## Output

Results are saved as JSON files in the output directory with:

- Per-question scores (correct/incorrect)
- Aggregate accuracy by type, difficulty, and category
- Model answers alongside reference answers for review

Example result filename: `bench_ollama_llama3-2_20260412_220855.json`

## Dataset

The bundled dataset (`dataset_v2.jsonl`) contains 441 questions across 6 types:

| Type          | Description                                  |
|---------------|----------------------------------------------|
| `qa`          | Knowledge questions about MLX APIs          |
| `mcq`         | Multiple choice                              |
| `true_false`  | True/false statements                        |
| `fill_blank`  | Code completion tasks                        |
| `coding`      | Full code writing tasks                      |
| `debug`       | Identify and fix bugs in MLX code            |

Covering 11 categories and 4 difficulty levels (easy, medium, hard, very-hard).

## License

MIT